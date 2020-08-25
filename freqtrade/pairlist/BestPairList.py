"""
Volume PairList provider

Provides dynamic pair list based on trade volumes
"""
import random
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List
from pandas import DataFrame, Series
import numpy as np
import talib.abstract as ta

from freqtrade.exceptions import OperationalException
from freqtrade.pairlist.IPairList import IPairList
from freqtrade.data.converter import ohlcv_to_dataframe
from technical.indicators import fibonacci_retracements

logger = logging.getLogger(__name__)

SORT_VALUES = ['askVolume', 'bidVolume', 'quoteVolume']

class BestPairList(IPairList):

    def __init__(self, exchange, pairlistmanager,
                 config: Dict[str, Any], pairlistconfig: Dict[str, Any],
                 pairlist_pos: int, prices_model: List[float]) -> None:
        super().__init__(exchange, pairlistmanager, config, pairlistconfig, pairlist_pos)

        if 'number_assets' not in self._pairlistconfig:
            raise OperationalException(
                '`number_assets` not specified. Please check your configuration '
                'for "pairlist.config.number_assets"')

        self._pairlistmanager = pairlistmanager
        self._stake_currency = config['stake_currency']
        self._number_pairs = np.random.randint(10, 18)
        self._sort_key = self._pairlistconfig.get('sort_key', 'quoteVolume')
        self._min_value = self._pairlistconfig.get('min_value', 0)
        self.refresh_period = 0.3*60*60
        self.timeframe = config['ticker_interval']
        self.prices_model = prices_model

        if not self._exchange.exchange_has('fetchTickers'):
            raise OperationalException(
                'Exchange does not support dynamic whitelist. '
                'Please edit your config and restart the bot.'
            )

        if not self._validate_keys(self._sort_key):
            raise OperationalException(
                f'key {self._sort_key} not in {SORT_VALUES}')

        if self._sort_key != 'quoteVolume':
            logger.warning(
                "DEPRECATED: using any key other than quoteVolume for BestPairList is deprecated."
                )

    @property
    def needstickers(self) -> bool:
        """
        Boolean property defining if tickers are necessary.
        If no Pairlist requries tickers, an empty List is passed
        as tickers argument to filter_pairlist
        """
        return True

    def _validate_keys(self, key):
        return key in SORT_VALUES

    def short_desc(self) -> str:
        """
        Short whitelist method description - used for startup-messages
        """
        return f"{self.name} - top {self._pairlistconfig['number_assets']} volume pairs."

    def gen_pairlist(self, cached_pairlist: List[str], tickers: Dict) -> List[str]:
        """
        Generate the pairlist
        :param cached_pairlist: Previously generated pairlist (cached)
        :param tickers: Tickers (from exchange.get_tickers()).
        :return: List of pairs
        """
        # Generate dynamic whitelist
        if self._last_refresh + self.refresh_period < datetime.now().timestamp():
            self._last_refresh = int(datetime.now().timestamp())
            since_ms = int(datetime.timestamp(datetime.now() - timedelta(days=2)) * 1000) # MS

            # Use fresh pairlist
            # Check if pair quote currency equals to the stake currency.
            pairlist = []
            tickers = self._exchange._api.fetch_tickers()
            markets = self._exchange._api.load_markets()
            for trading_pair in markets:
                if markets[trading_pair]['active'] is True:
                    quote = trading_pair.split('/')[1]
                    if quote == 'USDT':
                        pairlist.append(trading_pair)

            # Take only those with greater volume
            pairlist = sorted(pairlist, reverse=True, key=lambda pair: tickers[pair]["quoteVolume"])
            pairlist = pairlist[:110]

            print(f"Available pairs: {len(pairlist)}\n")

            print(f"Available price model: {self.prices_model}")

            best_pairs = []
            for pair in pairlist:
                new_data = self._exchange.get_historic_ohlcv(pair=pair, timeframe=self.timeframe, since_ms=since_ms)
                ohlcv = ohlcv_to_dataframe(new_data, self.timeframe, pair, fill_missing=False, drop_incomplete=True)
                count = 0
                profitable = 0
                if len(ohlcv) > 0:
                    ohlcv['rsi'] = ta.RSI(ohlcv)
                    buy_signal = [0] * len(ohlcv)
                    for i in range(6, len(ohlcv), 1):
                        coeff = np.corrcoef(ohlcv[i-6:i]['close'].values, self.prices_model)[1][0]
                        price_coeff = coeff > 0.80
                        buy_signal[i] = 1 if price_coeff and ohlcv.iloc[i]['rsi'] < 30 else 0
                    ohlcv['buy'] = buy_signal

                    sell_price = None
                    last_index = None
                    for index, row in ohlcv.iterrows():
                        if sell_price is None:
                            if row['buy'] == 1:
                                last_index = index
                                sell_price = row['close'] * 1.01 # 1% profit.
                                count += 1
                        else:
                            # More than just one candle have passed.
                            if index > last_index + 1 and row['close'] >= sell_price:
                                sell_price = None
                                last_index = None
                                profitable += 1
                best_pairs.append({
                    "pair": pair,
                    "count": count,
                    "profitable": profitable,
                    "percentage": (profitable/count)*100 if count != 0 else 0,
                    "rsi": ohlcv['rsi'].values[-1]
                })

            best_pairs = DataFrame(best_pairs)
            best_pairs.sort_values(by=['count'], ascending=False, inplace=True)
            best_pairs = best_pairs[best_pairs['percentage'] > 75]

            # 15 are the ones with most chances in last two days.
            best_pairs = best_pairs[:20]

            best_pairs = best_pairs[best_pairs['rsi'] < 40]
            pairlist = best_pairs['pair'].values.tolist()
            if len(pairlist) == 0:
                pairlist = cached_pairlist
        else:
            # Use the cached pairlist if it's not time yet to refresh
            pairlist = cached_pairlist

        return pairlist

    def filter_pairlist(self, pairlist: List[str], tickers: Dict) -> List[str]:
        """
        Filters and sorts pairlist and returns the whitelist again.
        Called on each bot iteration - please use internal caching if necessary
        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers()). May be cached.
        :return: new whitelist
        """
        # Validate whitelist to only have active market pairs
        pairs = self.verify_blacklist(pairlist, logger.info)

        self.log_on_refresh(logger.info, f"Searching {self._number_pairs} pairs: {pairs}")

        return pairs

    def verify_blacklist(self, pairlist: List[str], logmethod) -> List[str]:
        """
        Proxy method to verify_blacklist for easy access for child classes.
        :param pairlist: Pairlist to validate
        :param logmethod: Function that'll be called, `logger.info` or `logger.warning`.
        :return: pairlist - blacklisted pairs
        """
        return self._pairlistmanager.verify_blacklist(pairlist, logmethod)
