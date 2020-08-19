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
                 pairlist_pos: int) -> None:
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
        self.refresh_period = 0.5*60*60 # in seconds = 2 hours
        self.timeframe = config['ticker_interval']

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

            # Seek for the right pairs accordign to the next logic.
            pairs_performance = []
            for pair in pairlist:
                new_data = self._exchange.get_historic_ohlcv(pair=pair, timeframe=self.timeframe, since_ms=since_ms)
                ohlcv = ohlcv_to_dataframe(new_data, self.timeframe, pair, fill_missing=False, drop_incomplete=True)
                profits = 0
                count = 0
                average_pct_changes = 0
                if len(ohlcv) > 0:
                    ohlcv['atr'] = ta.ATR(ohlcv['high'], ohlcv['low'], ohlcv['close'])
                    ohlcv['fibonacci'] = fibonacci_retracements(ohlcv)
                    ohlcv['rsi'] = ta.RSI(ohlcv)
                    ohlcv["pct_change"] = ohlcv['close'].pct_change()
                    ohlcv["signal"] = np.where(ohlcv.rsi < 30, 1, 0)

                    for index, row in ohlcv.iterrows():
                        if row['signal'] == 1:
                            count += 1
                            pct_changes = 0
                            for i in range(1,5):
                                if index + 1 >= len(ohlcv):
                                    break
                                pct_changes += ohlcv.iloc[index + i]['pct_change']
                            average_pct_changes = pct_changes / 4
                            if average_pct_changes > 0:
                                profits += 1

                    rsi_down_trend = Series(ohlcv['rsi'].values[-5:]).is_monotonic_decreasing

                    pairs_performance.append({
                        "pair": pair,
                        "profits": profits,
                        "count": count,
                        "pattern_prob": np.round((profits/count)*100, 2) if count != 0 else -1,
                        "average_change": average_pct_changes,
                        "rsi_is_downtrend": rsi_down_trend,
                        "rsi": ohlcv["rsi"].values[-1],
                        "fibonacci": np.mean(ohlcv['fibonacci'].values[-2:])
                    })

            cols = ['pair', 'profits', 'count', 'pattern_prob', 'average_change', 'rsi_is_downtrend', 'rsi', 'fibonacci']
            best_pairs = DataFrame(pairs_performance, columns=cols)
            best_pairs.sort_values(by=['pattern_prob', 'count'], ascending=False, inplace=True)
            # Top 40 with more probability of having a pattern with RSI and support levels.
            best_pairs = best_pairs.head(40)
            best_pairs = best_pairs[best_pairs['average_change'] > 0]
            best_pairs = best_pairs[best_pairs['fibonacci'] <= 0.5]
            best_pairs = best_pairs[best_pairs['rsi'] <= 35]
            best_pairs = best_pairs[best_pairs['rsi_is_downtrend'] == True]
            pairlist = best_pairs['pair'].values.tolist()
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
