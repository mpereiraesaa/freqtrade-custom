"""
Volume PairList provider

Provides dynamic pair list based on trade volumes
"""
import random
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import talib.abstract as ta

from freqtrade.persistence import Trade
from freqtrade.exceptions import OperationalException
from freqtrade.pairlist.IPairList import IPairList
from freqtrade.data.converter import ohlcv_to_dataframe
from technical.indicators import fibonacci_retracements
pd.options.mode.chained_assignment = None
logger = logging.getLogger(__name__)

SORT_VALUES = ['askVolume', 'bidVolume', 'quoteVolume']

class BestPairList(IPairList):

    def __init__(self, exchange, pairlistmanager,
                 config: Dict[str, Any], pairlistconfig: Dict[str, Any],
                 pairlist_pos: int, regr: Any) -> None:
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
        self.refresh_period = 0.5*60*60
        self.timeframe = config['ticker_interval']
        self.regr = regr

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
            since_day_ms = int(datetime.timestamp(datetime.now() - timedelta(hours=48)) * 1000) # MS

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
            pairlist = pairlist[:130]

            pairlist = self.verify_blacklist(pairlist, logger.info)

            # Fetch already worked out Pairs today.
            pairs_today = Trade.get_closed_pairs_today()
            pairlist = [x for x in pairlist if x not in pairs_today]

            best_pairs = []
            for pair in pairlist:
                day_data = self._exchange.get_historic_ohlcv(pair=pair, timeframe='1h', since_ms=since_day_ms)
                ohlcv_hourly = ohlcv_to_dataframe(day_data, '1h', pair, fill_missing=False, drop_incomplete=False)
                # consolidation_ohlcv = ohlcv_hourly[-15:]

                # Find only those pairs within safe ranges during hours.
                # max_close = consolidation_ohlcv['close'].max()
                # min_close = consolidation_ohlcv['close'].min()
                # threshold = 1 - (6.5 / 100)
                # if min_close < (max_close * threshold):
                #     continue

                ohlcv_hourly['returns'] = ohlcv_hourly['close'].pct_change()
                returns_df = ohlcv_hourly[ohlcv_hourly['returns'].notnull()]
                returns_df.sort_values(by=['returns'], ascending=True, inplace=True)

                self.log_on_refresh(logger.info, f"{pair} 99% conf level VaR: {returns_df['returns'].quantile(0.01)}")

                # VaR ratio 99% confidence level. Possible losses must be lower than -0.03 to stay safe.
                if returns_df['returns'].quantile(0.01) < -0.03:
                    continue

                new_data = self._exchange.get_historic_ohlcv(pair=pair, timeframe=self.timeframe, since_ms=since_ms)
                ohlcv = ohlcv_to_dataframe(new_data, self.timeframe, pair, fill_missing=False, drop_incomplete=True)

                if len(ohlcv) > 0:
                    ohlcv['rsi'] = ta.RSI(ohlcv)
                    ohlcv['atr'] = ta.ATR(ohlcv['high'], ohlcv['low'], ohlcv['close'])
                    ohlcv = ohlcv[(ohlcv['rsi'].notnull()) | (ohlcv['atr'].notnull())]

                    ohlcv["pct_change"] = ohlcv['close'].pct_change()
                    ohlcv['atr_rank'] = ohlcv['atr'].rank(pct=True)
                    ohlcv['vol_rank'] = ohlcv['volume'].rank(pct=True)

                    count = 0
                    profitable = 0
                    ohlcv.dropna(inplace=True)
                    ohlcv.reset_index(drop=True, inplace=True)

                    buy_signal = [0] * len(ohlcv)
                    for i in range(6, len(ohlcv), 1):
                        # Retrieve features
                        pct_change = ohlcv['pct_change'].iloc[i-5:i+1].values
                        atr_rank = ohlcv['atr_rank'].iloc[i]
                        rsi = ohlcv['rsi'].iloc[i]/100
                        input_data = np.append(pct_change, [rsi, atr_rank])
                        input_data = input_data.reshape(-1, 8)

                        # Pass input to predictor
                        predict_threshold = self.regr.predict(input_data)[0][0] > 0.009
                        buy_signal[i] = 1 if predict_threshold else 0
                        ohlcv['buy_signal'] = buy_signal

                    sell_price = None
                    buy_date = None
                    for index, row in ohlcv.iterrows():
                        if sell_price is None:
                            if row['buy_signal'] == 1:
                                buy_date = row['date']
                                sell_price = row['close'] * 1.008 # 1% profit.
                                count += 1
                        else:
                            minutes_passed = (row['date'] - buy_date)  / timedelta(minutes=1)
                            if minutes_passed > 10 and row['close'] >= sell_price:
                                sell_price = None
                                buy_date = None
                                profitable += 1
                    best_pairs.append({
                        "pair": pair,
                        "count": count,
                        "profitable": profitable,
                        "percentage": (profitable/count)*100 if count != 0 else 0,
                        "rsi": ohlcv['rsi'].values[-1]
                    })

            best_pairs = DataFrame(best_pairs)
            best_pairs = best_pairs[best_pairs['percentage'] > 80]
            best_pairs.sort_values(by=['profitable'], ascending=False, inplace=True)

            self.log_on_refresh(logger.info, f"Predictive power: {best_pairs[:17]['percentage'].mean()}")

            best_pairs = best_pairs[best_pairs['profitable'] > 2]
            best_pairs = best_pairs[:15]

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
