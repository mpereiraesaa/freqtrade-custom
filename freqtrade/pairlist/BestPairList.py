"""
Volume PairList provider

Provides dynamic pair list based on trade volumes
"""
import random
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List
from pandas import DataFrame
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

        self._stake_currency = config['stake_currency']
        self._number_pairs = np.random.randint(10, 18)
        self._sort_key = self._pairlistconfig.get('sort_key', 'quoteVolume')
        self._min_value = self._pairlistconfig.get('min_value', 0)
        self.refresh_period = 7200 # seconds
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
            for trading_pair in self._exchange._api.load_markets():
                quote = trading_pair.split('/')[1]
                if quote == 'USDT':
                    pairlist.append(trading_pair)

            # Seek for the right pairs accordign to the next logic.
            pairs_performance = []
            for pair in pairlist:
                new_data = self._exchange.get_historic_ohlcv(pair=pair, timeframe=self.timeframe, since_ms=since_ms)
                ohlcv = ohlcv_to_dataframe(new_data, self.timeframe, pair,
                fill_missing=False, drop_incomplete=True)
                if len(ohlcv) > 0:
                    ohlcv['rate_change_%'] = ta.ROCP(ohlcv['close'], timeperiod=1)
                    ohlcv['atr'] = ta.ATR(ohlcv['high'], ohlcv['low'], ohlcv['close'], timeperiod=1)
                    ohlcv['fibonacci'] = fibonacci_retracements(ohlcv)

                    avg_rate_change = np.mean(ohlcv['rate_change_%'])
                    avg_atr = np.mean(ohlcv['atr'])

                    pairs_performance.append({
                        "pair": pair,
                        "avg_rate_change": avg_rate_change,
                        "avg_atr": avg_atr,
                        "last_fibonacci": ohlcv['fibonacci'].values[-1]
                    })

            best_pairs = DataFrame(pairs_performance, columns=['pair', 'avg_rate_change', 'avg_atr', 'last_fibonacci'])
            best_pairs = best_pairs[best_pairs['avg_rate_change'] > 0]
            best_pairs = best_pairs[best_pairs['last_fibonacci'] <= 0.618]
            best_pairs = best_pairs[best_pairs['avg_atr'] <= 2]
            best_pairs = best_pairs[best_pairs['avg_atr'] >= 0.0005]
            best_pairs.sort_values('avg_atr', ascending=False, inplace=True)
            best_pairs = best_pairs[:30]
            # Top 30 by Volatility
            best_pairs.sort_values('avg_rate_change', ascending=False, inplace=True)
            # Top 18 by positive rate of change
            best_pairs = best_pairs[:18]
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
