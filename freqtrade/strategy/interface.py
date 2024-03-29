"""
IStrategy interface
This module defines the interface to apply for strategies
"""
import logging
import warnings
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, NamedTuple, Optional, Tuple, List, Any

import pickle
import rapidjson
import talib as ta
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

import arrow
from pandas import DataFrame, read_csv

from freqtrade.data.dataprovider import DataProvider
from freqtrade.exceptions import StrategyError
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.persistence import Trade
from freqtrade.constants import ListPairsWithTimeframes, USERPATH_MODELS, USERPATH_TWEETS
from freqtrade.wallets import Wallets

diff = lambda x, y: x - y
abs_diff = lambda x, y: abs(x - y)

indicators = [
    ('RSI', ta.RSI, ['close']),
    ('ATR', ta.ATR, ['high', 'low', 'close']),
    ('MFI', ta.MFI, ['high', 'low', 'close', 'volume']),
    ('ULTOSC', ta.ULTOSC, ['high', 'low', 'close']),
    ('WILLR', ta.WILLR, ['high', 'low', 'close']),
    ('CCI', ta.CCI, ['high', 'low', 'close']),
    ('AD', ta.AD, ['high', 'low', 'close', 'volume']),
    ('ADOSC', ta.ADOSC, ['high', 'low', 'close', 'volume']),       
    ('TRIX', ta.TRIX, ['close']),
    ('PPO', ta.PPO, ['close']),
    ('CCI', ta.CCI, ['high', 'low', 'close']),
    ('ADX', ta.ADX, ['high', 'low', 'close']),
    ('OBV', ta.OBV, ['close', 'volume']),
    ('CDLENGULFING', ta.CDLENGULFING, ['open', 'high', 'low', 'close']),
    ('CDLHIKKAKE', ta.CDLHIKKAKE, ['open', 'high', 'low', 'close']),
    ('CDLHIKKAKEMOD', ta.CDLHIKKAKEMOD, ['open', 'high', 'low', 'close']),
    ('CDLBELTHOLD', ta.CDLBELTHOLD, ['open', 'high', 'low', 'close']),
    ('CDLEVENINGDOJISTAR', ta.CDLEVENINGDOJISTAR, ['open', 'high', 'low', 'close']),
    ('CDLEVENINGSTAR', ta.CDLEVENINGSTAR, ['open', 'high', 'low', 'close']),
    ('CDLLONGLINE', ta.CDLLONGLINE, ['open', 'high', 'low', 'close']),
    ('CDLCLOSINGMARUBOZU', ta.CDLCLOSINGMARUBOZU, ['open', 'high', 'low', 'close']),
    ('CDLSHORTLINE', ta.CDL3OUTSIDE, ['open', 'high', 'low', 'close']),
    ('CDLDOJI', ta.CDLDOJI, ['open', 'high', 'low', 'close']),
    ('CDLHARAMICROSS', ta.CDLHARAMICROSS, ['open', 'high', 'low', 'close'])        
]

features = ['open', 'high', 'low', 'close', 'volume', 'sentiment',
       'tw_count', 'RSI', 'ATR', 'MFI', 'ULTOSC', 'WILLR', 'CCI', 'AD',
       'ADOSC', 'TRIX', 'PPO', 'ADX', 'OBV', 'CDLENGULFING', 'CDLHIKKAKE',
       'CDLHIKKAKEMOD', 'CDLBELTHOLD', 'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR',
       'CDLLONGLINE', 'CDLCLOSINGMARUBOZU', 'CDLSHORTLINE', 'CDLDOJI',
       'CDLHARAMICROSS', 'macd_diff',
       'hband_distance', 'lband_distance', 'up_bband', 'down_bband',
        'pct_change', 'ret_lag_1', 'ret_lag_2', 'ret_lag_3', 'ret_lag_4', 'ret_lag_5']
N = 5

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """
    Enum to distinguish between buy and sell signals
    """
    BUY = "buy"
    SELL = "sell"


class SellType(Enum):
    """
    Enum to distinguish between sell reasons
    """
    ROI = "roi"
    STOP_LOSS = "stop_loss"
    STOPLOSS_ON_EXCHANGE = "stoploss_on_exchange"
    TRAILING_STOP_LOSS = "trailing_stop_loss"
    SELL_SIGNAL = "sell_signal"
    FORCE_SELL = "force_sell"
    EMERGENCY_SELL = "emergency_sell"
    NONE = ""


class SellCheckTuple(NamedTuple):
    """
    NamedTuple for Sell type + reason
    """
    sell_flag: bool
    sell_type: SellType

class IStrategy:
    """
    Interface for freqtrade strategies
    Defines the mandatory structure must follow any custom strategies

    Attributes you can use:
        minimal_roi -> Dict: Minimal ROI designed for the strategy
        stoploss -> float: optimal stoploss designed for the strategy
        ticker_interval -> str: value of the timeframe (ticker interval) to use with the strategy
    """
    # Strategy interface version
    # Default to version 2
    # Version 1 is the initial interface without metadata dict
    # Version 2 populate_* include metadata dict
    INTERFACE_VERSION: int = 2

    _populate_fun_len: int = 0
    _buy_fun_len: int = 0
    _sell_fun_len: int = 0
    # associated minimal roi
    minimal_roi: Dict

    # associated stoploss
    stoploss: float = -0.999

    # trailing stoploss
    trailing_stop: bool = False
    trailing_stop_positive: Optional[float] = None
    trailing_stop_positive_offset: float = 0.0
    trailing_only_offset_is_reached = False

    # associated ticker interval
    ticker_interval: str = '5m'

    # Optional order types
    order_types: Dict = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False,
        'stoploss_on_exchange_interval': 60,
    }

    # Optional time in force
    order_time_in_force: Dict = {
        'buy': 'gtc',
        'sell': 'gtc',
    }

    # run "populate_indicators" only for new candle
    process_only_new_candles: bool = False

    # Disable checking the dataframe (converts the error into a warning message)
    disable_dataframe_checks: bool = False

    # Count of candles the strategy requires before producing valid signals
    startup_candle_count: int = 0

    # Class level variables (intentional) containing
    # the dataprovider (dp) (access to other candles, historic data, ...)
    # and wallets - access to the current balance.
    dp: Optional[DataProvider] = None
    wallets: Optional[Wallets] = None

    # Definition of plot_config. See plotting documentation for more details.
    plot_config: Dict = {}

    def __init__(self, config: dict) -> None:
        self.config = config
        # Dict to determine if analysis is necessary
        self._last_candle_seen_per_pair: Dict[str, datetime] = {}
        self._pair_locked_until: Dict[str, datetime] = {}

    def add_indicators(self, df) -> pd.DataFrame:
        for name, f, arg_names in indicators:
            wrapper = lambda func, args: func(*args)
            args = [df[arg_name] for arg_name in arg_names]
            df[name] = wrapper(f, args)
            
        macd, macdsignal, macdhist = ta.MACD(df['close'])
        df['macd_diff'] = macd - macdsignal
        upperband, middleband, lowerband = ta.BBANDS(df['close'])
        df['hband_distance'] = upperband - df['close']
        df['lband_distance'] = df['close'] - lowerband
        df['up_bband'] = np.where(df['close'] > upperband, 1, 0)
        df['down_bband'] = np.where(df['close'] < lowerband, 1, 0)
        df['target'] = np.where(((df.close - df.close.shift(1)) / df.close) > 0.0005, 1, 0)
        df['pct_change'] = df['close'].pct_change()  
        
        df.fillna(method='bfill', inplace=True)
        return df

    # Add lagged indicators and target feature.
    def add_lagged_indicators(self, df):
        new_df = []
        for index, row in df.iterrows():
            curr_price = row['close']
            # Take some Up-Trend from next 6 candles
            max_next_price = df[index+1:index+7]['close'].max()
            future_roc = (max_next_price-curr_price)/curr_price
            agg_lags = {}
            for i in range(1, N+1):
                agg_lags[f"ret_lag_{i}"] = df['pct_change'].iloc[index-i]
            if future_roc > 0.005:
                agg_lags["target"] = 1
            else:
                agg_lags["target"] = 0
            new_df.append({ **dict(row), **agg_lags })
        return DataFrame(new_df)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate indicators that will be used in the Buy and Sell strategy
        :param dataframe: DataFrame with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        df_with_indicators = self.add_indicators(dataframe)
        df = self.add_lagged_indicators(df_with_indicators)

        return df

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        pair = str(metadata.get('pair'))
        symbol = pair.split('/')[0]
        tweets_metadata = None
        clf = None
        pca = None

        # Load twitter json
        with open(f"{self.config['user_data_dir']}/{USERPATH_TWEETS}/{symbol}.json", 'rb') as file:
            tweets_metadata = rapidjson.load(file)

        # Load Pair ML Model
        with open(f"{self.config['user_data_dir']}/{USERPATH_MODELS}/{symbol}.pkl", 'rb') as file:
            clf = pickle.load(file)

        # Load Pair PCA Model
        with open(f"{self.config['user_data_dir']}/{USERPATH_MODELS}/pca-{symbol}.pkl", 'rb') as file:
            pca = pickle.load(file)

        # Add twitter data
        dataframe['sentiment'] = tweets_metadata['sentiment']
        dataframe['tw_count'] = tweets_metadata['tw_count']

        # Scale data for ML
        scaler = MinMaxScaler()
        scaled_df = DataFrame(scaler.fit_transform(dataframe[features]), columns=[features])

        pca_output = pca.transform(scaled_df)

        # Get last index.
        x = pca_output[-1].reshape(-1, pca_output.shape[1])

        buy = clf.predict(x)[0]

        logger.info(f"Pair {str(pair)} X.shape {str(x.shape)} predicted value: {str(buy)} x: {str(x)}")

        pred_column = [0] * len(dataframe)
        pred_column[-1] = buy

        dataframe['buy'] = pred_column

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with sell column
        """
        dataframe['sell'] = 0

        return dataframe

    def check_buy_timeout(self, pair: str, trade: Trade, order: dict, **kwargs) -> bool:
        """
        Check buy timeout function callback.
        This method can be used to override the buy-timeout.
        It is called whenever a limit buy order has been created,
        and is not yet fully filled.
        Configuration options in `unfilledtimeout` will be verified before this,
        so ensure to set these timeouts high enough.

        When not implemented by a strategy, this simply returns False.
        :param pair: Pair the trade is for
        :param trade: trade object.
        :param order: Order dictionary as returned from CCXT.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True is returned, then the buy-order is cancelled.
        """
        return False

    def check_sell_timeout(self, pair: str, trade: Trade, order: dict, **kwargs) -> bool:
        """
        Check sell timeout function callback.
        This method can be used to override the sell-timeout.
        It is called whenever a limit sell order has been created,
        and is not yet fully filled.
        Configuration options in `unfilledtimeout` will be verified before this,
        so ensure to set these timeouts high enough.

        When not implemented by a strategy, this simply returns False.
        :param pair: Pair the trade is for
        :param trade: trade object.
        :param order: Order dictionary as returned from CCXT.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True is returned, then the sell-order is cancelled.
        """
        return False

    def informative_pairs(self) -> ListPairsWithTimeframes:
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def get_strategy_name(self) -> str:
        """
        Returns strategy class name
        """
        return self.__class__.__name__

    def lock_pair(self, pair: str, until: datetime) -> None:
        """
        Locks pair until a given timestamp happens.
        Locked pairs are not analyzed, and are prevented from opening new trades.
        Locks can only count up (allowing users to lock pairs for a longer period of time).
        To remove a lock from a pair, use `unlock_pair()`
        :param pair: Pair to lock
        :param until: datetime in UTC until the pair should be blocked from opening new trades.
                Needs to be timezone aware `datetime.now(timezone.utc)`
        """
        if pair not in self._pair_locked_until or self._pair_locked_until[pair] < until:
            self._pair_locked_until[pair] = until

    def unlock_pair(self, pair: str) -> None:
        """
        Unlocks a pair previously locked using lock_pair.
        Not used by freqtrade itself, but intended to be used if users lock pairs
        manually from within the strategy, to allow an easy way to unlock pairs.
        :param pair: Unlock pair to allow trading again
        """
        if pair in self._pair_locked_until:
            del self._pair_locked_until[pair]

    def is_pair_locked(self, pair: str) -> bool:
        """
        Checks if a pair is currently locked
        """
        if pair not in self._pair_locked_until:
            return False
        return self._pair_locked_until[pair] >= datetime.now(timezone.utc)

    def analyze_ticker(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Parses the given candle (OHLCV) data and returns a populated DataFrame
        add several TA indicators and buy signal to it
        :param dataframe: Dataframe containing data from exchange
        :param metadata: Metadata dictionary with additional data (e.g. 'pair')
        :return: DataFrame of candle (OHLCV) data with indicator data and signals added
        """
        logger.debug("TA Analysis Launched")
        dataframe = self.advise_indicators(dataframe, metadata)
        dataframe = self.advise_buy(dataframe, metadata)
        dataframe = self.advise_sell(dataframe, metadata)
        return dataframe

    def _analyze_ticker_internal(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Parses the given candle (OHLCV) data and returns a populated DataFrame
        add several TA indicators and buy signal to it
        WARNING: Used internally only, may skip analysis if `process_only_new_candles` is set.
        :param dataframe: Dataframe containing data from exchange
        :param metadata: Metadata dictionary with additional data (e.g. 'pair')
        :return: DataFrame of candle (OHLCV) data with indicator data and signals added
        """
        pair = str(metadata.get('pair'))

        # Test if seen this pair and last candle before.
        # always run if process_only_new_candles is set to false
        if (not self.process_only_new_candles or
                self._last_candle_seen_per_pair.get(pair, None) != dataframe.iloc[-1]['date']):
            # Defs that only make change on new candle data.
            dataframe = self.analyze_ticker(dataframe, metadata)
            self._last_candle_seen_per_pair[pair] = dataframe.iloc[-1]['date']
        else:
            logger.debug("Skipping TA Analysis for already analyzed candle")
            dataframe['buy'] = 0
            dataframe['sell'] = 0

        # Other Defs in strategy that want to be called every loop here
        # twitter_sell = self.watch_twitter_feed(dataframe, metadata)
        logger.debug("Loop Analysis Launched")

        return dataframe

    @staticmethod
    def preserve_df(dataframe: DataFrame) -> Tuple[int, float, datetime]:
        """ keep some data for dataframes """
        return len(dataframe), dataframe["close"].iloc[-1], dataframe["date"].iloc[-1]

    def assert_df(self, dataframe: DataFrame, df_len: int, df_close: float, df_date: datetime):
        """ make sure data is unmodified """
        message = ""
        if df_len != len(dataframe):
            message = "length"
        elif df_close != dataframe["close"].iloc[-1]:
            message = "last close price"
        elif df_date != dataframe["date"].iloc[-1]:
            message = "last date"
        if message:
            if self.disable_dataframe_checks:
                logger.warning(f"Dataframe returned from strategy has mismatching {message}.")
            else:
                raise StrategyError(f"Dataframe returned from strategy has mismatching {message}.")

    def get_signal(self, pair: str, interval: str, dataframe: DataFrame) -> Tuple[bool, bool]:
        """
        Calculates current signal based several technical analysis indicators
        :param pair: pair in format ANT/BTC
        :param interval: Interval to use (in min)
        :param dataframe: Dataframe to analyze
        :return: (Buy, Sell) A bool-tuple indicating buy/sell signal
        """
        if not isinstance(dataframe, DataFrame) or dataframe.empty:
            logger.warning('Empty candle (OHLCV) data for pair %s', pair)
            return False, False

        try:
            df_len, df_close, df_date = self.preserve_df(dataframe)
            dataframe = self._analyze_ticker_internal(dataframe, {'pair': pair})
            self.assert_df(dataframe, df_len, df_close, df_date)
        except StrategyError as error:
            logger.warning(f"Unable to analyze candle (OHLCV) data for pair {pair}: {error}")

            return False, False

        if dataframe.empty:
            logger.warning('Empty dataframe for pair %s', pair)
            return False, False

        latest_date = dataframe['date'].max()
        latest = dataframe.loc[dataframe['date'] == latest_date].iloc[-1]
        # Explicitly convert to arrow object to ensure the below comparison does not fail
        latest_date = arrow.get(latest_date)

        # Check if dataframe is out of date
        interval_minutes = timeframe_to_minutes(interval)
        offset = self.config.get('exchange', {}).get('outdated_offset', 5)
        if latest_date < (arrow.utcnow().shift(minutes=-(interval_minutes * 2 + offset))):
            logger.warning(
                'Outdated history for pair %s. Last tick is %s minutes old',
                pair,
                (arrow.utcnow() - latest_date).seconds // 60
            )
            return False, False

        (buy, sell) = latest[SignalType.BUY.value] == 1, latest[SignalType.SELL.value] == 1

        logger.debug(
            'trigger: %s (pair=%s) buy=%s sell=%s',
            latest['date'],
            pair,
            str(buy),
            str(sell)
        )
        return buy, sell

    def should_sell(self, trade: Trade, rate: float, date: datetime, buy: bool,
                    sell: bool, low: float = None, high: float = None,
                    force_stoploss: float = 0) -> SellCheckTuple:
        """
        This function evaluates if one of the conditions required to trigger a sell
        has been reached, which can either be a stop-loss, ROI or sell-signal.
        :param low: Only used during backtesting to simulate stoploss
        :param high: Only used during backtesting, to simulate ROI
        :param force_stoploss: Externally provided stoploss
        :return: True if trade should be sold, False otherwise
        """
        # Set current rate to low for backtesting sell
        # current_rate = low or rate
        # current_profit = trade.calc_profit_ratio(current_rate)

        # trade.adjust_min_max_rates(high or current_rate)

        # stoplossflag = self.stop_loss_reached(current_rate=current_rate, trade=trade,
        #                                       current_time=date, current_profit=current_profit,
        #                                       force_stoploss=force_stoploss, high=high)

        # if stoplossflag.sell_flag:
        #     logger.debug(f"{trade.pair} - Stoploss hit. sell_flag=True, "
        #                  f"sell_type={stoplossflag.sell_type}")
        #     return stoplossflag

        # Set current rate to high for backtesting sell
        # config_ask_strategy = self.config.get('ask_strategy', {})
        current_rate = high or rate
        current_profit = trade.calc_profit_ratio(current_rate)

        if self.roi_reached(trade=trade, current_profit=current_profit, current_time=date):
            logger.debug(f"{trade.pair} - Required profit reached. profit={current_profit}")
            return SellCheckTuple(sell_flag=True, sell_type=SellType.ROI)

        # if buy and config_ask_strategy.get('ignore_roi_if_buy_signal', False):
        #     # This one is noisy, commented out
        #     # logger.debug(f"{trade.pair} - Buy signal still active. sell_flag=False")
        #     return SellCheckTuple(sell_flag=False, sell_type=SellType.NONE)

        # # Check if minimal roi has been reached and no longer in buy conditions (avoiding a fee)
        # if self.min_roi_reached(trade=trade, current_profit=current_profit, current_time=date):
        #     logger.debug(f"{trade.pair} - Required profit reached. sell_flag=True, "
        #                  f"sell_type=SellType.ROI")
        #     return SellCheckTuple(sell_flag=True, sell_type=SellType.ROI)

        # if config_ask_strategy.get('sell_profit_only', False):
        #     # This one is noisy, commented out
        #     # logger.debug(f"{trade.pair} - Checking if trade is profitable...")
        #     if trade.calc_profit(rate=rate) <= 0:
        #         # This one is noisy, commented out
        #         # logger.debug(f"{trade.pair} - Trade is not profitable. sell_flag=False")
        #         return SellCheckTuple(sell_flag=False, sell_type=SellType.NONE)

        # if sell and not buy and config_ask_strategy.get('use_sell_signal', True):
        #     logger.debug(f"{trade.pair} - Sell signal received. sell_flag=True, "
        #                  f"sell_type=SellType.SELL_SIGNAL")
        #     return SellCheckTuple(sell_flag=True, sell_type=SellType.SELL_SIGNAL)

        # This one is noisy, commented out...
        # logger.debug(f"{trade.pair} - No sell signal. sell_flag=False")
        return SellCheckTuple(sell_flag=False, sell_type=SellType.NONE)

    def stop_loss_reached(self, current_rate: float, trade: Trade,
                          current_time: datetime, current_profit: float,
                          force_stoploss: float, high: float = None) -> SellCheckTuple:
        """
        Based on current profit of the trade and configured (trailing) stoploss,
        decides to sell or not
        :param current_profit: current profit as ratio
        """
        stop_loss_value = force_stoploss if force_stoploss else self.stoploss

        # Initiate stoploss with open_rate. Does nothing if stoploss is already set.
        trade.adjust_stop_loss(trade.open_rate, stop_loss_value, initial=True)

        if self.trailing_stop:
            # trailing stoploss handling
            sl_offset = self.trailing_stop_positive_offset

            # Make sure current_profit is calculated using high for backtesting.
            high_profit = current_profit if not high else trade.calc_profit_ratio(high)

            # Don't update stoploss if trailing_only_offset_is_reached is true.
            if not (self.trailing_only_offset_is_reached and high_profit < sl_offset):
                # Specific handling for trailing_stop_positive
                if self.trailing_stop_positive is not None and high_profit > sl_offset:
                    stop_loss_value = self.trailing_stop_positive
                    logger.debug(f"{trade.pair} - Using positive stoploss: {stop_loss_value} "
                                 f"offset: {sl_offset:.4g} profit: {current_profit:.4f}%")

                trade.adjust_stop_loss(high or current_rate, stop_loss_value)

        # evaluate if the stoploss was hit if stoploss is not on exchange
        # in Dry-Run, this handles stoploss logic as well, as the logic will not be different to
        # regular stoploss handling.
        if ((self.stoploss is not None) and
            (trade.stop_loss >= current_rate) and
                (not self.order_types.get('stoploss_on_exchange') or self.config['dry_run'])):

            sell_type = SellType.STOP_LOSS

            # If initial stoploss is not the same as current one then it is trailing.
            if trade.initial_stop_loss != trade.stop_loss:
                sell_type = SellType.TRAILING_STOP_LOSS
                logger.debug(
                    f"{trade.pair} - HIT STOP: current price at {current_rate:.6f}, "
                    f"stoploss is {trade.stop_loss:.6f}, "
                    f"initial stoploss was at {trade.initial_stop_loss:.6f}, "
                    f"trade opened at {trade.open_rate:.6f}")
                logger.debug(f"{trade.pair} - Trailing stop saved "
                             f"{trade.stop_loss - trade.initial_stop_loss:.6f}")

            return SellCheckTuple(sell_flag=True, sell_type=sell_type)

        return SellCheckTuple(sell_flag=False, sell_type=SellType.NONE)

    def min_roi_reached_entry(self, trade_dur: int) -> Tuple[Optional[int], Optional[float]]:
        """
        Based on trade duration defines the ROI entry that may have been reached.
        :param trade_dur: trade duration in minutes
        :return: minimal ROI entry value or None if none proper ROI entry was found.
        """
        # Get highest entry in ROI dict where key <= trade-duration
        roi_list = list(filter(lambda x: x <= trade_dur, self.minimal_roi.keys()))
        if not roi_list:
            return None, None
        roi_entry = max(roi_list)
        return roi_entry, self.minimal_roi[roi_entry]

    def roi_reached(self, trade: Trade, current_profit: float, current_time: datetime) -> bool:
        """
        Based on trade duration, current profit of the trade and ROI configuration,
        decides whether bot should sell.
        :param current_profit: current profit as ratio
        :return: True if bot should sell at current rate
        """
        # Check if time matches and current rate is above threshold
        trade_dur = int((current_time.timestamp() - trade.open_date.timestamp()) // 60)

        # 10 minutes (Convert to UTC-3) have passed and we seek profits greater than 0.7%
        sell = (trade_dur - 180) >= 10 and current_profit > 0.007

        # if (trade_dur - 180) > 1400 and current_profit > -0.02:
        #     sell = True

        return sell

    def min_roi_reached(self, trade: Trade, current_profit: float, current_time: datetime) -> bool:
        """
        Based on trade duration, current profit of the trade and ROI configuration,
        decides whether bot should sell.
        :param current_profit: current profit as ratio
        :return: True if bot should sell at current rate
        """
        # Check if time matches and current rate is above threshold
        trade_dur = int((current_time.timestamp() - trade.open_date.timestamp()) // 60)
        _, roi = self.min_roi_reached_entry(trade_dur)
        if roi is None:
            return False
        else:
            return current_profit > roi

    def ohlcvdata_to_dataframe(self, data: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
        """
        Creates a dataframe and populates indicators for given candle (OHLCV) data
        Used by optimize operations only, not during dry / live runs.
        Using .copy() to get a fresh copy of the dataframe for every strategy run.
        Has positive effects on memory usage for whatever reason - also when
        using only one strategy.
        """
        return {pair: self.advise_indicators(pair_data.copy(), {'pair': pair})
                for pair, pair_data in data.items()}

    def advise_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate indicators that will be used in the Machine learning model
        This method should not be overridden.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        logger.debug(f"Populating indicators for pair {metadata.get('pair')}.")

        return self.populate_indicators(dataframe, metadata)

    def advise_buy(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        This method should not be overridden.
        :param dataframe: DataFrame
        :param pair: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        logger.debug(f"Populating buy signals for pair {metadata.get('pair')}.")
        
        return self.populate_buy_trend(dataframe, metadata)  # type: ignore

    def advise_sell(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        This method should not be overridden.
        :param dataframe: DataFrame
        :param pair: Additional information, like the currently traded pair
        :return: DataFrame with sell column
        """
        logger.debug(f"Populating fake sell signals for pair {metadata.get('pair')}.")

        return self.populate_sell_trend(dataframe, metadata)
