import asyncio
import datetime
import sys

import nest_asyncio
from typing import List, Optional, Tuple, Dict, Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas_ta import rsi, macd, supertrend, ema, vwap, atr
import logging

from utils import (minutes_to_secs, days_to_secs, filter_by_array, get_curr_utc_2_timestamp, get_percent,
                   send_email_ele, send_email_all, reindex_df)
from consts import (DEFAULT_RES, LONG_STOCK_NAME, MACD_INDEX, MACD_SIGNAL_INDEX, SellStatus, CRITERIA, LOGGER_NAME,
                    STOP_LOSS_RANGE, TAKE_PROFIT_MULTIPLIER, SUPERTREND_COL_NAME, DEFAULT_RISK_UNIT,
                    DEFAULT_RISK_LIMIT, DEFAULT_START_CAPITAL, DEFAULT_CRITERIA_LIST, DEFAULT_USE_PYRAMID,
                    DEFAULT_GROWTH_PERCENT, DEFAULT_RU_PERCENTAGE, GAIN, LOSS, EMA_LENGTH, STOP_LOSS_PERCENTAGE_MARGIN,
                    SHORT_STOCK_NAME, STOP_LOSS_LOWER_BOUND, TRADE_NOT_COMPLETE, OUTPUT_PLOT, FILTER_STOCKS,
                    RUN_ROBOT, USE_RUN_WINS, RUN_WINS_TAKE_PROFIT_MULTIPLIER, RUN_WINS_PERCENT, TRADE_COMPLETE,
                    MACD_PARAMS, SUPERTREND_PARAMS, RSI_PARAMS, N_FIRST_CANDLES_OF_DAY, N_LAST_CANDLES_OF_DAY,
                    REAL_TIME, SELL_ON_TOUCH, ALWAYS_BUY, CANDLE_DATA_CSV_NAME, TRADE_DATA_CSV_NAME, VIX, DEBUG,
                    TRADE_SUMMARY_CSV_NAME, REAL_TIME_PERIOD, HISTORY_PERIOD, PYR_RISK_UNIT_CALCULATION_PERIOD,
                    USE_DL_MODEL, CLASSIFIER_THRESH_MIN,CLASSIFIER_THRESH_MAX, MACD_H_INDEX, ATR_MUL, ATR_PERIOD,
                    DATA_STOCKS_TO_TRADED_MAPPING)
from stock_client import StockClient
from dl_utils.data_generator import DataGenerator, RawDataSample
from dl_utils.fc_classifier import FcClassifier

AMOUNT = 0
NUM_STOCKS = 1
STOCK_PRICE = 2


def plot_total_gain_percentage(gains):
    xy = (np.random.random((10, 2))).cumsum(axis=0)
    bought = [0, 2]
    fig, ax = plt.subplots()
    for i, (start, stop) in enumerate(zip(xy[:-1], xy[1:])):
        x, y = zip(start, stop)
        ax.plot(x, y, color='black' if i in bought else 'red')
    plt.show()


class StockBot:
    def __init__(self, stock_clients: List[StockClient], start: int = None, end: int = None, period: str = None, rsi_win_size: int = 10,
                 ema_win_size: int = 10, risk_unit: float = None, risk_limit: float = None, start_capital: float = None,
                 use_pyr: bool = DEFAULT_USE_PYRAMID, ru_growth: float = None, monitor_revenue: bool = False, criteria: Optional[List[str]] = None,
                 log_to_file=False, tp_multiplier=TAKE_PROFIT_MULTIPLIER, sl_bound=STOP_LOSS_LOWER_BOUND, run_wins=USE_RUN_WINS,
                 rw_take_profit_multiplier=RUN_WINS_TAKE_PROFIT_MULTIPLIER, rw_percent=RUN_WINS_PERCENT, vix_client=None,
                 enable_history_log: bool = True, real_time: bool = REAL_TIME, model: Optional[FcClassifier] = None,
                 scalar = None, data_generator: Optional[DataGenerator] = None):
        self.clients = stock_clients
        self.vix_client: StockClient = vix_client
        self.gain_avg = None
        self.loss_avg = None
        self.resolution = DEFAULT_RES
        self.start_time = start
        self.end_time = end
        self.period = period
        
        self.enable_history_log = enable_history_log

        self.take_profit_multiplier = tp_multiplier
        self.stop_loss_bound = sl_bound

        self.ru_percentage = DEFAULT_RU_PERCENTAGE
        self.risk_unit = self.initial_risk_unit = risk_unit if risk_unit else DEFAULT_RISK_UNIT
        self.risk_limit = risk_limit if risk_limit else DEFAULT_RISK_LIMIT
        self.risk_growth = ru_growth if ru_growth else DEFAULT_GROWTH_PERCENT

        self.monitor_revenue = monitor_revenue
        self.capital = self.initial_capital = start_capital if start_capital else DEFAULT_START_CAPITAL

        self.latest_trade: List[List[int]] = [[-1 for i in range(3)] for i in range(len(self.clients))]  # sum paid, number of stock bought, stock price on latest buy trade

        self.use_pyramid = use_pyr

        # these attributes will be used when determining whether to sell
        self.stop_loss = None
        self.take_profit = None
        self.status = [SellStatus.NEITHER for i in range(len(self.clients))]

        self.data_changed = [True for i in range(len(self.clients))]  # indicates whether new candle data was fetched or not
        self.criteria_indices = [None for i in range(len(self.clients))]

        self.set_candle_data()

        # "let the wins run" strategy parameters
        self.rw = run_wins
        self.num_wins_thresh = 1
        self.rw_num_times_sold = 0
        self.rw_take_profit_multiplier = rw_take_profit_multiplier
        self.rw_percent = rw_percent

        num_candles = self.clients[0].get_num_candles()
        self.capital_history = np.zeros((2, num_candles))
        self.capital_history[:, 0] = [0, self.capital]
        self.gains = np.zeros(num_candles)
        self.eod_gains = np.zeros(num_candles)
        self.losses = np.zeros(num_candles)
        self.eod_losses = np.zeros(num_candles)
        self.num_gains = self.num_eod_gains = self.num_losses = self.num_eod_losses = 0
        self.trades_per_day = np.full(num_candles, -1)

        # we use this in order to check that criteria was not fulfilled at least 1 time between buy trades
        self.block_buy = False

        # this dict maps a criterion to the method that returns the candle indices that fulfil it
        self.criteria_func_data_mapping: Dict[CRITERIA, Callable[[int, int], np.ndarray]] =\
                                                            {
                                                                CRITERIA.RSI: self.get_rsi_criterion,
                                                                CRITERIA.SUPER_TREND: self.get_supertrend_criterion,
                                                                CRITERIA.MACD: self.get_macd_criterion,
                                                                CRITERIA.INSIDE_BAR: self.get_inside_bar_criterion,
                                                                CRITERIA.REVERSAL_BAR: self.get_reversal_bar_criterion
                                                            }

        self.criteria: List[CRITERIA] = []
        self.set_criteria(criteria)

        # Define the log file name and the log formatting, including the date format
        self.log_file = f'log_{datetime.datetime.now().strftime("%m_%d_%Y_%H-%M-%S")}.txt'
        self.log_format = '%(asctime)s - [%(name)s]    - [%(levelname)s] - %(message)s'
        self.date_format = '%Y-%m-%d %H:%M:%S'
        self.formatter = logging.Formatter(self.log_format, datefmt=self.date_format)
        # set log formatting
        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setFormatter(self.formatter)

        # Create a logger object
        self.logger = logging.getLogger('StockBot')
        self.logger.setLevel(logging.INFO)

        self.real_time = real_time
        self.sell_on_touch = SELL_ON_TOUCH

        # Add the FileHandler object to the logger object
        if self.real_time:
            if DEBUG:
                self.logger.addHandler(logging.StreamHandler(sys.stderr))
            else:
                # Create a FileHandler object and set the log file name
                self.file_handler = logging.FileHandler(self.log_file)
                self.file_handler.setFormatter(self.formatter)
                self.logger.addHandler(self.file_handler)
            self.logger.info("Real time mode")
        elif self.enable_history_log:
            self.logger.addHandler(logging.StreamHandler(sys.stderr))
            self.logger.info("History collection mode")

        # check which strategy is used is True and log it
        if self.is_bar_strategy() and (self.real_time or self.enable_history_log):
            self.logger.info("StockBot initialized with bar strategy")
        elif self.real_time or self.enable_history_log:
            self.logger.info("StockBot initialized with candle strategy")

        self.main_loop_sleep_time = 60

        # create dataframes to store runtime analysis candle data and trade data
        index_cols = ['date', 'name']
        cols = ['high', 'low', 'close', 'ema200', 'ema100', 'ema50', 'ema45', 'ema40', 'ema35',
                'ema30', 'ema25', 'ema20']
        if self.is_bar_strategy():
            cols += ['RSI', 'MACD', 'MACD_SIGNAL', 'MACDh', 'SUPERTREND']
        if not self.real_time:
            cols += ['vwap', VIX]

        self.data_collection_df = pd.DataFrame(columns=index_cols+cols)
        # used to store candle data for specific client before concatenation
        # to data_collection_df that contains data for all clients
        self.temp_client_candle_df = pd.DataFrame(columns=index_cols+cols)

        reindex_df(self.data_collection_df, ['date', 'name'])
        self.trades_df = pd.DataFrame(
            columns=index_cols + ['number', 'type', 'price', 'amount', 'capital', 'sell type'] + cols)
        self.curr_data_entry = {}

        self.logger.setLevel(logging.INFO)
        
        # these attributes are used for prediction using a DL classification model
        self.dl_model = model
        self.dl_scalar = scalar
        self.dl_data_generator = data_generator
        self.model_thresh_min = CLASSIFIER_THRESH_MIN
        self.model_thresh_max = CLASSIFIER_THRESH_MAX
        self.atr_mul = ATR_MUL
        self.atr_period = ATR_PERIOD
        self.num_candles_in_trend = 0  # number of candles in a continous trend slope
        self.percentage_diff = 0
        self.accumulated_percentage_diff = 0

    def _reset_temp_candle_df(self):
        self.temp_client_candle_df = self.temp_client_candle_df.iloc[0:0]

    def flush_candle_data_entry(self):
        """
        append a row to the self.data_collection_df- a dataframe that collects candle data
        """
        self.data_collection_df.loc[(self.curr_data_entry['date'], self.curr_data_entry['name']), :] =\
            [self.curr_data_entry.get(col, '') for col in self.data_collection_df.columns]
        self.data_collection_df.to_csv(CANDLE_DATA_CSV_NAME)  # flush to disk to save progress
        self.curr_data_entry = {}

    def flush_trade_entry(self, trade_entry: Dict):
        """
        adds a new row to the trade report
        """
        try:
            candle_data: Dict = self.data_collection_df.loc[(trade_entry['date'], trade_entry['name']), :].iloc[0].to_dict()
            row: Dict = dict(trade_entry, **candle_data)
            self.trades_df = self.trades_df.append(row, ignore_index=True)
            if self.real_time:  # for history mode, we write the csv to disk at the end of the run
                self.trades_df.to_csv(TRADE_DATA_CSV_NAME, index=False)  # flush to disk to save progress
        except KeyError:  # in case candle data is not found
            return

    def get_trade_close_price(self, client_index: int, index=-1):
        # get close price from trade stock
        closing_price = self.clients[client_index].get_trade_closing_price()
        return closing_price.iloc[index]

    def get_close_price(self, client_index: int, index=-1):
        # get close price from data stock
        closing_price = self.clients[client_index].get_closing_price()
        return closing_price.iloc[index]

    def get_open_price(self, client_index: int, index=-1):
        open_price = self.clients[client_index].get_opening_price()
        return open_price.iloc[index]

    def get_low_price(self, client_index: int, index=-1):
        low_price = self.clients[client_index].get_low_prices()
        return low_price.iloc[index]

    def get_high_price(self, client_index: int, index=-1):
        high_price = self.clients[client_index].get_high_prices()
        return high_price.iloc[index]

    def set_tp_multiplier(self, val: float):
        self.take_profit_multiplier = val

    def reset(self):
        self.num_gains = self.num_eod_gains = self.num_losses = self.num_eod_losses = 0
        self.status = [SellStatus.NEITHER for i in range(len(self.clients))]
        self.capital = self.initial_capital
        self.trades_per_day = np.full((self.clients[0].get_num_candles()), -1)

    # @lru_cache(maxsize=1)
    def set_candle_data(self):
        if self.period:
            for c in self.clients:
                c.set_candle_data(res=self.resolution, start=self.start_time, end=self.end_time, period=period)
            if self.vix_client is not None:
                self.vix_client.set_candle_data(res=self.resolution, start=self.start_time, end=self.end_time, period=period)
        for i in range(len(self.data_changed)):
            self.data_changed[i] = True

    def set_criteria(self, criteria: Optional[List[str]]) -> None:
        if criteria is None:
            self.criteria.append(CRITERIA.RSI)
            self.criteria.append(CRITERIA.MACD)
            self.criteria.append(CRITERIA.SUPER_TREND)
            return
        else:
            for name in criteria:
                c = CRITERIA.factory(name)
                self.criteria.append(c)

    def get_rsi_criterion(self, client_index: int, candle_index: int) -> np.ndarray:
        s_close_old = pd.Series(self.clients[client_index].get_closing_price())

        length = RSI_PARAMS

        rsi_results = np.array(rsi(s_close_old, length=length).round(2))

        if self.real_time:
            self.curr_data_entry['RSI'] = rsi_results[candle_index]
            if not self.use_dl_model:
                self.logger.info(f"\t\t[+] RSI condition is [{self.curr_data_entry['RSI'] > 50}]"
                                f" with [{self.curr_data_entry['RSI']} {'>' if self.curr_data_entry['RSI'] > 50 else '<'} 50]")
        else:
            self.temp_client_candle_df['RSI'] = rsi_results.tolist()

        return np.where(rsi_results > 50)

    def get_supertrend_criterion(self, client_index: int, candle_index: int) -> np.ndarray:
        high = self.clients[client_index].get_high_prices()
        low = self.clients[client_index].get_low_prices()
        close = self.clients[client_index].get_closing_price()

        length, multiplier = SUPERTREND_PARAMS

        supertrend_results_df = supertrend(pd.Series(high), pd.Series(low), pd.Series(close),
                                           length=length, multiplier=multiplier)
        supertrend_results = supertrend_results_df[SUPERTREND_COL_NAME].round(2)

        if self.real_time:
            self.curr_data_entry['SUPERTREND'] = supertrend_results[candle_index]
            if not self.use_dl_model:
                self.logger.info(f"\t\t[+] SUPERTREND condition is [{self.curr_data_entry['SUPERTREND'] < close[candle_index]}]"
                                f" with [{self.curr_data_entry['SUPERTREND']} (SUPERTREND) "
                                f"{'>' if self.curr_data_entry['SUPERTREND'] > close[candle_index] else '<'}"
                                f" {close[candle_index]} (close_price)]")
        else:
            self.temp_client_candle_df['SUPERTREND'] = supertrend_results.tolist()

        indices = np.where(supertrend_results < close)

        return np.array(indices)

    def get_macd_criterion(self, client_index: int, candle_index: int) -> np.ndarray:
        close_prices = self.clients[client_index].get_closing_price()

        fast, slow, signal = MACD_PARAMS
        macd_close = macd(close=pd.Series(close_prices), fast=fast, slow=slow, signal=signal)
        macd_data = np.array(macd_close.iloc[:, MACD_INDEX].round(2))
        signal_data = np.array(macd_close.iloc[:, MACD_SIGNAL_INDEX].round(2))
        macd_h_data = np.array(macd_close.iloc[:, MACD_H_INDEX].round(2))

        if self.real_time:
            self.curr_data_entry['MACD'] = macd_data[candle_index]
            self.curr_data_entry['MACD_SIGNAL'] = signal_data[candle_index]
            self.curr_data_entry['MACDh'] = macd_h_data[candle_index]
            if not self.use_dl_model:
                self.logger.info(
                    f"\t\t[+] First condition [{self.curr_data_entry['MACD'] > self.curr_data_entry['MACD_SIGNAL']}]"
                    f" with [{self.curr_data_entry['MACD']} (MACD) "
                    f"{'>' if self.curr_data_entry['MACD'] > self.curr_data_entry['MACD_SIGNAL'] else '<'}"
                    f" {self.curr_data_entry['MACD_SIGNAL']} (MACD_SIGNAL)]")
                self.logger.info(
                    f"\t\t[+] Second condition is [{self.curr_data_entry['MACD'] < 0}]"
                    f" with [{self.curr_data_entry['MACD']} (MACD) "
                    f"{'>' if self.curr_data_entry['MACD'] > 0 else '<'} 0]")
        else:
            self.temp_client_candle_df['MACD'] = macd_data.tolist()
            self.temp_client_candle_df['MACD_SIGNAL'] = signal_data.tolist()
            self.temp_client_candle_df['MACDh'] = macd_h_data.tolist()

        indices = np.where((macd_data > signal_data) & (macd_data < 0))
        indices_as_nd = np.array(indices)

        return indices_as_nd

    def get_ema(self, client_index: int, length=EMA_LENGTH):
        ema_data = ema(self.clients[client_index].get_closing_price(), length).round(2)
        return ema_data

    def get_vwap(self, client_index: int):
        high = self.clients[client_index].get_high_prices()
        low = self.clients[client_index].get_low_prices()
        close = self.clients[client_index].get_closing_price()
        volume = self.clients[client_index].get_volume()
        vwap_data = vwap(high, low, close, volume)
        return vwap_data

    def get_inside_bar_criterion(self, client_index: int, candle_index: int) -> np.ndarray:
        close_prices = self.clients[client_index].get_closing_price()
        open_prices = self.clients[client_index].get_opening_price()
        high_prices = self.clients[client_index].get_high_prices()
        low_prices = self.clients[client_index].get_low_prices()
        if self.real_time or self.enable_history_log:
            self.logger.info("calculating EMA")
        ema = self.get_ema(client_index)

        positive_candles = close_prices[:-1] - open_prices[:-1]
        high_price_diff = np.diff(high_prices)
        # 1st candle positive + 2nd candle in 1st candle price range [open, close]
        cond_1 = np.where((positive_candles > 0) & (high_price_diff < 0) & (np.diff(low_prices) > 0))[0] + 1
        # 3rd candle close bigger than both 1st candle close and ema
        cond_2 = np.where((close_prices < np.roll(close_prices, -2)) & (np.roll(close_prices, -2) > np.roll(ema, -2)))[0] + 1

        # candle 1 stop loss

        indices = np.intersect1d(cond_1, cond_2) + 2

        return indices

    def get_reversal_bar_criterion(self, client_index: int, candle_index: int) -> np.ndarray:
        close_prices = self.clients[client_index].get_closing_price()
        high_prices = self.clients[client_index].get_high_prices()
        low_prices = self.clients[client_index].get_low_prices()
        if self.real_time or self.enable_history_log:
            self.logger.info("calculating EMA")
        ema = self.get_ema(client_index)

        # 1st candle high+low bigger than 2nd candle high+low
        cond_1 = np.where((np.diff(high_prices) < 0) & (np.diff(low_prices) < 0))[0] + 1
        # 3rd candle close bigger than both 1st candle high and ema
        cond_2 = np.where((high_prices < np.roll(close_prices, -2)) & (np.roll(close_prices, -2) > np.roll(ema, -2)))[0] + 1
        cond_3 = np.where((low_prices < np.roll(low_prices, -1)))

        indices = np.intersect1d(cond_1, np.intersect1d(cond_2, cond_3)) + 2

        return indices

    def get_num_candles(self, client_index: int = 0):
        """
        returns the number of candles that are currently observed
        """
        return len(self.clients[client_index].get_closing_price())

    def _is_criteria(self, client_index: int, candle_index: int, cond_operation='&') -> np.ndarray:
        """
        :param cond_operation: '&' or '|'. Indicates the operation that will be performed on the given criteria.
        """
        indices = np.arange(self.get_num_candles(client_index)) if cond_operation == '&' else np.empty(0)
        for c in self.criteria:
            if self.real_time or self.enable_history_log:
                self.logger.info("Calculating criteria: {}".format(c))
            callback: Callable[[int, int], np.ndarray] = self.criteria_func_data_mapping[c]
            criterion_indices = callback(client_index, candle_index)
            if cond_operation == '&':
                indices = filter_by_array(indices, criterion_indices)
            else:
                criteria_union = np.concatenate((indices, criterion_indices))
                indices = np.unique(criteria_union)

        self.criteria_indices[client_index] = indices
        self.data_changed[client_index] = False  # criteria are now updated according to the latest change

        return indices

    def is_buy(self, client_index: int, index: int = None) -> bool:
        if self.real_time:
            self.logger.info("Performing is buy check in real time")
        if ALWAYS_BUY:
            if ALWAYS_BUY:
                self.logger.info("Always buy -> return [True]")
                return True
        if index is None:  # TODO means that we are in real time
            index = self.get_num_candles(client_index)-2 if index is None else index
        is_begin_day = self.clients[client_index].is_in_first_n_candles(n=N_FIRST_CANDLES_OF_DAY, candle_index=index)
        if is_begin_day:
            if self.real_time:
                self.logger.info("Blocking buy operation: beginning of day -> return [False]")
            return False
        is_end_day = self.clients[client_index].is_in_last_n_candles(n=N_LAST_CANDLES_OF_DAY, candle_index=index)
        if is_end_day:
            if self.real_time:
                self.logger.info("Blocking buy operation: end of day -> return [False]")
            return False

        op = '|' if self.is_bar_strategy() else '&'
        if self.data_changed[client_index]:
            approved_indices = self._is_criteria(cond_operation=op, client_index=client_index, candle_index=index)
            # data analysis of stock candles
            if self.real_time:
                self.extract_single_candle_data(client_index, index)
                self.logger.info(f"Current candle analysis data: {self.curr_data_entry}")
            else:  # in case we run in history mode, we can fetch the data for all candles at once
                self.extract_candle_data_table(client_index)
        else:
            approved_indices = self.criteria_indices[client_index]
        
        if self.use_dl_model:
            if self.real_time:
                self.logger.info("Using classification model to determine whether to buy or not")
            raw_sample_data = self.generate_dl_model_sample(client_index=client_index, candle_index=index)
            input_vec = self.dl_data_generator.generate_sample(raw_sample_data)
            scaled_input_vec = self.dl_scalar.transform(input_vec.to_numpy().reshape(1, -1))
            if np.isnan(scaled_input_vec).any():
                if self.real_time:
                    self.logger.info(f"data sample has Nan, returning false")
                ret_val = False
            else:
                classifier_out = float(self.dl_model.predict(scaled_input_vec))
                ret_val = (classifier_out >= self.model_thresh_min and classifier_out <= self.model_thresh_max)
                if self.real_time:
                    if ret_val:
                        self.logger.info(f"Classification model returned [{classifier_out}] between [{self.model_thresh_min}, {self.model_thresh_max}] -> return [{ret_val}]")
                    else:
                        self.logger.info(f"Classification model returned [{classifier_out}] outside [{self.model_thresh_min}, {self.model_thresh_max}] -> return [{ret_val}]")
        else:
            # check if latest index which represents the current candle meets the criteria
            ret_val = np.isin([index], approved_indices)[0]
        
        if self.real_time:
            # clear candle data only after generating model sample from it
            self.flush_candle_data_entry()
        
        if self.block_buy:
            if ret_val:
                if self.real_time:
                    self.logger.info("Citeria met and buy is blocked -> return [False]")
            else:
                if self.real_time:
                    self.logger.info("Citeria not met and buy is blocked, Unblocking buy -> return [False]")
                self.block_buy = False
            return False
        else:
            if ret_val:
                if self.real_time:
                    self.logger.info("Criteria met and buy is unblocked, blocking buy -> return [True]")
                self.block_buy = True
                return True
            else:
                if self.real_time:
                    self.logger.info("Criteria not met and buy is unblocked -> return [False]")
                return False

    @property
    def use_dl_model(self) -> bool:
        return self.dl_model is not None

    def extract_single_candle_data(self, client_index: int, candle_index: int):
        """
        builds a dict representing a single row in the candle data dataframe-
         containing info fields (low, high, ema etc..)- according to the given candle index
        """
        self.curr_data_entry['name'] = self.clients[client_index].name
        self.curr_data_entry['date'] = self.clients[client_index].get_candle_date(candle_index)
        self.curr_data_entry['high'] = self.clients[client_index].get_high_prices()[candle_index]
        self.curr_data_entry['low'] = self.clients[client_index].get_low_prices()[candle_index]
        self.curr_data_entry['close'] = self.clients[client_index].get_closing_price()[candle_index]
        if len(self.clients[client_index].candles) >= 21:
            self.curr_data_entry['ema20'] = self.get_ema(client_index, length=20)[candle_index]
        else:
            self.curr_data_entry['ema20'] = 0
        if len(self.clients[client_index].candles) >= 26:
            self.curr_data_entry['ema25'] = self.get_ema(client_index, length=25)[candle_index]
        else:
            self.curr_data_entry['ema25'] = 0
        if len(self.clients[client_index].candles) >= 31:
            self.curr_data_entry['ema30'] = self.get_ema(client_index, length=30)[candle_index]
        else:
            self.curr_data_entry['ema30'] = 0
        if len(self.clients[client_index].candles) >= 36:
            self.curr_data_entry['ema35'] = self.get_ema(client_index, length=35)[candle_index]
        else:
            self.curr_data_entry['ema35'] = 0
        if len(self.clients[client_index].candles) >= 41:
            self.curr_data_entry['ema40'] = self.get_ema(client_index, length=40)[candle_index]
        else:
            self.curr_data_entry['ema40'] = 0
        if len(self.clients[client_index].candles) >= 46:
            self.curr_data_entry['ema45'] = self.get_ema(client_index, length=45)[candle_index]
        else:
            self.curr_data_entry['ema45'] = 0
        if len(self.clients[client_index].candles) >= 201:
            self.curr_data_entry['ema200'] = self.get_ema(client_index, length=200)[candle_index]
        else:
            self.curr_data_entry['ema200'] = 0
        if len(self.clients[client_index].candles) >= 51:
            self.curr_data_entry['ema50'] = self.get_ema(client_index, length=50)[candle_index]
        else:
            self.curr_data_entry['ema50'] = 0
        if len(self.clients[client_index].candles) >= 101:
            self.curr_data_entry['ema100'] = self.get_ema(client_index, length=100)[candle_index]
        else:
            self.curr_data_entry['ema100'] = 0

    def extract_candle_data_table(self, client_index: int):
        """
        builds a dataframe containing info fields (low, high, ema etc..) about the given client candles.
        """
        self.data_collection_df.reset_index(inplace=True)

        self.temp_client_candle_df['name'] = self.clients[client_index].name
        self.temp_client_candle_df['date'] = self.clients[client_index].get_candle_dates().tolist()
        self.temp_client_candle_df['low'] = self.clients[client_index].get_low_prices().tolist()
        self.temp_client_candle_df['high'] = self.clients[client_index].get_high_prices().tolist()
        self.temp_client_candle_df['close'] = self.clients[client_index].get_closing_price().tolist()
        if len(self.clients[client_index].candles) >= 21:
            self.temp_client_candle_df['ema20'] = self.get_ema(client_index, length=20).tolist()
        if len(self.clients[client_index].candles) >= 26:
            self.temp_client_candle_df['ema25'] = self.get_ema(client_index, length=25).tolist()
        if len(self.clients[client_index].candles) >= 31:
            self.temp_client_candle_df['ema30'] = self.get_ema(client_index, length=30).tolist()
        if len(self.clients[client_index].candles) >= 36:
            self.temp_client_candle_df['ema35'] = self.get_ema(client_index, length=35).tolist()
        if len(self.clients[client_index].candles) >= 41:
            self.temp_client_candle_df['ema40'] = self.get_ema(client_index, length=40).tolist()
        if len(self.clients[client_index].candles) >= 46:
            self.temp_client_candle_df['ema45'] = self.get_ema(client_index, length=45).tolist()
        if len(self.clients[client_index].candles) >= 51:
            self.temp_client_candle_df['ema50'] = self.get_ema(client_index, length=50).tolist()
        if len(self.clients[client_index].candles) >= 101:
            self.temp_client_candle_df['ema100'] = self.get_ema(client_index, length=100).tolist()
        if len(self.clients[client_index].candles) >= 201:
            self.temp_client_candle_df['ema200'] = self.get_ema(client_index, length=200).tolist()
        self.temp_client_candle_df['vwap'] = self.get_vwap(client_index).tolist()
        # TODO ^VIX dates were sometimes inconsistent with other stock candle dates we got,
        #  the indexing below is a temporary patch. Should try to find a better fix.
        if self.vix_client is not None:
            self.temp_client_candle_df[VIX] = self.vix_client.get_closing_price()[self.clients[0].get_candle_dates()].tolist()

        self.data_collection_df = pd.concat([self.data_collection_df, self.temp_client_candle_df])
        self._reset_temp_candle_df()
        # now that the candle data was added, multiindex to make further operations faster
        if not self.real_time:
            reindex_df(self.data_collection_df, ['date', 'name'])

    def generate_dl_model_sample(self, client_index: int, candle_index: int) -> RawDataSample:
        curr_date = self.clients[client_index].get_candle_date(candle_index)
        curr_candle_data = self.data_collection_df.loc[(curr_date, self.clients[client_index].name)] if not self.real_time else None
        curr_candle_close = self.get_close_price(client_index, candle_index)
        
        raw_sample = RawDataSample(
            close=curr_candle_close,
            percentage_diff=self.percentage_diff,
            accumulated_percentage_diff=self.accumulated_percentage_diff,
            num_candles_in_trend=self.num_candles_in_trend,
            ema20=self.curr_data_entry['ema20'] if self.real_time else curr_candle_data['ema20'],
            ema50=self.curr_data_entry['ema50'] if self.real_time else curr_candle_data['ema50'],
            ema100=self.curr_data_entry['ema100'] if self.real_time else curr_candle_data['ema100'],
            ema200=self.curr_data_entry['ema200'] if self.real_time else curr_candle_data['ema200'],
            ema25=self.curr_data_entry['ema25'] if self.real_time else curr_candle_data['ema25'],
            ema30=self.curr_data_entry['ema30'] if self.real_time else curr_candle_data['ema30'],
            ema35=self.curr_data_entry['ema35'] if self.real_time else curr_candle_data['ema35'],
            ema40=self.curr_data_entry['ema40'] if self.real_time else curr_candle_data['ema40'],
            ema45=self.curr_data_entry['ema45'] if self.real_time else curr_candle_data['ema45'],
            rsi=self.curr_data_entry['RSI'] if self.real_time else curr_candle_data['RSI'],
            supertrend=self.curr_data_entry['SUPERTREND'] if self.real_time else curr_candle_data['SUPERTREND'],
            macd=self.curr_data_entry['MACD'] if self.real_time else curr_candle_data['MACD'],
            macd_signal=self.curr_data_entry['MACD_SIGNAL'] if self.real_time else curr_candle_data['MACD_SIGNAL'],
            macd_h=self.curr_data_entry['MACDh'] if self.real_time else curr_candle_data['MACDh'],
            date=curr_date,
        )
        return raw_sample

    def is_sell(self, client_index: int, index: int = None) -> bool:
        if index is None:
            index = self.get_num_candles(client_index)-1 if index is None else index
        if self.clients[client_index].is_day_last_transaction(index):
            return True
        curr_index = self.get_num_candles(client_index) - 1 if index is None else index
        if self.sell_on_touch:
            latest_low = self.get_low_price(client_index, curr_index)
            latest_high = self.get_high_price(client_index, curr_index)
            return latest_low <= self.stop_loss or latest_high >= self.take_profit
        else:
            latest_close = self.get_close_price(client_index, curr_index)
            return latest_close <= self.stop_loss or latest_close >= self.take_profit

    def get_num_trades(self):
        return self.num_gains + self.num_eod_gains + self.num_losses + self.num_eod_losses

    def get_latest_trade_stock_price(self, client_index: int) -> float:
        client_latest_trade = self.latest_trade[client_index]
        return client_latest_trade[STOCK_PRICE]

    def buy(self, client_index: int, index: int = None, real_time=False) -> int:
        if self.real_time or self.enable_history_log:
            self.logger.info("---Buying---")
        if index is None:  # real-time
            index = self.get_num_candles(client_index)-1
        low_prices = self.clients[client_index].get_low_prices()
        stock_price = self.get_trade_close_price(client_index, index)

        # calculate local minima of history
        if self.is_bar_strategy():  # strategy no.2
            local_min = self.get_close_price(client_index, index - 2)
        else:  # strategy no.1
            if len(low_prices) < STOP_LOSS_RANGE:
                stop_loss_range = low_prices
            else:
                stop_loss_range = low_prices[np.maximum(0, index-STOP_LOSS_RANGE):index]
            local_min = np.min(stop_loss_range)

        # in case of inconsistent data, local_min!=local_min when local_min is nan
        if not self.use_dl_model and (local_min != local_min or local_min > stock_price):
            self.logger.info("Local min is bigger than stock price, aborting... -> return [trade not complete]")
            return TRADE_NOT_COMPLETE

        # TODO don't buy if stop loss percentage is >= X, put in LS
        loss_percentage = 1-(local_min/stock_price)+STOP_LOSS_PERCENTAGE_MARGIN
        # if loss_percentage > self.stop_loss_bound and (self.real_time or self.enable_history_log):
        #     self.logger.info("Loss precentage is bigger than stop loss bound, aborting... -> return [trade not complete]")
        #     return TRADE_NOT_COMPLETE
        if self.use_dl_model:
            close_prices = self.clients[client_index].get_closing_price()
            high_prices = self.clients[client_index].get_high_prices()
            atr_val = round(atr(high_prices, low_prices, close_prices, length=self.atr_period)[index], 2)
            self.stop_loss = stock_price - (atr_val * self.atr_mul)
            self.take_profit = stock_price + (atr_val * ATR_MUL * self.take_profit_multiplier)
        else:
            self.stop_loss = stock_price * (1-loss_percentage)
            self.take_profit = stock_price*((loss_percentage * self.take_profit_multiplier)+1)

        if self.use_pyramid:
            ru_dollars = get_percent(self.capital, self.risk_unit*self.ru_percentage)
            num_stocks = np.floor(np.minimum((ru_dollars / (1-(self.stop_loss/stock_price)) / stock_price),
                                  self.capital / stock_price))
            if num_stocks == 0:
                self.logger.info("Inconsistent data pyramid, num of stocks to buy is zero, aborting... -> return [trade not complete]")
                return TRADE_NOT_COMPLETE
            self.latest_trade[client_index] = [num_stocks * stock_price, num_stocks, stock_price]
        else:
            num_stocks = np.floor(self.capital / stock_price)
            if num_stocks == 0:
                self.logger.info("Inconsistent data pyramid, num of stocks to buy is zero -> return [trade not complete]")
                return TRADE_NOT_COMPLETE
            self.latest_trade[client_index] = [self.capital, num_stocks, stock_price]

        self.capital -= self.latest_trade[client_index][AMOUNT]
        self.status[client_index] = SellStatus.BOUGHT
        if self.real_time:
            self.logger.info("SellStatus changed -> [BOUGHT]")

        trade_data = {"date": self.clients[client_index].get_candle_date(index),
                      "number": self.get_num_trades()+1,
                      "amount": self.latest_trade[client_index][AMOUNT],
                      "price": stock_price,
                      "type": "Buy",
                      "capital": self.capital,
                      "name": self.clients[client_index].name}
        self.flush_trade_entry(trade_data)

        if not self.real_time and self.enable_history_log:
            self.logger.info(f"Buy date: {trade_data['date']}")
            self.logger.info(f"Trade number: {trade_data['number']}")
            self.logger.info(f"Buy amount: {format(trade_data['amount'], '.2f')}")
            self.logger.info(f"Stock price: {stock_price}")

        if self.real_time:
            self.clients[client_index].buy_order(self.latest_trade[client_index][NUM_STOCKS],
                                                 stop_loss=float(format(self.stop_loss, '.2f')),
                                                 price=float(format(self.get_close_price(client_index, index-1), '.2f')))

        self.block_buy = True
        return TRADE_COMPLETE

    def set_take_profit(self, take_profit: float):
        self.take_profit = take_profit

    def set_stop_loss(self, stop_loss: float):
        self.logger.info(f"setting stop-loss to {stop_loss}")
        self.stop_loss = stop_loss

    def sell(self, client_index: int, index: int = None) -> int:
        if not self.real_time and self.enable_history_log:
            self.logger.info("Selling...")
        if index is None:
            index = self.get_num_candles(client_index)-1 if index is None else index
        if self.status[client_index] != SellStatus.BOUGHT and self.enable_history_log:
            self.logger.info("Tried selling before buying a stock")
            return 0

        curr_index = self.get_num_candles(client_index) - 1 if index is None else index

        # check if candle price reached take_profit / stop_loss- if in real time mode or sell_on_touch
        # option is used, set stock_price accordingly. Else, stock_price will be set to the candle close
        # price as a default
        stock_price = None
        if self.real_time or self.sell_on_touch:
            curr_low = self.get_low_price(client_index, curr_index)
            curr_high = self.get_high_price(client_index, curr_index)
            if curr_low <= self.stop_loss:
                stock_price = curr_low
            elif curr_high >= self.take_profit:
                stock_price = curr_high
        if stock_price is None:
            stock_price = self.get_close_price(client_index, curr_index)

        sell_price = stock_price*self.latest_trade[client_index][NUM_STOCKS]
        self.status[client_index] = SellStatus.SOLD
        if self.real_time:
            self.logger.info("SellStatus changed -> [SOLD]")

        is_eod = self.clients[client_index].is_day_last_transaction(index)
        profit = (sell_price / self.latest_trade[client_index][AMOUNT]) - 1

        add_to_log = True  # we don't want to add trades that are in the middle of rw streak

        if self.rw and not is_eod and profit > 0:
            if self.rw_num_times_sold < self.num_wins_thresh:
                self.rw_num_times_sold += 1
                self.status[client_index] = SellStatus.BOUGHT

                num_stocks_sold = np.floor(self.latest_trade[client_index][NUM_STOCKS] * self.rw_percent)
                sell_price = stock_price * num_stocks_sold
                prev_stock_price = self.get_latest_trade_stock_price(client_index)
                relative_prev_trade_amount = prev_stock_price * num_stocks_sold
                profit = (sell_price / relative_prev_trade_amount) - 1

                self.stop_loss = prev_stock_price
                loss_percentage = 1 - (self.stop_loss / stock_price)
                self.take_profit = stock_price * ((loss_percentage * self.rw_take_profit_multiplier) + 1)

                self.latest_trade[client_index] = [self.latest_trade[client_index][AMOUNT]-relative_prev_trade_amount,
                                                   self.latest_trade[client_index][NUM_STOCKS]-num_stocks_sold,
                                                   stock_price]
                if self.enable_history_log:
                    self.logger.info(f"Let the wins run no. {self.rw_num_times_sold}; Selling {self.rw_percent} of bought amount")
            else:
                self.rw_num_times_sold = 0
                add_to_log = False
        else:
            add_to_log = self.rw_num_times_sold == 0
            self.rw_num_times_sold = 0

        if not self.rw or (self.rw and self.rw_num_times_sold == 1) or add_to_log:
            if profit > 0:  # gain
                if self.use_pyramid:
                    self.risk_unit = self.initial_risk_unit
                if not is_eod:
                    self.gains[self.num_gains] = profit
                    self.num_gains += 1
                else:
                    self.eod_gains[self.num_eod_gains] = profit
                    self.num_eod_gains += 1
            else:  # loss
                if self.use_pyramid and (not self.rw or self.rw_num_times_sold == 0):  # we only use pyramid at while not in rw streak
                    self.risk_unit = np.minimum(self.risk_unit*self.risk_growth, self.risk_limit)
                if not is_eod:
                    self.losses[self.num_losses] = -profit
                    self.num_losses += 1
                else:
                    self.eod_losses[self.num_eod_losses] = -profit
                    self.num_eod_losses += 1
        self.capital += sell_price
        if self.rw_num_times_sold == 0:  # make sure that we are not in the middle of rw
            self.capital_history[:, self.get_num_trades()] = [index, self.capital]

        trade_data = {"date": self.clients[client_index].get_candle_date(index),
                      "number": self.get_num_trades(),
                      "amount": sell_price,
                      "price": stock_price,
                      "type": "Sell",
                      "capital": self.capital,
                      "sell_type": 'gain' if profit > 0 else 'loss',
                      "name": self.clients[client_index].name}
        self.flush_trade_entry(trade_data)

        if not self.real_time and self.enable_history_log:
            self.logger.info(f"\tSale date: {trade_data['date']}\n\tSale amount: {format(sell_price, '.2f')}\n"
                            f"\tTrade no.: {self.get_num_trades()}\n"
                            f"\tStock price: {stock_price}")

        return profit

    def cancel_trade(self, client_index: int):
        # TODO fill missing logic
        if self.status == SellStatus.BOUGHT:
            self.capital += self.latest_trade[client_index][AMOUNT]  # trade cancelled -> money was not spent
            self.status = SellStatus.SOLD
        else:
            pass

    def is_bar_strategy(self):
        return CRITERIA.REVERSAL_BAR in self.criteria and\
               CRITERIA.INSIDE_BAR in self.criteria and\
               len(self.criteria) == 2

    def get_profit(self) -> float:
        return self.capital - self.initial_capital

    def get_profit_percentage(self) -> float:
        return self.get_profit() / self.initial_capital

    def is_client_occupied(self):
        """
        True if there's a client that still needs to sell
        """
        return any([self.status[k] == SellStatus.BOUGHT for k in range(len(self.clients))])
    
    def update_dl_attributes(self, client_index: int, candle_index: int) -> None:
        """
        This function calculates the relative changes between candle prices in order to
        be able to derive the features of the deep learning classification model.
        Attributes which are calculated:
            - percentage differance from the latest candle
            - accumulated percentage differance of the most recent tred
            - number of candles in the most recent trend
        """
        if self.real_time:
            self.logger.info("Calculating classification model attributes")
        # reset attributes at the start of each trading day
        if self.clients[client_index].is_in_first_n_candles(candle_index=candle_index, n=1):
            self.accumulated_percentage_diff = self.percentage_diff = self.num_candles_in_trend = 0
            return
        
        prev_candle_close: float = 0
        curr_candle_close = self.get_close_price(client_index, candle_index)
        
        if (not self.real_time and candle_index > 0) or (self.real_time and self.get_num_candles(client_index) > 1): 
            prev_candle_close = self.get_close_price(client_index, candle_index-1)
        diff_from_prev_candle = curr_candle_close - prev_candle_close
        percentage_diff_from_last_candle=(diff_from_prev_candle / prev_candle_close)
        
        if self.num_candles_in_trend >= 0 and percentage_diff_from_last_candle > 0 and self.accumulated_percentage_diff >= 0:
            self.num_candles_in_trend += 1
            self.accumulated_percentage_diff = (1 + percentage_diff_from_last_candle) * (1 + self.accumulated_percentage_diff) - 1
        elif self.num_candles_in_trend <= 0 and percentage_diff_from_last_candle < 0 and self.accumulated_percentage_diff <= 0:
            self.num_candles_in_trend -= 1
            self.accumulated_percentage_diff = -1 * ((-1 + percentage_diff_from_last_candle) * (-1 + self.accumulated_percentage_diff)) + 1
        else:
            self.num_candles_in_trend = 0
            self.accumulated_percentage_diff = percentage_diff_from_last_candle
            
        self.percentage_diff = percentage_diff_from_last_candle
        

    def main_loop_history(self, candle_range: Tuple[int, int] = None) -> float:
        if candle_range:
            start, end = candle_range
        else:
            start, end = 0, self.get_num_candles() - 1

        for i in range(start, end):
            for j in range(len(self.clients)):
                if self.use_dl_model:
                    self.update_dl_attributes(client_index=j, candle_index=i)
                if self.status[j] in (SellStatus.SOLD, SellStatus.NEITHER) and not self.is_client_occupied():  # try to buy
                    condition = self.is_buy(index=i, client_index=j)
                    if condition:
                        ret_val = self.buy(index=i, client_index=j)
                        if ret_val == TRADE_COMPLETE and self.enable_history_log:  # in case trade was not complete
                            self.logger.info(f"Current capital: {format(self.capital, '.2f')}\nStocks bought: {int(self.latest_trade[j][NUM_STOCKS])}\nStock name: {self.clients[j].trading_name}\n")
                elif self.status[j] == SellStatus.BOUGHT:  # try to sell
                    condition = self.is_sell(index=i, client_index=j)
                    if condition:
                        price = self.sell(index=i, client_index=j)
                        if self.enable_history_log:
                            self.logger.info(f"Current capital: {format(self.capital, '.2f')}\nSell type: {GAIN if price > 0 else LOSS}\nStock name: {self.clients[j].trading_name}\n")
            # update the number of trades per day
            if i > 0 and self.clients[0].is_day_last_transaction(i-1):  # first candle of the day
                curr_day_index = np.where(self.trades_per_day == -1)[0][0]
                self.trades_per_day[curr_day_index] = self.get_num_trades() - (np.sum(self.trades_per_day[:curr_day_index]))
                
        if self.enable_history_log:
            self.log_summary()
        return self.capital

    def add_candle(self, client_index: int) -> bool:
        """
        add the latest available candle and retunr True if it is a new candle
        """
        lastest_time = self.clients[client_index].get_candle_date(-1)
        self.clients[client_index].add_candle()
        if self.clients[client_index].get_candle_date(-1) != lastest_time:
            self.data_changed[client_index] = True
            self.logger.info(f"New candle added!")
            return True
        return False

    async def main_loop_real_time(self) -> float:
        self.logger.info("------------------ Main loop ----------------------\n")
        if not DEBUG:
            send_email_all("Starting main loop ELE", f"Let's earn some money, Voovos!!!\nStocks: {[c.trading_name for c in self.clients]}")
        is_eod = False
        trading_started = False
        while not is_eod or self.is_client_occupied():
            self.logger.info("------------------ New iteration ------------------")
            for j in range(len(self.clients)):
                self.logger.info(f"Total cash: [{self.clients[j].get_cash()}$]")
                self.logger.info(f"Stock holdings: {self.clients[j].get_stock_holdings()}")
                new_candle_added = self.add_candle(client_index=j)
                if new_candle_added:
                    if self.use_dl_model:
                        self.update_dl_attributes(client_index=j, candle_index=-1)
                self.update_dl_attributes(client_index=j, candle_index=i, )
                self.logger.info(
                    f"Current candle: {self.clients[j].get_candle_date(-1)}; Stock: {self.clients[j].name}")

                # try to buy only when new candle was added and no client is in a trade
                if self.status[j] in (SellStatus.SOLD, SellStatus.NEITHER) and not self.is_client_occupied()\
                        and self.data_changed[j]:
                    condition = self.is_buy(client_index=j)  # does client need to buy
                    if condition:
                        ret_val = self.buy(client_index=j)

                # reached last candle of the day while there's an open trade -> sell all stocks using sell market order
                elif self.status[j] == SellStatus.BOUGHT and self.clients[j].is_day_last_transaction(-1):
                    self.clients[j].sell_order()

                # we check if eod is reached after candles in trading hours were obsereved
                # because otherwise we could end the loop before even reaching trading hours 
                if self.clients[j].is_day_last_transaction(-1):
                    if trading_started:
                        is_eod = True
                else:
                    trading_started = True
            await asyncio.sleep(self.main_loop_sleep_time)
            self.logger.info(f"------------------ End of iteration [{self.main_loop_sleep_time}] seconds --\n\n")
        if not DEBUG:
            send_email_all("Main loop ELE ended", "Thank you...Voovo!!!")
        return self.capital

    def log_summary(self):
        risk_chance = (np.average(self.gains[:self.num_gains]) / np.average(self.losses[:self.num_losses]))\
                        if self.num_gains > 0 and self.num_losses > 0 else 0
        tpd_valid = self.trades_per_day[:np.where(self.trades_per_day == -1)[0][0]]
        gains = self.gains[:self.num_gains]
        losses = self.losses[:self.num_losses]
        self.logger.info(f"Win percentage: {format((self.num_gains / (self.num_gains + self.num_losses)) if (self.num_gains + self.num_losses) > 0 else 0, '.3f')}\n"
                         f"Total trades: {self.get_num_trades()}\n"
                         f"Winning trades: {self.num_gains}\nLosing trades: {self.num_losses}\n"
                         f"Winning end of day trades: {self.num_eod_gains}\n"
                         f"Losing end of day trades: {self.num_eod_losses}\n"
                         f"End of day gain / loss: {format((self.num_eod_gains / (self.num_eod_gains + self.num_eod_losses)) if (self.num_eod_gains + self.num_eod_losses) > 0 else 0, '.3f')}\n"
                         f"Gain average: {format(np.average(gains) if self.num_gains > 0 else 0, '.3f')}\n"
                         f"Gain max: {format(np.max(gains) if self.num_gains > 0 else 0, '.3f')}\n"
                         f"Gain min: {format(np.min(gains) if self.num_gains > 0 else 0, '.3f')}\n"
                         f"Loss average: {format(np.average(losses) if self.num_losses > 0 else 0, '.3f')}\n"
                         f"Loss max: {format(np.max(losses) if self.num_losses > 0 else 0, '.3f')}\n"
                         f"Loss min: {format(np.min(losses) if self.num_losses > 0 else 0, '.3f')}\n"
                         f"Max number of trades per day: {np.max(tpd_valid)}\n"
                         f"Min number of trades per day: {np.min(tpd_valid)}\n"
                         f"Average number of trades per day: {format(np.average(tpd_valid), '.3f')}\n"
                         f"Most common number of trades per day: {np.bincount(tpd_valid).argmax()}\n"
                         f"Risk / Chance: "
                         f"{format(risk_chance, '.3f')}\n"
                         f"Total profit: {format(self.get_profit(), '.3f')}\n")

    def update_time(self, n: int = 15):
        """
        advance current time window by given number of minutes
        """
        period = minutes_to_secs(n)
        self.start_time += period
        self.end_time += period


def plot_capital_history(sb: StockBot, vals, path: str = None):
    capital_dic_array = {-i: -1 for i in range(4)}

    for val in vals:
        sb.reset()
        sb.set_tp_multiplier(val)
        sb.main_loop_history()
        res = sb.capital_history[1, :sb.get_num_trades()]
        max_val = np.max(res)
        curr_min = min(capital_dic_array.values())
        if max_val > curr_min:
            for key, _ in capital_dic_array.items():
                if _ == curr_min:
                    capital_dic_array.pop(key)
                    break
            capital_dic_array[val] = max_val

    for val in capital_dic_array:
        print(val)


def plot_capital(sb: StockBot):
    # this inner function converts a pandas timestamp into a string date of format %d/%m
    def convert_date(date: pd.Timestamp):
        return date.to_pydatetime().strftime('%d/%m')

    candle_dates: pd.Series = sb.clients[0].get_candle_dates()
    x_points = [f"#{int(i)+1}: {convert_date(candle_dates[int(candle_num)])}" for i, candle_num
                in enumerate(sb.capital_history[0, :sb.get_num_trades()+1])]
    y_points = sb.capital_history[1, :sb.get_num_trades()+1]

    plt.xticks(np.arange(0, len(x_points), 5))
    plt.gcf().autofmt_xdate() # beautify the x-labels
    plt.plot(x_points, y_points)
    plt.xlabel('no. / date of trades')
    plt.ylabel('capital')
    plt.savefig("graph.png",  bbox_inches='tight')


def filter_stocks(stocks: List[str], val: float):
    res = []
    for stock in stocks:
        sb = StockBot(stock_clients=[StockClientYfinance(name=stock)], period=period, use_pyr=True, criteria=DEFAULT_CRITERIA_LIST)
        sb.main_loop_history()
        if sb.get_profit_percentage() >= val:
            res.append(stock)
    return res


def generate_trade_summary(sb: StockBot) -> pd.DataFrame:
    """
    this fucntion will generate a trade summary from the trade data of the given bot
    """
    trade_summary: pd.DataFrame = pd.DataFrame(columns=["name",
                                                        "buy date",
                                                        "sell date",
                                                        "number",
                                                        "entry price",
                                                        "exit price",
                                                        "capital before",
                                                        "capital after",
                                                        "profit %",
                                                        "profit $"])
    trade_actions: pd.DataFrame = sb.trades_df
    for i in range(0, trade_actions.shape[0], 2):
        if i == trade_actions.shape[0]-1:
            break  # in case last trade wasn't sold yet
        
        buy_action = trade_actions.iloc[i, :]
        sell_action = trade_actions.iloc[i+1, :]
        
        if buy_action["type"].lower() != 'buy' or sell_action["type"].lower() != 'sell':
            break  # each buy action should be followed buy a sell action
        
        trade_row = {
            "name": buy_action["name"],
            "buy date": buy_action["date"],
            "sell date": sell_action["date"],
            "number": buy_action["number"],
            "entry price": buy_action["price"],
            "exit price": sell_action["price"],
            "capital before": buy_action["capital"] + buy_action["amount"],
            "capital after": sell_action["capital"],
        }
        trade_row["profit $"] = trade_row["capital after"] - trade_row["capital before"]
        trade_row["profit %"] = (trade_row["profit $"] / trade_row["capital before"]) * 100
        
        trade_summary = trade_summary.append(trade_row, ignore_index=True)
    
    return trade_summary


if __name__ == '__main__':
    curr_time = get_curr_utc_2_timestamp()  # current time in utc+2
    # fetch candles from 3 days for real-time and 60 days for history run
    period = f'{f"{REAL_TIME_PERIOD} D" if REAL_TIME else f"{HISTORY_PERIOD} D"}'

    from stock_client import StockClientYfinance, StockClientInteractive

    if FILTER_STOCKS[2]:
        stock_list, filter_val = FILTER_STOCKS[:2]
        filtered = filter_stocks(stock_list, filter_val)
        print(f"stocks that were filtered: {filtered}")

    if RUN_ROBOT:
        real_time_client = StockClientInteractive.init_client() if REAL_TIME else None
        clients = [StockClientInteractive(data_stock=dname, trade_stock=tname, client=real_time_client, client_id=i+1)
                   for i, (dname, tname) in enumerate(DATA_STOCKS_TO_TRADED_MAPPING.items())]
        # this is used in order to get the value of ^VIX
        # vix_client = StockClientInteractive(name=VIX, client_id=len(clients)+1) if not REAL_TIME else None
        vix_client = None  # todo need to make this work with interactive
        
        # declare optional variables used with dl model
        dl_model: Optional[FcClassifier] = None
        scalar = None
        data_generator: Optional[DataGenerator] = None

        if USE_DL_MODEL:
            data_generator = DataGenerator(clients[0])
            raw_data = data_generator.get_training_data()
            X, y, scalar = DataGenerator.pre_process(raw_data)
            dl_model = FcClassifier(X, y, weights_from_file=True)
            dl_model.train_nn_model()

        sb = StockBot(stock_clients=clients, period=period, criteria=DEFAULT_CRITERIA_LIST, vix_client=vix_client,
                      model=dl_model, data_generator=data_generator, scalar=scalar)
        if sb.real_time:
            if DEFAULT_USE_PYRAMID:
                # calculate updated risk unit before running real-time
                sb.logger.info("performing risk unit calculation...")
                stock_bot_history_mode = StockBot(stock_clients=clients,
                                                  period=f"{PYR_RISK_UNIT_CALCULATION_PERIOD} D",
                                                  criteria=DEFAULT_CRITERIA_LIST,
                                                  enable_history_log=False,
                                                  real_time=False)
                stock_bot_history_mode.main_loop_history()
                sb.risk_unit = stock_bot_history_mode.risk_unit
                sb.logger.info(f"calculation finished with [risk_unit == {sb.risk_unit}]")
                sb.set_candle_data()
            
            nest_asyncio.apply()
            for i, client in enumerate(clients):
                if type(client) == StockClientInteractive:
                    client.bind_tp_observer(sb.set_take_profit)
                    client.bind_sell_observer(lambda i=i: sb.sell(client_index=i))
                    sb.logger.info("--run robot--")
                    client.bind_cancel_observer(lambda i=i: sb.cancel_trade(client_index=i))
            res = asyncio.run(sb.main_loop_real_time())
        else:
            res = sb.main_loop_history()
            sb.data_collection_df.to_csv(CANDLE_DATA_CSV_NAME)
            sb.trades_df.to_csv(TRADE_DATA_CSV_NAME, index=False)
            trade_summary: pd.DataFrame = generate_trade_summary(sb)
            trade_summary.to_csv(TRADE_SUMMARY_CSV_NAME, index=False)

        if OUTPUT_PLOT:
            plot_capital(sb)
