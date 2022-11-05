import time
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas_ta import rsi, macd, supertrend, ema
import logging

from utils import (minutes_to_secs, days_to_secs, filter_by_array, get_curr_utc_2_timestamp, get_percent, send_email)
from consts import (DEFAULT_RES, LONG_STOCK_NAME, MACD_INDEX, MACD_SIGNAL_INDEX, SellStatus, CRITERIA, LOGGER_NAME,
                    STOP_LOSS_RANGE, TAKE_PROFIT_MULTIPLIER, SUPERTREND_COL_NAME, DEFAULT_RISK_UNIT,
                    DEFAULT_RISK_LIMIT, DEFAULT_START_CAPITAL, DEFAULT_CRITERIA_LIST, DEFAULT_USE_PYRAMID,
                    DEFAULT_GROWTH_PERCENT, DEFAULT_RU_PERCENTAGE, GAIN, LOSS, EMA_LENGTH, STOP_LOSS_PERCENTAGE_MARGIN,
                    SHORT_STOCK_NAME, STOP_LOSS_LOWER_BOUND, TRADE_NOT_COMPLETE, OUTPUT_PLOT, STOCKS, FILTER_STOCKS,
                    RUN_ROBOT, USE_RUN_WINS, RUN_WINS_TAKE_PROFIT_MULTIPLIER, RUN_WINS_PERCENT, TRADE_COMPLETE,
                    MACD_PARAMS, SUPERTREND_PARAMS, RSI_PARAMS, N_FIRST_CANDLES_OF_DAY, N_LAST_CANDLES_OF_DAY,
                    REAL_TIME, SELL_ON_TOUCH, ALWAYS_BUY)
from stock_client import StockClient

API_KEY = "c76vsr2ad3iaenbslifg"

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
                 rw_take_profit_multiplier=RUN_WINS_TAKE_PROFIT_MULTIPLIER, rw_percent=RUN_WINS_PERCENT):
        self.clients = stock_clients
        self.gain_avg = None
        self.loss_avg = None
        self.resolution = DEFAULT_RES
        self.start_time = start
        self.end_time = end
        self.period = period
        self.rsi_window_size, self.ema_window_size = rsi_win_size, ema_win_size

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

        self.set_candle_data()

        # "let the wins run" strategy parameters
        self.rw = run_wins
        self.num_wins_thresh = 1
        self.rw_num_times_sold = 0
        self.rw_take_profit_multiplier = rw_take_profit_multiplier
        self.rw_percent = rw_percent

        num_candles = self.clients[0].get_num_candles()
        self.capital_history = np.zeros(num_candles)
        self.capital_history[0] = self.capital
        self.gains = np.zeros(num_candles)
        self.eod_gains = np.zeros(num_candles)
        self.losses = np.zeros(num_candles)
        self.eod_losses = np.zeros(num_candles)
        self.num_gains = self.num_eod_gains = self.num_losses = self.num_eod_losses = 0
        self.trades_per_day = np.full(num_candles, -1)

        self.data_changed = True  # indicates whether new candle data was fetched or not
        self.criteria_indices = None
        self.block_buy = False

        # this dict maps a criterion to the method that returns the candle indices that fulfil it
        self.criteria_func_data_mapping = {CRITERIA.RSI: self.get_rsi_criterion,
                                           CRITERIA.SUPER_TREND: self.get_supertrend_criterion,
                                           CRITERIA.MACD: self.get_macd_criterion,
                                           CRITERIA.INSIDE_BAR: self.get_inside_bar_criterion,
                                           CRITERIA.REVERSAL_BAR: self.get_reversal_bar_criterion}
        self.ema = [None for i in range(len(self.clients))]

        self.criteria: List[CRITERIA] = []
        self.set_criteria(criteria)

        logger_options = {"format": '%(message)s',
                          "datefmt": '%H:%M:%S',
                          "level": logging.INFO}
        if log_to_file:
            logger_options["filename"] = "STOCKBOT_LOG"
            logger_options["filemode"] = 'w'

        # initialize logger
        logging.basicConfig(**logger_options)
        self.logger = logging.getLogger('start logging')
        self.logger.setLevel(logging.INFO)

    def get_close_price(self, client_index: int, index=-1):
        closing_price = self.clients[client_index].get_closing_price()
        return closing_price.iloc[index]

    def get_open_price(self, client_index: int, index=-1):
        open_price = self.clients[client_index].get_opening_price()
        return open_price.iloc[index]

    def get_low_price(self, client_index: int, index=-1):
        low_price = self.clients[client_index].get_low_price()
        return low_price.iloc[index]

    def get_high_price(self, client_index: int, index=-1):
        high_price = self.clients[client_index].get_high_price()
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
        self.data_changed = True

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

    def get_rsi_criterion(self, client_index: int):
        s_close_old = pd.Series(self.clients[client_index].get_closing_price())

        length = RSI_PARAMS

        rsi_results = np.array(rsi(s_close_old, length=length))
        return np.where(rsi_results > 50)

    def get_supertrend_criterion(self, client_index: int):
        high = self.clients[client_index].get_high_price()
        low = self.clients[client_index].get_low_price()
        close = self.clients[client_index].get_closing_price()

        length, multiplier = SUPERTREND_PARAMS

        supertrend_results_df = supertrend(pd.Series(high), pd.Series(low), pd.Series(close),
                                           length=length, multiplier=multiplier)
        supertrend_results = supertrend_results_df[SUPERTREND_COL_NAME]

        indices = np.where(supertrend_results < close)

        return np.array(indices)

    def get_macd_criterion(self, client_index: int):
        close_prices = self.clients[client_index].get_closing_price()

        fast, slow, signal = MACD_PARAMS
        macd_close = macd(close=pd.Series(close_prices), fast=fast, slow=slow, signal=signal)
        macd_data = np.array(macd_close.iloc[:, MACD_INDEX])
        signal_data = np.array(macd_close.iloc[:, MACD_SIGNAL_INDEX])

        indices = np.where((macd_data > signal_data) & (macd_data < 0))
        indices_as_nd = np.array(indices)

        return indices_as_nd

    def get_ema(self, client_index: int):
        self.ema[client_index] = ema(self.clients[client_index].get_closing_price(), EMA_LENGTH)
        return self.ema[client_index]

    def get_inside_bar_criterion(self, client_index: int):
        close_prices = self.clients[client_index].get_closing_price()
        open_prices = self.clients[client_index].get_opening_price()
        high_prices = self.clients[client_index].get_high_price()
        low_prices = self.clients[client_index].get_low_price()
        ema = self.ema[client_index] if self.ema[client_index] is not None else self.get_ema(client_index)

        positive_candles = close_prices[:-1] - open_prices[:-1]
        high_price_diff = np.diff(high_prices)
        # 1st candle positive + 2nd candle in 1st candle price range [open, close]
        cond_1 = np.where((positive_candles > 0) & (high_price_diff < 0) & (np.diff(low_prices) > 0))[0] + 1
        # 3rd candle close bigger than both 1st candle close and ema
        cond_2 = np.where((close_prices < np.roll(close_prices, -2)) & (np.roll(close_prices, -2) > np.roll(ema, -2)))[0] + 1

        # candle 1 stop loss

        indices = np.intersect1d(cond_1, cond_2) + 2

        return indices

    def get_reversal_bar_criterion(self, client_index: int):
        close_prices = self.clients[client_index].get_closing_price()
        high_prices = self.clients[client_index].get_high_price()
        low_prices = self.clients[client_index].get_low_price()
        ema = self.ema[client_index] if self.ema[client_index] is not None else self.get_ema(client_index)

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

    def _is_criteria(self, client_index: int, cond_operation='&') -> np.ndarray:
        """
        :param cond_operation: '&' or '|'. Indicates the operation that will be performed on the given criteria.
        """
        indices = np.arange(self.get_num_candles(client_index)) if cond_operation == '&' else np.empty(0)
        for c in self.criteria:
            callback = self.criteria_func_data_mapping[c]
            criterion_indices = callback(client_index)
            if cond_operation == '&':
                indices = filter_by_array(indices, criterion_indices)
            else:
                criteria_union = np.concatenate((indices, criterion_indices))
                indices = np.unique(criteria_union)

        self.criteria_indices = indices
        self.data_changed = False  # criteria are now updated according to the latest change

        return indices

    def is_buy(self, client_index: int, index: int = None) -> bool:
        if ALWAYS_BUY:
            return True
        if index is None:
            index = self.get_num_candles(client_index)-1 if index is None else index
        is_begin_day = self.clients[client_index].is_in_first_n_candles(n=N_FIRST_CANDLES_OF_DAY, candle_index=index)
        if is_begin_day:
            return False
        is_end_day = self.clients[client_index].is_in_last_n_candles(n=N_LAST_CANDLES_OF_DAY, candle_index=index)
        if is_end_day:
            return False
        op = '|' if self.is_bar_strategy() else '&'
        approved_indices = self._is_criteria(cond_operation=op, client_index=client_index) if self.data_changed else self.criteria_indices
        curr_index = self.get_num_candles()-1 if index is None else index
        # check if latest index which represents the current candle meets the criteria
        ret_val = np.isin([curr_index], approved_indices)[0]
        if self.block_buy:
            if not ret_val:
                self.block_buy = False
            else:
                self.logger.info(f"Blocking buy operation; criteria already met\n")
        return ret_val and not self.block_buy

    def is_sell(self, client_index: int, index: int = None, sell_on_touch=SELL_ON_TOUCH) -> bool:
        if index is None:
            index = self.get_num_candles(client_index)-1 if index is None else index
        if self.clients[client_index].is_day_last_transaction(index):
            return True
        curr_index = self.get_num_candles(client_index) - 1 if index is None else index
        if sell_on_touch:
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

    def buy(self, client_index: int, index: int = None, sl_min_rel_pos=None, real_time=False) -> int:
        if index is None:
            index = self.get_num_candles(client_index)-1 if index is None else index
        low_prices = self.clients[client_index].get_low_price()
        stock_price = self.get_close_price(client_index, index)

        if sl_min_rel_pos is None:
            if len(low_prices) < STOP_LOSS_RANGE:
                stop_loss_range = low_prices
            else:
                stop_loss_range = low_prices[np.maximum(0, index-STOP_LOSS_RANGE):index]
            local_min = np.min(stop_loss_range)
        else:
            local_min = self.get_close_price(client_index, index+sl_min_rel_pos)

        if local_min > stock_price:  # in case of inconsistent data
            return TRADE_NOT_COMPLETE

        # TODO don't buy if stop loss percentage is >= X, put in LS
        loss_percentage = 1-(local_min/stock_price)+STOP_LOSS_PERCENTAGE_MARGIN
        if loss_percentage > self.stop_loss_bound:
            return TRADE_NOT_COMPLETE
        self.stop_loss = stock_price * (1-loss_percentage)
        self.take_profit = stock_price*((loss_percentage * self.take_profit_multiplier)+1)

        if self.use_pyramid:
            ru_dollars = get_percent(self.capital, self.risk_unit*self.ru_percentage)
            num_stocks = np.floor(np.minimum((ru_dollars / (1-(self.stop_loss/stock_price)) / stock_price),
                                  self.capital / stock_price))
            if num_stocks == 0:
                return TRADE_NOT_COMPLETE
            self.latest_trade[client_index] = [num_stocks * stock_price, num_stocks, stock_price]
        else:
            num_stocks = np.floor(self.capital / stock_price)
            if num_stocks == 0:
                return TRADE_NOT_COMPLETE
            self.latest_trade[client_index] = [self.capital, num_stocks, stock_price]

        self.capital -= self.latest_trade[client_index][AMOUNT]
        self.status[client_index] = SellStatus.BOUGHT
        self.logger.info(f"Buy date: {self.clients[client_index].get_candle_date(index)}\n"
                         f"Trade no.: {self.get_num_trades()+1}\n"
                         f"Buy amount: {self.latest_trade[client_index][AMOUNT]}\n"
                         f"Stock price: {stock_price}")

        if real_time:
            self.clients[client_index].buy_order(self.latest_trade[client_index][NUM_STOCKS],
                                                 float(format(self.take_profit, '.2f')),
                                                 float(format(self.stop_loss, '.2f')),
                                                 price=float(format(self.get_close_price(client_index, index-1), '.2f')))

        self.block_buy = True
        return TRADE_COMPLETE

    def sell(self, client_index: int, index: int = None, real_time=False) -> int:
        if index is None:
            index = self.get_num_candles(client_index)-1 if index is None else index
        if self.status[client_index] != SellStatus.BOUGHT:
            self.logger.info("Tried selling before buying a stock")
            return 0

        curr_index = self.get_num_candles(client_index) - 1 if index is None else index
        stock_price = self.get_close_price(client_index, curr_index)
        sell_price = stock_price*self.latest_trade[client_index][NUM_STOCKS]
        self.status[client_index] = SellStatus.SOLD

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
            self.capital_history[self.get_num_trades()] = self.capital
        self.logger.info(f"Sale date: {self.clients[client_index].get_candle_date(index)}\nSale amount: {sell_price}\n"
                         f"Trade no.: {self.get_num_trades()}\n"
                         f"Stock price: {stock_price}")

        return profit

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

    def calc_revenue(self, candle_range: Tuple[int, int] = None) -> float:
        if candle_range:
            start, end = candle_range
        else:
            start, end = 0, self.get_num_candles() - 1

        for i in range(start, end):
            for j in range(len(self.clients)):
                if self.status[j] in (SellStatus.SOLD, SellStatus.NEITHER) and not self.is_client_occupied():  # try to buy
                    condition = self.is_buy(index=i, client_index=j)
                    if condition:
                        ret_val = self.buy(index=i, sl_min_rel_pos=-2 if self.is_bar_strategy() else None, client_index=j)
                        if ret_val == TRADE_COMPLETE:  # in case trade was not complete
                            self.logger.info(f"Current capital: {self.capital}\nStocks bought: {int(self.latest_trade[j][NUM_STOCKS])}\nStock name: {self.clients[j].name}\n")
                elif self.status[j] == SellStatus.BOUGHT:  # try to sell
                    condition = self.is_sell(index=i, client_index=j)
                    if condition:
                        price = self.sell(index=i, client_index=j)
                        self.logger.info(f"Current capital: {self.capital}\nSell type: {GAIN if price > 0 else LOSS}\nStock name: {self.clients[j].name}\n")
            # update the number of trades per day
            if i > 0 and self.clients[0].is_day_last_transaction(i-1):  # first candle of the day
                curr_day_index = np.where(self.trades_per_day == -1)[0][0]
                self.trades_per_day[curr_day_index] = self.get_num_trades() - (np.sum(self.trades_per_day[:curr_day_index]))

        self.log_summary()
        return self.capital

    def main_loop(self) -> float:
        is_eod = False
        while not is_eod:
            for j in range(len(self.clients)):
                self.clients[j].add_candle()
                self.logger.info(
                    f"Current candle: {self.clients[j].get_candle_date(-1)}; Stock: {self.clients[j].name}")
                if self.status[j] in (SellStatus.SOLD, SellStatus.NEITHER) and not self.is_client_occupied():  # try to buy
                    condition = self.is_buy(client_index=j)
                    if condition:
                        ret_val = self.buy(sl_min_rel_pos=-2 if self.is_bar_strategy() else None, client_index=j, real_time=True)
                        if ret_val == TRADE_COMPLETE:  # in case trade was not complete
                            self.logger.info(f"Current capital: {self.capital}\nStocks bought: {int(self.latest_trade[j][NUM_STOCKS])}\nStock name {self.clients[j].name}\n")
                elif self.status[j] == SellStatus.BOUGHT:  # try to sell
                    condition = self.is_sell(client_index=j)
                    if condition:
                        profit = self.sell(client_index=j, real_time=True)
                        self.logger.info(f"Current capital: {self.capital}\nSell type: {GAIN if profit > 0 else LOSS}\nStock name {self.clients[j].name}\n")
                if self.clients[j].is_day_last_transaction(-1):
                    is_eod = True
            time.sleep(10)
        return self.capital

    def log_summary(self):
        risk_chance = (np.average(self.gains[:self.num_gains]) / np.average(self.losses[:self.num_losses]))\
                        if self.num_gains > 0 and self.num_losses > 0 else 0
        tpd_valid = self.trades_per_day[:np.where(self.trades_per_day == -1)[0][0]]
        gains = self.gains[:self.num_gains]
        losses = self.losses[:self.num_losses]
        self.logger.info(f"Win percentage: {format(self.num_gains / (self.num_gains + self.num_losses), '.3f')}\n"
                         f"Total trades: {self.get_num_trades()}\n"
                         f"Winning trades: {self.num_gains}\nLosing trades: {self.num_losses}\n"
                         f"Winning end of day trades: {self.num_eod_gains}\n"
                         f"Losing end of day trades: {self.num_eod_losses}\n"
                         f"End of day gain / loss: {format(self.num_eod_gains / (self.num_eod_gains + self.num_eod_losses), '.3f')}\n"
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
        sb.calc_revenue()
        res = sb.capital_history[:sb.get_num_trades()]
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
    plt.plot(sb.capital_history[:sb.get_num_trades()])
    plt.xlabel('no. of trades')
    plt.ylabel('capital')
    plt.savefig("graph.png",  bbox_inches='tight')


def filter_stocks(stocks: List[str], val: float):
    res = []

    for stock in stocks:
        sb = StockBot(stock_clients=[StockClientYfinance(name=stock)], period=period, use_pyr=True, criteria=DEFAULT_CRITERIA_LIST)
        sb.calc_revenue()
        if sb.get_profit_percentage() >= val:
            res.append(stock)
    return res


if __name__ == '__main__':
    curr_time = get_curr_utc_2_timestamp()  # current time in utc+2
    period = f'{"3 D" if REAL_TIME else "60d"}'

    from stock_client import StockClientYfinance, StockClientInteractive

    if FILTER_STOCKS[2]:
        stock_list, filter_val = FILTER_STOCKS[:2]
        filtered = filter_stocks(stock_list, filter_val)
        print(f"stocks that were filtered: {filtered}")

    if RUN_ROBOT:
        real_time_client = StockClientInteractive.init_client() if REAL_TIME else None
        clients = [StockClientYfinance(name=name) if not REAL_TIME else StockClientInteractive(name=name, client=real_time_client) for name in STOCKS]

        sb = StockBot(stock_clients=clients, period=period, criteria=DEFAULT_CRITERIA_LIST)
        if REAL_TIME:
            res = sb.main_loop()
        else:
            res = sb.calc_revenue()

        if OUTPUT_PLOT:
            plot_capital(sb)
