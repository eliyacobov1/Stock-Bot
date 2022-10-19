from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas_ta import rsi, macd, supertrend, ema
import logging

from utils import (minutes_to_secs, days_to_secs, filter_by_array, get_curr_utc_2_timestamp, get_percent)
from consts import (DEFAULT_RES, LONG_STOCK_NAME, MACD_INDEX, MACD_SIGNAL_INDEX, SellStatus, CRITERIA, LOGGER_NAME,
                    STOP_LOSS_RANGE, TAKE_PROFIT_MULTIPLIER, SUPERTREND_COL_NAME, DEFAULT_RISK_UNIT,
                    DEFAULT_RISK_LIMIT, DEFAULT_START_CAPITAL, DEFAULT_CRITERIA_LIST, DEFAULT_USE_PYRAMID,
                    DEFAULT_GROWTH_PERCENT, DEFAULT_RU_PERCENTAGE, GAIN, LOSS, EMA_LENGTH, STOP_LOSS_PERCENTAGE_MARGIN,
                    SHORT_STOCK_NAME, STOP_LOSS_LOWER_BOUND, TRADE_NOT_COMPLETE, OUTPUT_PLOT, STOCKS)
from stock_client import StockClient

API_KEY = "c76vsr2ad3iaenbslifg"


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
                 use_pyr: bool = True, ru_growth: float = None, monitor_revenue: bool = False, criteria: Optional[List[str]] = None,
                 log_to_file=False, tp_multiplier=TAKE_PROFIT_MULTIPLIER, sl_bound=STOP_LOSS_LOWER_BOUND):
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

        self.latest_trade: List[Tuple[int, int]] = [(-1, -1) for i in range(len(self.clients))]  # sum paid and number of stock bought on latest buy trade

        self.use_pyramid = use_pyr

        # these attributes will be used when determining whether to sell
        self.stop_loss = None
        self.take_profit = None
        self.status = [SellStatus.NEITHER for i in range(len(self.clients))]

        self.set_candle_data()

        num_candles = self.clients[0].get_num_candles()
        self.capital_history = np.zeros(num_candles)
        self.capital_history[0] = self.capital
        self.gains = np.zeros(num_candles)
        self.eod_gains = np.zeros(num_candles)
        self.losses = np.zeros(num_candles)
        self.eod_losses = np.zeros(num_candles)
        self.num_gains = self.num_eod_gains = self.num_losses = self.num_eod_losses = 0

        self.data_changed = True  # indicates whether new candle data was fetched or not
        self.criteria_indices = None

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
        return closing_price[index]

    def set_tp_multiplier(self, val: float):
        self.take_profit_multiplier = val

    def reset(self):
        self.num_gains = self.num_eod_gains = self.num_losses = self.num_eod_losses = 0
        self.status = [SellStatus.NEITHER for i in range(len(self.clients))]
        self.capital = self.initial_capital

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
        rsi_results = np.array(rsi(s_close_old))
        return np.where(rsi_results > 50)

    def get_supertrend_criterion(self, client_index: int):
        high = self.clients[client_index].get_high_price()
        low = self.clients[client_index].get_low_price()
        close = self.clients[client_index].get_closing_price()
        supertrend_results_df = supertrend(pd.Series(high), pd.Series(low), pd.Series(close))
        supertrend_results = supertrend_results_df[SUPERTREND_COL_NAME]

        indices = np.where(supertrend_results < close)

        return np.array(indices)

    def get_macd_criterion(self, client_index: int):
        close_prices = self.clients[client_index].get_closing_price()

        macd_close = macd(close=pd.Series(close_prices), fast=12, slow=26, signal=9)
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
        op = '|' if self.is_bar_strategy() else '&'
        approved_indices = self._is_criteria(cond_operation=op, client_index=client_index) if self.data_changed else self.criteria_indices
        curr_index = self.get_num_candles()-1 if index is None else index
        # check if latest index which represents the current candle meets the criteria
        return np.isin([curr_index], approved_indices)[0]

    def is_sell(self, client_index: int, index: int = None) -> bool:
        if self.clients[client_index].is_day_last_transaction(index):
            return True
        curr_index = self.get_num_candles(client_index) - 1 if index is None else index
        latest = self.get_close_price(client_index, curr_index)
        return latest <= self.stop_loss or latest >= self.take_profit

    def get_num_trades(self):
        return self.num_gains + self.num_eod_gains + self.num_losses + self.num_eod_losses

    def buy(self, client_index: int, index: int = None, sl_min_rel_pos=None) -> Tuple[int, int]:
        closing_prices = self.clients[client_index].get_closing_price()
        stock_price = self.get_close_price(client_index, index)

        if sl_min_rel_pos is None:
            if len(closing_prices) < STOP_LOSS_RANGE:
                stop_loss_range = closing_prices
            else:
                stop_loss_range = closing_prices[np.maximum(0, index-STOP_LOSS_RANGE):index]
            local_min = np.min(stop_loss_range)
        else:
            local_min = self.get_close_price(client_index, index+sl_min_rel_pos)

        if local_min > stock_price:  # in case of inconsistent data
            return TRADE_NOT_COMPLETE, TRADE_NOT_COMPLETE

        # TODO don't buy if stop loss percentage is >= X, put in LS
        loss_percentage = 1-(local_min/stock_price)+STOP_LOSS_PERCENTAGE_MARGIN
        if loss_percentage > self.stop_loss_bound:
            return TRADE_NOT_COMPLETE, TRADE_NOT_COMPLETE
        self.stop_loss = stock_price * (1-loss_percentage)
        self.take_profit = stock_price*((loss_percentage * self.take_profit_multiplier)+1)

        if self.use_pyramid:
            ru_dollars = get_percent(self.capital, self.risk_unit*self.ru_percentage)
            num_stocks = np.floor(np.minimum((ru_dollars / (1-(self.stop_loss/stock_price)) / stock_price),
                                  self.capital / stock_price))
            if num_stocks == 0:
                return TRADE_NOT_COMPLETE, TRADE_NOT_COMPLETE
            self.latest_trade[client_index] = (num_stocks * stock_price, num_stocks)
        else:
            num_stocks = self.capital / stock_price
            if num_stocks == 0:
                return TRADE_NOT_COMPLETE, TRADE_NOT_COMPLETE
            self.latest_trade[client_index] = (self.capital, num_stocks)

        self.capital -= self.latest_trade[client_index][0]
        self.status[client_index] = SellStatus.BOUGHT
        self.logger.info(f"Buy date: {self.clients[client_index].get_candle_date(index)}\n"
                         f"Buy price: {self.latest_trade[client_index][0]}")

        return self.latest_trade[client_index]

    def sell(self, client_index: int, index: int = None) -> int:
        if self.status[client_index] != SellStatus.BOUGHT:
            self.logger.info("Tried selling before buying a stock")
            return 0

        curr_index = self.get_num_candles(client_index) - 1 if index is None else index
        stock_price = self.get_close_price(client_index, curr_index)
        sell_price = stock_price*self.latest_trade[client_index][1]
        self.status[client_index] = SellStatus.SOLD
        profit = (sell_price / self.latest_trade[client_index][0]) - 1
        is_eod = self.clients[client_index].is_day_last_transaction(index)

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
            if self.use_pyramid:
                self.risk_unit = np.minimum(self.risk_unit*self.risk_growth, self.risk_limit)
            if not is_eod:
                self.losses[self.num_losses] = -profit
                self.num_losses += 1
            else:
                self.eod_losses[self.num_eod_losses] = -profit
                self.num_eod_losses += 1

        self.capital += sell_price
        self.capital_history[self.get_num_trades()] = self.capital
        self.logger.info(f"Sale date: {self.clients[client_index].get_candle_date(index)}\nSale Price: {sell_price}")

        return sell_price

    def is_bar_strategy(self):
        return CRITERIA.REVERSAL_BAR in self.criteria and\
               CRITERIA.INSIDE_BAR in self.criteria and\
               len(self.criteria) == 2

    def calc_revenue(self, candle_range: Tuple[int, int] = None):
        if candle_range:
            start, end = candle_range
        else:
            start, end = 0, self.get_num_candles() - 1

        for i in range(start, end):
            for j in range(len(self.clients)):
                if self.status[j] in (SellStatus.SOLD, SellStatus.NEITHER):  # try to buy
                    condition = self.is_buy(index=i, client_index=j)
                    if condition:
                        price, num_stocks = self.buy(index=i, sl_min_rel_pos=-2 if self.is_bar_strategy() else None, client_index=j)
                        if price != -1:  # in case trade was not complete
                            self.logger.info(f"Current capital: {self.capital}\nStocks bought: {int(num_stocks)}\nStock name {self.clients[j].name}\n")
                elif self.status[j] == SellStatus.BOUGHT:  # try to sell
                    condition = self.is_sell(index=i, client_index=j)
                    if condition:
                        price = self.sell(index=i, client_index=j)
                        self.logger.info(f"Current capital: {self.capital}\nSell type: {GAIN if price >= self.latest_trade[j][0] else LOSS}\nStock name {self.clients[j].name}\n")

        self.log_summary()
        return self.capital

    def log_summary(self):
        risk_chance = (np.average(self.gains[:self.num_gains]) / np.average(self.losses[:self.num_losses]))\
                        if self.num_gains > 0 and self.num_losses > 0 else 0
        self.logger.info(f"Win percentage: {self.num_gains / (self.num_gains + self.num_losses)}\n"
                         f"Winning trades: {self.num_gains}\nLosing trades: {self.num_losses}\n"
                         f"Winning end of day trades: {self.num_eod_gains}\n"
                         f"Losing end of day trades: {self.num_eod_losses}\n"
                         f"End of day gain / loss: {self.num_eod_gains / (self.num_eod_gains + self.num_eod_losses)}\n"
                         f"Gain average: {np.average(self.gains[:self.num_gains])}\n"
                         f"Gain max: {np.max(self.gains[:self.num_gains])}\n"
                         f"Gain min: {np.min(self.gains[:self.num_gains])}\n"
                         f"Loss average: {np.average(self.losses[:self.num_losses])}\n"
                         f"Loss max: {np.max(self.losses[:self.num_losses])}\n"
                         f"Loss min: {np.min(self.losses[:self.num_losses])}\n"
                         f"Risk / Chance: "
                         f"{risk_chance}\n"
                         f"Total profit: {self.capital-self.initial_capital}\n")

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


if __name__ == '__main__':
    curr_time = get_curr_utc_2_timestamp()  # current time in utc+2
    period = '60d'

    from stock_client import StockClientYfinance
    clients = [StockClientYfinance(name=name) for name in STOCKS]

    sb = StockBot(stock_clients=clients, period=period, use_pyr=True, criteria=DEFAULT_CRITERIA_LIST)
    res = sb.calc_revenue()

    if OUTPUT_PLOT:
        plot_capital(sb)
