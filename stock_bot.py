from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas_ta import rsi, macd, supertrend, ema
import logging

from utils import (minutes_to_secs, days_to_secs, filter_by_array, get_curr_utc_2_timestamp, get_percent)
from consts import (DEFAULT_RES, DEFAULT_STOCK_NAME, MACD_INDEX, MACD_SIGNAL_INDEX, SellStatus, CRITERIA, LOGGER_NAME,
                    STOP_LOSS_RANGE, TAKE_PROFIT_MULTIPLIER, SUPERTREND_COL_NAME, DEFAULT_RISK_UNIT,
                    DEFAULT_RISK_LIMIT, DEFAULT_START_CAPITAL, DEFAULT_CRITERIA_LIST, DEFAULT_USE_PYRAMID,
                    DEFAULT_GROWTH_PERCENT, DEFAULT_RU_PERCENTAGE, GAIN, LOSS, EMA_LENGTH)
from stock_client import StockClient

RESOLUTIONS = {15}

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
    def __init__(self, stock_client: StockClient, start: int = None, end: int = None, period: str = None, rsi_win_size: int = 10,
                 ema_win_size: int = 10, risk_unit: float = None, risk_limit: float = None, start_capital: float = None,
                 use_pyr: bool = True, ru_growth: float = None, monitor_revenue: bool = False, criteria: Optional[List[str]] = None,
                 log_to_file=False):
        self.ema = None
        self.prices = None
        self.gain_avg = None
        self.loss_avg = None
        self.resolution = DEFAULT_RES
        self.start_time = start
        self.end_time = end
        self.period = period
        self.rsi_window_size, self.ema_window_size = rsi_win_size, ema_win_size

        self.ru_percentage = DEFAULT_RU_PERCENTAGE
        self.risk_unit = self.initial_risk_unit = risk_unit if risk_unit else DEFAULT_RISK_UNIT
        self.risk_limit = risk_limit if risk_limit else DEFAULT_RISK_LIMIT
        self.risk_growth = ru_growth if ru_growth else DEFAULT_GROWTH_PERCENT

        self.monitor_revenue = monitor_revenue
        self.capital = self.initial_capital = start_capital if start_capital else DEFAULT_START_CAPITAL

        self.latest_trade: Tuple[int, int] = (-1, -1)  # sum paid and number of stock bought on latest buy trade

        self.use_pyramid = use_pyr

        # these attributes will be used when determining whether to sell
        self.stop_loss = None
        self.take_profit = None
        self.status = SellStatus.NEITHER

        self.client = stock_client

        self.set_candle_data()

        num_candles = self.client.get_num_candles()
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
                                           CRITERIA.REVERSAL_BAR: self.get_inside_bar_criterion}
        self.ema = None

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

    def get_close_price(self, index=-1):
        closing_price = self.client.get_closing_price()
        return closing_price[index]

    # @lru_cache(maxsize=1)
    def set_candle_data(self):
        if self.period:
            self.client.set_candle_data(res=self.resolution, start=self.start_time, end=self.end_time, period=period)
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

    def get_rsi_criterion(self):
        s_close_old = pd.Series(self.client.get_closing_price())
        rsi_results = np.array(rsi(s_close_old))
        return np.where(rsi_results > 50)

    def get_supertrend_criterion(self):
        high = self.client.get_high_price()
        low = self.client.get_low_price()
        close = self.client.get_closing_price()
        supertrend_results_df = supertrend(pd.Series(high), pd.Series(low), pd.Series(close))
        supertrend_results = supertrend_results_df[SUPERTREND_COL_NAME]

        indices = np.where(supertrend_results < close)

        return np.array(indices)

    def get_macd_criterion(self):
        close_prices = self.client.get_closing_price()

        macd_close = macd(close=pd.Series(close_prices), fast=12, slow=26, signal=9)
        macd_data = np.array(macd_close.iloc[:, MACD_INDEX])
        signal_data = np.array(macd_close.iloc[:, MACD_SIGNAL_INDEX])

        indices = np.where((macd_data > signal_data) & (macd_data < 0))
        indices_as_nd = np.array(indices)

        return indices_as_nd

    def get_ema(self):
        self.ema = ema(self.client.get_closing_price(), EMA_LENGTH)
        return self.ema

    def get_inside_bar_criterion(self):
        close_prices = self.client.get_closing_price()
        open_prices = self.client.get_opening_price()
        ema = self.ema if self.ema is not None else self.get_ema()

        positive_candles = close_prices[:-1] - open_prices[:-1]
        close_price_diff = np.diff(close_prices)
        # 1st candle positive + 2nd candle in 1st candle price range [open, close]
        cond_1 = np.where((positive_candles > 0) & (close_price_diff < 0) & (np.diff(open_prices) > 0))[0] + 1
        # 3rd candle close bigger than both 1st candle close and ema
        cond_2 = np.where((close_prices < np.roll(close_prices, -2)) & (np.roll(close_prices, -2) > np.roll(ema, -2)))[0] + 1

        indices = np.intersect1d(cond_1, cond_2)

        return indices

    def get_reversal_bar_criterion(self):
        close_prices = self.client.get_closing_price()
        high_prices = self.client.get_high_price()
        low_prices = self.client.get_low_price()
        ema = self.ema if self.ema is not None else self.get_ema()

        # 1st candle high+low bigger than 2nd candle high+low
        cond_1 = np.where((np.diff(high_prices) < 0) & (np.diff(low_prices) < 0))[0] + 1
        # 3rd candle close bigger than both 1st candle high and ema
        cond_2 = np.where((high_prices < np.roll(close_prices, -2)) & (np.roll(close_prices, -2) > np.roll(ema, -2)))[0] + 1

        indices = np.intersect1d(cond_1, cond_2)

        return indices

    def get_num_candles(self):
        """
        returns the number of candles that are currently observed
        """
        return len(self.client.get_closing_price())

    def _is_criteria(self) -> np.ndarray:
        indices = np.arange(self.get_num_candles())
        for c in self.criteria:
            callback = self.criteria_func_data_mapping[c]
            criterion_indices = callback()
            indices = filter_by_array(indices, criterion_indices)

        self.criteria_indices = indices
        self.data_changed = False  # criteria are now updated according to the latest change

        return indices

    def is_buy(self, index: int = None) -> bool:
        approved_indices = self._is_criteria() if self.data_changed else self.criteria_indices
        curr_index = self.get_num_candles()-1 if index is None else index
        # check if latest index which represents the current candle meets the criteria
        return np.isin([curr_index], approved_indices)[0]

    def is_sell(self, index: int = None) -> bool:
        if self.client.is_day_last_transaction(index):
            return True
        curr_index = self.get_num_candles() - 1 if index is None else index
        latest = self.get_close_price(curr_index)
        return latest <= self.stop_loss or latest >= self.take_profit

    def buy(self, index: int = None) -> Tuple[int, int]:
        closing_prices = self.client.get_closing_price()
        stock_price = self.get_close_price(index)

        if len(closing_prices) < STOP_LOSS_RANGE:
            stop_loss_range = closing_prices
        else:
            stop_loss_range = closing_prices[np.maximum(0, index-STOP_LOSS_RANGE):index]

        # TODO don't buy if stop loss percentage is >= X, put in LS
        local_min = np.min(stop_loss_range)
        loss_percentage = 1-(local_min/stock_price)+0.005
        self.stop_loss = stock_price * (1-loss_percentage)
        self.take_profit = stock_price*((loss_percentage * TAKE_PROFIT_MULTIPLIER)+1)

        if self.use_pyramid:
            ru_dollars = get_percent(self.capital, self.risk_unit*self.ru_percentage)
            num_stocks = (ru_dollars / (1-(self.stop_loss/stock_price))) / stock_price  # TODO check this calculation
            self.latest_trade = (num_stocks * stock_price, num_stocks)
        else:
            num_stocks = self.capital / stock_price
            self.latest_trade = (self.capital, num_stocks)

        self.status = SellStatus.BOUGHT
        self.logger.info(f"Buy date: {self.client.get_candle_date(index)}\n"
                         f"Buy price: {self.latest_trade[0]}")

        return self.latest_trade

    def sell(self, index: int = None) -> int:
        if self.status != SellStatus.BOUGHT:
            self.logger.info("Tried selling before buying a stock")
            return 0

        curr_index = self.get_num_candles() - 1 if index is None else index
        stock_price = self.get_close_price(curr_index)
        sell_price = stock_price*self.latest_trade[1]
        self.status = SellStatus.SOLD
        profit = (sell_price / self.latest_trade[0]) - 1
        is_eod = self.client.is_day_last_transaction(index)

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

        self.logger.info(f"Sale date: {self.client.get_candle_date(index)}\nSale Price: {sell_price}")

        return sell_price

    def calc_revenue(self, candle_range: Tuple[int, int] = None):
        if candle_range:
            start, end = candle_range
        else:
            start, end = 0, self.get_num_candles() - 1

        for i in range(start, end):
            if self.status in (SellStatus.SOLD, SellStatus.NEITHER):  # try to buy
                condition = self.is_buy(index=i)
                if condition:
                    price, num_stocks = self.buy(index=i)
                    self.capital -= price
                    self.logger.info(f"Current capital: {self.capital}\nStocks bought: {num_stocks}")
            elif self.status == SellStatus.BOUGHT:  # try to sell
                condition = self.is_sell(index=i)
                if condition:
                    price = self.sell(index=i, )
                    self.capital += price
                    self.logger.info(f"Current capital: {self.capital}\nSell type: {GAIN if price >= self.latest_trade[0] else LOSS}\n")

        self.log_summary()
        return self.capital

    def log_summary(self):
        self.logger.info(f"Win percentage: {self.num_gains / (self.num_gains + self.num_losses)}\n"
                         f"Winning trades: {self.num_gains}\nLosing trades: {self.num_losses}\n"
                         f"Winning end of day trades: {self.num_eod_gains}\n"
                         f"Losing end of day trades: {self.num_eod_losses}\n"
                         f"End of day gain / loss: {self.num_eod_gains / (self.num_eod_gains + self.num_eod_losses)}\n"
                         f"Risk / Chance: "
                         f"{np.average(self.gains[:self.num_gains]) / np.average(self.losses[:self.num_losses])}\n"
                         f"Total profit: {self.capital-self.initial_capital}\n")

    def update_time(self, n: int = 15):
        """
        advance current time window by given number of minutes
        """
        period = minutes_to_secs(n)
        self.start_time += period
        self.end_time += period


if __name__ == '__main__':
    curr_time = get_curr_utc_2_timestamp()  # current time in utc+2
    period = '60d'

    from stock_client import StockClientYfinance
    client: StockClient = StockClientYfinance(name=DEFAULT_STOCK_NAME)

    sb = StockBot(stock_client=client, period=period, use_pyr=True, criteria=DEFAULT_CRITERIA_LIST)
    res = sb.calc_revenue()
