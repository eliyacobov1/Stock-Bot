import time
from typing import List, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import finnhub
from pandas_ta import rsi, macd, supertrend
from functools import lru_cache
import logging

from utils import (get_closing_price, get_high_price, get_low_price, minutes_to_secs, filter_by_array)
from consts import (DEFAULT_RES, DEFAULT_STOCK_NAME, MACD_INDEX, MACD_SIGNAL_INDEX, EMA_SMOOTHING,
                    SellStatus, INITIAL_EMA_WINDOW_SIZE, INITIAL_RSI_WINDOW_SIZE, CRITERIA, LOGGER_NAME,
                    STOP_LOSS_RANGE, TAKE_PROFIT_MULTIPLIER)

RESOLUTIONS = {15}

API_KEY = "c76vsr2ad3iaenbslifg"


def add_candle(close_price, status=SellStatus.NEITHER):
    pass


def plot_total_gain_percentage(gains):
    xy = (np.random.random((10, 2))).cumsum(axis=0)
    bought = [0, 2]
    fig, ax = plt.subplots()
    for i, (start, stop) in enumerate(zip(xy[:-1], xy[1:])):
        x, y = zip(start, stop)
        ax.plot(x, y, color='black' if i in bought else 'red')
    plt.show()


class StockBot:
    def __init__(self, stock_name: str, start: int, end: int, rsi_win_size: int = 10, ema_win_size: int = 10,
                 criteria: Optional[List[str]] = None):
        self.stock_name = stock_name
        self.ema = None
        self.prices = None
        self.gain_avg = None
        self.loss_avg = None
        self.resolution = str(DEFAULT_RES)
        self.start_time = start
        self.end_time = end
        self.rsi_window_size, self.ema_window_size = rsi_win_size, ema_win_size

        # these attributes will be used when determining whether to sell
        self.stop_loss = None
        self.take_profit = None
        self.status = SellStatus.NEITHER

        self.client = finnhub.Client(api_key=API_KEY)

        self.candles = None
        self.set_candle_data()

        # this dict maps a criterion to the method that returns the candle indices that fulfil it
        self.criteria_func_data_mapping = {CRITERIA.RSI: self.get_rsi_criterion,
                                           CRITERIA.SUPER_TREND: self.get_supertrend_criterion,
                                           CRITERIA.MACD: self.get_macd_criterion}

        self.criteria: List[CRITERIA] = []
        self.set_criteria(criteria)

        # initialize logger
        logging.basicConfig(filename=LOGGER_NAME,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)
        self.logger = logging.getLogger('start logging')
        self.logger.setLevel(logging.INFO)

    def get_latest_close_price(self):
        closing_price = get_closing_price(self.candles)
        return closing_price[-1]

    @lru_cache(maxsize=1)
    def set_candle_data(self):
        self.candles = self.client.stock_candles(self.stock_name, self.resolution, self.start_time, self.end_time)

    def set_criteria(self, criteria: Optional[List[str]]) -> None:
        if criteria is None:
            self.criteria.append(CRITERIA.RSI)
            return
        else:
            for name in criteria:
                c = CRITERIA.factory(name)
                self.criteria.append(c)

    def get_rsi_criterion(self):
        s_close_old = pd.Series(get_closing_price(self.candles))
        rsi_results = np.array(rsi(s_close_old))
        return np.where(rsi_results > 50)

    def get_supertrend_criterion(self):
        # TODO need to finish implementing this
        high = get_high_price(data=self.candles)
        low = get_low_price(data=self.candles)
        close = get_closing_price(data=self.candles)
        supertrend_results = supertrend(high, low, close)
        return supertrend_results

    def get_macd_criterion(self):
        close_prices = get_closing_price(self.candles)

        macd_close = macd(close=close_prices, fast=12, slow=26, signal=9)
        macd_data = np.array(macd_close.iloc[:, MACD_INDEX])
        signal_data = np.array(macd_close.iloc[:, MACD_SIGNAL_INDEX])

        macd_data_diff = macd_data - signal_data
        macd_data_diff_shifted = macd_data_diff[:-1]
        macd_data_diff = macd_data_diff[1:]
        macd_data = macd_data[1:]

        indices = np.where((macd_data_diff > 0) & (macd_data < 0) & (macd_data_diff_shifted <= 0))

        return indices + 1

    def get_num_candles(self):
        """
        returns the number of candles that are currently observed
        """
        return len(get_closing_price(self.candles))

    def _is_criteria(self) -> np.ndarray:
        indices = np.arange(self.get_num_candles())
        for c in self.criteria:
            callback = self.criteria_func_data_mapping[c]
            criterion_indices = callback()
            indices = filter_by_array(indices, criterion_indices)
        return indices

    def is_buy(self) -> bool:
        approved_indices = self._is_criteria()
        latest_index = self.get_num_candles()-1
        # check if latest index which represents the current candle meets the criteria
        return np.isin([latest_index], approved_indices)[0]

    def is_sell(self) -> bool:
        latest = self.get_latest_close_price()
        return latest <= self.stop_loss or latest >= self.take_profit

    def buy(self) -> None:
        closing_prices = get_closing_price(self.candles)
        latest = self.get_latest_close_price()

        if len(closing_prices) < STOP_LOSS_RANGE:
            stop_loss_range = closing_prices
        else:
            stop_loss_range = closing_prices[:(-1*STOP_LOSS_RANGE)]

        self.stop_loss = np.min(stop_loss_range)
        loss_percentage = 1-(self.stop_loss/latest)
        self.take_profit = latest*(loss_percentage*TAKE_PROFIT_MULTIPLIER)

        self.status = SellStatus.BOUGHT

    def sell(self) -> None:
        if self.status == SellStatus.BOUGHT:
            self.logger.info("Tried selling before buying a stock")
            return

    def update_time(self, n: int = 15):
        """
        advance current time window by given number of minutes
        """
        period = minutes_to_secs(n)
        self.start_time += period
        self.end_time += period

    @staticmethod
    def calc_shift_percentage(val1, val2):
        """
        Calculates the relative percentage between val1 and val2
        """
        return abs((val1 / val2) - 1) * 100

    @staticmethod
    def get_next_avg(curr_avg, val, window_size):
        """
        This function returns the weighted average of 'curr_avg' along with the new value 'val'
         according to the given window size
        """
        return (curr_avg * (window_size - 1) + val) / window_size

    def update_ema(self, val):
        """
        Calculates and updates the next EMA value according to the given new stock value
        """
        coefficient = EMA_SMOOTHING / (1 + self.ema_window_size)
        self.ema = val * coefficient + self.ema * (1 - coefficient)
        return self.ema

    def get_initial_rsi(self, initial_stock_values):
        """
        Calculates the initial RSI value according to a given starting period window of the stock.
        :return the initial RSI value, gain average and loss average of the stock
        """
        losses, gains, period_size = [], [], len(initial_stock_values)
        for open_price, close_price in initial_stock_values:
            diff_percentage = self.calc_shift_percentage(close_price, open_price)
            if close_price >= open_price:
                gains.append(diff_percentage)
            else:
                losses.append(diff_percentage)
        self.gain_avg, self.loss_avg = sum(gains) / len(gains), sum(losses) / len(losses)
        rsi = 100 - (100 / (1 + (self.gain_avg / self.loss_avg)))
        return rsi

    def update_rsi(self, gain, loss, window_size):
        """
        Calculates and updates the next RSI value according to the given new gain and loss
        """
        self.gain_avg = self.get_next_avg(self.gain_avg, gain, window_size)
        self.loss_avg = self.get_next_avg(self.loss_avg, loss, window_size)
        return self.get_rsi_criterion()

    def main(self):
        pass


if __name__ == '__main__':
    curr_time = int(time.time())
    period = minutes_to_secs(4000)
    sb = StockBot(stock_name=DEFAULT_STOCK_NAME, start=curr_time-period, end=curr_time)
    res = sb.is_buy()
    x = 0
