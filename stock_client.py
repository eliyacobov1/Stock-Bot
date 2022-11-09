import datetime
import logging

from functools import lru_cache

import yfinance as yf
import ib_insync
from typing import List, Union
from abc import ABC, abstractmethod
import pandas as pd

from consts import TimeRes
from utils import convert_timestamp_format


class StockClient(ABC):
    __slots__: List[str] = ['name', 'candles', '_client', '_res']

    @property
    def client(self):
        return self._client

    @abstractmethod
    def set_candle_data(self, res: TimeRes, period: Union[str, int] = None, start: int = None, end: int = None):
        pass

    @staticmethod
    @abstractmethod
    def res_to_str(res: TimeRes) -> str:
        pass

    @abstractmethod
    def get_closing_price(self):
        pass

    @abstractmethod
    def get_opening_price(self) -> pd.Series:
        pass

    @abstractmethod
    def get_high_price(self):
        pass

    @abstractmethod
    def get_candle_date(self, i: int) -> str:
        pass

    @abstractmethod
    def get_low_price(self):
        pass

    @abstractmethod
    def is_day_last_transaction(self, i: int) -> bool:
        pass

    @abstractmethod
    def get_num_candles(self) -> int:
        pass

    @staticmethod
    @abstractmethod
    def res_to_period(res: TimeRes) -> str:
        pass

    @abstractmethod
    def parse_date(self, date: str = None, candle_index: int = None) -> datetime.time:
        pass

    @abstractmethod
    @lru_cache(maxsize=None)
    def _n_last_candles(self, n: int = 3) -> datetime.time:
        pass

    @abstractmethod
    @lru_cache(maxsize=None)
    def _n_first_candles(self, n: int = 3) -> datetime.time:
        pass

    def is_in_first_n_candles(self, candle_index: int, n: int = 3) -> bool:
        candle_date = self.parse_date(candle_index=candle_index)
        return candle_date < self._n_first_candles(n=n)

    def is_in_last_n_candles(self, candle_index: int, n: int = 3) -> bool:
        candle_date = self.parse_date(candle_index=candle_index)
        return candle_date >= self._n_last_candles(n=n)

    def is_close_date(self, date_str: str = None, candle_index: int = None):
        if date_str is None:
            date = self.parse_date(candle_index=candle_index)
        else:
            date = self.parse_date(date=date_str)
        if self._res == TimeRes.MINUTE_1:
            return date.second == 0
        if self._res == TimeRes.MINUTE_5:
            return date.minute % 5 == 0 and date.second == 0
        elif self._res == TimeRes.MINUTE_15:
            return date.minute % 15 == 0 and date.second == 0

    def add_candle(self) -> True:
        pass

    @abstractmethod
    def buy_order(self, quantity: float, stop_loss: float, take_profit: float, price: float = None):
        pass

    @abstractmethod
    def sell_order(self):
        pass


class StockClientFinhub(StockClient):
    def __init__(self, name: str):
        from secret import FINHUB_API_KEY  # you need to create secret.py and put your finhub api key there
        import finnhub

        super(StockClient, self).__init__()
        self.name = name
        self._client = finnhub.Client(api_key=FINHUB_API_KEY)
        self.candles = None

    def get_closing_price(self) -> pd.Series:
        return self.candles['c']

    def get_opening_price(self) -> pd.Series:
        return self.candles['o']

    def get_high_price(self) -> pd.Series:
        return self.candles['h']

    def get_num_candles(self) -> int:
        return self.candles.shape[0]

    def get_candle_date(self, i: int) -> str:
        pass

    def get_low_price(self) -> pd.Series:
        return self.candles['l']

    def is_day_last_transaction(self, i: int) -> bool:
        pass

    @staticmethod
    def res_to_str(res: TimeRes) -> str:
        if res == TimeRes.MINUTE_5:
            return '5'
        elif res == TimeRes.MINUTE_15:
            return '15'

    def set_candle_data(self, res: TimeRes, period: Union[str, int] = None, start: int = None, end: int = None) -> None:
        parsed_res = self.res_to_str(res)
        if period:
            pass
        else:
            self.candles = self._client.stock_candles(symbol=self.name, resolution=parsed_res, _from=start, to=end)


class StockClientYfinance(StockClient):
    def __init__(self, name: str):
        super(StockClient, self).__init__()
        self.name = name
        self._client = yf.Ticker(name)
        self._res = None

    def get_closing_price(self) -> pd.Series:
        return self.candles['Close']

    def get_opening_price(self) -> pd.Series:
        return self.candles['Open']

    def get_high_price(self) -> pd.Series:
        return self.candles['High']

    def get_low_price(self) -> pd.Series:
        return self.candles['Low']

    def is_day_last_transaction(self, i: int) -> bool:
        last_transaction_time: str = str()
        if self._res == TimeRes.MINUTE_5:
            last_transaction_time = "15:55"
        elif self._res == TimeRes.MINUTE_15:
            last_transaction_time = "15:45"
        elif self._res == TimeRes.MINUTE_1:
            last_transaction_time = "15:59"
        return last_transaction_time in self.get_candle_date(i)

    def get_candle_date(self, i: int) -> str:
        return str(self.candles.iloc[i].name)

    @staticmethod
    def res_to_str(res: TimeRes) -> str:
        if res == TimeRes.MINUTE_1:
            return '1m'
        if res == TimeRes.MINUTE_5:
            return '5m'
        elif res == TimeRes.MINUTE_15:
            return '15m'

    def get_num_candles(self) -> int:
        return self.candles.shape[0]

    def set_candle_data(self, res: TimeRes, period: Union[str, int] = None, start: int = None, end: int = None):
        self._res = res
        parsed_res = self.res_to_str(res)
        if start is not None:
            formatted_start = convert_timestamp_format(start)
            formatted_end = convert_timestamp_format(end)
            self.candles = self._client.history(start=formatted_start, end=formatted_end, interval=parsed_res)
        else:
            self.candles = self._client.history(period=period, interval=parsed_res)

    @staticmethod
    def res_to_period(res: TimeRes) -> str:
        if res == TimeRes.MINUTE_1:
            return '1m'
        if res == TimeRes.MINUTE_5:
            return '5m'
        elif res == TimeRes.MINUTE_15:
            return '15m'

    def parse_date(self, date: str = None, candle_index: int = None) -> datetime.time:
        if date is None:
            date_str = str(self.candles.iloc[candle_index].name).split("-")[-2].split()[-1].split(":")
        else:
            date_str = date.split("-")[-2].split()[-1].split(":")
        h, m, s = date_str
        return datetime.time(hour=int(h), minute=int(m), second=int(s))

    @lru_cache(maxsize=None)
    def _n_last_candles(self, n: int = 3) -> datetime.time:
        res_to_int = 1 if self._res == TimeRes.MINUTE_1 else (5 if self._res == TimeRes.MINUTE_5 else 15)
        h = (res_to_int * n) // 60
        m = (res_to_int * n) % 60
        end_time = datetime.time(hour=15-h + (60-m) // 60, minute=(60-m) % 60, second=0)
        return end_time

    @lru_cache(maxsize=None)
    def _n_first_candles(self, n: int = 3) -> datetime.time:
        res_to_int = 1 if self._res == TimeRes.MINUTE_1 else (5 if self._res == TimeRes.MINUTE_5 else 15)
        h = (res_to_int * n) // 60
        m = (res_to_int * n) % 60
        start_time = datetime.time(hour=9+h+(30+m)//60, minute=(30+m) % 60, second=0)
        return start_time

    def add_candle(self) -> True:
        candles = self._client.history(period=self.res_to_period(self._res), interval=self.res_to_str(self._res))
        latest_date = self.parse_date(candle_index=-1)
        if self.parse_date(str(candles.iloc[-1].name)) <= latest_date:
            return False
        df_rows = [candles.iloc[-1]]
        if not self.is_close_date(candle_index=len(self.candles) - 1):
            self.candles.drop(self.candles.tail(n=1).index, inplace=True)
        for i in range(len(candles) - 2, -1, -1):
            row_date = str(self.candles.iloc[i].name)
            if self.is_close_date(date_str=row_date) and self.parse_date(row_date) > latest_date:
                df_rows = [candles.iloc[i]] + df_rows
        for row in df_rows:
            if self.is_close_date(date_str=str(row.name)):
                self.candles.drop(self.candles.head(n=1).index, inplace=True)
            self.candles = self.candles.append(row)
        return True

    def buy_order(self, quantity: float, stop_loss: float, take_profit: float, price: float = None):
        pass

    def sell_order(self):
        pass


class StockClientInteractive(StockClient):
    def __init__(self, name: str, demo=True, client: ib_insync.IB = None):
        super(StockClient, self).__init__()
        if client is not None:
            ib = client
        else:
            ib = ib_insync.IB()
        if not ib.isConnected():
            ib.connect('127.0.0.1', 7497 if demo else 7496, clientId=1)
        self.name = name
        self._stock = ib_insync.Stock(self.name, 'SMART', 'USD')
        self._client = ib
        self._res = None
        self.current_orders = {}
        self.demo = demo
        self.logger = logging.getLogger('__main__')

    @staticmethod
    def init_client() -> ib_insync.IB:
        ib = ib_insync.IB()
        return ib

    def get_closing_price(self) -> pd.Series:
        return self.candles['close']

    def get_opening_price(self) -> pd.Series:
        return self.candles['open']

    def get_high_price(self) -> pd.Series:
        return self.candles['high']

    def get_low_price(self) -> pd.Series:
        return self.candles['low']

    def is_day_last_transaction(self, i: int) -> bool:
        last_transaction_time: str = str()
        if self._res == TimeRes.MINUTE_5:
            last_transaction_time = "22:55:"
        elif self._res == TimeRes.MINUTE_15:
            last_transaction_time = "22:45:"
        elif self._res == TimeRes.MINUTE_1:
            last_transaction_time = "22:59:"
        return last_transaction_time in self.get_candle_date(i)

    def get_candle_date(self, i: int) -> str:
        return str(self.candles.iloc[i].date)

    def buy_order(self, quantity: float, take_profit: float, stop_loss: float, price: float = None, market_order=True):
        action = 'BUY'
        reverse_action = 'BUY' if action == 'SELL' else 'SELL'
        if market_order:
            parent = ib_insync.MarketOrder(
                action, quantity,
                orderId=self._client.client.getReqId(),
                transmit=False,
                outsideRth=True)
        else:
            parent = ib_insync.LimitOrder(
                action, quantity, price,
                orderId=self._client.client.getReqId(),
                transmit=True,
                outsideRth=True)
        tp_order = ib_insync.LimitOrder(
            reverse_action, quantity, take_profit,
            orderId=self._client.client.getReqId(),
            transmit=False,
            parentId=parent.orderId,
            outsideRth=True)
        sl_order = ib_insync.StopOrder(
            reverse_action, quantity, stop_loss,
            orderId=self._client.client.getReqId(),
            transmit=True,
            parentId=parent.orderId,
            outsideRth=True)

        parent_trade = self._execute_order(parent, wait_until_done=False)
        tp_trade = self._execute_order(tp_order)
        sl_trade = self._execute_order(sl_order)

        # parent_trade.fillEvent += lambda x, y: print(x)
        parent_trade.fillEvent += self.analyze_output
        tp_trade.fillEvent += lambda x, y: print("sold")
        sl_trade.fillEvent += lambda x, y: print("sold")

        return ib_insync.BracketOrder(parent, tp_order, sl_order)

    def analyze_output(self, trade: ib_insync.Trade, fill: ib_insync.Fill):
        price = fill.execution.price
        status = trade.orderStatus.status
        self.current_orders[trade.order.orderId] = trade.order
        print(f"API update: order bought for {price} with status {status}")

    def sell_order(self):
        pass

    def _execute_order(self, order: ib_insync.Order, wait_until_done=False) -> ib_insync.Trade:
        trade = self.client.placeOrder(self._stock, order)
        if wait_until_done:
            while not trade.isDone():
                self._client.sleep(1)
                print("waiting")
                self._client.waitOnUpdate()
        return trade

    @staticmethod
    def res_to_str(res: TimeRes) -> str:
        if res == TimeRes.MINUTE_1:
            return '1 min'
        if res == TimeRes.MINUTE_5:
            return '5 mins'
        elif res == TimeRes.MINUTE_15:
            return '15 mins'

    @staticmethod
    def res_to_period(res: TimeRes) -> str:
        if res == TimeRes.MINUTE_1:
            return '60 S'
        if res == TimeRes.MINUTE_5:
            return '300 S'
        elif res == TimeRes.MINUTE_15:
            return '900 S'

    def parse_date(self, date: str = None, candle_index: int = None) -> datetime.time:
        if date is None:
            date_str = str(self.candles.iloc[candle_index].date).split("-")[2].split()[-1].split(":")
        else:
            date_str = date.split("-")[2].split()[-1].split(":")
        h, m, s = date_str
        return datetime.time(hour=int(h), minute=int(m), second=int(s))

    @lru_cache(maxsize=None)
    def _n_last_candles(self, n: int = 3) -> datetime.time:
        res_to_int = 1 if self._res == TimeRes.MINUTE_1 else (5 if self._res == TimeRes.MINUTE_5 else 15)
        h = (res_to_int * n) // 60
        m = (res_to_int * n) % 60
        end_time = datetime.time(hour=22 - h + (60 - m) // 60, minute=(60 - m) % 60, second=0)
        return end_time

    @lru_cache(maxsize=None)
    def _n_first_candles(self, n: int = 3) -> datetime.time:
        res_to_int = 1 if self._res == TimeRes.MINUTE_1 else (5 if self._res == TimeRes.MINUTE_5 else 15)
        h = (res_to_int * n) // 60
        m = (res_to_int * n) % 60
        start_time = datetime.time(hour=16 + h + (30 + m) // 60, minute=(30 + m) % 60, second=0)
        return start_time

    def get_num_candles(self) -> int:
        return self.candles.shape[0]

    def add_candle(self) -> True:
        self.logger.info("fetching latest candle data...\n")
        candles = self._client.reqHistoricalData(self._stock, endDateTime='', durationStr=self.res_to_period(self._res),
                                                 barSizeSetting=self.res_to_str(self._res), whatToShow='MIDPOINT', useRTH=False)
        while len(candles) == 0:
            print("waiting for data")
            self._client.sleep(2)
        df_candles = ib_insync.util.df(candles)
        latest_date = self.parse_date(candle_index=-1)
        if self.parse_date(str(df_candles.iloc[-1].date)) == latest_date:
            self.candles.drop(self.candles.tail(n=1).index, inplace=True)
        elif len(df_candles) > 1 and self.parse_date(str(df_candles.iloc[-2].date)) == latest_date:
            self.candles.drop(self.candles.tail(n=1).index, inplace=True)
            self.candles = self.candles.append(df_candles.iloc[-2], ignore_index=True)
        elif len(df_candles) > 1:
            self.candles = self.candles.append(df_candles.iloc[-2], ignore_index=True)
        self.candles = self.candles.append(df_candles.iloc[-1], ignore_index=True)
        return True

    def set_candle_data(self, res: TimeRes, period: Union[str, int] = None, start: int = None, end: int = None):
        self.logger.info("fetching initial candle data...\n")
        self._res = res
        parsed_res = self.res_to_str(res)

        formatted_end = convert_timestamp_format(end) if end is not None else ''
        candles = self._client.reqHistoricalData(self._stock, endDateTime=formatted_end, durationStr=period,
                                                 barSizeSetting=parsed_res, whatToShow='MIDPOINT', useRTH=True)
        while len(candles) == 0:
            self._client.waitOnUpdate()
        self.candles = ib_insync.util.df(candles)
