import datetime

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
    def parse_date(self, candle_index: int) -> datetime.time:
        pass

    @abstractmethod
    @lru_cache(maxsize=None)
    def _n_last_candles(self, n: int = 3) -> datetime.time:
        pass

    @abstractmethod
    @lru_cache(maxsize=None)
    def _n_first_candles(self, n: int = 3) -> datetime.time:
        pass

    @abstractmethod
    def is_in_first_n_candles(self, candle_index: int, n: int = 3) -> bool:
        pass

    @abstractmethod
    def is_in_last_n_candles(self, candle_index: int, n: int = 3) -> bool:
        pass

    @abstractmethod
    def is_close_date(self, candle_index: int):
        pass

    @abstractmethod
    def add_candle(self):
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

    def parse_date(self, candle_index: int) -> datetime.time:
        date_str = str(self.candles.iloc[candle_index].name).split("-")[-2].split()[-1].split(":")
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

    def is_in_first_n_candles(self, candle_index: int, n: int = 3) -> bool:
        candle_date = self.parse_date(candle_index)
        return candle_date < self._n_first_candles(n=n)

    def is_in_last_n_candles(self, candle_index: int, n: int = 3) -> bool:
        candle_date = self.parse_date(candle_index)
        return candle_date >= self._n_last_candles(n=n)

    def is_close_date(self, candle_index: int):
        date = self.parse_date(candle_index)
        if self._res == TimeRes.MINUTE_1:
            return date.second == 0
        if self._res == TimeRes.MINUTE_5:
            return date.minute % 5 == 0 and date.second == 0
        elif self._res == TimeRes.MINUTE_15:
            return date.minute % 15 == 0 and date.second == 0


class StockClientInteractive(StockClient):
    def __init__(self, name: str):
        super(StockClient, self).__init__()
        ib = ib_insync.IB()
        ib.connect('127.0.0.1', 7496, clientId=1)
        self.name = name
        self._stock = ib_insync.Stock(self.name, 'SMART', 'USD')
        self._client = ib
        self._res = None

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
        return str(self.candles.iloc[i].name)

    @staticmethod
    def res_to_str(res: TimeRes) -> str:
        if res == TimeRes.MINUTE_1:
            return '1 min'
        if res == TimeRes.MINUTE_5:
            return '5 min'
        elif res == TimeRes.MINUTE_15:
            return '15 min'

    @staticmethod
    def res_to_period(res: TimeRes) -> str:
        if res == TimeRes.MINUTE_1:
            return '60 S'
        if res == TimeRes.MINUTE_5:
            return '300 S'
        elif res == TimeRes.MINUTE_15:
            return '900 S'

    def get_num_candles(self) -> int:
        return self.candles.shape[0]

    def set_candle_data(self, res: TimeRes, period: Union[str, int] = None, start: int = None, end: int = None):
        self._res = res
        parsed_res = self.res_to_str(res)

        formatted_end = convert_timestamp_format(end) if end is not None else ''
        candles = self._client.reqHistoricalData(self._stock, endDateTime=formatted_end, durationStr=period,
                                                 barSizeSetting=parsed_res, whatToShow='MIDPOINT', useRTH=True)
        self.candles = ib_insync.util.df(candles)
