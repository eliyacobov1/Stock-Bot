import yfinance as yf
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
