import yfinance as yf
from typing import List, Union
from abc import ABC, abstractmethod
import pandas as pd

from consts import TimeRes
from utils import convert_timestamp_format


class StockClient(ABC):
    __slots__: List[str] = ['name', 'candles', '_client']

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
    def get_low_price(self):
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

    def get_low_price(self) -> pd.Series:
        return self.candles['l']

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

    def get_closing_price(self) -> pd.Series:
        return self.candles['Close']

    def get_high_price(self) -> pd.Series:
        return self.candles['High']

    def get_low_price(self) -> pd.Series:
        return self.candles['Low']

    @staticmethod
    def res_to_str(res: TimeRes) -> str:
        if res == TimeRes.MINUTE_5:
            return '5m'
        elif res == TimeRes.MINUTE_15:
            return '15m'

    def set_candle_data(self, res: TimeRes, period: Union[str, int] = None, start: int = None, end: int = None):
        parsed_res = self.res_to_str(res)
        if start is not None:
            formatted_start = convert_timestamp_format(start)
            formatted_end = convert_timestamp_format(end)
            self.candles = self._client.history(start=formatted_start, end=formatted_end, interval=parsed_res)
        else:
            self.candles = self._client.history(period=period, interval=parsed_res)
