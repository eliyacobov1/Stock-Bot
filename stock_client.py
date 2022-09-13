import numpy as np
import yfinance as yf
from typing import List, Union
from abc import ABC, abstractmethod
import pandas as pd

from utils import minutes_to_secs


class StockClient(ABC):
    __slots__: List[str] = ['name', 'candles', '_client']

    @property
    def client(self):
        return self._client

    @abstractmethod
    def set_candle_data(self, res: str, period: Union[str, int] = None, start: int = None, end: int = None):
        pass

    @abstractmethod
    def get_closing_price(self):
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

    def set_candle_data(self, res: str, period: Union[str, int] = None, start: int = None, end: int = None) -> None:
        if period:
            pass
        else:
            self.candles = self._client.stock_candles(symbol=self.name, resolution=res, _from=start, to=end)


class StockClientYfinance(StockClient):
    def __init__(self, name: str):
        super(StockClient, self).__init__()
        self.name = name
        self._client = yf.Ticker(name)

    def get_closing_price(self):
        return self.candles['Close']

    def set_candle_data(self, res: str, period: Union[str, int] = None, start: int = None, end: int = None):
        if start is not None:
            self.candles = self._client.history(start=start, end=end, interval=res)
        else:
            self.candles = self._client.history(period=period, interval=res)
