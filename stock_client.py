import datetime
import logging

from functools import lru_cache

import yfinance as yf
import ib_insync
from typing import List, Union, Optional, Callable
from abc import ABC, abstractmethod
import pandas as pd

from consts import TimeRes, TradeTypes, SellStatus
from utils import convert_timestamp_format, get_take_profit


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
    def buy_order(self, quantity: float, stop_loss: float, price: float = None, market_order=True):
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

    def buy_order(self, quantity: float, stop_loss: float, price: float = None, market_order=True):
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

        self._take_profit_observers: List[Callable[[float], None]] = []  # will be triggered when take profit is calculated
        self._sell_observer: Optional[Callable[[int], int]] = None  # will be triggered when sell is performed
        self._cancel_observer: Optional[Callable[[int], None]] = None  # will be triggered on cancel
        self._res = None

        self.current_trades = {TradeTypes.BUY: [], TradeTypes.SELL: []}

        self.demo = demo
        self.logger = logging.getLogger('__main__')

    def bind_tp_observer(self, callback: Callable[[float], None]):
        self._take_profit_observers.append(callback)

    def bind_sell_observer(self, callback: Callable[[int], int]):
        self._sell_observer = callback

    def bind_cancel_observer(self, callback: Callable[[SellStatus], None]):
        self._cancel_observer = callback

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

    def buy_order(self, quantity: float, stop_loss: float, price: float = None, market_order=True):
        action = 'BUY'
        reverse_action = 'SELL'
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
                transmit=False,
                outsideRth=True)
        sl_order = ib_insync.StopOrder(
            reverse_action, quantity, stop_loss,
            orderId=self._client.client.getReqId(),
            transmit=True,
            parentId=parent.orderId,
            outsideRth=True)

        parent_trade = self._execute_order(parent, wait_until_done=False)
        parent_trade.filledEvent += self.buy_callback
        self._set_trade(trade_type=TradeTypes.BUY, trade=parent_trade, append=True)

        sl_trade = self._execute_order(sl_order)
        sl_trade.filledEvent += self.sell_callback
        sl_trade.cancelledEvent += self.cancel_callback
        self._set_trade(trade_type=TradeTypes.SELL, trade=sl_trade, append=True)

        return parent_trade, sl_trade

    def _resubmit_trade(self, trade: ib_insync.Trade):
        order = trade.order
        order.orderId = self._client.client.getReqId()

        updated_trade = self._execute_order(order, wait_until_done=False)
        updated_trade.filledEvent += self.sell_callback
        self._set_trade(trade_type=TradeTypes.SELL, trade=updated_trade, replace=True)

    def cancel_callback(self, trade: ib_insync.Trade):
        self.logger.info("Order Cancelled (client)!")
        action = trade.order.action
        if action == 'BUY':
            self._cancel_observer()
        elif action == 'SELL':
            self._resubmit_trade(trade)

    def _set_trade(self, trade_type: TradeTypes, trade: ib_insync.Trade, append=False, replace=False):
        if replace:
            # first, remove the existing trade
            for i in range(len(self.current_trades[trade_type])):
                if self.current_trades[trade_type][i].order.orderId == trade.order.orderId:
                    del self.current_trades[trade_type][i]
                    break
        if append or replace:
            self.current_trades[trade_type].append(trade)
        else:
            self.current_trades[trade_type] = [trade]

    def _reset_trades(self, trade_type: TradeTypes):
        self.current_trades[trade_type].clear()

    def get_curr_buy_price(self) -> Optional[float]:
        trade: ib_insync.Trade = self.current_trades[TradeTypes.BUY][0]
        return trade.fills[0].execution.avgPrice if trade.fills else None

    def get_curr_stop_loss_price(self) -> float:
        trade: ib_insync.Trade = [trade for trade in self.current_trades[TradeTypes.SELL] if type(trade.order) == ib_insync.StopOrder][0]
        return trade.order.auxPrice

    def get_curr_take_profit_price(self) -> float:
        trade: ib_insync.Trade = [trade for trade in self.current_trades[TradeTypes.SELL] if type(trade.order) == ib_insync.LimitOrder][0]
        return trade.order.auxPrice

    def buy_callback(self, trade: ib_insync.Trade):
        # price = fill.execution.avgPrice
        price = trade.orderStatus.avgFillPrice
        stop_loss = self.get_curr_stop_loss_price()

        status = trade.orderStatus.status

        self.logger.info(f"!!!!!! landing ing analyze buy with avg price of {price}, status {status}")

        # transmit the take-profit sell order
        take_profit = float(format(get_take_profit(curr_price=price, stop_loss=stop_loss), '.2f'))
        for callback in self._take_profit_observers:  # update observers with the newly calculated take-profit
            callback(take_profit)

        quantity = trade.order.totalQuantity
        trade_id = trade.order.orderId
        tp_order = ib_insync.LimitOrder(
            'SELL', quantity, take_profit,
            orderId=self._client.client.getReqId(),
            # parentId=trade_id,  # TODO check this
            transmit=True,
            outsideRth=True)

        tp_trade = self._execute_order(tp_order)
        tp_trade.filledEvent += self.sell_callback
        tp_trade.cancelledEvent += self.cancel_callback
        self._set_trade(trade_type=TradeTypes.SELL, trade=tp_trade, append=True)

        self.logger.info(f"API update: order bought for {price} with status {status}")

    def get_cash(self):
        cash = [i.value for i in self._client.accountValues() if (i.tag == 'TotalCashBalance' and i.currency == 'USD')][0]
        return cash

    def sell_callback(self, trade: ib_insync.Trade):
        self._reset_trades(trade_type=TradeTypes.BUY)

        price = trade.orderStatus.avgFillPrice  # TODO check this
        status = trade.orderStatus.status

        # assert that all other sales are canceled and if not, cancel them
        for t in self.current_trades[TradeTypes.SELL]:
            if t.order.orderId != trade.order.orderId:  # and not t.isActive()
                self._client.cancelOrder(t.order)

        self._reset_trades(trade_type=TradeTypes.SELL)
        self._sell_observer()  # TODO

        self.logger.info(f"API update: order sold for {price} with status {status}")

    def get_current_quantity(self) -> Optional[float]:
        buy_trade: ib_insync.Trade = self.current_trades[TradeTypes.BUY][0] if len(self.current_trades[TradeTypes.BUY]) else None

        if buy_trade:
            return buy_trade.order.totalQuantity
        else:
            return None

    def sell_order(self):
        """
        executes a sell market order
        """
        # TODO use this function
        quantity = self.get_current_quantity()
        self._sell_observer()

        if quantity:
            order = ib_insync.MarketOrder(
                'SELL', quantity,
                orderId=self._client.client.getReqId(),
                transmit=True,
                outsideRth=True)
            trade = self._execute_order(order)
        else:
            self.logger.info("could not execute sell market trade; Bought quantity is undefined")

    def _execute_order(self, order: ib_insync.Order, wait_until_done=False) -> ib_insync.Trade:
        trade = self.client.placeOrder(self._stock, order)
        if wait_until_done:
            while not trade.isDone():
                self._client.sleep(1)
                self.logger.info("waiting")
                self._client.waitOnUpdate()
        self.logger.info(f"Order of type {order.orderType} executed!")
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
            self.logger.info("waiting for data")
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
