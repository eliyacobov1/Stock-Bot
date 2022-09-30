from enum import Enum

MACD_INDEX, MACD_SIGNAL_INDEX = 0, 2
SUPERTREND_COL_NAME = 'SUPERT_7_3.0'
EMA_SMOOTHING, INITIAL_RSI_WINDOW_SIZE, INITIAL_EMA_WINDOW_SIZE = 2, 14, 200
DEFAULT_STOCK_NAME = "SOXL"
LOGGER_NAME = "STOCKBOT_LOG"

# sell parameters
STOP_LOSS_RANGE = 20
TAKE_PROFIT_MULTIPLIER = 1.5


class TimeRes(Enum):
    MINUTE_1 = 0
    MINUTE_5 = 1
    MINUTE_15 = 2


DEFAULT_RES = TimeRes.MINUTE_15


class SellStatus(Enum):
    BOUGHT = 0
    SOLD = 1
    NEITHER = 2


class CRITERIA(Enum):
    MACD = 1
    RSI = 2
    SUPER_TREND = 3

    @staticmethod
    def factory(name: str):
        normalized_name = name.lower().strip()
        if normalized_name == "rsi":
            return CRITERIA.RSI
        elif normalized_name == "macd":
            return CRITERIA.MACD
        elif normalized_name == "supertrend":
            return CRITERIA.SUPER_TREND
        else:  # TODO create and raise custom exception
            pass
