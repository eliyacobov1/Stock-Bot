from enum import Enum

MACD_INDEX, MACD_SIGNAL_INDEX = 0, 2
SUPERTREND_COL_NAME = 'SUPERT_7_3.0'
DEFAULT_STOCK_NAME = "SOXL"
LOGGER_NAME = "STOCKBOT_LOG"

GAIN, LOSS = "Gain", "Loss"

# sell parameters
STOP_LOSS_RANGE = 20
STOP_LOSS_PERCENTAGE_MARGIN = 0.0035
TAKE_PROFIT_MULTIPLIER = 1.5

DEFAULT_CRITERIA_LIST = ["rsi", "macd", "supertrend"]

DEFAULT_RU_PERCENTAGE = 2
DEFAULT_RISK_UNIT = 1
DEFAULT_RISK_LIMIT = 5
DEFAULT_GROWTH_PERCENT = 1.2

DEFAULT_USE_PYRAMID = True  # set to False if you don't want to use pyramid

DEFAULT_START_CAPITAL = 1000

DEFAULT_CANDLE_SIZE = 5  # 1, 5 or 15

EMA_LENGTH = 200


class TimeRes(Enum):
    MINUTE_1 = 0
    MINUTE_5 = 1
    MINUTE_15 = 2


def int_to_res(val: int) -> TimeRes:
    if val == 1:
        return TimeRes.MINUTE_1
    elif val == 5:
        return TimeRes.MINUTE_5
    else:
        return TimeRes.MINUTE_15


DEFAULT_RES = int_to_res(DEFAULT_CANDLE_SIZE)


class SellStatus(Enum):
    BOUGHT = 0
    SOLD = 1
    NEITHER = 2


class CRITERIA(Enum):
    MACD = 1
    RSI = 2
    SUPER_TREND = 3
    INSIDE_BAR = 4
    REVERSAL_BAR = 5

    @staticmethod
    def factory(name: str):
        normalized_name = name.lower().strip()
        if normalized_name == "rsi":
            return CRITERIA.RSI
        elif normalized_name == "macd":
            return CRITERIA.MACD
        elif normalized_name == "supertrend":
            return CRITERIA.SUPER_TREND
        elif normalized_name == "insidebar":
            return CRITERIA.INSIDE_BAR
        elif normalized_name == "reversalbar":
            return CRITERIA.REVERSAL_BAR
        else:  # TODO create and raise custom exception
            pass
