from enum import Enum

MACD_INDEX, MACD_SIGNAL_INDEX = 0, 2
SUPERTREND_COL_NAME = 'SUPERT_7_3.0'
LONG_STOCK_NAME = "SOXL"
SHORT_STOCK_NAME = "SOXS"
LOGGER_NAME = "STOCKBOT_LOG"

GAIN, LOSS = "Gain", "Loss"
TRADE_NOT_COMPLETE = -1
TRADE_COMPLETE = 0

# sell parameters
STOP_LOSS_RANGE = 20
STOP_LOSS_PERCENTAGE_MARGIN = 0.0035
TAKE_PROFIT_MULTIPLIER = 0.8
STOP_LOSS_LOWER_BOUND = 0.04

OUTPUT_PLOT = True

FILTER_STOCKS = (["AMD", "soxl", "soxs", "wix", "aapl"], 0.09, False)
RUN_ROBOT = True

USE_RUN_WINS = True
RUN_WINS_TAKE_PROFIT_MULTIPLIER = 0.5
RUN_WINS_PERCENT = 0.5

MACD_PARAMS = (12, 26, 9)  # fast, slow, signal
SUPERTREND_PARAMS = (7, 3)  # length, multiplier
RSI_PARAMS = 14  # length

STRATEGY_1 = ["rsi", "supertrend", "macd"]
STRATEGY_2 = ["insidebar", "reversalbar"]
DEFAULT_CRITERIA_LIST = STRATEGY_2

STOCKS = [LONG_STOCK_NAME, SHORT_STOCK_NAME]

REAL_TIME = False

SELL_ON_TOUCH = True

DEFAULT_RU_PERCENTAGE = 2
DEFAULT_RISK_UNIT = 1
DEFAULT_RISK_LIMIT = 5
DEFAULT_GROWTH_PERCENT = 1.2

DEFAULT_USE_PYRAMID = True  # set to False if you don't want to use pyramid

DEFAULT_START_CAPITAL = 1000

DEFAULT_CANDLE_SIZE = 5  # 1, 5 or 15

N_FIRST_CANDLES_OF_DAY = 0
N_LAST_CANDLES_OF_DAY = 1

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
