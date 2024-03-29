from enum import Enum

MACD_INDEX, MACD_SIGNAL_INDEX, MACD_H_INDEX = 0, 2, 1
SUPERTREND_COL_NAME = 'SUPERT_7_3.0'
LONG_STOCK_NAME = "SOXL"
SHORT_STOCK_NAME = "SOXS"
LOGGER_NAME = "STOCKBOT_LOG"

GAIN, LOSS = "Gain", "Loss"
TRADE_NOT_COMPLETE = -1
TRADE_COMPLETE = 0

USE_DL_MODEL = True
CLASSIFIER_THRESH_MIN = 0
CLASSIFIER_THRESH_MAX = 0.3
WEIGHT_FILE_PATH = "./dl_utils/weights.h5"
DEFUALT_AI_DATA_PATH = "./ai_csv.csv"

# sell parameters
STOP_LOSS_RANGE = 10
STOP_LOSS_PERCENTAGE_MARGIN = 0.002
TAKE_PROFIT_MULTIPLIER = 1
STOP_LOSS_LOWER_BOUND = 0.07
ATR_MUL = 2.5
ATR_PERIOD = 14

OUTPUT_PLOT = True

FILTER_STOCKS = (["AMD", "soxl", "soxs", "wix", "aapl"], 0.09, False)
RUN_ROBOT = True

USE_RUN_WINS = False
RUN_WINS_TAKE_PROFIT_MULTIPLIER = 0.5
RUN_WINS_PERCENT = 0.5

MACD_PARAMS = (12, 26, 9)  # fast, slow, signal
SUPERTREND_PARAMS = (7, 3)  # length, multiplier
RSI_PARAMS = 14  # length

#  TODO run each stock with it's own strategy
#   send sell command on first sell of rw strategy
#   enable selection of date range
#   add candles every 5 minutes by checking datetime time
#   add max loss/win streak
#   buy when reaching a certain margin from stop-loss
#   how to set conditions dynamically according to user input
#   further optimize logs- criteria calculations, central logic functions
#   wait for buy order to initiate before sending sell orders- Done
#   change limit sell order to market on eod
#   ib.portfolio()
#   validate current trades vs. ib_trades.api
#   check how to write to dashboard datasheet csv
#   convert all not self.real_time into an encapsulated property

STRATEGY_1 = ["rsi", "supertrend", "macd"]
STRATEGY_2 = ["insidebar", "reversalbar"]
DEFAULT_CRITERIA_LIST = STRATEGY_1

DATA_STOCKS_TO_TRADED_MAPPING = {"SOXL": "SOXS"}

REAL_TIME = True
ALWAYS_BUY = False  # for debugging

DEBUG = False

SELL_ON_TOUCH = True

DEFAULT_RU_PERCENTAGE = 4
DEFAULT_RISK_UNIT = 1
DEFAULT_RISK_LIMIT = 3
DEFAULT_GROWTH_PERCENT = 1.25

DEFAULT_USE_PYRAMID = True  # set to False if you don't want to use pyramid

DEFAULT_START_CAPITAL = 1000

DEFAULT_CANDLE_SIZE = 5  # 1, 5 or 15

N_FIRST_CANDLES_OF_DAY = 1
N_LAST_CANDLES_OF_DAY = 3

EMA_LENGTH = 200
CANDLE_DATA_CSV_NAME = 'candle_data.csv'
TRADE_DATA_CSV_NAME = 'trade_data.csv'
TRADE_SUMMARY_CSV_NAME = 'trade_summary.csv'
VIX = "^VIX"

# period of candle history to fetch on init
REAL_TIME_PERIOD = 10
HISTORY_PERIOD = 60
PYR_RISK_UNIT_CALCULATION_PERIOD = 30


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


class TimeZones(Enum):
    ISRAEL = 0
    AMERICA = 1


class TradeTypes(Enum):
    BUY = 0
    SELL = 1


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
