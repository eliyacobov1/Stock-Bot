from typing import Dict
from ib_insync import *
from pandas_ta import *

import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler

from stock_client import StockClient

# parameters for historical data fetching
DURATION = '2 M'
CANDLE_SIZE = 5
BAR_SIZE = "5 mins"


ATR_PERIOD = 14
RSI_PERIOD = 14
RSI_COLUMN = "RSI_" + str(RSI_PERIOD)
SUPERTREND_PERIOD = 7
SUPERTREND_MULTIPLIER = 3.0
SUPERTREND_COLUMN = "SUPERT_" + str(SUPERTREND_PERIOD) + "_" + str(SUPERTREND_MULTIPLIER)
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MACD_COLUMN = "MACD_" + str(MACD_FAST) + "_" + str(MACD_SLOW) + "_" + str(MACD_SIGNAL)
MACDh_COLUMN = "MACDh_" + str(MACD_FAST) + "_" + str(MACD_SLOW) + "_" + str(MACD_SIGNAL)
MACDs_COLUMN = "MACDs_" + str(MACD_FAST) + "_" + str(MACD_SLOW) + "_" + str(MACD_SIGNAL)
EMA_PERIODS = [200, 100, 50, 20]
EMA_RIBBON_PERIODS = [20, 25, 30, 35, 40, 45, 50]
EMA_RIBBON_COLUMNS = ["EMA_RIBBON_" + str(x) for x in EMA_RIBBON_PERIODS]

OUTPUT_COLS = ['pct_change','acc_pct_change','pos_change_in_a_row','neg_change_in_a_row',RSI_COLUMN,'Close_Minus_SuperTrend',
                MACD_COLUMN, MACDh_COLUMN,MACDs_COLUMN,'Close_Minus_EMA_200','Close_Minus_EMA_100','Close_Minus_EMA_50','Close_Minus_EMA_20',
                'IS_EMA_Ribbon', 'Minute_Number']
PREDICTION_FEATURE = 'is_win'

STOP_LOSS = 0.9975
TAKE_PROFIT = 0.3
ATR_MUL = 2.5


@dataclass
class RawDataSample:
    close: float
    ema20: float
    ema50: float
    ema100: float
    ema200: float
    rsi: float
    supertrend: float
    macd: float
    macd_signal: float
    macd_h: float
    date: str
    ema25: float = 0
    ema30: float = 0
    ema35: float = 0
    ema40: float = 0
    ema45: float = 0
    num_candles_in_trend: float = 0  # number of candles in a continous trend slope
    percentage_diff: float = 0
    accumulated_percentage_diff: float = 0
    

def retrieve_candles(ib, contract) -> pd.DataFrame:
    print(f"[+] Retrieving historical data for [{contract.symbol}] for [{DURATION}] and [{BAR_SIZE}] bar size")
    # Set the time period for historical data
    end_time = datetime.datetime.now()
    # change the hour of the end_time to 22:55:00
    end_time = end_time.replace(hour=23, minute=00, second=0, microsecond=0)
    # get end_time minus 1 day
    end_time = end_time - datetime.timedelta(days=1)

    # Request historical data
    bars = ib.reqHistoricalData(contract,endDateTime=end_time.strftime('%Y%m%d %H:%M:%S'),durationStr=DURATION,barSizeSetting=BAR_SIZE,whatToShow='TRADES',useRTH=True)

    #convert to pandas dataframe
    df = util.df(bars)
    
    # Fix winter hours
    print(f"[+] Fixing winter hours")
    # if df['date'] is is between 31/10 - 4/11, then add 1 hour to the date column
    df['date'] = df['date'].apply(lambda x: x + datetime.timedelta(hours=1) if x >= datetime.datetime(2022, 10, 31, 0, 0, 0) and x <= datetime.datetime(2022, 11, 5, 0, 0, 0) else x)
    # if df['date'] is is between 25.11.2022 - 26.11.2022 the remove the row from the dataframe
    df = df[(df['date'] < datetime.datetime(2022, 11, 25, 0, 0, 0)) | (df['date'] > datetime.datetime(2022, 11, 26, 0, 0, 0))]
    
    df['date'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    # df.set_index('date', inplace=True)
    df.drop(['volume', 'average', 'barCount'], axis=1, inplace=True)

    # calculatr atr and add to dataframe
    print(f"[+] Calculating ATR for [{ATR_PERIOD}] period")
    df['ATR'] = atr(df['high'], df['low'], df['close'], length=ATR_PERIOD).apply(lambda x: round(x, 2) if x is not None else x)
 
    # percentage change
    print(f"[+] Calculating percent change")
    df['pct_change'] = df['close'].pct_change().apply(lambda x: round(x, 3) if x is not None else x)
    df['pct_change'] = df.apply(lambda x: 0 if x['date'].split(' ')[1] == '16:30:00' else x['pct_change'], axis=1)

    # accumulate percent
    print(f"[+] Calculating accumulate percent change")
    df['acc_pct_change'] = accum_reset(df['pct_change'].values)

    # positive change in a row and negative change in a row
    print(f"[+] Calculating positive and negative change in a row")
    df['pos_change_in_a_row'] = df['pct_change'].apply(lambda x: 1 if x > 0 else 0) != 0
    df['neg_change_in_a_row'] = df['pct_change'].apply(lambda x: 1 if x < 0 else 0) != 0
    df['pos_change_in_a_row'] = df['pos_change_in_a_row'].cumsum()-df['pos_change_in_a_row'].cumsum().where(~df['pos_change_in_a_row']).ffill().fillna(0).astype(int)
    df['neg_change_in_a_row'] = df['neg_change_in_a_row'].cumsum()-df['neg_change_in_a_row'].cumsum().where(~df['neg_change_in_a_row']).ffill().fillna(0).astype(int)

    # rsi
    print(f"[+] Calculating RSI for [{RSI_PERIOD}] period")
    rsi_data = rsi(df['close'], length=RSI_PERIOD)
    if rsi_data is not None:
        df[RSI_COLUMN] = rsi_data.apply(lambda x: round(x, 2) if x is not None else x)

        # add column that indicates if rsi is above 50 or not 
        df['Rsi_Above_50'] = df[RSI_COLUMN].apply(lambda x: True if x > 50 else False)
    
    # supertrend
    print(f"[+] Calculating Supertrend for [{SUPERTREND_PERIOD}] period and [{SUPERTREND_MULTIPLIER}] multiplier")
    supertrend_data = supertrend(df['high'], df['low'], df['close'], length=SUPERTREND_PERIOD, multiplier=SUPERTREND_MULTIPLIER)
    if supertrend_data is not None:
        df[SUPERTREND_COLUMN] = supertrend_data[SUPERTREND_COLUMN].apply(lambda x: round(x, 2) if x is not None else x)

        # add column that indicates if supertrend is above or below close price 
        df['SuperTrend_Below_Close'] = (df[SUPERTREND_COLUMN] < df['close']).apply(lambda x: True if x else False)
        df['Close_Minus_SuperTrend'] = df['close'] - df[SUPERTREND_COLUMN]

    # macd
    print(f"[+] Calculating MACD for [{MACD_FAST}], [{MACD_SLOW}] and [{MACD_SIGNAL}]")
    macd_data = macd(df['close'], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    if macd_data is not None:
        df[MACD_COLUMN] = macd_data[MACD_COLUMN].apply(lambda x: round(x, 3) if x is not None else x)
        df[MACDh_COLUMN] = macd_data[MACDh_COLUMN].apply(lambda x: round(x, 3) if x is not None else x)
        df[MACDs_COLUMN] = macd_data[MACDs_COLUMN].apply(lambda x: round(x, 3) if x is not None else x)

        # add column that indicates if macd is below signal line or not
        df['MACD_Above_Signal'] = (df[MACD_COLUMN] > df[MACDs_COLUMN]).apply(lambda x: True if x else False)

        # add column that indicates if macd is below zero or not
        df['MACD_Below_Zero'] = (df[MACD_COLUMN] < 0).apply(lambda x: True if x else False)

    # ema 
    for EMA_PERIOD in EMA_PERIODS:
        print(f"[+] Calculating EMA for [{EMA_PERIOD}] period")
        ema_data = ema(df['close'], length=EMA_PERIOD)
        if ema_data is not None:
            EMA_COLUMN = "EMA_" + str(EMA_PERIOD)
            df[EMA_COLUMN] = ema_data.apply(lambda x: round(x, 3) if x is not None else x)
            df['Close_Minus_'+EMA_COLUMN] = (df['close'] - df[EMA_COLUMN]).apply(lambda x: round(x, 3) if x is not None else x)

    # ema ribbon
    for EMA_RIBBON_PERIOD in EMA_RIBBON_PERIODS:
        print(f"[+] Calculating EMA Ribbon for [{EMA_RIBBON_PERIOD}] period")
        ema_ribbon_data = ema(df['close'], length=EMA_RIBBON_PERIOD)
        if ema_ribbon_data is not None:
            EMA_RIBBON_COLUMN = "EMA_RIBBON_" + str(EMA_RIBBON_PERIOD)
            df[EMA_RIBBON_COLUMN] = ema_ribbon_data.apply(lambda x: round(x, 3) if x is not None else x)


    df["IS_EMA_Ribbon"] = (df[EMA_RIBBON_COLUMNS[0]].astype(float) > df[EMA_RIBBON_COLUMNS[1]].astype(float)) & (df[EMA_RIBBON_COLUMNS[1]].astype(float) > df[EMA_RIBBON_COLUMNS[2]].astype(float)) & (df[EMA_RIBBON_COLUMNS[2]].astype(float) > df[EMA_RIBBON_COLUMNS[3]].astype(float)) & (df[EMA_RIBBON_COLUMNS[3]].astype(float) > df[EMA_RIBBON_COLUMNS[4]].astype(float)) & (df[EMA_RIBBON_COLUMNS[4]].astype(float) > df[EMA_RIBBON_COLUMNS[5]].astype(float)) &  (df[EMA_RIBBON_COLUMNS[5]].astype(float) > df[EMA_RIBBON_COLUMNS[6]].astype(float))

    return df


def accum_reset(values):
    acc = 0
    result = np.zeros_like(values)
    for i in range(len(values)):
        acc += values[i]
        if i > 0:
            if (values[i-1] >= 0 and values[i] < 0) or (values[i-1] < 0 and values[i] >= 0):
                acc = values[i]
        result[i] = acc
    return result

def parse_stop_loss_and_take_profit(df: pd.DataFrame):
    print(f"[+] Calculating stop loss and take profit for STOP_LOSS: [{STOP_LOSS}] and TAKE_PROFIT: [{TAKE_PROFIT}]")
    df['Stop_Loss'] = (df['close'] - df['ATR']*ATR_MUL).apply(lambda x: round(x, 2) if x is not None else x)
    df['Take_Profit'] = (( df['close'] +  df['ATR'] * ATR_MUL *TAKE_PROFIT)).apply(lambda x: round(x, 2) if x is not None else x)

    return df

def parse_stop_loss_hit_date(df: pd.DataFrame):
    print(f"[+] Calculating stop loss hit date")
    df['Stop_Loss_Hit_Date'] = None
    stop_loss_hit_date = np.empty(len(df), dtype= object)
    for i in range(len(df)):
        stop_loss_hit = np.where(df.iloc[i+1:]['low'] < df.iloc[i]['Stop_Loss'])[0]
        if len(stop_loss_hit) > 0:
            hit_date = df.iloc[stop_loss_hit[0]+i+1]['date']
            if(hit_date.split(' ')[0] == df.iloc[i]['date'].split(' ')[0]):
                stop_loss_hit_date[i] = hit_date
    df['Stop_Loss_Hit_Date'] = stop_loss_hit_date
    return df

def get_stop_loss_hit_row_number(df: pd.DataFrame):
    print(f"[+] Calculating stop loss hit row number")
    size = len(df)
    df['Stop_Loss_Hit_Row_Number'] = df['Stop_Loss_Hit_Date'].apply(lambda x: df[df['date'] == x].index[0] if (x is not None and df[df['date'] == (x)].index.size > 0 ) else size)
    return df

def get_take_profit_hit_date(df: pd.DataFrame):
    print(f"[+] Calculating take profit hit date")
    df['Take_Profit_Hit_Date'] = None
    take_profit_hit_date = np.empty(len(df), dtype= object)
    for i in range(len(df)):
        take_profit_hit = np.where(df.iloc[i+1:]['high'] > df.iloc[i]['Take_Profit'])[0]
        if len(take_profit_hit) > 0:
            hit_date = df.iloc[take_profit_hit[0]+i+1]['date']
            if(hit_date.split(' ')[0] == df.iloc[i]['date'].split(' ')[0]):
                take_profit_hit_date[i] = hit_date

    df['Take_Profit_Hit_Date'] = take_profit_hit_date
    return df

def get_take_profit_hit_row_number(df: pd.DataFrame):
    size = len(df)
    df['Take_Profit_Hit_Row_Number'] = df['Take_Profit_Hit_Date'].apply(lambda x: df[df['date'] == x].index[0] if (x is not None and df[df['date'] == x].index.size > 0 ) else size)
    return df

def fill_profit_trade(df) -> pd.DataFrame:
    print(f"[+] Calculating if trade is profit trade")
    df['Profit_Trade'] = np.where((df['Take_Profit_Hit_Date'].notnull()) & (df['Stop_Loss_Hit_Date'].notnull()),
                                (df['Take_Profit_Hit_Row_Number'] < df['Stop_Loss_Hit_Row_Number']).astype(str),
                                np.where(df['Take_Profit_Hit_Date'].notnull(), 'True',
                                        np.where(df['Stop_Loss_Hit_Date'].notnull(), 'False', 'End_Of_Day')))
    return df


# create a function that exchange the values of the columns 'Profit_Trade' instead 'end_of_day' the date of the candle 
def get_profit_trade_date(df: pd.DataFrame):
    print(f"[+] Calculating profit trade date")
    df['Profit_Trade_Date'] = np.where(df['Profit_Trade'] == 'True', 'True',
                                    np.where(df['Profit_Trade'] == 'False', 'False', df['date']))

    df['Profit_Trade_Date'] = df['Profit_Trade_Date'].apply(lambda x: get_close_price_at_end_of_day(df,x) if x != 'True' and x != 'False' else x)

    df['Profit_Trade_Date'] = np.where(df['Profit_Trade'] == 'True', df['Take_Profit'],
                                    np.where(df['Profit_Trade'] == 'False', df['Stop_Loss'], df['Profit_Trade_Date']))

    return df

def get_profit_trade_pct(df: pd.DataFrame):
    print(f"[+] Calculating profit trade pct")
    df['Profit_Trade_Pct'] = (df['Profit_Trade_Date'] / df['close']).apply(lambda x: round(x, 3) if x is not None else x) - 1
    return df

def get_close_price_at_end_of_day(df: pd.DataFrame, date):
    try:
        last_candle_date = date.split(' ')[0] + ' 22:55:00'
        return df[df['date'] == last_candle_date]['close'].values[0]
    except Exception as e:
        print(f"[-] Error in get_close_price_at_end_of_day with date {date} ", e)


def final_profit_trade(df: pd.DataFrame):
    print(f"[+] Calculating final profit trade")
    df['Final_Profit_Trade'] = df['Profit_Trade_Pct'].apply(lambda x: True if x > 0 else False)
    return df

def is_buy(df: pd.DataFrame):
    print(f"[+] Calculating if trade is buy")
    df['Number_Of_Trues'] = df[['Rsi_Above_50', 'SuperTrend_Below_Close', 'MACD_Above_Signal', 'MACD_Below_Zero','IS_EMA_Ribbon']].sum(axis=1)
    df['Is_Buy'] = (df['Number_Of_Trues'] >= 4).apply(lambda x: True if x else False)
    return df

def print_summary(df: pd.DataFrame):
    print(f"[+] Printing summary")
    print(f"\n[+] Bar Size: {BAR_SIZE}")
    print(f"[+] Total Candles Analyzed: {len(df)}")
    print(f"[+] Total Profit Trades: {len(df[df['Profit_Trade'] == 'True'])}")
    print(f"[+] Total Loss Trades: {len(df[df['Profit_Trade'] == 'False'])}")
    print(f"[+] Total End Of Day Trades: {len(df[df['Profit_Trade'] == 'End_Of_Day'])}")
    print(f"[+] Total Buy Trades: {len(df[df['Is_Buy'] == True])}")
    print(f"[+] Total Winning Trades: {len(df[(df['Profit_Trade'] == 'True') & (df['Is_Buy'] == True)])}")
    print(f"[+] Total Losing Trades: {len(df[(df['Profit_Trade'] == 'False') & (df['Is_Buy'] == True)])}")
    print(f"[+] Total End Of Day Trades: {len(df[(df['Profit_Trade'] == 'End_Of_Day') & (df['Is_Buy'] == True)])}")

    print(f"\n[+] Total Winning Trades Final: {len(df[(df['Final_Profit_Trade'] == True) & (df['Is_Buy'] == True)])}")
    print(f"[+] Total Losing Trades Final: {len(df[(df['Final_Profit_Trade'] == False) & (df['Is_Buy'] == True)])}")

def save_to_csv(df: pd.DataFrame):
    print(f"[+] Saving to csv")
    df.to_csv(f'{BAR_SIZE}.csv')

# create a function that create a new column name minute_number that will
# be the number of the minute of the day that start from 16:30:00 to 22:55:00 
def get_minute_number(df: pd.DataFrame):
    print(f"[+] Calculating minute number")
    df['Minute_Number'] = df['date'].apply(lambda x: get_minute_number_from_date(x))
    return df

def get_minute_number_from_date(date):
    hour = int(date.split(' ')[1].split(':')[0])
    minute = int(date.split(' ')[1].split(':')[1])
    return (((hour-16)*60 + minute) / CANDLE_SIZE)- 5

def create_ai_csv(df: pd.DataFrame) -> pd.DataFrame:
    print(f"[+] Creating AI CSV")

    # droptown tht first 200 rows
    df = df.iloc[200:]
    # change IS_EMA_Ribbon to 1 and 0 if true or false
    df['IS_EMA_Ribbon'] = df['IS_EMA_Ribbon'].apply(lambda x: 1 if x else 0)
    # add colmn is_win if pr Profit_Trade_Pct > 0 
    df['is_win'] = df['Profit_Trade_Pct'].apply(lambda x: 1 if x > 0 else 0)
    
    # filter the training features from the csv columns
    df = df[OUTPUT_COLS + [PREDICTION_FEATURE]]
    return df 

 
class DataGenerator:
    """
    this class is used in order to generate dataframes
    with training data for the DL classsification models
    """
    def __init__(self, client: StockClient) -> None:
        self.client = client._client
        self.contract = client._stock
    
    def get_training_data(self) -> pd.DataFrame:
        df = retrieve_candles(self.client, self.contract)
        df = parse_stop_loss_and_take_profit(df)
        df = parse_stop_loss_hit_date(df)
        df = get_stop_loss_hit_row_number(df)
        df = get_take_profit_hit_date(df)
        df = get_take_profit_hit_row_number(df)
        df = fill_profit_trade(df)
        df = get_profit_trade_date(df)
        df = get_profit_trade_pct(df)
        df = final_profit_trade(df)
        df = is_buy(df)
        df = get_minute_number(df)
        df = create_ai_csv(df)
        return df
    
    @staticmethod
    def generate_sample(raw_sample: RawDataSample) -> pd.Series:
        ret_dict = {}
        ret_dict["IS_EMA_Ribbon"] = int((raw_sample.ema20 > raw_sample.ema25) and\
                                          (raw_sample.ema25 > raw_sample.ema30) and\
                                          (raw_sample.ema30 > raw_sample.ema35) and\
                                          (raw_sample.ema30 > raw_sample.ema35) and\
                                          (raw_sample.ema35 > raw_sample.ema40) and\
                                          (raw_sample.ema40 > raw_sample.ema45) and\
                                          (raw_sample.ema45 > raw_sample.ema50))
        ret_dict['SuperTrend_Below_Close'] = int(raw_sample.supertrend < raw_sample.close)
        ret_dict['Close_Minus_SuperTrend'] = raw_sample.close - raw_sample.supertrend
        ret_dict[MACD_COLUMN] = raw_sample.macd
        ret_dict[MACDh_COLUMN] = raw_sample.macd_h
        ret_dict[MACDs_COLUMN] = raw_sample.macd_signal
        ret_dict[RSI_COLUMN] = raw_sample.rsi
        ret_dict['pct_change'] = raw_sample.percentage_diff
        ret_dict['acc_pct_change'] = raw_sample.accumulated_percentage_diff
        ret_dict['pos_change_in_a_row'] = max(raw_sample.num_candles_in_trend, 0)
        ret_dict['neg_change_in_a_row'] = -1 * min(raw_sample.num_candles_in_trend, 0)
        ret_dict['Minute_Number'] = get_minute_number_from_date(raw_sample.date)
        for EMA_PERIOD in EMA_PERIODS:
            EMA_COLUMN = "EMA_" + str(EMA_PERIOD)
            ema_val = getattr(raw_sample, f"ema{EMA_PERIOD}")
            ret_dict['Close_Minus_'+ EMA_COLUMN] = round((raw_sample.close - ema_val), 3) if ema_val else None
        
        output_sample = pd.Series(ret_dict, index=OUTPUT_COLS)
        return output_sample
        
    
    @staticmethod
    def pre_process(raw_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, StandardScaler]:
        X = raw_data.drop(columns=['is_win']).values
        y = raw_data['is_win'].values
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X, y, scaler
