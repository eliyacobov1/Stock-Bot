import logging
import pandas as pd

from stock_client import StockClientInteractive
from dl_utils.data_generator import DataGenerator
from dl_utils.fc_classifier import FcClassifier


DEFAULT_PERIOD = 90
NUM_MODELS_TO_TRAIN = 10
STATS_FILE_NAME = "df_stats.csv"


def train_and_test_period() -> pd.DataFrame:
    client = StockClientInteractive(config_logger=False)
    model = FcClassifier()
    data_generator = DataGenerator(client=client)
    
    data = data_generator.get_training_data()
    res_df = model.train_test_model_over_time_period(period=DEFAULT_PERIOD, dataset=data)
    
    return res_df

def generate_model_train_n_times(n: int = NUM_MODELS_TO_TRAIN):
    """
    this function will train the defualt model as implemented in FcClassifier the given
    number of times. It will then write to disk the weights of the model with best test
    accuracy
    """
    client = StockClientInteractive()
    data_generator = DataGenerator(client=client)
    data = data_generator.get_training_data()
    X, y = data[:, :-1], data[:, -1]

    model = FcClassifier(X, y, split_test=True)
    
    model.train_model_n_times(n)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    df: pd.DataFrame = train_and_test_period()
    df.to_csv(STATS_FILE_NAME, index=False)
