import logging
import pandas as pd

from stock_client import StockClientInteractive
from dl_utils.data_generator import DataGenerator
from dl_utils.fc_classifier import FcClassifier, DEFAULT_NUM_MODELS_TO_TRAIN


DEFAULT_PERIOD = 90
STATS_FILE_NAME = "df_stats.csv"


def train_and_test_period() -> pd.DataFrame:
    client = StockClientInteractive(run_offline=True)
    model = FcClassifier()
    data_generator = DataGenerator(client=client)
    
    data = data_generator.get_training_data(from_file=True)
    res_df = model.train_test_model_over_time_period(period=DEFAULT_PERIOD, dataset=data)
    
    return res_df

def generate_model_train_n_times(n: int = DEFAULT_NUM_MODELS_TO_TRAIN):
    """
    this function will train the defualt model as implemented in FcClassifier the given
    number of times. It will then write to disk the weights of the model with best test
    accuracy
    """
    client = StockClientInteractive(run_offline=True)
    data_generator = DataGenerator(client=client)
    data = data_generator.get_training_data(from_file=True)
    model = FcClassifier(data.iloc[:, :-1], data.iloc[:, -1], split_test=True)
    
    # nomalize the model's train and test set
    X, _, scalar = DataGenerator.pre_process(data[:model.X.shape[0]])
    model.X = X
    model.X_test = scalar.transform(model.X_test)
    
    model.train_model_n_times(n)


if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)
    # df: pd.DataFrame = train_and_test_period()
    # df.to_csv(STATS_FILE_NAME, index=False)
    generate_model_train_n_times()
