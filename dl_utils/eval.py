import logging
import pandas as pd

from stock_client import StockClientInteractive
from dl_utils.data_generator import DataGenerator
from dl_utils.fc_classifier import FcClassifier


DEFAULT_PERIOD = 90


def train_and_test_period() -> pd.DataFrame:
    client = StockClientInteractive(config_logger=False)
    model = FcClassifier()
    data_generator = DataGenerator(client=client)
    
    data = data_generator.get_training_data()
    res_df = model.train_test_model_over_time_period(period=DEFAULT_PERIOD, dataset=data)
    
    return res_df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    df: pd.DataFrame = train_and_test_period()
    df.to_csv("df_stats.csv", index=False)
