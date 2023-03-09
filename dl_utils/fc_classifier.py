import logging
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from typing import List, Optional
from keras.optimizers import Adam
from keras.models import Sequential, clone_and_build_model
from keras.layers import Dense, Dropout
from keras.initializers import GlorotUniform, HeNormal
from keras import backend as K
from multiprocessing import Pool

from consts import WEIGHT_FILE_PATH

DEFAULT_CANDLE_SIZE = 5
LAYER_INIT_SEED = 42
NUM_TRADING_HOURS = 6.5
NUM_MINUTES_IN_HOUR = 60
STATS_FILE_NAME = "df_stats_model_selection.csv"
ACCURACY_COL_NAME = "0_0.30_ratio"
ACCURACY_COL_INDEX = -1


class FcClassifier:
    __slots__: List[str] = ['model', 'X', 'y', 'X_test', 'y_test', 'logger']
    
    def __init__(self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None, split_test: bool = False) -> None:
        self.init_model(X)
        
        if split_test:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
            self.X = X_train
            self.y = y_train
            self.X_test = X_test
            self.y_test = y_test
        else:
            self.X = X
            self.y = y
            
        self.logger = logging.getLogger('StockClient')

    def init_model(self, X: Optional[pd.DataFrame] = None):
        from dl_utils.data_generator import OUTPUT_COLS
        n_in: int = X.shape[1] if X is not None else len(OUTPUT_COLS)
        
        # Create a neural network model using the Sequential API
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(n_in,)))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))

        # set an initializer for each of the model's layers in order to make the model converge
        FcClassifier.initialize_model(model)

        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy', 'Precision', 'Recall'])
        self.model = model

    @staticmethod
    def train_model(model: Sequential, train: pd.DataFrame, pred: pd.Series):
        model.summary()

        # Train the model
        print("[+] Training the model...")
        model.fit(train, pred, epochs=1, batch_size=32, verbose=1)

    def train_nn_model(self):
        FcClassifier.train_model(self.model, self.X, self.y)
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series):
        # Evaluate the model
        print("[+] Evaluating the model...")
        self.model.evaluate(X_test, y_test, batch_size=32, verbose=1)

        # Predict the model
        print("[+] Predicting the model...")
        y_pred = self.model.predict(X_test)

        # Print the confusion matrix
        print("[+] Confusion Matrix:")
        print(pd.crosstab(y_test, y_pred[:,0].round(), rownames=['Actual'], colnames=['Predicted']))

        # Get confusion matrix values
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred[:,0].round()).ravel()
        print(f"[+] tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}")
        # Calculate the accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        print(f"[+] Accuracy: {accuracy}")
        # Calculate the precision
        precision = tp / (tp + fp)
        print(f"[+] Precision: {precision}")
        # Calculate the recall
        recall = tp / (tp + fn)
        print(f"[+] Recall: {recall}")
        # Calculate the F1 score
        f1 = 2 * (precision * recall) / (precision + recall)
        print(f"[+] F1: {f1}")
    
    def get_model(self) -> Sequential:
        return self.model
    
    def reset_model(self):
        self.init_model(self.X)

    @staticmethod
    def _num_candles_in_day(candle_size: int):
        return (NUM_TRADING_HOURS*NUM_MINUTES_IN_HOUR)/candle_size
    
    def train_model_n_times(self, n, write_results: bool = True):
        weight_dict = {}
        res_df = FcClassifier._init_acc_stats_df()
        
        # Define a function to train the model and save the validation accuracy
        def train_and_eval(i):
            model = clone_and_build_model(self.model)
            # Reset the weights of the model
            model.reset_states()
            
            # Train the model
            FcClassifier.train_model(model, self.X, self.y)
            
            # Evaluate the model on the test set and save acc and weights
            accuracy_data = self._dump_predictions(self.X_test, self.y_test)
            res_df.loc[len(res_df)] = accuracy_data
            self.logger.info(f"testing {i} done; got {accuracy_data}")
            val_acc = accuracy_data[ACCURACY_COL_INDEX]
            
            weight_dict[i] = (val_acc, model.get_weights())
        
        # Train the model multiple times in parallel and save the weights with the best test accuracy
        # with Pool(processes=n) as pool:
        #     pool.map(train_and_eval, range(n))
        
        for i in range(n):
            train_and_eval(i)
            
        # Save the weights if they have the best test accuracy
        best_acc_val = -1
        best_model_number = 0
        for i, (acc_val, weights) in enumerate(weight_dict.values()):
            import pdb; pdb.set_trace()
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                best_model_number = i
                self.model.set_weights(weights)

        logging.info(f"best model {best_model_number} with acc {best_acc_val}")
        self.model.save_weights(WEIGHT_FILE_PATH)
        
        if write_results:
            res_df.to_csv(STATS_FILE_NAME, index=False)
    
    @staticmethod
    def initialize_model(model: Sequential):
        """
        Initializes the weights of a Keras sequential model according to some best practices
        """
        for layer in model.layers:
            if isinstance(layer, Dense):
                n_in, n_out = K.get_value(layer.kernel.shape[0]), K.get_value(layer.kernel.shape[1])
                if layer.activation.__name__ == 'relu':
                    scale = 2.0 / (n_in + n_out)
                    initializer = HeNormal(seed=LAYER_INIT_SEED)
                else:
                    scale = 1.0
                    initializer = GlorotUniform(seed=LAYER_INIT_SEED)
                layer.kernel_initializer = initializer
                layer.bias_initializer = initializer
                stddev = scale * K.sqrt(K.constant(scale))
                weights = initializer(shape=layer.get_weights()[0].shape)
                weights *= K.constant(stddev, dtype=K.floatx()) / K.std(weights)
                layer.set_weights([weights, layer.get_weights()[1]])
    
    def _dump_predictions(self, X_test: pd.DataFrame, y_test: pd.Series):
        y_pred = self.model.predict(X_test)
        df_pred = pd.DataFrame(y_test)
        df_pred['y_pred'] = y_pred
        row = []
        low = 0
        for i in range(1, 7):
            high = (i * 0.05)
            high = round(high, 3)
            try:
                y_pred_between_low_high_is_win_0 = df_pred[(df_pred['y_pred'] >= low) & (df_pred['y_pred'] <= high) & (df_pred['is_win'] == 0)].shape[0]
                row.append(round(y_pred_between_low_high_is_win_0,3))

                y_pred_between_low_high_is_win_1 = df_pred[(df_pred['y_pred'] >= low) & (df_pred['y_pred'] <= high) & (df_pred['is_win'] == 1)].shape[0]
                row.append(round(y_pred_between_low_high_is_win_1,3))

                ratio = y_pred_between_low_high_is_win_0 / (y_pred_between_low_high_is_win_0+ y_pred_between_low_high_is_win_1)
                row.append(round(ratio,3))
            except Exception as e:
                row.append("Null")
        return row

    @staticmethod
    def _init_acc_stats_df() -> pd.DataFrame:
        # initialize the results dataframe
        stats = [0.05,0.1,0.15,0.2,0.25,0.3]
        cols_0 = [f"0_{s:.2f}_0" for s in stats]
        cols_1 = [f"0_{s:.2f}_1" for s in stats]
        cols_ratio = [f"0_{s:.2f}_ratio" for s in stats]
        cols = [col for tup in zip(cols_0, cols_1, cols_ratio) for col in tup]
        res_df = pd.DataFrame(columns=cols)
        return res_df
    
    def train_test_model_over_time_period(self, period: int, dataset: pd.DataFrame, candle_size: int = DEFAULT_CANDLE_SIZE) -> pd.DataFrame:
        """
        Train and test a TensorFlow sequential model over a time period.
        
        Parameters:
            period (int): The length of the time period to train over in days.
            start_date (datetime): The starting date for the time period.
            model (tf.keras.Sequential): The TensorFlow sequential model to train and test.
            dataset (np.array): The dataset to use for training and testing.
        """
        num_candles_per_day: int = FcClassifier._num_candles_in_day(candle_size)
        train_set_size = num_candles_per_day * period
        num_training_samples = dataset.shape[0]
        num_train_iterations = int((num_training_samples - train_set_size) / num_candles_per_day)
        
        self.logger.info(f"total number of day iteration is {num_train_iterations}")
        
        res_df = FcClassifier._init_acc_stats_df()
        
        self.logger.info(f"train and test over {period} days period with dataset of {len(dataset)} rows")
        
        # Loop over each day in the time period
        for i in range(num_train_iterations):
            start_index = num_candles_per_day * i
            
            if start_index+train_set_size+num_candles_per_day > len(dataset):
                logging.error(f"exceeding dataset indices; existing main loop")
                break
            
            self.X = dataset.iloc[start_index:start_index+train_set_size, :-1]
            self.y = dataset.iloc[start_index:start_index+train_set_size, -1]
            self.logger.info(f"training over day {i} with dataset rows [{start_index}, {start_index+train_set_size}]")

            # Train the model on the training data according to input period
            self.train_nn_model()
            
            # set test set to be the following trading day candle data
            test_set = dataset.iloc[start_index+train_set_size:start_index+train_set_size+num_candles_per_day, :-1]
            test_pred = dataset.iloc[start_index+train_set_size:start_index+train_set_size+num_candles_per_day, -1]
            
            # evaluate the models accuracy over the test set and record the results
            self.logger.info(f"testing over day {i} with dataset rows [{start_index+train_set_size}, {start_index+train_set_size+num_candles_per_day}]")
            pred_iter_stats_row = self._dump_predictions(test_set, test_pred)
            res_df.loc[len(res_df)] = pred_iter_stats_row
            self.logger.info(f"testing done; got {pred_iter_stats_row}")
            
            # Reset the model weights before training on the next time period
            self.logger.info("resetting weights")
            self.reset_model()
        
        return res_df
    
    def predict(self, sample: pd.Series) -> float:
        return self.model(sample)
