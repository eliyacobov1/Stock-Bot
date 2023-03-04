import logging
import pandas as pd
from sklearn.metrics import confusion_matrix
from typing import Optional
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.initializers import GlorotUniform, HeNormal
from keras import backend as K

DEFAULT_CANDLE_SIZE = 5
LAYER_INIT_SEED = 42


class FcClassifier:
    def __init__(self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None) -> None:
        
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
        self.X = X
        self.y = y
        self.pre_training_weights = None

    def train_nn_model(self):
        self.model.summary()
        if self.pre_training_weights is None:
            self.pre_training_weights = self.model.get_weights()  # save weigths to allow restoring them after training

        # Train the model
        print("[+] Training the model...")
        self.model.fit(self.X, self.y, epochs=50, batch_size=32, verbose=1)
    
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
        self.model.set_weights(self.pre_training_weights)

    @staticmethod
    def _num_candles_in_day(candle_size: int):
        return int((6.5*60)//candle_size)
    
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
        
        logging.info(f"total number of day iteration is {num_train_iterations}")
        
        # initialize the results dataframe
        stats = [0.05,0.1,0.15,0.2,0.25,0.3]
        cols_0 = [f"0_{s:.2f}_0" for s in stats]
        cols_1 = [f"0_{s:.2f}_1" for s in stats]
        cols_ratio = [f"0_{s:.2f}_ratio" for s in stats]
        cols = [col for tup in zip(cols_0, cols_1, cols_ratio) for col in tup]
        res_df = pd.DataFrame(columns=cols)
        
        logging.info(f"train and test over {period} days period with dataset of {len(dataset)} rows")
        
        # Loop over each day in the time period
        for i in range(num_train_iterations):
            start_index = num_candles_per_day * i
            
            if start_index+train_set_size+num_candles_per_day > len(dataset):
                logging.error(f"exceeding dataset indices; existing main loop")
                break
            
            self.X = dataset.iloc[start_index:start_index+train_set_size, :-1]
            self.y = dataset.iloc[start_index:start_index+train_set_size, -1]
            logging.info(f"training over day {i} with dataset rows [{start_index}, {start_index+train_set_size}]")

            # Train the model on the training data according to input period
            self.train_nn_model()
            
            # set test set to be the following trading day candle data
            test_set = dataset.iloc[start_index+train_set_size:start_index+train_set_size+num_candles_per_day, :-1]
            test_pred = dataset.iloc[start_index+train_set_size:start_index+train_set_size+num_candles_per_day, -1]
            
            # evaluate the models accuracy over the test set and record the results
            logging.info(f"testing over day {i} with dataset rows [{start_index+train_set_size}, {start_index+train_set_size+num_candles_per_day}]")
            pred_iter_stats_row = self._dump_predictions(test_set, test_pred)
            res_df.loc[len(res_df)] = pred_iter_stats_row
            logging.info(f"testing done; got {pred_iter_stats_row}")
            
            # Reset the model weights before training on the next time period
            logging.info("resetting weights")
            self.reset_model()
        
        return res_df
    
    def predict(self, sample: pd.Series) -> float:
        return self.model(sample)
