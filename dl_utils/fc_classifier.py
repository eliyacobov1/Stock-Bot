import pandas as pd
from sklearn.metrics import confusion_matrix
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense


class FcClassifier:
    def __init__(self, X: pd.DataFrame, y: pd.Series) -> None:
        # Create a neural network model using the Sequential API
        self.X = X
        self.y = y
        self.model = Sequential()
        self.model.add(Dense(32, input_shape=(X.shape[1],), activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        self.model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.07), metrics=['accuracy'])

    def train_nn_model(self):
        self.model.summary()

        # Train the model
        print("[+] Training the model...")
        self.model.fit(self.X, self.y, epochs=100, batch_size=32, verbose=1)
    
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
    
    def predict(self, sample: pd.Series) -> float:
        return self.model(sample)
