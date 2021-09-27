# fromhttps://towardsdatascience.com/time-series-of-price-anomaly-detection-with-lstm-11a12ba4f6d9

from tensorflow import keras
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
import numpy as np


class LSTM_autoencoder():
    model = Sequential()

    def __init__(self, X_shape, threshold=0.65):
        self.model.add(LSTM(128, input_shape=(X_shape[0], X_shape[1])))
        self.model.add(Dropout(rate=0.2))
        self.model.add(RepeatVector(X_shape[0]))
        self.model.add(LSTM(128, return_sequences=True))
        self.model.add(Dropout(rate=0.2))
        self.model.add(TimeDistributed(Dense(X_shape[1])))
        self.model.compile(optimizer='adam', loss='mae')
        self.model.summary()

        self.threshold = threshold

    def fit(self, X_train, y_train):
        self.model = self.model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1,
                            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')],
                            shuffle=False)

    def predict(self, X):
        prediction = self.model.predict(X)
        loss = np.mean(np.abs(X - prediction), axis=1)
        y_pred = [0 if loss[idx] <= self.threshold else 1 for idx in range(X.shape[2])]
        return y_pred
