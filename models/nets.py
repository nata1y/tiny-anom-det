# fromhttps://towardsdatascience.com/time-series-of-price-anomaly-detection-with-lstm-11a12ba4f6d9
import funcy
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
import numpy as np


class LSTM_autoencoder():
    model = None
    loss = []

    def __init__(self, X_shape, dataset, datatype, filename, threshold=0.85):
        self.model = Sequential()
        self.model.add(LSTM(128, input_shape=(X_shape[0], X_shape[1])))
        self.model.add(Dropout(rate=0.2))
        self.model.add(RepeatVector(X_shape[0]))
        self.model.add(LSTM(128, return_sequences=True))
        self.model.add(Dropout(rate=0.2))
        self.model.add(TimeDistributed(Dense(X_shape[1])))
        self.model.compile(optimizer='adam', loss='mae')
        self.model.summary()

        self.threshold = threshold
        self.datatype = datatype
        self.dataset = dataset
        self.filename = filename
        self.loss = []

    def fit(self, X_train, y_train):
        self.history = self.model.fit(X_train, y_train, epochs=100, batch_size=32,
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')],
                                      validation_split=0.1,  shuffle=False)

    def predict(self, X):
        prediction = self.model.predict(X)
        loss = np.abs(X - prediction).ravel()

        y_pred = [0 if loss[idx] <= self.threshold else 1 for idx in range(len(loss))]
        self.loss += loss.flatten().tolist()
        return y_pred

    def plot(self, timestamp):
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=timestamp[: len(self.loss)], y=self.loss, name='Test loss'))
        fig.add_trace(go.Scatter(x=timestamp[: len(self.loss)], y=[self.threshold for _ in range(len(self.loss))],
                                 name='Threshold'))
        fig.update_layout(showlegend=True, title='Test loss vs. Threshold')
        fig.write_image(f'results/imgs/{self.dataset}/{self.datatype}/lstm/lstm_{self.filename}_full.png')

        fig.data = []
