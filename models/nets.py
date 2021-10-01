# fromhttps://towardsdatascience.com/time-series-of-price-anomaly-detection-with-lstm-11a12ba4f6d9
import funcy
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

import plotly.graph_objects as go

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
import numpy as np

print(f'Running on GPU {tf.test.is_built_with_cuda()}. Devices: {tf.config.list_physical_devices("GPU")}')


class LSTM_autoencoder():
    model = None
    loss = []

    def __init__(self, X_shape, dataset, datatype, filename, threshold=7.0):
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
        self.threshold = 5.0 * np.max(self.history.history['loss'])
        print(self.threshold)

    def predict(self, X):
        mean_val = np.mean(X.flatten())
        Xf = funcy.lflatten(X.flatten())
        for idx in range(60 - X.shape[1]):
            Xf.append(mean_val)

        Xf = np.array(Xf).reshape((1, 60, 1))
        prediction = self.model.predict(Xf)
        loss = np.abs(Xf - prediction).ravel()[:X.shape[1]]

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
