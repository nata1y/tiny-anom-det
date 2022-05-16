# fromhttps://towardsdatascience.com/time-series-of-price-anomaly-detection-with-lstm-11a12ba4f6d9

import funcy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as st
import tensorflow as tf
from keras.engine.input_layer import InputLayer
from scipy import stats
from sklearn.cluster import DBSCAN
from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.models import Sequential
import antropy as ant

from settings import entropy_params

print(f'Running on GPU {tf.test.is_built_with_cuda()}. Devices: {tf.config.list_physical_devices("GPU")}')


class LSTM_autoencoder:
    model = None
    loss = []

    def __init__(self, X_shape, dataset, datatype, filename, magnitude=1.5):

        self.model = Sequential()
        self.model.add(LSTM(128, input_shape=(X_shape[0], X_shape[1])))
        self.model.add(Dropout(rate=0.2))
        self.model.add(RepeatVector(X_shape[0]))
        self.model.add(LSTM(128, return_sequences=True))
        self.model.add(Dropout(rate=0.2))
        self.model.add(TimeDistributed(Dense(X_shape[1])))
        self.model.compile(optimizer='adam', loss='mae')
        self.model.summary()

        self.magnitude = magnitude
        self.datatype = datatype
        self.dataset = dataset
        self.filename = filename
        self.loss = []
        self.predicted = []
        self.window = X_shape[0]
        self.entr_factor = entropy_params[f'{dataset}_{datatype}']['factor']

    def fit(self, X_train, y_train, timestamp, Xf):
        self.history = self.model.fit(X_train, y_train, epochs=100, batch_size=32,
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')],
                                      validation_split=0.1,  shuffle=False)

        self.threshold = self.magnitude * st.t.interval(alpha=0.99, df=len(self.history.history['loss'])-1,
                                                        loc=np.mean(self.history.history['loss']),
                                                        scale=st.sem(self.history.history['loss']))[1]

        self.svd_entropies = []

        for start in range(0, len(Xf), self.window):
            try:
                self.svd_entropies.append(
                    ant.svd_entropy(Xf[start:start + self.window], normalize=True))
            except:
                pass
        self.boundary_bottom = np.mean([v for v in self.svd_entropies if pd.notna(v)]) - \
                               self.entr_factor * np.std([v for v in self.svd_entropies if pd.notna(v)])
        self.boundary_up = np.mean([v for v in self.svd_entropies if pd.notna(v)]) + \
                           self.entr_factor * np.std([v for v in self.svd_entropies if pd.notna(v)])
        print(self.boundary_up, self.boundary_bottom)

    def get_pred_mean(self):
        pred_thr = pd.DataFrame([])
        pred_thr['upper value'] = [self.threshold for _ in range(len(self.loss))]
        pred_thr['lower value'] = 0.0

        return pred_thr

    def predict(self, X, timestamp):
        mean_val = np.mean(X.flatten())
        Xf = funcy.lflatten(X.flatten())
        for idx in range(self.window - X.shape[1]):
            Xf.append(mean_val)

        entropy = ant.svd_entropy(Xf, normalize=True)

        Xf = np.array(Xf).reshape((1, self.window, 1))
        prediction = self.model.predict(Xf)
        loss = np.abs(Xf - prediction).ravel()[:X.shape[1]]

        y_pred1 = [0 if loss[idx] <= self.threshold else 1 for idx in range(len(loss))]
        if self.boundary_bottom <= entropy <= self.boundary_up:
            y_pred2 = [0 if loss[idx] <= self.threshold else 1 for idx in range(len(loss))]
        else:
            extent = stats.percentileofscore(self.svd_entropies, entropy) / 100.0
            extent = 1.5 - max(extent, 1.0 - extent)
            threshold_adapted = self.threshold * extent
            y_pred2 = [0 if loss[idx] <= threshold_adapted else 1 for idx in range(len(loss))]

        self.loss += loss.flatten().tolist()
        self.predicted += y_pred1
        return y_pred1, y_pred2

    def plot(self, datatest):
        loss_df = pd.DataFrame([])
        loss_df['value'] = self.loss
        loss_df['timestamp'] = datatest['timestamp'].tolist()
        arranged_loss = loss_df

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=arranged_loss.index.tolist(), y=arranged_loss['value'].tolist(), name='Test loss'))
        fig.add_trace(go.Scatter(x=arranged_loss.index.tolist(), y=[self.threshold for _ in range(arranged_loss.shape[0])],
                                 name='Threshold'))

        x_fp, y_fp = [], []
        x_fn, y_fn = [], []

        datatest.reset_index(inplace=True)
        for tm, row in loss_df.iterrows():
            if tm in datatest.index:
                if self.predicted[tm] == 1 and datatest.loc[tm, 'is_anomaly'] == 0:
                    x_fp.append(tm)
                    y_fp.append(row['value'])
                if self.predicted[tm] == 0 and datatest.loc[tm, 'is_anomaly'] == 1:
                    x_fn.append(tm)
                    y_fn.append(row['value'])

        if x_fp:
            fig.add_trace(go.Scatter(x=x_fp, y=y_fp, name='FP', mode="markers"))
        if x_fn:
            fig.add_trace(go.Scatter(x=x_fn, y=y_fn, name='FN', mode="markers"))
        fig.update_layout(showlegend=True, title='Test loss vs. Threshold')
        fig.write_image(f'results/imgs/{self.dataset}/{self.datatype}/lstm_{self.filename.replace(".csv", "")}_full.png')

        fig.data = []
