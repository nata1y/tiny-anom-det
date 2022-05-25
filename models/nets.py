# fromhttps://towardsdatascience.com/time-series-of-price-anomaly-detection-with-lstm-11a12ba4f6d9
import time
from datetime import datetime

import funcy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as st
import tensorflow as tf
from scipy import stats
from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.models import Sequential
import antropy as ant

from drift_detectors.ecdd import ECDD
from settings import entropy_params
from utils import create_dataset

print(f'Running on GPU {tf.test.is_built_with_cuda()}. Devices: {tf.config.list_physical_devices("GPU")}')


class LSTM_autoencoder:
    model = None
    loss = []

    def __init__(self, X_shape, dataset, datatype, filename, magnitude=1.5,
                 w=0.25, c=0.25, drift_count_limit=10):

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
        self.filename = filename.replace(".csv", "")
        self.loss = []
        self.predicted = []
        self.window = X_shape[0]
        self.entr_factor = entropy_params[f'{dataset}_{datatype}']['factor']
        self.drift_detector = ECDD(0.2, w, c)
        self.is_drift = False
        self.drift_alerting_cts = 0
        self.drift_count_limit = drift_count_limit
        self.data_for_retrain = []
        self.dynamic_thresholds = []

    def fit(self, data_train, phase='fit'):
        X_train, y_train = create_dataset(data_train[['value']], data_train[['value']], self.window)
        data_train = data_train['value'].tolist()

        self.history = self.model.fit(X_train, y_train, epochs=100, batch_size=32,
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')],
                                      validation_split=0.1,  shuffle=False)

        self.threshold = self.magnitude * st.t.interval(alpha=0.99, df=len(self.history.history['loss'])-1,
                                                        loc=np.mean(self.history.history['loss']),
                                                        scale=st.sem(self.history.history['loss']))[1]

        self.svd_entropies = []

        # record entropy if in initial phase
        if phase != 'retrain':
            for start in range(0, len(data_train), self.window):
                try:
                    self.svd_entropies.append(
                        ant.svd_entropy(data_train[start:start + self.window], normalize=True))
                except:
                    pass
            self.boundary_bottom = np.mean([v for v in self.svd_entropies if pd.notna(v)]) - \
                                   self.entr_factor * np.std([v for v in self.svd_entropies if pd.notna(v)])
            self.boundary_up = np.mean([v for v in self.svd_entropies if pd.notna(v)]) + \
                               self.entr_factor * np.std([v for v in self.svd_entropies if pd.notna(v)])

        # record value for drift detection
        self.drift_detector.record(np.mean(self.history.history['loss']),
                                   np.std(self.history.history['loss']))

    def get_pred_mean(self):
        pred_thr = pd.DataFrame([])
        pred_thr['upper value'] = [self.threshold for _ in range(len(self.loss))]
        pred_thr['lower value'] = 0.0

        return pred_thr

    def predict(self, window):
        X = window['value'].to_numpy().reshape(1, len(window['value'].tolist()), 1)
        timestamp = window['timestamp'].tolist()

        mean_val = np.mean(X.flatten())
        Xf = funcy.lflatten(X.flatten())
        for idx in range(self.window - X.shape[1]):
            Xf.append(mean_val)

        entropy = ant.svd_entropy(Xf, normalize=True)

        Xf = np.array(Xf).reshape((1, self.window, 1))
        prediction = self.model.predict(Xf)
        loss = np.abs(Xf - prediction).ravel()[:X.shape[1]]

        for error, t in zip(loss, timestamp):
            if isinstance(t, str):
                t = datetime.strptime(t, '%Y-%m-%d %H:%M:%S').timestamp()

            self.drift_detector.update_ewma(error=error, t=t)
            response = self.drift_detector.monitor()
            if response == self.drift_detector.drift:
                self.drift_alerting_cts += 1

        # use static threshold
        y_pred1 = [0 if loss[idx] <= self.threshold else 1 for idx in range(len(loss))]

        # use dynamic threshold
        if self.boundary_bottom <= entropy <= self.boundary_up:
            y_pred2 = [0 if loss[idx] <= self.threshold else 1 for idx in range(len(loss))]
            self.dynamic_thresholds += [self.threshold] * len(y_pred2)
        else:
            extent = stats.percentileofscore(self.svd_entropies, entropy) / 100.0
            extent = 1.5 - max(extent, 1.0 - extent)
            threshold_adapted = self.threshold * extent
            y_pred2 = [0 if loss[idx] <= threshold_adapted else 1 for idx in range(len(loss))]
            self.dynamic_thresholds += [threshold_adapted] * len(y_pred2)

        self.loss += loss.flatten().tolist()
        self.predicted += y_pred1

        # perform drift adaptation
        if self.drift_alerting_cts >= self.drift_count_limit:
            if len(self.data_for_retrain) > self.window * 5:
                print('Drift occured')
                start_time = time.time()
                tmp = pd.DataFrame(columns=['values'])
                tmp['values'] = self.data_for_retrain
                self.fit(tmp, 'retrain')
                end_time = time.time()
                diff = end_time - start_time
                print(f"Trained lstm for {diff}")
                self.data_for_retrain = []
                self.drift_alerting_cts = 0
            else:
                self.data_for_retrain += funcy.lflatten(X.flatten())
                print('?&', len(self.data_for_retrain))
        return y_pred1, y_pred2

    def plot(self, datatest, threshold_type='dynamic'):
        loss_df = pd.DataFrame([])
        loss_df['value'] = self.loss
        loss_df['timestamp'] = datatest['timestamp'].tolist()
        arranged_loss = loss_df

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=arranged_loss.index.tolist(), y=arranged_loss['value'].tolist(), name='Test loss'))
        if threshold_type == 'dynamic':
            fig.add_trace(go.Scatter(x=arranged_loss.index.tolist(),
                                     y=self.dynamic_thresholds,
                                     name='Threshold'))
        else:
            fig.add_trace(go.Scatter(x=arranged_loss.index.tolist(),
                                     y=[self.threshold for _ in range(arranged_loss.shape[0])],
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
        fig.write_image(f'results/imgs/{self.dataset}/{self.datatype}/lstm_{self.filename}_reconstruction_loss.png')

        fig.data = []
