# fromhttps://towardsdatascience.com/time-series-of-price-anomaly-detection-with-lstm-11a12ba4f6d9
import math

import funcy
import torch
import scipy.stats as st
from keras.engine.input_layer import InputLayer
from keras.layers import Conv2D, Conv2DTranspose, Reshape, Conv1D, MaxPooling1D, Flatten
from sklearn.cluster import DBSCAN
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from pytorch_forecasting.models.deepar import DeepAR

import plotly.graph_objects as go

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
import numpy as np
import pandas as pd

# print(f'Torch cuda is avaliable {torch.cuda.is_available()}')
from analysis.preanalysis import periodicity_analysis
from models.sr.spectral_residual import SpectralResidual, MAG_WINDOW, SCORE_WINDOW, THRESHOLD
from models.sr.util import DetectMode
from models.statistical_models import SARIMA
from utils import adjust_range

print(f'Running on GPU {tf.test.is_built_with_cuda()}. Devices: {tf.config.list_physical_devices("GPU")}')


class SeqSeq:

    def __init__(self, seq_len, latent_dim):
        self.threshold_net_seqseq = tf.keras.Sequential(
            [
                InputLayer(input_shape=(seq_len, latent_dim)),
                Dense(64, activation=tf.nn.relu),
                Dense(64, activation=tf.nn.relu),
            ])


class Vae:

    def __init__(self, latent_dim, n_features=1):
        self.encoder_net_vae = tf.keras.Sequential(
          [
              InputLayer(input_shape=(60,)),
              Dense(25, activation=tf.nn.relu),
              Dense(10, activation=tf.nn.relu),
              Dense(5, activation=tf.nn.relu)
          ])

        self.decoder_net_vae = tf.keras.Sequential(
          [
              InputLayer(input_shape=(latent_dim,)),
              Dense(5, activation=tf.nn.relu),
              Dense(10, activation=tf.nn.relu),
              Dense(25, activation=tf.nn.relu),
              Dense(60, activation=None)
          ])


class LSTM_autoencoder:
    model = None
    loss = []

    def __init__(self, X_shape, dataset, datatype, filename, magnitude=1.5, window=60, threshold_model_type=None):
        # self.model = Sequential()
        # self.model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_shape[0], X_shape[1])))
        # self.model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        # self.model.add(MaxPooling1D(pool_size=2))
        # self.model.add(Flatten())
        # self.model.add(RepeatVector(X_shape[0]))
        # self.model.add(LSTM(200, activation='relu', return_sequences=True))
        # self.model.add(TimeDistributed(Dense(100, activation='relu')))
        # self.model.add(TimeDistributed(Dense(1)))
        # self.model.compile(loss='mse', optimizer='adam')

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
        self.window = window
        self.threshold_model_type = threshold_model_type

    def fit(self, X_train, y_train, timestamp, Xf):
        self.history = self.model.fit(X_train, y_train, epochs=100, batch_size=32,
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')],
                                      validation_split=0.1,  shuffle=False)

        self.threshold = self.magnitude * st.t.interval(alpha=0.99, df=len(self.history.history['loss'])-1,
                                                        loc=np.mean(self.history.history['loss']),
                                                        scale=st.sem(self.history.history['loss']))[1]

        if self.threshold_model_type == 'sarima':
            self.threshold_model = SARIMA(1.3)
            self.threshold_model.dataset = self.dataset
            self.threshold_model.threshold_modelling = True
            threshold_train_data = pd.DataFrame([])
            prediction_full = self.model.predict(X_train)
            loss = np.abs(X_train - prediction_full).ravel()
            loss = [loss[idx * self.window] for idx in range(X_train.shape[0])]

            mean_val = np.mean(Xf.tolist())
            Xf = funcy.lflatten(Xf.tolist())
            for idx in range(self.window - len(Xf)):
                Xf.append(mean_val)
            Xf = np.array(Xf).reshape((1, self.window, 1))
            prediction = self.model.predict(Xf)
            loss += np.abs(Xf - prediction).ravel().tolist()

            threshold_train_data['value'] = loss[-2000:]
            threshold_train_data['timestamp'] = timestamp.tolist()[-2000:]
            # model = SpectralResidual(series=threshold_train_data[['value', 'timestamp']], threshold=THRESHOLD,
            #                          mag_window=MAG_WINDOW, score_window=SCORE_WINDOW,
            #                          sensitivity=99, detect_mode=DetectMode.anomaly_only)
            # loss = model.detect()['score'].tolist()
            # threshold_train_data['value'] = loss
            self.threshold_model.fit(threshold_train_data, self.dataset)
        elif self.threshold_model_type == 'dbscan':
            self.threshold_model = DBSCAN(eps=1, min_samples=2, metric='euclidean')

    def get_pred_mean(self):
        pred_thr = pd.DataFrame([])
        if self.threshold_model_type == 'sarima':
            pred_thr['upper value'] = self.threshold_model.get_pred_mean()
        else:
            pred_thr['upper value'] = [self.threshold for _ in range(len(self.loss))]
        pred_thr['lower value'] = 0.0

        return pred_thr

    def predict(self, X, timestamp):
        y_pred = []
        mean_val = np.mean(X.flatten())
        Xf = funcy.lflatten(X.flatten())
        for idx in range(self.window - X.shape[1]):
            Xf.append(mean_val)

        Xf = np.array(Xf).reshape((1, self.window, 1))
        prediction = self.model.predict(Xf)
        loss = np.abs(Xf - prediction).ravel()[:X.shape[1]]

        threshold_test_data = pd.DataFrame([])
        threshold_test_data['value'] = loss.flatten().tolist()
        threshold_test_data['timestamp'] = timestamp.tolist()

        # model = SpectralResidual(series=threshold_test_data[['value', 'timestamp']], threshold=THRESHOLD,
        #                          mag_window=MAG_WINDOW, score_window=SCORE_WINDOW,
        #                          sensitivity=99, detect_mode=DetectMode.anomaly_only)
        # loss = model.detect()['score'].tolist()
        # threshold_test_data['value'] = loss

        if not self.threshold_model_type:
            y_pred = [0 if loss[idx] <= self.threshold else 1 for idx in range(len(loss))]
        elif self.threshold_model_type == 'sarima':
            y_pred = self.threshold_model.predict(threshold_test_data)
        elif self.threshold_model_type == 'dbscan':
            y_pred = self.threshold_model.fit_predict(threshold_test_data[['value']])
        elif self.threshold_model_type == 'sr':
            # apply spectral residuals
            model = SpectralResidual(series=threshold_test_data[['value', 'timestamp']], threshold=THRESHOLD,
                                     mag_window=MAG_WINDOW, score_window=SCORE_WINDOW,
                                     sensitivity=99, detect_mode=DetectMode.anomaly_only)
            loss = model.detect()['score'].tolist()
            threshold_test_data['value'] = loss
            try:
                y_pred = [1 if x else 0 for x in model.detect()['isAnomaly'].tolist()]
            except:
                y_pred = [0 for _ in range(threshold_test_data.shape[0])]

        self.loss += loss.flatten().tolist()
        return y_pred

    def plot(self, timestamps, dataset, datatype, filename, data_test):
        loss_df = pd.DataFrame([])
        loss_df['value'] = self.loss
        loss_df['timestamp'] = timestamps
        try:
            arranged_loss = self.threshold_model.plot_threshold(loss_df, dataset, datatype, filename, data_test, 'lstm')
            arranged_loss = arranged_loss.dropna(subset=['value'])
        except:
            arranged_loss = loss_df

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=arranged_loss.index.tolist(), y=arranged_loss['value'].tolist(), name='Test loss'))
        fig.add_trace(go.Scatter(x=arranged_loss.index.tolist(), y=[self.threshold for _ in range(arranged_loss.shape[0])],
                                 name='Threshold'))
        fig.update_layout(showlegend=True, title='Test loss vs. Threshold')
        fig.write_image(f'results/imgs/{self.dataset}/{self.datatype}/lstm/lstm_{self.filename.replace(".csv", "")}_full.png')

        fig.data = []
