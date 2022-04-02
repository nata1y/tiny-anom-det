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

# print(f'Torch cuda is avaliable {torch.cuda.is_available()}')
from models.sr.spectral_residual import SpectralResidual, MAG_WINDOW, SCORE_WINDOW, THRESHOLD
from models.sr.util import DetectMode
from models.statistical_models import SARIMA

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
        # self.threshold = np.percentile(self.history.history['loss'], 90)

        self.svd_entropies = []

        for start in range(0, len(Xf), 60):
            try:
                self.svd_entropies.append(
                    ant.svd_entropy(Xf[start:start + 60], normalize=True))
            except:
                pass
        self.boundary_bottom = np.mean([v for v in self.svd_entropies if pd.notna(v)]) - \
                               2.5 * np.std([v for v in self.svd_entropies if pd.notna(v)])
        self.boundary_up = np.mean([v for v in self.svd_entropies if pd.notna(v)]) + \
                           2.5 * np.std([v for v in self.svd_entropies if pd.notna(v)])
        print(self.boundary_up, self.boundary_bottom)

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

        entropy = ant.svd_entropy(Xf, normalize=True)
        print(entropy)

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
        return y_pred1, y_pred2

    def plot(self, timestamps, dataset, datatype, filename, data_test, drifts):
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
