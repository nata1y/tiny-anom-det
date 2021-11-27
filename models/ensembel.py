import copy
from collections import Counter

import funcy
from catch22 import catch22_all
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from skopt.space import Integer, Categorical, Real

from models.nets import LSTM_autoencoder
from models.sr.spectral_residual import SpectralResidual
from models.sr.util import THRESHOLD, MAG_WINDOW, SCORE_WINDOW, DetectMode
from models.statistical_models import SARIMA

from tslearn.metrics import dtw
from pycaret.anomaly import *

from utils import create_dataset, plot_dbscan


class Ensemble:
    model_selector = RandomForestClassifier(max_depth=10, random_state=0)
    current_model = None
    current_model_name = ''
    models = {
      'sarima': (SARIMA, [Real(low=0.5, high=5.0, name="conf_top"), Real(low=0.5, high=5.0, name="conf_botton")], [1.15, 1.15]),
      'lstm': (LSTM_autoencoder, [Real(low=0.0, high=20.0, name='threshold')], [1.5]),
      'dbscan': (DBSCAN, [Integer(low=1, high=100, name='eps'), Integer(low=1, high=100, name='min_samples'),
                 Categorical(['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan',
                'nan_euclidean', dtw], name='metric')], [1, 2, dtw]),
      'sr': (SpectralResidual, [Real(low=0.01, high=0.99, name='THRESHOLD'),
                                    Integer(low=1, high=30, name='MAG_WINDOW'),
                                    Integer(low=5, high=1000, name='SCORE_WINDOW'),
                                    Integer(low=1, high=100, name='sensitivity')],
                                [THRESHOLD, MAG_WINDOW, SCORE_WINDOW, 99]),
    }
    dataset, datatype = '', ''
    all_labels_dbscan = []

    def __init__(self, filename, memory_size=3000, anomaly_window=60):
        self.memory_size = memory_size
        self.data_in_memory = pd.DataFrame([])
        self.end_train_time = 0
        self.anomaly_window = anomaly_window
        self.filename = filename

    def fit_selector(self, X, y, X_test, y_test):
        self.model_selector.fit(X, y)
        y_pred = self.model_selector.predict(X_test)
        print(classification_report(y_test, y_pred))

    def fit(self, data, dataset, datatype):
        self.dataset = dataset
        self.datatype = datatype
        properties = catch22_all(data['value'].tolist())

        model = self.model_selector.predict(properties.to_numpy())
        # TODO: dummy, update
        if model == 'lstm':
            X, y = create_dataset(data[['value']], data[['value']], self.anomaly_window)
            self.current_model = self.models['lstm'][0]([self.anomaly_window, 1], self.dataset, self.datatype,
                                                         self.filename.replace(".csv", ""), self.models['lstm'][2][0],
                                                         self.anomaly_window)
            self.current_model.fit(X, y, data['timestamp'], data['value'][-self.anomaly_window:])
            self.current_model_name = 'lstm'
        elif model == 'sarima':
            self.current_model = self.models['sarima'][0](self.models['sarima'][2][0], self.models['sarima'][2][1])
            self.current_model.fit(data[['timestamp', 'value']], self.dataset)
            self.current_model_name = 'sarima'
        elif model == 'dbscan':
            self.current_model = self.models['dbscan'][0](self.models['dbscan'][2][0],
                                                          self.models['dbscan'][2][1],
                                                          self.models['dbscan'][2][2])
            self.current_model_name = 'dbscan'
        elif model == 'sr':
            self.current_model = SpectralResidual(series=None,
                                                  threshold=self.models['dbscan'][2][0],
                                                  mag_window=self.models['dbscan'][2][1],
                                                  score_window=self.models['dbscan'][2][2],
                                                  sensitivity=self.models['dbscan'][2][3],
                                                  detect_mode=DetectMode.anomaly_only)
            self.current_model == 'sr'

    def predict(self, window):
        self.data_in_memory = pd.concat([self.data_in_memory, window])[-self.memory_size:]
        y_pred = []
        if self.current_model_name == 'sarima':
            y_pred = self.current_model.predict(window[['timestamp', 'value']])
        elif self.current_model_name == 'lstm':
            X = window['value']
            y_pred = self.current_model.predict(X.to_numpy().
                                                reshape(1, len(window['value'].tolist()), 1), window['timestamp'])
        elif self.current_model_name == 'dbscan':
            window_ = copy.deepcopy(self.data_in_memory[['value']])
            y_pred = self.current_model.fit_predict(window_)[-window.shape[0]:]
            core_smpls = np.array([i - window.index.min() for i in self.current_model.core_sample_indices_
                                   if window.index.min() <= i <= window.index.max()])
            common = Counter(funcy.lflatten(y_pred)[:window.shape[0]]).most_common(1)[0][0]
            y_pred = [0 if val == common else 1 for val in funcy.lflatten(y_pred)[:window.shape[0]]]
            self.all_labels_dbscan += [(copy.deepcopy(self.current_model.labels_[-window.shape[0]:]), core_smpls)]
        elif self.current_model_name == 'sr':
            self.current_model.__series__ = self.data_in_memory[['value', 'timestamp']]
            try:
                y_pred = [1 if x else 0 for x in self.current_model.detect(window)['isAnomaly'].tolist()[-window.shape[0]:]]
            except Exception as e:
                print(e)
                y_pred = [0 for _ in range(window.shape[0])]

        return y_pred

    def plot(self, data_test):
        if self.current_model_name == 'sarima':
            self.current_model.plot(data_test[['timestamp', 'value']], self.dataset, self.datatype, self.filename,
                                    data_test, True)
        elif self.current_model_name == 'lstm':
            self.current_model.plot(data_test['timestamp'].tolist(), self.dataset, self.datatype, self.filename,
                                    data_test, True)
        elif self.current_model_name == 'dbscan':
            plot_dbscan(self.all_labels_dbscan, self.dataset, self.datatype, self.filename,
                        data_test[['value', 'timestamp']], self.anomaly_window)

