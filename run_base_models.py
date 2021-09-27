import copy
import os
import time

import funcy
from funcy import flatten
from matplotlib import pyplot
from sklearn import metrics, preprocessing
from sklearn.cluster import DBSCAN, KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor, KNeighborsClassifier
from sklearn.svm import OneClassSVM
from skopt import gp_minimize
from skopt.space import Integer, Categorical, Real
from statsmodels.tsa._stl import STL
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

from analysis.preanalysis import visualize, series_analysis
from analysis.postanalysis import confusion_visualization
from analysis.time_series_feature_analysis import analyse_series_properties
from models.nets import LSTM_autoencoder
from models.statistical_models import SARIMA
from statsmodels.tsa.api import ExponentialSmoothing
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from utils import create_dataset

models = {
          # scale, n_clusters = 2
          # 'knn': (KMeans, []),
          # don't scale novelty=True
          # 'lof': (LocalOutlierFactor, [Integer(low=1, high=10, name='n_neighbors')]),
          # scale gamma='scale'
          # 'ocsvm': (OneClassSVM,
          #           [Real(low=0.001, high=0.999, name='nu'), Categorical(['linear', 'rbf', 'poly'], name='kernel')]),
          # 'dbscan': DBSCAN(eps=1, min_samples=3),
          # egads hyperparams, no normalization
          # 'dbscan': (DBSCAN, [Integer(low=1, high=1000, name='eps'), Integer(low=1, high=10, name='min_samples')]),
          'sarima': (SARIMA, []),
          # no norm
          # 'isolation_forest': (IsolationForest, [Integer(low=1, high=1000, name='n_estimators')]),
          # 'es': (ExponentialSmoothing, []),
          # 'stl': (STL, []),
          # 'lstm': (LSTM_autoencoder, [Real(low=0.001, high=0.999, name='threshold')]),
          # 'sr-cnn': []
          }
root_path = os.getcwd()
le = preprocessing.LabelEncoder().fit([-1, 1])
anomaly_window = 60


def fit_base_model(model, data, filename, model_params, for_optimization=False):
    # 50% train-test split
    data_train, data_test = np.array_split(data, 2)
    # standardize
    scaler = preprocessing.StandardScaler().fit(data_train[['value']])
    data_train['value'] = scaler.transform(data_train[['value']])
    data_test['value'] = scaler.transform(data_test[['value']])

    # create models with hyper-parameters
    if name == 'knn':
        model = KMeans(n_clusters=2)
    elif name == 'lof':
        model = LocalOutlierFactor(novelty=True, n_neighbors=model_params[0])
    elif name == 'ocsvm':
        model = OneClassSVM(gamma='scale', nu=model_params[0], kernel=model_params[1])
    elif name == 'dbscan':
        model = DBSCAN(eps=model_params[0], min_samples=model_params[1])
    elif name == 'sarima':
        model = SARIMA()
    elif name == 'isolation_forest':
        model = IsolationForest(n_estimators=model_params[0])
    elif name == 'lstm':
        model = LSTM_autoencoder([anomaly_window, 1], threshold=model_params[0])

    # visualize(data)
    trend, seasonality, autocrr, non_lin, skewness, kurtosis, hurst = \
        series_analysis(data)

    if name == 'sarima':

        start_time = time.time()
        model.fit(data_train[['timestamp', 'value']], dataset)
        end_time = time.time()

        diff = end_time - start_time
        print(f"Trained model {name} on {filename} for {diff}")
    elif name == 'lstm':
        X, y = create_dataset(data_train[['value']], data_train.value, anomaly_window)
        start_time = time.time()
        model.fit(X)
        end_time = time.time()

        diff = end_time - start_time
        print(f"Trained model {name} on {filename} for {diff}")
    else:
        num_windows = data_train.shape[0] // anomaly_window + 1
        for window in np.array_split(data_train, num_windows):
            X, y = window[['value']], window['is_anomaly']

            start_time = time.time()
            model.fit(X)
            end_time = time.time()

            diff = end_time - start_time
            print(f"Trained model {name} on {filename} for {diff}")

    num_windows = data_test.shape[0] // anomaly_window + 1
    precision, recall, f1 = [], [], []
    time_total, value_total, y_total, y_pred_total = [], [], [], []
    for window in np.array_split(data_test, num_windows):
        X, y = window['value'], window['is_anomaly']
        value_total += X.tolist()
        time_total += window['timestamp'].tolist()
        y_total += y.tolist()

        if name in ['sarima']:
            y_pred = model.predict(window[['timestamp', 'value']])
        elif name == 'dbscan':
            y_pred = model.fit_predict(X)
        else:
            y_pred = model.predict(X)

        try:
            y_pred = le.transform(y_pred).tolist()
        except:
            pass

        y_pred_total += list(y_pred)
        # print(metrics.classification_report(y, y_pred))
        met = precision_recall_fscore_support(y, y_pred, average='weighted')
        precision.append(met[0])
        recall.append(met[1])
        f1.append(met[2])
        print(f"Model {name} has f1-score {f1}")

    if not for_optimization:
        try:
            stats = pd.read_csv(f'results/yahoo_{type}_stats_{name}.csv')
        except:
            stats = pd.DataFrame([])

        if name == 'sarima':
            model.plot(data_test[['timestamp', 'value']], dataset, type, filename, data_test)

        confusion_visualization(time_total, value_total, y_total, y_pred_total,
                                dataset, name, filename.replace('.csv', ''), type)

        quit()
        stats = stats.append({
            'model': name,
            'dataset': filename.replace('.csv', ''),
            # 'periodicity': periodicity,
            'trend': trend,
            'seasonality': seasonality,
            'autocorrelation': autocrr,
            'non-linearity': non_lin,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'hurst': hurst,
            'mean_f1': np.mean(f1),
            'min_f1': np.min(f1),
            'mean_precision': np.mean(precision),
            'min_precision': np.min(precision),
            'mean_recall': np.mean(recall),
            'min_recall': np.min(recall),
            'num_windows': num_windows
        }, ignore_index=True)
        stats.to_csv(f'results/yahoo_{type}_stats_{name}.csv', index=False)

    return np.mean(f1)


# monitor function for BO optimization intermediate results
def monitor(res):
    print('run_score: %s' % str(res.func_vals[-1]))
    print('run_parameters: %s' % str(res.x_iters[-1]))


def explore_dataset(model_params, for_optimization=True):
    mean_f1 = []
    for filename in os.listdir(train_data_path):
        f = os.path.join(train_data_path, filename)
        if os.path.isfile(f):
            print(f"Training model {name} with data {filename}")
            data = pd.read_csv(f)

            mean_f1.append(fit_base_model(model, data, filename, model_params, for_optimization=for_optimization))

    return np.mean(mean_f1)


if __name__ == '__main__':
    dataset, type = 'yahoo', 'synthetic'
    for name, (model, bo_space) in models.items():
        train_data_path = root_path + '/datasets/' + dataset + '/' + type + '/'

        if name not in ['knn', 'sarima']:
        ################ Bayesian optimization of hyperparams ###################################################
            bo_result = gp_minimize(explore_dataset, bo_space, callback=[monitor], n_calls=20, random_state=1,
                                    verbose=False)

            print(f"Found hyper parameters for {name}: {bo_result.x}")

            explore_dataset(bo_result.x, for_optimization=False)
        else:
            explore_dataset(None, for_optimization=False)

        analyse_series_properties(dataset, type, name)
