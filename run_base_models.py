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
from models.statistical_models import SARIMA, ExpSmoothing
from statsmodels.tsa.api import ExponentialSmoothing
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from utils import create_dataset

models = {
          # scale, n_clusters = 2
          # 'knn': (KMeans, [], []),
          # don't scale novelty=True
          # 'lof': (LocalOutlierFactor, [Integer(low=1, high=20, name='n_neighbors')], [5]),
          # scale gamma='scale'
          # 'ocsvm': (OneClassSVM,
          #           [Real(low=0.001, high=0.999, name='nu'), Categorical(['linear', 'rbf', 'poly'], name='kernel')],
          #           [0.85, 'poly']),
          # 'dbscan': DBSCAN(eps=1, min_samples=3),
          # egads hyperparams, no normalization
          # 'dbscan': (DBSCAN, [Integer(low=1, high=100, name='eps'), Integer(low=1, high=10, name='min_samples')], [1, 3]),
          'sarima': (SARIMA, [], []),
          # no norm
          'isolation_forest': (IsolationForest, [Integer(low=1, high=1000, name='n_estimators')], [100]),
          'es': (ExpSmoothing, [], []),
          # 'stl': (STL, []),
          'lstm': (LSTM_autoencoder, [Real(low=0.0, high=20.0, name='threshold')], [7.0]),
          # 'sr-cnn': []
          }
root_path = os.getcwd()
le = preprocessing.LabelEncoder().fit([-1, 1])
anomaly_window = 60


def fit_base_model(model_params, for_optimization=True):
    # 50% train-test split
    data_train, data_test = np.array_split(data, 2)
    model = None
    # standardize
    if name not in ['sarima', 'lof', 'isolation_forest', 'es']:
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
    elif name == 'es':
        model = ExpSmoothing()
    elif name == 'isolation_forest':
        model = IsolationForest(n_estimators=model_params[0])
    elif name == 'lstm':
        model = LSTM_autoencoder([anomaly_window, 1],  dataset, type, filename.replace(".csv", ""), threshold=model_params[0])

    if name in ['sarima', 'es']:

        start_time = time.time()
        model.fit(data_train[['timestamp', 'value']], dataset)
        end_time = time.time()

        diff = end_time - start_time
        print(f"Trained model {name} on {filename} for {diff}")
    elif name == 'lstm':
        X, y = create_dataset(data_train[['value']], data_train[['value']], anomaly_window)
        start_time = time.time()
        model.fit(X, y)
        end_time = time.time()

        diff = end_time - start_time
        print(f"Trained model {name} on {filename} for {diff}")
    else:
        X, y = data_test[['value']], data_test['is_anomaly']
        start_time = time.time()
        model.fit(X)
        end_time = time.time()

        diff = end_time - start_time
        print(f"Trained model {name} on {filename} for {diff}")
        # for start in range(0, data_test.shape[0], anomaly_window):
        #     window = data_test.iloc[start:start + anomaly_window]
        #     X, y = window[['value']], window['is_anomaly']
        #     start_time = time.time()
        #     model.fit(X)
        #     end_time = time.time()
        #
        #     diff = end_time - start_time
        #     print(f"Trained model {name} on {filename} for {diff}")

    precision, recall, f1 = [], [], []
    time_total, value_total, y_total, y_pred_total = [], [], [], []
    idx = 0
    stacked_res = data_train[['timestamp', 'value']]

    pred_time = []
    for start in range(0, data_test.shape[0], anomaly_window):
        start_time = time.time()
        try:
            window = data_test.iloc[start:start + anomaly_window]
            X, y = window['value'], window['is_anomaly']
            value_total += X.tolist()
            time_total += window['timestamp'].tolist()
            y_total += y.tolist()

            if name in ['sarima']:
                y_pred = model.predict(window[['timestamp', 'value']])
            elif name == 'es':
                y_pred = model.predict(window[['timestamp', 'value']], anomaly_window)
                stacked_res = pd.concat([stacked_res, window[['value', 'timestamp']]])
                model.fit(stacked_res, dataset)
            elif name == 'lstm':
                y_pred = model.predict(X.to_numpy().reshape(1, len(window['value'].tolist()), 1))
            elif name == 'dbscan':
                y_pred = model.fit_predict(window[['value']])
            else:
                y_pred = model.predict(window[['value']])

            try:
                y_pred = le.transform(y_pred).tolist()
            except:
                pass

            idx += 1
            y_pred_total += funcy.lflatten(y_pred)

            # print(metrics.classification_report(y, y_pred))
            met = precision_recall_fscore_support(y, y_pred[:len(list(y))], average='weighted')
            precision.append(met[0])
            recall.append(met[1])
            f1.append(met[2])
            print(f"Model {name} has f1-score {f1[-1]}")
        except Exception as e:
            print(e)
            raise e
        end_time = time.time()
        pred_time.append((end_time - start_time))

    if not for_optimization:
        # visualize(data)
        trend, seasonality, autocrr, non_lin, skewness, kurtosis, hurst, lyapunov = \
            series_analysis(data)

        try:
            stats = pd.read_csv(f'results/yahoo_{type}_stats_{name}.csv')
        except:
            stats = pd.DataFrame([])

        if name in ['sarima', 'es']:
            model.plot(data_test[['timestamp', 'value']], dataset, type, filename, data_test)
        elif name == 'lstm':
            model.plot(data['timestamp'].tolist())

        confusion_visualization(time_total, value_total, y_total[:len(y_pred_total)], y_pred_total,
                                dataset, name, filename.replace('.csv', ''), type)

        stats = stats.append({
            'model': name,
            'dataset': filename.replace('.csv', ''),
            'trend': trend,
            'seasonality': seasonality,
            'autocorrelation': autocrr,
            'non-linearity': non_lin,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'hurst': hurst,
            'max_lyapunov_e': lyapunov,
            'mean_f1': np.mean(f1),
            'min_f1': np.min(f1),
            'mean_precision': np.mean(precision),
            'min_precision': np.min(precision),
            'mean_recall': np.mean(recall),
            'min_recall': np.min(recall),
            'prediction_time': np.mean(pred_time)
        }, ignore_index=True)
        stats.to_csv(f'results/yahoo_{type}_stats_{name}.csv', index=False)

    return np.mean(f1)


def monitor(res):
    print('run_score: %s' % str(res.func_vals[-1]))
    print('run_parameters: %s' % str(res.x_iters[-1]))


if __name__ == '__main__':
    dataset, type = 'yahoo', 'synthetic'
    for name, (model, bo_space, def_params) in models.items():
        train_data_path = root_path + '/datasets/' + dataset + '/' + type + '/'
        for filename in os.listdir(train_data_path):
            f = os.path.join(train_data_path, filename)
            if os.path.isfile(f):
                print(f"Training model {name} with data {filename}")
                data = pd.read_csv(f)

                fit_base_model(def_params, for_optimization=False)

                # if name not in ['knn', 'sarima', 'es']:
                # ################ Bayesian optimization ###################################################
                #     bo_result = gp_minimize(fit_base_model, bo_space, callback=[monitor], n_calls=10, random_state=13,
                #                             verbose=False)
                #
                #     print(f"Found hyper parameters for {name}: {bo_result.x}")
                #
                #     fit_base_model(bo_result.x, for_optimization=False)
                # else:
                #     fit_base_model(def_params, for_optimization=False)

        analyse_series_properties(dataset, type, name)
        try:
            stats = pd.read_csv(f'results/yahoo_{type}_stats_{name}.csv')
            stats = stats.append({
                'model': name,
                'dataset': 'all-' + type,
                'trend': np.mean(stats['trend'].tolist()),
                'seasonality': np.mean(stats['seasonality'].tolist()),
                'autocorrelation': np.mean(stats['autocorrelation'].tolist()),
                'non-linearity': np.mean(stats['non-linearity'].tolist()),
                'skewness': np.mean(stats['skewness'].tolist()),
                'kurtosis': np.mean(stats['kurtosis'].tolist()),
                'hurst': np.mean(stats['hurst'].tolist()),
                'mean_lyapunov_e': np.mean(stats['mean_lyapunov_e'].tolist()),
                'mean_f1': np.mean(stats['mean_f1'].tolist()),
                'min_f1': np.min(stats['min_f1'].tolist()),
                'mean_precision': np.mean(stats['mean_precision'].tolist()),
                'min_precision': np.min(stats['min_precision'].tolist()),
                'mean_recall': np.mean(stats['mean_recall'].tolist()),
                'min_recall': np.min(stats['min_recall'].tolist()),
                'prediction_time': np.mean(stats['prediction_time'].tolist())
            }, ignore_index=True)
            stats.to_csv(f'results/yahoo_{type}_stats_{name}.csv', index=False)
        except:
            pass
