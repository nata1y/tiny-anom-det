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
from models.sr.main import detect_anomaly
from models.statistical_models import SARIMA, ExpSmoothing
from statsmodels.tsa.api import ExponentialSmoothing
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from utils import create_dataset
from tslearn.metrics import dtw
from models.sr.spectral_residual import THRESHOLD, MAG_WINDOW, SCORE_WINDOW, DetectMode, SpectralResidual

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
          'dbscan': (DBSCAN,
                     [Integer(low=1, high=100, name='eps'), Integer(low=1, high=10, name='min_samples'),
                      Categorical(['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan',
                                   'nan_euclidean', dtw], name='metric')],
                     [1, 3, 'euclidean']),
          'sarima': (SARIMA, [Real(low=0.5, high=5.0, name="conf")], [1.5]),
          # no norm
          # 'isolation_forest': (IsolationForest, [Integer(low=1, high=1000, name='n_estimators')], [100]),
          'es': (ExpSmoothing, [Integer(low=10, high=1000, name='sims')], [100]),
          # 'stl': (STL, []),
          'lstm': (LSTM_autoencoder, [Real(low=0.0, high=20.0, name='threshold')], [7.0]),
          # 'sr-cnn': []
          'sr': (SpectralResidual, [Real(low=0.01, high=0.99, name='THRESHOLD'),
                                    Integer(low=1, high=30, name='MAG_WINDOW'),
                                    Integer(low=5, high=50, name='SCORE_WINDOW'),
                                    Integer(low=1, high=100, name='sensitivity')],
                 [THRESHOLD, MAG_WINDOW, SCORE_WINDOW, 99, DetectMode.anomaly_only])
          }
root_path = os.getcwd()
le = preprocessing.LabelEncoder().fit([-1, 1])
anomaly_window = 60 #1024
data_test = None


def fit_base_model(model_params, for_optimization=True):
    global data_test
    # 50% train-test split
    if dataset == 'kpi':
        data_train = data
    else:
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
        model = DBSCAN(eps=model_params[0], min_samples=model_params[1], metric=model_params[2])
    elif name == 'sarima':
        model = SARIMA(model_params[0])
    elif name == 'es':
        model = ExpSmoothing(model_params[0])
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
    elif name == 'sr':
        model = SpectralResidual(series=data_train[['value', 'timestamp']], threshold=model_params[0], mag_window=model_params[1],
                                 score_window=model_params[2], sensitivity=model_params[3],
                                 detect_mode=DetectMode.anomaly_only)
        # print(model.detect())
    else:
        if name != 'dbscan':
            try:
                X, y = data_test[['value']], data_test['is_anomaly']
                start_time = time.time()
                model.fit(X)
                end_time = time.time()
            except:
                X, y = data_test[['value']], data_test['is_anomaly']
                start_time = time.time()
                model.fit(X[-25000:])
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
            elif name == 'sr':
                model = SpectralResidual(series=window[['value', 'timestamp']], threshold=model_params[0],
                                         mag_window=model_params[1], score_window=model_params[2],
                                         sensitivity=model_params[3], detect_mode=DetectMode.anomaly_only)
                try:
                    y_pred = [1 if x else 0 for x in model.detect()['isAnomaly'].tolist()]
                except:
                    y_pred = [0 for _ in range(window.shape[0])]

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
            y_pred_total += [0 if v != 1 else 1 for v in funcy.lflatten(y_pred[:len(list(y))])]

            # print(metrics.classification_report(y, y_pred))
            met = precision_recall_fscore_support(y, y_pred[:len(list(y))], average='weighted')
            precision.append(met[0])
            recall.append(met[1])
            f1.append(met[2])
            print(f"Model {name} has f1-score {f1[-1]} on window {start}")
        except Exception as e:
            print(e)
            raise e
        end_time = time.time()
        pred_time.append((end_time - start_time))

    met_total = precision_recall_fscore_support(data_test['is_anomaly'], y_pred_total, average='binary')

    if not for_optimization:
        # visualize(data)
        trend, seasonality, autocrr, non_lin, skewness, kurtosis, hurst, lyapunov = \
            series_analysis(data)

        try:
            stats = pd.read_csv(f'results/{dataset}_{type}_stats_{name}.csv')
        except:
            stats = pd.DataFrame([])

        if name in ['sarima', 'es']:
            model.plot(data_test[['timestamp', 'value']], dataset, type, filename, data_test)
        elif name == 'lstm':
            model.plot(data['timestamp'].tolist())

        confusion_visualization(time_total, value_total, y_total[:len(y_pred_total)], y_pred_total,
                                dataset, name, filename.replace('.csv', ''), type)

        try:
            tn, fp, fn, tp = confusion_matrix(data_test['is_anomaly'], y_pred_total).ravel()
            specificity = tn / (tn + fp)
        except:
            specificity = None
            tn, fp, fn, tp = None, None, None, None

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
            'f1': met_total[2],
            'precision': met_total[0],
            'recall': met_total[1],
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'tn': tn,
            'specificity': specificity,
            'total_anomalies': data_test['is_anomaly'].tolist().count(1),
            'prediction_time': np.mean(pred_time),
            'total_points': data_test.shape[0]
        }, ignore_index=True)
        stats.to_csv(f'results/{dataset}_{type}_stats_{name}.csv', index=False)

    return met_total[2]


def monitor(res):
    print('run_score: %s' % str(res.func_vals[-1]))
    print('run_parameters: %s' % str(res.x_iters[-1]))


if __name__ == '__main__':
    dataset, type = 'yahoo', 'real'
    for name, (model, bo_space, def_params) in models.items():
        train_data_path = root_path + '/datasets/' + dataset + '/' + type + '/'
        for filename in os.listdir(train_data_path):
            f = os.path.join(train_data_path, filename)
            if os.path.isfile(f):
                print(f"Training model {name} with data {filename}")
                data = pd.read_csv(f)
                if dataset == 'kpi':
                    data_test = pd.read_csv(os.path.join(root_path + '/datasets/' + dataset + '/' + 'test' + '/', filename))

                fit_base_model(def_params, for_optimization=False)

                # if name not in ['knn']:
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
        # try:
        #     stats = pd.read_csv(f'results/{dataset}_{type}_stats_{name}.csv')
        #     stats = stats.append({
        #         'model': name,
        #         'dataset': 'all-' + type,
        #         'trend': np.mean(stats['trend'].tolist()),
        #         'seasonality': np.mean(stats['seasonality'].tolist()),
        #         'autocorrelation': np.mean(stats['autocorrelation'].tolist()),
        #         'non-linearity': np.mean(stats['non-linearity'].tolist()),
        #         'skewness': np.mean(stats['skewness'].tolist()),
        #         'kurtosis': np.mean(stats['kurtosis'].tolist()),
        #         'hurst': np.mean(stats['hurst'].tolist()),
        #         'mean_lyapunov_e': np.mean(stats['max_lyapunov_e'].tolist()),
        #         'mean_f1': np.mean(stats['f1'].tolist()),
        #         'min_f1': np.min(stats['min_f1'].tolist()),
        #         'mean_precision': np.mean(stats['mean_precision'].tolist()),
        #         'min_precision': np.min(stats['min_precision'].tolist()),
        #         'mean_recall': np.mean(stats['mean_recall'].tolist()),
        #         'min_recall': np.min(stats['min_recall'].tolist()),
        #         'prediction_time': np.mean(stats['prediction_time'].tolist())
        #     }, ignore_index=True)
        #     stats.to_csv(f'results/{dataset}_{type}_stats_{name}.csv', index=False)
        # except Exception as e:
        #     print(e)
