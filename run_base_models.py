import copy
import os
import time

from sklearn import metrics, preprocessing
from sklearn.cluster import DBSCAN, KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor, KNeighborsClassifier
from sklearn.svm import OneClassSVM
from statsmodels.tsa._stl import STL
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

from analysis.preanalysis import visualize, series_analysis
from analysis.postanalysis import confusion_visualization
from analysis.time_series_feature_analysis import analyse_series_properties
from models.statistical_models import fit_sarima
from statsmodels.tsa.api import ExponentialSmoothing
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


models = {
          # scale
          # 'knn': KMeans(n_clusters=2),
          # don't scale
          # 'lof': LocalOutlierFactor(n_neighbors=4, novelty=True),
          # scale
          # 'ocsvm': OneClassSVM(kernel='poly', gamma='scale', nu=0.9),
          'dbscan': DBSCAN(eps=1, min_samples=3),
          # egads hyperparams, no normalization
          # 'dbscan': DBSCAN(eps=500, min_samples=2),
          # 'sarima': SARIMAX,
          # no norm
          # 'isolation_forest': IsolationForest(),
          # 'es': ExponentialSmoothing()
          # 'stl': STL,
          # 'lstm': [],
          # 'sr-cnn': []
          }
root_path = os.getcwd()
le = preprocessing.LabelEncoder().fit([-1, 1])
anomaly_window = 60


def fit_base_model():
    # visualize(data)
    trend, seasonality, autocrr, non_lin, skewness, kurtosis, hurst = \
        series_analysis(data)

    num_windows = data_train.shape[0] // anomaly_window + 1
    for window in np.array_split(data_train, num_windows):
        X, y = window['value'], window['is_anomaly']
        X = X.values.reshape(-1, 1)
        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)

        if name in ['sarima']:
            model_ = copy.deepcopy(model(X))
            if name == 'sarima':
                start_time = time.time()
                results = fit_sarima(X)
                end_time = time.time()

                diff = end_time - start_time
                print(f"Trained model {name} on {filename} for {diff}")
        else:
            start_time = time.time()
            model.fit(X)
            end_time = time.time()

            diff = end_time - start_time
            print(f"Trained model {name} on {filename} for {diff}")

    num_windows = data_test.shape[0] // anomaly_window + 1
    f1 = []
    precision, recall = [], []
    time_total, value_total, y_total, y_pred_total = [], [], [], []
    for window in np.array_split(data_test, num_windows):
        X, y = window['value'], window['is_anomaly']
        value_total += X.tolist()
        X = X.values.reshape(-1, 1)
        time_total += window['timestamp'].tolist()
        y_total += y.tolist()
        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)

        if name in ['sarima']:
            continue
            print(results.summary().tables[1])
            results.plot_diagnostics(figsize=(18, 8))
            pyplot.show()
        else:
            if name == 'dbscan':
                y_pred = model.fit_predict(X)
            else:
                y_pred = model.predict(X)
            try:
                y_pred = le.transform(y_pred)
            except:
                pass

            y_pred_total += y_pred.tolist()
            # print(metrics.classification_report(y, y_pred))
            met = precision_recall_fscore_support(y, y_pred, average='weighted')
            precision.append(met[0])
            recall.append(met[1])
            f1.append(met[2])
            print(f"Model {name} has f1-score {f1}")

    try:
        stats = pd.read_csv(f'results/yahoo_{type}_stats.csv')
    except:
        stats = pd.DataFrame([])

    confusion_visualization(time_total, value_total, y_total, y_pred_total,
                            dataset, name, filename.replace('.csv', ''), type)

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
    stats.to_csv(f'results/yahoo_{type}_stats.csv', index=False)


if __name__ == '__main__':
    dataset, type = 'yahoo', 'real'
    for name, model in models.items():
        train_data_path = root_path + '/datasets/' + dataset + '/' + type + '/'
        for filename in os.listdir(train_data_path):
            f = os.path.join(train_data_path, filename)
            if os.path.isfile(f):
                print(f"Training model {name} with data {filename}")
                data = pd.read_csv(f)
                # 50% train-test split
                data_train, data_test = np.array_split(data, 2)

                fit_base_model()

        analyse_series_properties(pd.read_csv(f'results/yahoo_{type}_stats.csv'))
