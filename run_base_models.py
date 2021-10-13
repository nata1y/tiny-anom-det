import copy
import os
import time

import funcy
from alibi_detect.utils.data import create_outlier_batch
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
from models.nets import LSTM_autoencoder, Vae, SeqSeq
from models.sr.main import detect_anomaly
from models.statistical_models import SARIMA, ExpSmoothing
from statsmodels.tsa.api import ExponentialSmoothing
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from utils import create_dataset
from tslearn.metrics import dtw
from models.sr.spectral_residual import THRESHOLD, MAG_WINDOW, SCORE_WINDOW, DetectMode, SpectralResidual
from alibi_detect.od import SpectralResidual as SR, OutlierVAE
from alibi_detect.od import OutlierProphet, OutlierSeq2Seq


root_path = os.getcwd()
le = preprocessing.LabelEncoder().fit([-1, 1])
anomaly_window = 1024
step = 1024
data_test = None
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
          # 'dbscan': (DBSCAN,
          #            [Integer(low=1, high=100, name='eps'), Integer(low=1, high=10, name='min_samples'),
          #             Categorical(['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan',
          #                          'nan_euclidean', dtw], name='metric')],
          #            [1, 3, 'euclidean']),
          'sarima': (SARIMA, [Real(low=0.5, high=5.0, name="conf")], [1.5]),
          # no norm
          # 'isolation_forest': (IsolationForest, [Integer(low=1, high=1000, name='n_estimators')], [100]),
          # 'es': (ExpSmoothing, [Integer(low=10, high=1000, name='sims')], [100]),
          # 'stl': (STL, []),
          # 'lstm': (LSTM_autoencoder, [Real(low=0.0, high=20.0, name='threshold')], [1.5]),
          # 'prophet': (OutlierProphet, [Real(low=0.01, high=5.0, name='threshold'),
          #                              Categorical(['linear', 'logistic'], name='growth')], [0.9, 'linear']),
          # 'sr_alibi': (SR, [Real(low=0.01, high=10.0, name='threshold'),
          #                   Integer(low=1, high=anomaly_window, name='window_amp'),
          #                   Integer(low=1, high=anomaly_window, name='window_local'),
          #                   Integer(low=1, high=anomaly_window, name='n_est_points'),
          #                   Integer(low=1, high=anomaly_window, name='n_grad_points'),
          #                   Real(low=0.5, high=0.999, name='percent_anom')],
          #              [1.0, 20, 20, 10, 5, 0.95]),
          # 'seq2seq': (OutlierSeq2Seq, [Integer(low=1, high=100, name='latent_dim'),
          #                              Real(low=0.5, high=0.999, name='percent_anom')], [2, 0.95]),
          # 'sr-cnn': []
          # 'sr': (SpectralResidual, [Real(low=0.01, high=0.99, name='THRESHOLD'),
          #                           Integer(low=1, high=30, name='MAG_WINDOW'),
          #                           Integer(low=5, high=50, name='SCORE_WINDOW'),
          #                           Integer(low=1, high=100, name='sensitivity')],
          #        [THRESHOLD, MAG_WINDOW, SCORE_WINDOW, 99]),
          # 'vae': (OutlierVAE, [Real(low=0.01, high=0.99, name='threshold'),
          #                      Integer(low=2, high=anomaly_window, name='latent_dim'),
          #                      Integer(low=1, high=100, name='samples'),
          #                      Real(low=0.5, high=0.999, name='percent_anom')],
          #         [0.9, 100, 10, 0.95])
          }


def fit_base_model(model_params, for_optimization=True):
    global data_test
    percent_anomaly = 0.95
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
        model = LSTM_autoencoder([anomaly_window, 1],  dataset, type, filename.replace(".csv", ""),
                                 magnitude=model_params[0], window=anomaly_window)
    elif name == 'seq2seq':
        seqseq_net = SeqSeq(anomaly_window, 2 * model_params[0])
        model = OutlierSeq2Seq(n_features=1, seq_len=anomaly_window, threshold=None,
                               threshold_net=seqseq_net.threshold_net_seqseq, seq2seq=None,
                               latent_dim=int(2 * model_params[0]))
        percent_anomaly = model_params[1]
    elif name == 'prophet':
        model = OutlierProphet(threshold=model_params[0], growth=model_params[1])
    elif name == 'sr_alibi':
        model = SR(threshold=model_params[0], window_amp=model_params[1], window_local=model_params[2],
                   n_est_points=model_params[3], n_grad_points=model_params[4])
        percent_anomaly = model_params[5]

    elif name == 'vae':
        vae_net = Vae(model_params[1])
        model = OutlierVAE(threshold=model_params[0], encoder_net=vae_net.encoder_net_vae,
                           decoder_net=vae_net.decoder_net_vae, latent_dim=model_params[1], samples=model_params[2])
        percent_anomaly = model_params[3]

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
    elif name in ['seq2seq', 'vae']:
        X, y = create_dataset(data_train[['value']], data_train[['value']], anomaly_window)
        if name == 'vae':
            X = X.reshape(X.shape[0], anomaly_window)
        start_time = time.time()
        model.fit(X, epochs=50)
        end_time = time.time()
        model.infer_threshold(X, threshold_perc=percent_anomaly)

        diff = end_time - start_time
        print(f"Trained model {name} on {filename} for {diff}")
    elif name == 'sr_alibi':
        model.infer_threshold(data_train[['value']].to_numpy(), threshold_perc=percent_anomaly)
    else:
        if name not in ['dbscan']:
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
    y_pred_total = []
    idx = 0
    stacked_res = data_train[['timestamp', 'value']]

    pred_time = []
    for start in range(0, data_test.shape[0], step):
        try:
            start_time = time.time()
            window = data_test.iloc[start:start + anomaly_window]
            X, y = window['value'], window['is_anomaly']
            if y.tolist():
                if name in ['sarima']:
                    y_pred = model.predict(window[['timestamp', 'value']])
                    print(y_pred)
                    print(y)
                    print(len(y))
                    print(len(y_pred))
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
                elif name == 'vae':
                    mean_val = np.mean(X.to_numpy().flatten())
                    Xf = funcy.lflatten(X.to_numpy().flatten())
                    for idx in range(anomaly_window - len(Xf)):
                        Xf.append(mean_val)

                    Xf = np.array(Xf).reshape((1, anomaly_window))
                    y_pred = model.predict(Xf, outlier_type='feature')
                    y_pred = [x for x in y_pred['data']['is_outlier'][0][:len(y)]]

                elif name == 'sr_alibi':
                    mean_val = np.mean(X.to_numpy().flatten())
                    Xf = funcy.lflatten(X.to_numpy().flatten())
                    for idx in range(anomaly_window - len(Xf)):
                        Xf.append(mean_val)

                    Xf = np.asarray(Xf).reshape(anomaly_window, 1)

                    y_pred = model.predict(Xf)
                    y_pred = [x for x in y_pred['data']['is_outlier'][:len(y)]]
                elif name == 'seq2seq':
                    mean_val = np.mean(X.to_numpy().flatten())
                    Xf = funcy.lflatten(X.to_numpy().flatten())
                    for idx in range(anomaly_window - len(Xf)):
                        Xf.append(mean_val)

                    Xf = np.array(Xf).reshape(anomaly_window, 1)
                    y_pred = model.predict(Xf, outlier_type='feature')
                    y_pred = [x[0] for x in y_pred['data']['is_outlier'][:len(y)]]
                elif name == 'dbscan':
                    y_pred = model.fit_predict(window[['value']])
                else:
                    y_pred = model.predict(window[['value']])

                try:
                    y_pred = le.transform(y_pred).tolist()
                except:
                    pass

                idx += 1
                if anomaly_window == step:
                    y_pred_total += [0 if val != 1 else 1 for val in funcy.lflatten(y_pred)[:window.shape[0]]]
                else:
                    if y_pred_total:
                        if len(y_pred) == anomaly_window:
                            y_pred_total = y_pred_total[:-anomaly_window + step] + \
                                           [0 if (o != 1 and n != 1) else 1 for o, n in
                                            zip(y_pred_total[-anomaly_window + step:] + [0 for _ in range(step)],
                                                funcy.lflatten(y_pred))]
                        else:
                            y_pred_total = y_pred_total[:-anomaly_window + step] + \
                                           [0 if (o != 1 and n != 1) else 1 for o, n in
                                            zip(y_pred_total[-anomaly_window + step:] +
                                                [0 for _ in range(step - (anomaly_window - len(y_pred)))],
                                                funcy.lflatten(y_pred))]
                            break
                    else:
                        y_pred_total = [0 if val != 1 else 1 for val in funcy.lflatten(y_pred)]

                # print(metrics.classification_report(y, y_pred))
                # met = precision_recall_fscore_support(y, y_pred[:len(list(y))], average='weighted')
                # precision.append(met[0])
                # recall.append(met[1])
                # f1.append(met[2])
                # print(f"Model {name} has f1-score {f1[-1]} on window {start}")
                end_time = time.time()
                pred_time.append((end_time - start_time))
        except Exception as e:
            raise e

    met_total = precision_recall_fscore_support(data_test['is_anomaly'], y_pred_total, average='binary')

    if not for_optimization:
        # visualize(data)
        trend, seasonality, autocrr, non_lin, skewness, kurtosis, hurst, lyapunov = \
            series_analysis(data[:25000])

        try:
            stats = pd.read_csv(f'results/{dataset}_{type}_stats_{name}.csv')
        except:
            stats = pd.DataFrame([])

        confusion_visualization(data_test['timestamp'].tolist(), data_test['value'].tolist(),
                                data_test['is_anomaly'].tolist(), y_pred_total,
                                dataset, name, filename.replace('.csv', ''), type)

        if name in ['sarima', 'es']:
            model.plot(data_test[['timestamp', 'value']], dataset, type, filename, data_test)
        elif name == 'lstm':
            model.plot(data['timestamp'].tolist())

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

    return 1.0 - met_total[2]


def monitor(res):
    print('run_score: %s' % str(res.func_vals[-1]))
    print('run_parameters: %s' % str(res.x_iters[-1]))


if __name__ == '__main__':
    dataset, type = 'kpi', 'train'
    for name, (model, bo_space, def_params) in models.items():
        train_data_path = root_path + '/datasets/' + dataset + '/' + type + '/'
        for filename in os.listdir(train_data_path):
            f = os.path.join(train_data_path, filename)
            if os.path.isfile(f):
                print(f"Training model {name} with data {filename}")
                data = pd.read_csv(f)[-3000:]
                if dataset == 'kpi':
                    data_test = pd.read_csv(os.path.join(root_path + '/datasets/' + dataset + '/' + 'test' + '/', filename))[:2000]

                fit_base_model(def_params, for_optimization=False)
                quit()

                # if name not in ['knn']:
                # ################ Bayesian optimization ###################################################
                #     bo_result = gp_minimize(fit_base_model, bo_space, callback=[monitor], n_calls=20, random_state=13,
                #                             verbose=False, x0=def_params)
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
