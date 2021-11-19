import copy
import os
import time
from collections import Counter

import funcy
from alibi_detect.utils.data import create_outlier_batch
from funcy import flatten
from matplotlib import pyplot
from pytorch_forecasting import DeepAR
from sklearn import metrics, preprocessing
from sklearn.cluster import DBSCAN, KMeans

from models.decompose_model import DecomposeResidual
from models.ensembel import Ensemble
from utils import Stopper, handle_missing_values_kpi, preprocess_nab_labels, preprocess_telemanom_datatset, load_ucr_ts, \
    load_ts_prediction_performance, split_ensemble_stats, analyze_ensemble_stats
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor, KNeighborsClassifier
from sklearn.svm import OneClassSVM
from skopt import gp_minimize
from skopt.callbacks import EarlyStopper
from skopt.space import Integer, Categorical, Real
from statsmodels.tsa._stl import STL
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

from analysis.preanalysis import visualize, series_analysis, periodicity_analysis, full_analyzis, \
    analyse_dataset_catch22, compare_dataset_properties, calculate_dists, dist_between_sets, compare_dataset_distances, \
    compare_feature_samples_from_same_dist, check_low_variance_features, check_dist_sample
from analysis.postanalysis import confusion_visualization, weighted_f_score
from analysis.time_series_feature_analysis import analyse_series_properties
from models.nets import LSTM_autoencoder, Vae, SeqSeq
from models.sr.main import detect_anomaly
from models.statistical_models import SARIMA, ExpSmoothing
from models.mogaal import MOGAAL
from statsmodels.tsa.api import ExponentialSmoothing
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from utils import create_dataset, plot_dbscan
from tslearn.metrics import dtw
from models.sr.spectral_residual import THRESHOLD, MAG_WINDOW, SCORE_WINDOW, DetectMode, SpectralResidual
from alibi_detect.od import SpectralResidual as SR, OutlierVAE
from alibi_detect.od import OutlierProphet, OutlierSeq2Seq
from pycaret.anomaly import *


root_path = os.getcwd()
le = preprocessing.LabelEncoder().fit([-1, 1])
anomaly_window = 60
step = 60
data_test = None


models = {
          # scale, n_clusters = 2
          # 'knn': (KMeans, [], []),
          # don't scale novelty=True
          'mogaal': (MOGAAL, [], [])
          # 'dbscan': (DBSCAN,
          #            [Integer(low=1, high=100, name='eps'), Integer(low=1, high=100, name='min_samples'),
          #            Categorical(['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan',
          #                        'nan_euclidean', dtw], name='metric')],
          #            [1, 2, dtw]),
          # 'lof': (LocalOutlierFactor, [Integer(low=1, high=1000, name='n_neighbors'),
          #                              Real(low=0.001, high=0.5, name="contamination")], [5, 0.1]),
          # 'lof': (LocalOutlierFactor, [Real(low=0.01, high=0.5, name='fraction')], [0.1]),
          # scale gamma='scale'
          # 'ocsvm': (OneClassSVM,
          #           [Real(low=0.001, high=0.999, name='nu'), Categorical(['linear', 'rbf', 'poly'], name='kernel')],
          #           [0.85, 'poly']),
          # no norm
          # 'isolation_forest': (IsolationForest, [Integer(low=1, high=1000, name='n_estimators')], [100]),
          # 'isolation_forest': (IsolationForest, [Real(low=0.01, high=0.99, name='fraction')], [0.1]),
          # 'es': (ExpSmoothing, [Integer(low=10, high=1000, name='sims')], [10]),
          # 'stl': (STL, []),
          # 'lstm': (LSTM_autoencoder, [Real(low=0.0, high=20.0, name='threshold')], [1.5]),
          # 'deepar': (DeepAR, [], []),
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
          #                           Integer(low=5, high=1000, name='SCORE_WINDOW'),
          #                           Integer(low=1, high=100, name='sensitivity')],
          #        [THRESHOLD, MAG_WINDOW, SCORE_WINDOW, 99]),
          # 'seasonal_decomp': (DecomposeResidual, [], []),
          # 'sarima': (SARIMA, [Real(low=0.5, high=5.0, name="conf_top"), Real(low=0.5, high=5.0, name="conf_botton")],
          #            [1.2, 1.2]),
          # 'ensemble': (Ensemble, [], []),
          # 'vae': (OutlierVAE, [Real(low=0.01, high=0.99, name='threshold'),
          #                      Integer(low=2, high=anomaly_window, name='latent_dim'),
          #                      Integer(low=1, high=100, name='samples'),
          #                      Real(low=0.5, high=0.999, name='percent_anom')],
          #         [0.9, 100, 10, 0.95])
          }


def fit_base_model(model_params, for_optimization=True):
    percent_anomaly = 0.95
    # 50% train-test split
    if dataset == 'kpi':
        data_train = data
        if not for_optimization:
            global data_test
    else:
        data_train, data_test = np.array_split(data, 2)
    model = None

    if for_optimization:
        data_test = copy.deepcopy(data_train)

    if for_optimization:
        data_in_memory = pd.DataFrame([])
    else:
        data_in_memory = copy.deepcopy(data_train)

    # standardize
    if name not in ['sarima', 'es', 'isolation_forest', 'lof']:
        scaler = preprocessing.StandardScaler().fit(data_train[['value']])
        data_train['value'] = scaler.transform(data_train[['value']])
        data_test['value'] = scaler.transform(data_test[['value']])

    # create models with hyper-parameters
    if name == 'knn':
        model = KMeans(n_clusters=2)
    # elif name == 'lof':
    #     model = LocalOutlierFactor(n_neighbors=model_params[0], contamination=model_params[1])
    elif name == 'ocsvm':
        model = OneClassSVM(gamma='scale', nu=model_params[0], kernel=model_params[1])
    elif name == 'dbscan':
        model = DBSCAN(eps=model_params[0], min_samples=model_params[1], metric=model_params[2])
    elif name == 'sarima':
        model = SARIMA(model_params[0], model_params[1])
    elif name == 'es':
        model = ExpSmoothing(model_params[0])
    elif name == 'isolation_forest':
        model = IsolationForest(n_estimators=model_params[0])
    elif name == 'lstm':
        model = LSTM_autoencoder([anomaly_window, 1], dataset, type, filename.replace(".csv", ""),
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
                   n_est_points=model_params[3], n_grad_points=model_params[4]) # dont forget dt param!!!!
        percent_anomaly = model_params[5]

    elif name == 'vae':
        vae_net = Vae(model_params[1])
        model = OutlierVAE(threshold=model_params[0], encoder_net=vae_net.encoder_net_vae,
                           decoder_net=vae_net.decoder_net_vae, latent_dim=model_params[1], samples=model_params[2])
        percent_anomaly = model_params[3]
    elif name == 'deepar':
        model = DeepAR()
    elif name == 'seasonal_decomp':
        model = DecomposeResidual()
    elif name == 'ensemble':
        model = Ensemble(filename=filename, anomaly_window=anomaly_window)
    elif name == 'mogaal':
        model = MOGAAL()
        # need anomalies to properly train mogaal?
        if data_train['is_anomaly'].tolist().count(1) == 0:
            return

    if name in ['sarima', 'es']:

        start_time = time.time()
        model.fit(data_train[['timestamp', 'value']], dataset)
        end_time = time.time()

        diff = end_time - start_time
        print(f"Trained model {name} on {filename} for {diff}")
    elif name == 'lstm':
        X, y = create_dataset(data_train[['value']], data_train[['value']], anomaly_window)
        start_time = time.time()
        model.fit(X, y, data_train['timestamp'], data_train['value'][-anomaly_window:])
        end_time = time.time()

        diff = end_time - start_time
        print(f"Trained model {name} on {filename} for {diff}")
    elif name == 'sr':
        # DT param!!!!!!!!!!
        model = SpectralResidual(series=data_train[['value', 'timestamp']], threshold=model_params[0], mag_window=model_params[1],
                                 score_window=model_params[2], sensitivity=model_params[3],
                                 detect_mode=DetectMode.anomaly_only)
        if model.dynamic_threshold:
            model.fit()
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
    elif name == 'seasonal_decomp':
        model.fit(data_train[['timestamp', 'value']], dataset)
    elif name == 'ensemble':
        model.fit(data_train, dataset, type)
    elif name == 'mogaal':
        model.fit(data=data_train, batch_size=anomaly_window)
    elif name == 'deepar':
        X, y = create_dataset(data_train[['value']], data_train[['value']], anomaly_window)
        start_time = time.time()
        model.fit(X)
        end_time = time.time()

        diff = end_time - start_time
        print(f"Trained model {name} on {filename} for {diff}")
    else:
        if name not in ['dbscan', 'isolation_forest', 'lof']:
            try:
                X, y = data_test[['value', 'timestamp']], data_test['is_anomaly']
                start_time = time.time()
                model.fit(X)
                end_time = time.time()
            except:
                X, y = data_test[['value', 'timestamp']], data_test['is_anomaly']
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

    f1 = []
    all_labels = []
    y_pred_total = []
    idx = 0
    stacked_res = data_train[['timestamp', 'value']]

    pred_time = []

    for start in range(0, data_test.shape[0], step):
        try:
            start_time = time.time()
            window = data_test.iloc[start:start + anomaly_window]
            data_in_memory = pd.concat([data_in_memory, window])[-3000:]
            # if deseasonalize:
            #     window_ = copy.deepcopy(data_in_memory)
            #     window_['value'] = window_['value'] + shift
            #     result_mul = seasonal_decompose(window_['value'], model='additive', extrapolate_trend='freq',
            #                                     period=period)
            #     deseasonalized_memory = window_.value.values / result_mul.seasonal
            #     window['value'] = deseasonalized_memory[-anomaly_window:]

            X, y = window['value'], window['is_anomaly']
            if y.tolist():
                if name in ['sarima']:
                    if for_optimization:
                        y_pred = model.predict(window[['timestamp', 'value']], optimization=True)
                    else:
                        y_pred = model.predict(window[['timestamp', 'value']])
                elif name == 'es':
                    y_pred = model.predict(window[['timestamp', 'value']], anomaly_window)
                    stacked_res = pd.concat([stacked_res, window[['value', 'timestamp']]])
                    model.fit(stacked_res, dataset)
                elif name == 'ensemble':
                    y_pred = model.predict(window)
                elif name == 'sr':
                    model.__series__ = data_in_memory[['value', 'timestamp']]
                    if model.dynamic_threshold:
                        try:
                            # update dynamic threshold here
                            y_pred = [1 if x else 0 for x in model.detect_dynamic_threshold(window[['value', 'timestamp']])['isAnomaly'].tolist()]
                        except Exception as e:
                            print(e)
                            y_pred = [0 for _ in range(window.shape[0])]
                    else:
                        try:
                            y_pred = [1 if x else 0 for x in model.detect(anomaly_window)['isAnomaly'].tolist()[-len(y):]]
                        except Exception as e:
                            print(e)
                            y_pred = [0 for _ in range(window.shape[0])]

                elif name == 'lstm':
                    y_pred = model.predict(X.to_numpy().reshape(1, len(window['value'].tolist()), 1), window['timestamp'])
                elif name == 'mogaal':
                    y_pred = model.predict(window)
                    print(y_pred)
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
                elif name in ['lof']:
                    window = data_in_memory[['value', 'timestamp']].set_index('timestamp')
                    s = setup(data_train, session_id=13, silent=True)
                    lof = create_model('lof', fraction=model_params[0])
                    lof = assign_model(lof)
                    y_pred = lof['Anomaly'].tolist()[-len(y):]
                elif name in ['dbscan']:
                    window_ = copy.deepcopy(data_in_memory[['value']])
                    y_pred = model.fit_predict(window_)[-len(y):]
                    core_smpls = np.array([i - window.index.min() for i in model.core_sample_indices_
                                                if window.index.min() <= i <= window.index.max()])
                elif name == 'isolation_forest':
                    window = data_in_memory[['value', 'timestamp']].set_index('timestamp')
                    s = setup(window, session_id=13, silent=True)
                    iforest = create_model('iforest', fraction=model_params[0])
                    iforest_results = assign_model(iforest)
                    y_pred = iforest_results['Anomaly'].tolist()[-len(y):]
                else:
                    y_pred = model.predict(window[['value', 'timestamp']])

                try:
                    y_pred = le.transform(y_pred).tolist()
                except:
                    pass

                idx += 1
                if anomaly_window == step:
                    # Predict the labels (1 inlier, -1 outlier) of X according to LOF.
                    common = Counter(funcy.lflatten(y_pred)[:window.shape[0]]).most_common(1)[0][0]
                    y_pred_total += [0 if val == common else 1 for val in funcy.lflatten(y_pred)[:window.shape[0]]]
                    if name == 'dbscan':
                        all_labels += [(copy.deepcopy(model.labels_[-len(y):]), core_smpls)]
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

                end_time = time.time()
                pred_time.append((end_time - start_time))
        except Exception as e:
            raise e

    met_total = precision_recall_fscore_support(data_test['is_anomaly'], y_pred_total, average='binary')
    # Do f1 score smoothing
    # smoothed_f1 = pd.DataFrame(f1).ewm(com=0.5).mean()

    if name in ['sarima', 'lstm']:
        f = weighted_f_score(data_test['is_anomaly'].tolist(), y_pred_total, model.get_pred_mean(), data_test['value'].tolist())
        print(f"My f-score: {f} vs standard f score {met_total[2]}")

    if not for_optimization:
        # visualize(data)

        try:
            stats = pd.read_csv(f'results/{dataset}_{type}_stats_{name}.csv')
        except:
            stats = pd.DataFrame([])

        try:
            confusion_visualization(data_test['timestamp'].tolist(), data_test['value'].tolist(),
                                    data_test['is_anomaly'].tolist(), y_pred_total,
                                    dataset, name, filename.replace('.csv', ''), type)
        except Exception as e:
            print(e)

        try:
            if name in ['sarima', 'es']:
                model.plot(data_test[['timestamp', 'value']], dataset, type, filename, data_test)
            elif name in ['lstm']:
                model.plot(data_test['timestamp'].tolist(), dataset, type, filename, data_test)
            # elif name == 'ensemble':
            #     model.plot(dataset)
            elif name == 'sr':
                model.plot(dataset, type, filename, data_test)
                # if model.dynamic_threshold:
                #     model.plot_dynamic_threshold(data_test['timestamp'].tolist(), dataset, type, filename, data_test)
            elif name == 'dbscan':
                plot_dbscan(all_labels, dataset, type, filename, data_test[['value', 'timestamp']], anomaly_window)
            # elif name == 'seasonal_decomp':
            #     model.plot(type, filename, )
        except Exception as e:
            print(e)

        try:
            tn, fp, fn, tp = confusion_matrix(data_test['is_anomaly'], y_pred_total).ravel()
            specificity = tn / (tn + fp)
        except:
            specificity = None
            tn, fp, fn, tp = None, None, None, None

        # try:
        #     trend, seasonality, autocrr, non_lin, skewness, kurtosis, hurst, lyapunov, tf = \
        #         series_analysis(data)
        # except:
        #         try:
        #             trend, seasonality, autocrr, non_lin, skewness, kurtosis, hurst, lyapunov, tf = \
        #                 series_analysis(data[:25000])
        #         except:
        #             trend, seasonality, autocrr, non_lin, skewness, kurtosis, hurst, lyapunov, tf = \
        #                 None, None, None, None, None, None, None, None, pd.DataFrame([])

        stats = stats.append({
            'model': name,
            'dataset': filename.replace('.csv', ''),
            # 'trend': np.mean(data['trend'].tolist()) if 'trend' in data.columns else trend,
            # 'seasonality': np.mean(data['seasonality1'].tolist()) if 'seasonality1' in data.columns else seasonality,
            # 'autocorrelation': autocrr,
            # 'non-linearity': non_lin,
            # 'skewness': skewness,
            # 'kurtosis': kurtosis,
            # 'hurst': hurst,
            # 'max_lyapunov_e': lyapunov,
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
        # tf.to_csv(f'results/{dataset}_{type}_{filename.replace(".csv", "")}_full_stats.csv', index=False)

    return 1.0 - met_total[2]


if __name__ == '__main__':
    for dataset, type in [('yahoo', 'real')]:
                          # ('yahoo', 'real'),
                          # ('kpi', 'train'), ('yahoo', 'A4Benchmark'),
                          # ('yahoo', 'A3Benchmark'),
                          # ('NAB', 'relevant'),
                          # ('yahoo', 'synthetic')
        for name, (model, bo_space, def_params) in models.items():
            train_data_path = root_path + '/datasets/' + dataset + '/' + type + '/'
            for filename in os.listdir(train_data_path):
                f = os.path.join(train_data_path, filename)
                res_data_path = root_path + f'/results/imgs/{dataset}/{type}/{name}'
                if os.path.isfile(f):# and f'{name}_{filename.split(".")[0]}.png' in os.listdir(res_data_path):
                    print(f"Training model {name} with data {filename}")
                    data = pd.read_csv(f)
                    data.rename(columns={'timestamps': 'timestamp', 'anomaly': 'is_anomaly'}, inplace=True)

                    if dataset == 'kpi':
                        data_test = pd.read_csv(os.path.join(root_path + '/datasets/' + dataset + '/' + 'test' + '/', filename))

                    # try:
                    #     fit_base_model(def_params, for_optimization=False)
                    # except Exception as e:
                    #     raise e
                    #     print(f'Error on {filename}')
                    # continue

                    try:
                        if name not in ['knn', 'sarima', 'mogaal']:
                        ################ Bayesian optimization ###################################################
                            bo_result = gp_minimize(fit_base_model, bo_space, callback=Stopper(), n_calls=11,
                                                    random_state=13, verbose=False, x0=def_params)

                            print(f"Found hyper parameters for {name}: {bo_result.x}")

                            fit_base_model(bo_result.x, for_optimization=False)
                        else:
                            fit_base_model(def_params, for_optimization=False)
                    except Exception as e:
                        print(f'Error on {filename}')
