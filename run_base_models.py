import ast
import copy
import os
import ssl
from urllib.request import urlopen

import mock
from collections import Counter

import funcy
from catch22 import catch22_all
from requests import Request
from sklearn import preprocessing
from skopt.callbacks import EarlyStopper

from analysis.ts_analysis import complexity_entropy_analysis, performance
from models.decompose_model import DecomposeResidual
from models.entropy_model import entropy_modelling

from utils import drift_metrics, plot_general, plot_change, score_per_anomaly_group, analyze_anomalies, avg_batch_f1, \
    loss_behavior, ks_stat, preprocess_nab_labels

from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, mean_absolute_error, hamming_loss, f1_score, cohen_kappa_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from skopt import gp_minimize

from analysis.postanalysis import confusion_visualization
from models.nets import Vae, SeqSeq
from models.statistical_models import SARIMA, ExpSmoothing
from models.mogaal.mogaal import MOGAAL
from sklearn.metrics import precision_recall_fscore_support

from utils import create_dataset
from models.sr.spectral_residual import DetectMode
from alibi_detect.od import OutlierVAE
from alibi_detect.od import OutlierSeq2Seq
from pycaret.anomaly import *
from model_collection import *
from settings import *


root_path = os.getcwd()
le = preprocessing.LabelEncoder().fit([-1, 1])
data_test = None
data_in_memory_sz = 3000
can_model = True
traintime = 0
predtime = 0
bwa = []
batched_f1_score = []
batch_metrices = []


class Stopper(EarlyStopper):
    def __call__(self, result):
        ret = False
        if result.func_vals[-1] == 0.0 or not can_model:
            ret = True
        return ret


def fit_base_model(model_params, for_optimization=True):
    global data_in_memory_sz, can_model, traintime, predtime, batched_f1_score, batch_metrices, bwa
    percent_anomaly = 0.95
    is_normal_entropy = True
    # 50% train-test split
    if dataset == 'kpi':
        data_train = data
        if not for_optimization:
            global data_test
    else:
        data_train, data_test = np.array_split(copy.deepcopy(data), 2)
    model = None

    if for_optimization:
        data_test = copy.deepcopy(data_train)

    if for_optimization:
        data_in_memory = pd.DataFrame([])
    else:
        data_in_memory = copy.deepcopy(data_train)

    # standardize
    # if name not in ['sarima', 'es', 'isolation_forest', 'lof']:
    #     scaler = preprocessing.StandardScaler().fit(data_train[['value']])
    #     data_train['value'] = scaler.transform(data_train[['value']])
    #     data_test['value'] = scaler.transform(data_test[['value']])


    start = time.time()
    # create models with hyper-parameters
    if name == 'lof':
        model = LocalOutlierFactor(n_neighbors=model_params[0], contamination=model_params[1])
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

    elif name == 'vae':
        vae_net = Vae(model_params[1])
        model = OutlierVAE(threshold=model_params[0], encoder_net=vae_net.encoder_net_vae,
                           decoder_net=vae_net.decoder_net_vae, latent_dim=model_params[1], samples=model_params[2])
        percent_anomaly = model_params[3]
    elif name == 'naive':
        model = NaiveDetector(model_params[0], model_params[1], model_params[2], model_params[3], model_params[4])
    elif name == 'mogaal':
        model = MOGAAL(dataset=(dataset, type, filename))
        # need anomalies to properly train mogaal?
        if data_train['is_anomaly'].tolist().count(1) == 0:
            return

    if name == 'mmd':
        model = MMDDrift(data_train[['value']].to_numpy(), backend='tensorflow', data_type='time-series',
                         p_val=0.05)
        data_in_memory_sz = model_params[0]
    elif name == 'adwin':
        model = ADWIN(delta=model_params[0])
    elif name == 'ddm':
        model = DDM(out_control_level=model_params[0])
    elif name == 'eddm':
        model = EDDM()
    elif name == 'hddma':
        model = HDDM_A(drift_confidence=model_params[0])
    elif name == 'hddmw':
        model = HDDM_W(drift_confidence=model_params[0], lambda_option=model_params[1])
    elif name == 'kswin':
        model = KSWIN(alpha=model_params[0], window_size=model_params[1])
    elif name == 'ph':
        model = PageHinkley(delta=model_params[0], threshold=model_params[1])
    # drift_model = MMDDrift(data_train[['value']].to_numpy(), backend='tensorflow', data_type='time-series',
    #                        p_val=0.05)

    if name in ['sarima', 'es']:

        start_time = time.time()
        model.fit(copy.deepcopy(data_train[['timestamp', 'value']]), dataset)
        end_time = time.time()

        diff = end_time - start_time
        print(f"Trained model {name} on {filename} for {diff}")

    elif name == 'lstm':
        X, y = create_dataset(data_train[['value']], data_train[['value']], anomaly_window)
        start_time = time.time()
        model.fit(X, y, data_train['timestamp'], data_train['value'].tolist())
        end_time = time.time()

        diff = end_time - start_time
        print(f"Trained model {name} on {filename} for {diff}")
    elif name == 'dp':
        model.fit(copy.deepcopy(data_train[['timestamp', 'value']]))

        # decide if can model via sarima
        if model.anomaly_detector.model.fittedvalues.tolist().count(0.0) == \
                len(model.anomaly_detector.model.fittedvalues.tolist()):
            can_model = False
    elif name == 'sr':
        # DT param!!!!!!!!!!
        # model = SpectralResidual(series=data_train[['value', 'timestamp']], threshold=model_params[0], mag_window=model_params[1],
        #                          score_window=model_params[2], sensitivity=model_params[3],
        #                          detect_mode=DetectMode.anomaly_only)
        model = SpectralResidual(series=data_train[['value', 'timestamp']], threshold=model_params[0],
                                 mag_window=MAG_WINDOW,
                                 score_window=SCORE_WINDOW, sensitivity=99,
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
        if name not in ['dbscan', 'isolation_forest', 'lof', 'mmd', 'lsdd', 'mmd-online',
                        'adwin', 'ddm', 'eddm', 'hddma', 'hddmw', 'kswin', 'ph', 'naive']:
            try:
                X, y = data_train[['value', 'timestamp']], data_test['is_anomaly']
                start_time = time.time()
                model.fit(X)
                end_time = time.time()
            except:
                X, y = data_train[['value', 'timestamp']], data_test['is_anomaly']
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

    end = time.time()
    traintime = end - start

    batch_metrices_kc = []
    batch_metrices_kc_entropy = []
    batch_metrices_hamming = []
    batch_metrices_hamming_entropy = []
    all_labels = []
    y_pred_total = []
    y_pred_total_noe, y_pred_total_e = [], []
    batches_with_anomalies = []
    idx = 0
    stacked_res = data_train[['timestamp', 'value']]

    pred_time = []
    drift_windows = []

    batches_predicted = []
    batches_real = []

    for start in range(0, data_test.shape[0], step):
        try:
            start_time = time.time()
            window = data_test.iloc[start:start + anomaly_window]
            data_in_memory = pd.concat([data_in_memory, window])[-data_in_memory_sz:]

            X, y = window['value'], window['is_anomaly']
            if y.tolist():
                # then apply anomaly detectors #########################################################################
                if name in ['sarima']:
                    if for_optimization:
                        y_pred, is_normal_entropy = model.predict(window[['timestamp', 'value']], optimization=True)
                    else:
                        y_pred, is_normal_entropy = model.predict(window[['timestamp', 'value']])

                elif name == 'es':
                    y_pred = model.predict(window[['timestamp', 'value']], anomaly_window)
                    stacked_res = pd.concat([stacked_res, window[['value', 'timestamp']]])
                    model.fit(stacked_res, dataset)
                elif name == 'ensemble':
                    y_pred = model.predict(window)
                elif name == 'naive':
                    # enough data in window
                    if data_in_memory.shape[0] <= model.k:
                        y_pred = [0]
                    else:
                        y_pred = [model.predict(data_in_memory[['value']])]
                elif name == 'sr':
                    model.__series__ = data_in_memory[['value', 'timestamp']]
                    if model.dynamic_threshold:
                        try:
                            res = model.detect_dynamic_threshold(window[['value', 'timestamp']])
                            # update dynamic threshold here
                            y_pred_noe = [1 if x else 0 for x in res['isAnomaly'].tolist()]

                            y_pred_e = [1 if x else 0 for x in res['isAnomaly_e'].tolist()]
                        except Exception as e:
                            y_pred_noe = [0 for _ in range(window.shape[0])]
                            y_pred_e = [0 for _ in range(window.shape[0])]
                    else:
                        try:
                            y_pred = [1 if x else 0 for x in model.detect(anomaly_window)
                                                             ['isAnomaly'].tolist()[-len(y):]]
                        except Exception as e:
                            y_pred = [0 for _ in range(window.shape[0])]

                elif name == 'lstm':
                    y_pred_noe, y_pred_e = model.predict(X.to_numpy().reshape(1, len(window['value'].tolist()), 1),
                                           window['timestamp'])
                    y_pred_noe, y_pred_e = y_pred_noe[:window.shape[0]], y_pred_e[:window.shape[0]]
                elif name in ['dbscan']:
                    window_ = copy.deepcopy(data_in_memory[['value']])
                    y_pred = model.fit_predict(window_)[-len(y):]
                    core_smpls = np.array([i - window.index.min() for i in model.core_sample_indices_
                                                if window.index.min() <= i <= window.index.max()])
                elif name in ['adwin', 'ddm', 'eddm', 'hddma', 'hddmw', 'kswin', 'ph']:
                    y_pred = []
                    for idx, row in window.iterrows():
                        model.add_element(row['value'])
                        if model.detected_warning_zone():
                            pass
                        if model.detected_change():
                            model.reset()
                            try:
                                drift_windows.append((datetime.datetime.strptime(row['timestamp'], '%m/%d/%Y %H:%M'),
                                                      datetime.datetime.strptime(row['timestamp'], '%m/%d/%Y %H:%M')))
                            except:
                                try:
                                    drift_windows.append((datetime.datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S'),
                                                          datetime.datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S')))
                                except:
                                    drift_windows.append((row['timestamp'], row['timestamp']))
                            y_pred.append(1)
                        else:
                            y_pred.append(0)
                else:
                    y_pred = model.predict(window[['value', 'timestamp']])

                try:
                    y_pred = le.transform(y_pred).tolist()
                except:
                    pass

                idx += 1
                y_pred_total_noe += [0 if val != 1 else 1 for val in funcy.lflatten(y_pred_noe)][:window.shape[0]]
                y_pred_total_e += [0 if val != 1 else 1 for val in funcy.lflatten(y_pred_e)][:window.shape[0]]
                y_pred_noe = y_pred_noe[:window.shape[0]]
                y_pred_e = y_pred_e[:window.shape[0]]

                batch_metrices_hamming.append(1.0 - hamming_loss(window['is_anomaly'].tolist(), y_pred_noe))
                batch_metrices_hamming_entropy.append(1.0 - hamming_loss(window['is_anomaly'].tolist(), y_pred_e))
                batch_metrices_kc.append(cohen_kappa_score(window['is_anomaly'].tolist(), y_pred_noe))
                batch_metrices_kc_entropy.append(cohen_kappa_score(window['is_anomaly'].tolist(), y_pred_e))

                end_time = time.time()
                pred_time.append((end_time - start_time))
        except Exception as e:
            raise e

    # if not for_optimization:
    #     plot_general(model, dataset, type, name, data_test, y_pred_total, filename, drift_windows)

    print('saving results')
    predtime = np.mean(pred_time)
    bwa = batches_with_anomalies

    if not test_drifts:
        met_total_noe = precision_recall_fscore_support(data_test['is_anomaly'], y_pred_total_noe[:data_test.shape[0]])
        met_total_e = precision_recall_fscore_support(data_test['is_anomaly'], y_pred_total_e[:data_test.shape[0]])

        if not for_optimization:
            try:
                stats = pd.read_csv(f'results/{dataset}_{type}_stats_{name}_per_batch_{anomaly_window}_entropy_batched.csv')
            except:
                stats = pd.DataFrame([])

            stats = stats.append({
                # 'model': name,
                # 'dataset': filename.replace('.csv', ''),
                'f1_entropy': met_total_e[2][-1],
                'precision_entropy': met_total_e[0][-1],
                'recall_entropy': met_total_e[1][-1],
                'f1': met_total_noe[2][-1],
                'precision': met_total_noe[0][-1],
                'recall': met_total_noe[1][-1],
                'cohen_kappa': batch_metrices_kc,
                'cohen_kappa_entropy': batch_metrices_kc_entropy,
                'hamming': batch_metrices_hamming,
                'hamming_entropy': batch_metrices_hamming,
                'ts': filename.replace('.csv', ''),
                'step': anomaly_window,
                'model': name,
                'total_anomalies': batches_real.count(1)
            }, ignore_index=True)
            stats.to_csv(f'results/{dataset}_{type}_stats_{name}_per_batch_{anomaly_window}_entropy_batched.csv', index=False)
            # plot_change(batch_metrices, batches_with_anomalies, name, filename.replace('.csv', ''), dataset)
        # return specificity if no anomalies, else return f1 score
        return
        if data_test['is_anomaly'].tolist().count(1) == 0:
            return 1.0 - specificity
        else:
            return 1.0 - met_total[2]

    else:
        fp, latency, misses = drift_metrics(data_test['is_anomaly'].tolist(), y_pred_total)
        if not for_optimization:
            try:
                stats = pd.read_csv(f'results/{dataset}_{type}_stats_{name}.csv')
            except:
                stats = pd.DataFrame([])

            stats = stats.append({
                'model': name,
                'dataset': filename.replace('.csv', ''),
                'fp': fp,
                'avg_latency': latency,
                'misses': misses,
                'total_anomalies': data_test['is_anomaly'].tolist().count(1),
                'prediction_time': np.mean(pred_time),
                'total_points': data_test.shape[0]
            }, ignore_index=True)
            stats.to_csv(f'results/{dataset}_{type}_stats_{name}.csv', index=False)
        return latency + fp + misses


if __name__ == '__main__':
    # preprocess_nab_labels(root_path)
    # loss_behavior()
    avg_batch_f1()
    # entropy_modelling()
    quit()
    if test_drifts:
        learners_to_loop = drift_detectors
    else:
        learners_to_loop = models

    try:
        hp = pd.read_csv('hyperparams.csv')
    except:
        hp = pd.DataFrame([])

    continuer = True
    for dataset, type in [('kpi', 'train')]:
                          # ('kpi', 'train'), ('kpi', 'test')]:
                          # ('yahoo', 'real'),
                          # ('kpi', 'train'), ('yahoo', 'A4Benchmark'),
                          # ('yahoo', 'A3Benchmark'),
                          # ('NAB', 'relevant'),
                          # ('yahoo', 'synthetic')
        for name, (model, bo_space, def_params) in learners_to_loop.items():
            train_data_path = root_path + '/datasets/' + dataset + '/' + type + '/'

            try:
                statf = pd.read_csv(f'results/{dataset}_{type}_stats_{name}_per_batch_{anomaly_window}_entropy_batched.csv')
            except:
                statf = pd.DataFrame([])

            for filename in os.listdir(train_data_path):
                print(filename)
                f = os.path.join(train_data_path, filename)
                res_data_path = root_path + f'/results/imgs/{dataset}/{type}/{name}'
                if os.path.isfile(f) and filename: #\
                        # and (statf.shape[0] == 0 or filename.replace('.csv', '') not in statf['ts'].tolist()):
                    # and f'{name}_{filename.split(".")[0]}.png' in os.listdir(res_data_path):
                    data = pd.read_csv(f)
                    data.rename(columns={'timestamps': 'timestamp', 'anomaly': 'is_anomaly'}, inplace=True)
                    data.drop_duplicates(subset=['timestamp'], keep=False, inplace=True)
                    data.dropna(inplace=True)
                    data['timestamp'] = data['timestamp'].apply(lambda x: datetime.datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))

                    print(f"Training model {name} with data {filename}")
                    print(data.head())
                    continue

                    if dataset == 'kpi':
                        import matplotlib.pyplot as plt
                        data_test = pd.read_csv(os.path.join(root_path + '/datasets/' + dataset + '/test/', filename))
                        data = data

                    # if dataset != 'kpi' and np.array_split(copy.deepcopy(data), 2)[-1]['is_anomaly'].tolist().count(1) == 0:
                    #     continue
                    # elif dataset == 'kpi' and data_test['is_anomaly'].tolist().count(1) == 0:
                    #     continue

                    can_model = True
                    # try:
                    #     fit_base_model(def_params, for_optimization=False)
                    # except Exception as e:
                    #     raise e

                    # continue
                    # teams anomaly detection file

                    try:
                        if def_params and name not in ['knn', 'mogaal', 'mmd-online']:
                        ################ Bayesian optimization ###################################################
                            if hp[(hp['filename'] == filename.replace('.csv', '')) & (hp['model'] == name)].empty:

                                try:
                                    bo_result = gp_minimize(fit_base_model, bo_space, callback=Stopper(), n_calls=11,
                                                            random_state=13, verbose=False, x0=def_params)
                                except ValueError as e:
                                    # error is rised when function yields constant value and does not converge
                                    print(e)
                                    bo_result = mock.Mock()
                                    bo_result.x = def_params

                                print(f"Found hyper parameters for {name}: {bo_result.x}")

                                if hp.empty or filename.replace('.csv', '') not in hp[hp['model'] == name]['filename'].tolist():
                                    hp = hp.append({
                                        'filename': filename.replace('.csv', ''),
                                        'model': name,
                                        'hp': bo_result.x
                                    }, ignore_index=True)

                                    hp.to_csv('hyperparams.csv', index=False)
                            else:
                                bo_result = mock.Mock()
                                bo_result.x = ast.literal_eval(hp[(hp['filename'] == filename.replace('.csv', '')) & (hp['model'] == name)]['hp'].tolist()[0])

                            if can_model:
                                fit_base_model(bo_result.x, for_optimization=False)
                        else:
                            fit_base_model(def_params, for_optimization=False)

                    except Exception as e:
                        print('Error:', e)
                        raise e
                        # try:
                        #     fit_base_model(def_params, for_optimization=False)
                        # except:
                        #     pass
