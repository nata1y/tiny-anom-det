import ast
import copy
import os
import mock
from collections import Counter

import funcy
from catch22 import catch22_all
from sklearn import preprocessing
from skopt.callbacks import EarlyStopper

from analysis.ts_analysis import complexity_entropy_analysis, performance
from models.decompose_model import DecomposeResidual
from models.entropy_model import entropy_modelling
from utils import drift_metrics, plot_general, plot_change, score_per_anomaly_group, analyze_anomalies, avg_batch_f1

from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, mean_absolute_error, hamming_loss, f1_score
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


class Stopper(EarlyStopper):
    def __call__(self, result):
        ret = False
        if result.func_vals[-1] == 0.0 or not can_model:
            ret = True
        return ret


def fit_base_model(model_params, for_optimization=True):
    global data_in_memory_sz, can_model
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
        model.fit(X, y, data_train['timestamp'], data_train['value'][-anomaly_window:])
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

    batch_metrices = []
    batched_f1 = []
    all_labels = []
    y_pred_total = []
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
            # if deseasonalize:
            #     window_ = copy.deepcopy(data_in_memory)
            #     window_['value'] = window_['value'] + shift
            #     result_mul = seasonal_decompose(window_['value'], model='additive', extrapolate_trend='freq',
            #                                     period=period)
            #     deseasonalized_memory = window_.value.values / result_mul.seasonal
            #     window['value'] = deseasonalized_memory[-anomaly_window:]

            X, y = window['value'], window['is_anomaly']
            if y.tolist():
                # check for drifts #####################################################################################
                # if not for_optimization:
                #     drift_pred = drift_model.predict(window[['value']].to_numpy())
                #     if drift_pred['data']['is_drift'] == 1:
                #         # drift_windows += window['timestamp'].tolist()
                #         drift_windows.append((window['timestamp'].min(), window['timestamp'].max()))

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
                            # update dynamic threshold here
                            y_pred = [1 if x else 0 for x in model.detect_dynamic_threshold(
                                window[['value', 'timestamp']])['isAnomaly'].tolist()]
                        except Exception as e:
                            print(e)
                            y_pred = [0 for _ in range(window.shape[0])]
                    else:
                        try:
                            y_pred = [1 if x else 0 for x in model.detect(anomaly_window)
                                                             ['isAnomaly'].tolist()[-len(y):]]
                        except Exception as e:
                            print(e)
                            y_pred = [0 for _ in range(window.shape[0])]

                elif name == 'lstm':
                    y_pred = model.predict(X.to_numpy().reshape(1, len(window['value'].tolist()), 1),
                                           window['timestamp'])
                elif name == 'dp':
                    if for_optimization:
                        y_pred = model.predict(window[['timestamp', 'value']], optimization=True)
                    else:
                        y_pred = model.predict(window[['timestamp', 'value']])

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
                elif name in ['lsdd', 'mmd', 'mmd-online']:
                    drift_pred = model.predict(window[['value']].astype('float32').to_numpy())
                    if drift_pred['data']['is_drift'] == 1 and not for_optimization:
                        # drift_windows += window['timestamp'].tolist()
                        time_window = [datetime.datetime.strptime(x, '%m/%d/%Y %H:%M')
                                       for x in window['timestamp'].tolist()]
                        wd = (min(time_window), max(time_window))
                        drift_windows.append(wd)

                    if drift_pred['data']['is_drift'] == 1 and y.tolist().count(1) > 0:
                        y_pred = y.tolist()
                    elif drift_pred['data']['is_drift'] == 1 and y.tolist().count(1) == 0:
                        y_pred = [1 for _ in range(len(y.tolist()))]
                    elif drift_pred['data']['is_drift'] == 0 and y.tolist().count(1) == 0:
                        y_pred = [0 for _ in range(len(y.tolist()))]
                    elif drift_pred['data']['is_drift'] == 0 and y.tolist().count(1) > 0:
                        y_pred = [0 for _ in range(len(y.tolist()))]
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

                err = hamming_loss(window['is_anomaly'], y_pred)
                batch_metrices.append(err)
                met = precision_recall_fscore_support(window['is_anomaly'], y_pred, average='binary')
                batched_f1.append(met[2])
                if window['is_anomaly'].tolist().count(1) > 0:
                    batches_with_anomalies.append(start // step)

                # integrate entropy model into anomaly identification
                if y_pred.count(1) > 2: # and not is_normal_entropy:
                    batches_predicted.append(1)
                else:
                    batches_predicted.append(0)

                if window['is_anomaly'].tolist().count(1) > 0:
                    batches_real.append(1)
                else:
                    batches_real.append(0)

                idx += 1
                if name not in ['lsdd', 'mmd', 'mmd-online', 'adwin', 'ddm', 'eddm', 'hddma', 'hddmw', 'kswin', 'ph',
                                'dp']:
                    if anomaly_window == step:
                        # Predict the labels (1 inlier, -1 outlier) of X according to LOF.
                        common = Counter(funcy.lflatten(y_pred)[:window.shape[0]]).most_common(1)[0][0]
                        before = len(y_pred_total)
                        y_pred_total += [0 if val == common else 1 for val in funcy.lflatten(y_pred)[:window.shape[0]]]
                        diff = len(y_pred_total) - before
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
                else:
                    if anomaly_window == step or name == 'dp':
                        y_pred_total += [val for val in funcy.lflatten(y_pred)[:window.shape[0]]]
                    else:
                        anomaly_window_ = data_in_memory
                        step_ = anomaly_window
                        if y_pred_total:
                            if len(y_pred) == anomaly_window_:
                                y_pred_total = y_pred_total[:-anomaly_window_ + step_] + \
                                               [0 if (o != 1 and n != 1) else 1 for o, n in
                                                zip(y_pred_total[-anomaly_window_ + step_:] + [0 for _ in range(step_)],
                                                    funcy.lflatten(y_pred))]
                            else:
                                y_pred_total = y_pred_total[:-anomaly_window_ + step_] + \
                                               [0 if (o != 1 and n != 1) else 1 for o, n in
                                                zip(y_pred_total[-anomaly_window_ + step_:] +
                                                    [0 for _ in range(step_ - (anomaly_window_ - len(y_pred)))],
                                                    funcy.lflatten(y_pred))]
                                break
                        else:
                            y_pred_total = [0 if val != 1 else 1 for val in funcy.lflatten(y_pred)]

                end_time = time.time()
                pred_time.append((end_time - start_time))
        except Exception as e:
            raise e

    # if not for_optimization:
    #     plot_general(model, dataset, type, name, data_test, y_pred_total, filename, drift_windows)

    print('saving results')
    if not test_drifts:
        met_total = precision_recall_fscore_support(data_test['is_anomaly'], y_pred_total, average='binary')

        try:
            tn, fp, fn, tp = confusion_matrix(data_test['is_anomaly'], y_pred_total).ravel()
            specificity = tn / (tn + fp)
        except:
            specificity = y_pred_total.count(0) / data_test['is_anomaly'].tolist().count(0)
            tn, fp, fn, tp = None, None, None, None

        if not for_optimization:
            try:
                stats = pd.read_csv(f'results/{dataset}_{type}_stats_{name}_per_batch_2threshold.csv')
            except:
                stats = pd.DataFrame([])

            try:
                btn, bfp, bfn, btp = confusion_matrix(batches_real, batches_predicted).ravel()
            except:
                btn, bfp, bfn, btp = None, None, None, None

            stats = stats.append({
                # 'model': name,
                # 'dataset': filename.replace('.csv', ''),
                # 'f1': met_total[2],
                # 'precision': met_total[0],
                # 'recall': met_total[1],
                # 'fp': fp,
                # 'fn': fn,
                # 'tp': tp,
                # 'tn': tn,
                # 'specificity': specificity,
                # 'batch_metrics': batch_metrices,
                # 'batched_f1_score': batched_f1,
                # 'anomaly_idxs': batches_with_anomalies,
                # 'total_anomalies': data_test['is_anomaly'].tolist().count(1),
                # 'prediction_time': np.mean(pred_time),
                # 'total_points': data_test.shape[0]
                'ts': filename.replace('.csv', ''),
                'y_pred': batches_predicted,
                'y_true': batches_real,
                'f1': f1_score(batches_real, batches_predicted, average='binary'),
                'hamming': hamming_loss(batches_real, batches_predicted),
                'step': 100,
                'fp': bfp,
                'fn': bfn,
                'tp': btp,
                'tn': btn,
                'per_item_f1': met_total[2],
                'per_item_fp': fp,
                'per_item_fn': fn,
                'per_item_tp': tp,
                'per_item_tn': tn,
                'model': name,
                'total_anomalies': batches_real.count(1)
            }, ignore_index=True)
            stats.to_csv(f'results/{dataset}_{type}_stats_{name}_per_batch_2threshold.csv', index=False)
            # plot_change(batch_metrices, batches_with_anomalies, name, filename.replace('.csv', ''), dataset)
        # return specificity if no anomalies, else return f1 score
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
    # entropy_modelling()
    # quit()
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
            # if dataset == 'NAB':
            #     train_data_path = root_path + '/datasets/machine_metrics/nab_drift/'
            # else:
            #     train_data_path = root_path + '/datasets/' + dataset + '/nab_drift/' + type + '/'
            try:
                statf = pd.read_csv(f'results/{dataset}_{type}_stats_{name}_per_batch_2threshold.csv')
            except:
                statf = pd.DataFrame([])

            for filename in os.listdir(train_data_path):
                print(filename)
                # if '43115f2a' in filename or '43115f2a' in filename and '9c639a46' not in filename: # and continuer:
                #     # continuer = False
                #     continue
                f = os.path.join(train_data_path, filename)
                res_data_path = root_path + f'/results/imgs/{dataset}/{type}/{name}'
                # machine_data_path = f'datasets/machine_metrics/relevant/'
                if os.path.isfile(f) and filename\
                        and (statf.shape[0] == 0 or filename.replace('.csv', '') not in statf['ts'].tolist()):
                    # and f'{name}_{filename.split(".")[0]}.png' in os.listdir(res_data_path):
                    data = pd.read_csv(f)
                    data.rename(columns={'timestamps': 'timestamp', 'anomaly': 'is_anomaly'}, inplace=True)
                    data.drop_duplicates(subset=['timestamp'], keep=False, inplace=True)
                    data.dropna(inplace=True)

                    print(f"Training model {name} with data {filename}")

                    if dataset == 'kpi':
                        import matplotlib.pyplot as plt
                        data_test = pd.read_csv(os.path.join(root_path + '/datasets/' + dataset + '/test/', filename))
                        data = data[-3000:]

                    if dataset != 'kpi' and np.array_split(copy.deepcopy(data), 2)[-1]['is_anomaly'].tolist().count(1) == 0:
                        continue
                    elif dataset == 'kpi' and data_test['is_anomaly'].tolist().count(1) == 0:
                        continue

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
                        raise e
                        print('Error:', e)
                        # try:
                        #     fit_base_model(def_params, for_optimization=False)
                        # except:
                        #     pass
