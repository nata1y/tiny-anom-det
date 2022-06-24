import ast
import copy
import os

import funcy
import mock
from skopt.callbacks import EarlyStopper

from analysis.postanalysis import test_statistics
from utils import plot_general
from sklearn.metrics import hamming_loss, cohen_kappa_score
from skopt import gp_minimize

from sklearn.metrics import precision_recall_fscore_support

from models.sr.spectral_residual import DetectMode
import pandas as pd
import datetime
import time
import numpy as np
from model_collection import *
from settings import entropy_params, data_in_memory_sz, no_optimization
from drift_detectors.drift_detector_wrapper import DriftDetectorWrapper


root_path = os.getcwd()


class Stopper(EarlyStopper):
    def __call__(self, result):
        ret = False
        if result.func_vals[-1] == 0.0:
            ret = True
        return ret


def fit_base_model(model_params, for_optimization=True):
    global can_model, traintime, predtime, batched_f1_score, batch_metrices, bwa, dataset, dd

    # 50% fit-test split
    if dataset == 'kpi':
        data_train = copy.deepcopy(data)
        if not for_optimization:
            global data_test_
            data_test = copy.deepcopy(data_test_)
    else:
        data_train, data_test = np.array_split(copy.deepcopy(data), 2)
    model = None
    if for_optimization:
        data_test = copy.deepcopy(data_train)

    if for_optimization:
        data_in_memory = pd.DataFrame([])
    else:
        data_in_memory = copy.deepcopy(data_train)

    start = time.time()
    drift_detector = dd
    # create models with hyper-parameters
    if name == 'sarima':
        model = SARIMA(dataset, type, filename, drift_detector, use_drift_adapt, model_params[0], model_params[1])
    elif name == 'lstm':
        model = LSTM_autoencoder([anomaly_window, 1], dataset, type, filename, drift_detector, use_drift_adapt,
                                 magnitude=model_params[0])
    elif name == 'pso-elm':
        # take pso-elm fit size as variable
        n = min(model_params[0], data_train.shape[0])
        model = PSO_ELM_anomaly(dataset, type, filename,  drift_detector, use_drift_adapt,
                                n=n,
                                magnitude=model_params[1],
                                entropy_window=anomaly_window,
                                error_threshold=model_params[2])

    if name == 'sarima':
        start_time = time.time()
        model.fit(data_train)
        end_time = time.time()

        diff = end_time - start_time
        print(f"Trained model {name} on {filename} for {diff}")

    elif name == 'lstm':
        start_time = time.time()
        model.fit(data_train)
        end_time = time.time()

        diff = end_time - start_time
        print(f"Trained model {name} on {filename} for {diff}")
    elif name == 'sr':
        start_time = time.time()
        model = SpectralResidual(series=data_train[['value', 'timestamp']], use_drift=use_drift_adapt,
                                 threshold=model_params[0], mag_window=model_params[1],
                                 score_window=model_params[2], sensitivity=model_params[3],
                                 detect_mode=DetectMode.anomaly_only, dataset=dataset,
                                 datatype=type, filename=filename, drift_detector=drift_detector)
        model.fit()
        end_time = time.time()

        diff = end_time - start_time
        print(f"Trained model {name} on {filename} for {diff}")
    elif name == 'pso-elm':
        start_time = time.time()
        model.fit(data_train[-model.n:])
        end_time = time.time()

        diff = end_time - start_time
        print(f"Trained model {name} on {filename} for {diff}")
    else:
        exit(f'Sorry! Model {name} is not from the collection.')

    batch_metrices_f1_e = []
    batch_metrices_f1_noe = []
    y_pred_total_noe, y_pred_total_e = [], []
    batches_with_anomalies = []
    idx = 0

    pred_time = []

    for start in range(0, data_test.shape[0], step):
        try:
            start_time = time.time()
            window = data_test.iloc[start:start + anomaly_window]
            data_in_memory = pd.concat([data_in_memory, window])[-data_in_memory_sz:]

            X, y = window['value'], window['is_anomaly']
            if y.tolist():
                if name in ['sarima']:
                    y_pred_noe, y_pred_e = model.predict(window, optimization=for_optimization)

                elif name == 'sr':
                    model.__series__ = data_in_memory
                    try:
                        res = model.predict(data_in_memory, window.shape[0])
                        y_pred_noe = [1 if x else 0 for x in res['isAnomaly'].tolist()]

                        y_pred_e = [1 if x else 0 for x in res['isAnomaly_e'].tolist()]
                    except Exception as e:
                        y_pred_noe = [0 for _ in range(window.shape[0])]
                        y_pred_e = [0 for _ in range(window.shape[0])]

                elif name == 'lstm':
                    y_pred_noe, y_pred_e = model.predict(window)
                    y_pred_noe, y_pred_e = y_pred_noe[:window.shape[0]], y_pred_e[:window.shape[0]]
                elif name == 'pso-elm':
                    y_pred_noe, y_pred_e = model.predict(window[['value', 'timestamp']])

                idx += 1
                y_pred_total_noe += [0 if val != 1 else 1 for val in funcy.lflatten(y_pred_noe)][:window.shape[0]]
                y_pred_total_e += [0 if val != 1 else 1 for val in funcy.lflatten(y_pred_e)][:window.shape[0]]
                y_pred_noe = y_pred_noe[:window.shape[0]]
                y_pred_e = y_pred_e[:window.shape[0]]

                end_time = time.time()
                pred_time.append((end_time - start_time))
        except Exception as e:
            raise e

    if not for_optimization:
        print('Plotting..........')
        plot_general(model, dataset, type, name, data_test,
                     y_pred_total_e, filename)
        return

    print('saving results')
    predtime = np.mean(pred_time)
    bwa = batches_with_anomalies

    # calculate batched metrics per 60 timepoints
    # exact batch size may vary...
    # can also smooth batch metrics via rolling metrices
    for i in range(0, len(data_test['is_anomaly']), 60):
        met_total = precision_recall_fscore_support(data_test['is_anomaly'][i:i+60],
                                                    y_pred_total_e[:data_test.shape[0]][i:i+60])
        batch_metrices_f1_e.append(met_total[2][-1])

        met_total = precision_recall_fscore_support(data_test['is_anomaly'][i:i+60],
                                                    y_pred_total_noe[:data_test.shape[0]][i:i+60])
        batch_metrices_f1_noe.append(met_total[2][-1])

    met_total_noe = precision_recall_fscore_support(data_test['is_anomaly'],
                                                    y_pred_total_noe[:data_test.shape[0]])
    met_total_e = precision_recall_fscore_support(data_test['is_anomaly'],
                                                  y_pred_total_e[:data_test.shape[0]])

    if not for_optimization:
        try:
            stats_full = pd.read_csv(f'results/{dataset}_{type}_stats_{name}_drift_{dname}.csv')
        except:
            stats_full = pd.DataFrame([])

        stats_full = stats_full.append({
            'model': name,
            'dataset': filename.replace('.csv', ''),
            'window': anomaly_window,
            'f1-e': met_total_e[2][-1],
            'f1-noe': met_total_noe[2][-1],
        }, ignore_index=True)
        stats_full.to_csv(f'results/{dataset}_{type}_stats_{name}_drift_{dname}.csv', index=False)
    # return specificity if no anomalies, else return f1 score
    # can also just ignore tss with no anomalies...
    if data_test['is_anomaly'].tolist().count(1) == 0:
        return 1.0 - met_total_noe[0][-1]
    else:
        return 1.0 - met_total_noe[2][-1]


if __name__ == '__main__':
    # load already found hyperparameters
    try:
       hp = pd.read_csv('hyperparams.csv')
    except:
       hp = pd.DataFrame([])

    for dname, (dd, use_drift_adapt) in drift_detectors.items():
        if dname == 'ecdd':
            dd = dd()
        elif use_drift_adapt:
            dd = DriftDetectorWrapper(dd)
        else:
            dd = None

        dname = 'test'
        for dataset, type in [('yahoo', 'A1Benchmark'), ('NAB', 'windows'), ('kpi', 'train'),
                              ('yahoo', 'A2Benchmark'), ('yahoo', 'A3Benchmark'), ('yahoo', 'A4Benchmark')]:

            anomaly_window = step = entropy_params[f'{dataset}_{type}']['window']

            for name, (model, bo_space, def_params) in models.items():
                train_data_path = root_path + '/datasets/' + dataset + '/' + type + '/'

                try:
                    stats_full = pd.read_csv(f'results/{dataset}_{type}_stats_{name}_drift_{dname}.csv')
                except:
                    stats_full = pd.DataFrame([])

                try:
                    stats_batched = pd.read_csv(
                        f'results/entropy_addition/{dataset}_{type}_stats_{name}_batched_test.csv')
                except:
                    stats_batched = pd.DataFrame([])

                for filename in os.listdir(train_data_path):
                    f = os.path.join(train_data_path, filename)
                    res_data_path = root_path + f'/results/imgs/{dataset}/{type}/{name}'
                    data = pd.read_csv(f)
                    if os.path.isfile(f) and (stats_full.shape[0] == 0 or filename.replace('.csv', '')
                                              not in stats_full['dataset'].tolist()):
                        data = pd.read_csv(f)
                        print('Working with current time series:', filename)
                        data.rename(columns={'timestamps': 'timestamp', 'anomaly': 'is_anomaly'}, inplace=True)
                        data.drop_duplicates(subset=['timestamp'], keep=False, inplace=True)

                        if dataset == 'kpi':
                            data_test = pd.read_csv(os.path.join(root_path + '/datasets/' + dataset + '/test/', filename))
                            data_test['timestamp'] = data_test['timestamp'].apply(
                                lambda x: datetime.datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
                            data['timestamp'] = data['timestamp'].apply(
                                lambda x: datetime.datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
                            data_test_ = data_test
                            # if not enough memory -- can use smaller ts
                            # data = data[-3000:]
                            # data_test_ = data_test[:25000]

                        if no_optimization:
                            try:
                                bo_result = mock.Mock()
                                if not hp.empty and not hp[(hp['filename'] == filename.replace('.csv', '')) &
                                                           (hp['model'] == name)].empty:
                                    bo_result.x = ast.literal_eval(
                                        hp[(hp['filename'] == filename.replace('.csv', ''))
                                           & (hp['model'] == name)]['hp'].tolist()[0])
                                else:
                                    bo_result.x = def_params

                                fit_base_model(bo_result.x, for_optimization=False)
                            except Exception as e:
                                raise e
                        else:
                            try:
                                # ################ Bayesian optimization ###################################################
                                if True or hp.empty or hp[(hp['filename'] == filename.replace('.csv', '')) & (hp['model'] == name)].empty:
                                    print(bo_space)
                                    try:
                                        bo_result = gp_minimize(fit_base_model, bo_space, callback=Stopper(), n_calls=11,
                                                                random_state=3, verbose=True, x0=def_params)
                                    except ValueError as e:
                                        # error is rised when function yields constant value and does not converge
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
                                    bo_result.x = ast.literal_eval(
                                        hp[(hp['filename'] == filename.replace('.csv', ''))
                                           & (hp['model'] == name)]['hp'].tolist()[0])

                                fit_base_model(bo_result.x, for_optimization=False)

                            except Exception as e:
                                raise e
