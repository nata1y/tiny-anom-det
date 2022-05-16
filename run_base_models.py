import ast
import copy
import os

import funcy
import mock
from sklearn import preprocessing
from skopt.callbacks import EarlyStopper

from utils import plot_general
from sklearn.metrics import hamming_loss, cohen_kappa_score
from skopt import gp_minimize

from sklearn.metrics import precision_recall_fscore_support

from utils import create_dataset
from models.sr.spectral_residual import DetectMode
import pandas as pd
import datetime
import time
import numpy as np
from model_collection import *
from settings import entropy_params, data_in_memory_sz


root_path = os.getcwd()
le = preprocessing.LabelEncoder().fit([-1, 1])
data_test = None
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
    global can_model, traintime, predtime, batched_f1_score, batch_metrices, bwa, dataset

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

    start = time.time()
    # create models with hyper-parameters
    if name == 'sarima':
        model = SARIMA(dataset, type, model_params[0], model_params[1])
    elif name == 'lstm':
        model = LSTM_autoencoder([anomaly_window, 1], dataset, type, filename.replace(".csv", ""),
                                 magnitude=model_params[0])

    if name == 'sarima':

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
    elif name == 'sr':
        model = SpectralResidual(series=data_train[['value', 'timestamp']], threshold=model_params[0], mag_window=model_params[1],
                                 score_window=model_params[2], sensitivity=model_params[3],
                                 detect_mode=DetectMode.anomaly_only, dataset=dataset, datatype=type)
        model.fit()

    end = time.time()
    traintime = end - start

    batch_metrices_kc = []
    batch_metrices_kc_entropy = []
    batch_metrices_hamming = []
    batch_metrices_hamming_entropy = []
    batch_metrices_f1_e = []
    batch_metrices_f1_noe = []
    y_pred_total = []
    y_pred_total_noe, y_pred_total_e = [], []
    batches_with_anomalies = []
    idx = 0

    pred_time = []
    drift_windows = []

    for start in range(0, data_test.shape[0], step):
        try:
            start_time = time.time()
            window = data_test.iloc[start:start + anomaly_window]
            data_in_memory = pd.concat([data_in_memory, window])[-data_in_memory_sz:]

            X, y = window['value'], window['is_anomaly']
            if y.tolist():
                if name in ['sarima']:
                    if for_optimization:
                        y_pred_noe, y_pred_e = model.predict(window[['timestamp', 'value']], optimization=True)
                    else:
                        y_pred_noe, y_pred_e = model.predict(window[['timestamp', 'value']])

                elif name == 'sr':
                    model.__series__ = data_in_memory[['value', 'timestamp']]
                    if model.dynamic_threshold:
                        try:
                            res = model.detect_dynamic_threshold(data_in_memory[['value', 'timestamp']])
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
                else:
                    y_pred = model.predict(window[['value', 'timestamp']])

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

    if not for_optimization:
        print('Plotting..........')
        plot_general(model, dataset, type, name, data_test,
                     y_pred_total_e, filename, drift_windows)

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
            stats_full = pd.read_csv(f'results/entropy_addition/{dataset}_{type}_stats_{name}_test.csv')
        except:
            stats_full = pd.DataFrame([])

        stats_full = stats_full.append({
            'model': name,
            'dataset': filename.replace('.csv', ''),
            'window': anomaly_window,
            'f1-e': met_total_e[2][-1],
            'f1-noe': met_total_noe[2][-1],
        }, ignore_index=True)
        stats_full.to_csv(f'results/entropy_addition/{dataset}_{type}_stats_{name}_test.csv', index=False)
    # return specificity if no anomalies, else return f1 score
    if data_test['is_anomaly'].tolist().count(1) == 0:
        return 1.0 - met_total_noe[0][-1]
    else:
        return 1.0 - met_total_noe[2][-1]


if __name__ == '__main__':
    # load already found hyperparameters
    try:
       hp = pd.read_csv('hyperparams_test.csv')
    except:
       hp = pd.DataFrame([])

    continuer = True
    for dataset, type in [('NAB', 'windows')]:
        # options:
        # ('kpi', 'train'), ('NAB', 'windows'), ('NAB', 'relevant'),
        # ('yahoo', 'real'), ('yahoo', 'synthetic'), ('yahoo', 'A3Benchmark'), ('yahoo', 'A4Benchmark')

        anomaly_window = step = entropy_params[f'{dataset}_{type}']['window']

        for name, (model, bo_space, def_params) in models.items():
            train_data_path = root_path + '/datasets/' + dataset + '/' + type + '/'

            try:
                stats_full = pd.read_csv(f'results/entropy_addition/{dataset}_{type}_stats_{name}_test.csv')
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
                    print(filename)
                    data.rename(columns={'timestamps': 'timestamp', 'anomaly': 'is_anomaly'}, inplace=True)
                    data.drop_duplicates(subset=['timestamp'], keep=False, inplace=True)

                    if dataset == 'kpi':
                        data_test = pd.read_csv(os.path.join(root_path + '/datasets/' + dataset + '/test/', filename))
                        data_test['timestamp'] = data_test['timestamp'].apply(
                            lambda x: datetime.datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
                        data['timestamp'] = data['timestamp'].apply(
                            lambda x: datetime.datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))

                    try:
                        ################ Bayesian optimization ###################################################
                        # if hp.empty or hp[(hp['filename'] == filename.replace('.csv', '')) & (hp['model'] == name)].empty:
                        #     print(bo_space)
                        #     try:
                        #         bo_result = gp_minimize(fit_base_model, bo_space, callback=Stopper(), n_calls=11,
                        #                                 random_state=3, verbose=True, x0=def_params)
                        #     except ValueError as e:
                        #         # error is rised when function yields constant value and does not converge
                        #         bo_result = mock.Mock()
                        #         bo_result.x = def_params
                        #
                        #     print(f"Found hyper parameters for {name}: {bo_result.x}")
                        #
                        #     if hp.empty or filename.replace('.csv', '') not in hp[hp['model'] == name]['filename'].tolist():
                        #         hp = hp.append({
                        #             'filename': filename.replace('.csv', ''),
                        #             'model': name,
                        #             'hp': bo_result.x
                        #         }, ignore_index=True)
                        #
                        #         hp.to_csv('hyperparams.csv', index=False)
                        # else:
                        #     bo_result = mock.Mock()
                        #     bo_result.x = ast.literal_eval(
                        #         hp[(hp['filename'] == filename.replace('.csv', ''))
                        #            & (hp['model'] == name)]['hp'].tolist()[0])

                        bo_result = mock.Mock()
                        bo_result.x = def_params
                        fit_base_model(bo_result.x, for_optimization=False)

                    except Exception as e:
                        raise e
