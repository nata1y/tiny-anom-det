import json
from math import sqrt

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow
from catch22 import catch22_all
from skopt.callbacks import EarlyStopper

from analysis.postanalysis import confusion_visualization


class Stopper(EarlyStopper):
    def __call__(self, result):
        ret = False
        if result.func_vals[-1] == 0.0:
            ret = True
        return ret


def create_dataset(X, y, time_steps=60):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i:(i + time_steps)].values)

    return np.array(Xs), np.array(ys)


def create_dataset_keras(X, y, time_steps=60):
    print(f'{X.shape} shapeeee')
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i:(i + time_steps)].values)

    return tensorflow.convert_to_tensor(Xs), tensorflow.convert_to_tensor(ys)


def adjust_range(val, oper, factor):
    if val < 0:
        sign = -1.0
        val *= -1.0
        if oper == 'div':
            val = val * factor * sign
        else:
            val = sign * val / factor

        return val

    if oper == 'div':
        val = val / factor
    else:
        val = val * factor

    return val


def preprocess_kpi(cwd):
    dataset, type = 'kpi', 'synthetic'

    for t in ['train', 'test']:
        train_data_path = cwd + '\\' + dataset + '\\' + t + '\\'
        for filename in os.listdir(train_data_path):
            f = os.path.join(train_data_path, filename)
            if os.path.isfile(f):
                try:
                    data = pd.read_csv(f)
                except:
                    data = pd.read_hdf(f)

                for kpi_id in data['KPI ID'].unique().tolist():
                    df = data[data['KPI ID'] == kpi_id].reset_index(drop=True)[['timestamp', 'value', 'label']]
                    df.rename(columns={"label": "is_anomaly"}, inplace=True)
                    df.to_csv(cwd + '\\' + dataset + '\\' + t + '\\' + kpi_id + '.csv')


def plot_dbscan(preds, dataset, type, filename, data_test, window_sz):
    data_test.reset_index(inplace=True)
    for (label, core_sample_indice), start in zip(preds, range(0, data_test.shape[0], window_sz)):
        window = data_test.iloc[start:start + window_sz][['value', 'timestamp']]

        core_samples_mask = np.zeros_like(label, dtype=bool)
        core_samples_mask[core_sample_indice] = True
        n_clusters_ = len(set(label)) - (1 if -1 in label else 0)
        unique_labels = set(label)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (label == k)

            xy = window[class_member_mask & core_samples_mask]
            plt.plot(xy['timestamp'].tolist(), xy['value'].tolist(), 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=5)

            xy = window[class_member_mask & ~core_samples_mask]
            plt.plot(xy['timestamp'].tolist(), xy['value'].tolist(), 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=1)

        plt.axvline(x=np.max(window['timestamp'].tolist()))

    plt.savefig(
        f'results/imgs/{dataset}/{type}/dbscan/dbscan_{filename.replace(".csv", "")}_groups.png')


def handle_missing_values_kpi(data, start=None):
    data.loc[:, 'timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
    data.set_index('timestamp', inplace=True)
    data = data.asfreq('T')
    if start:
        dti = pd.DataFrame([])
        dti.loc[:, 'timestamp'] = pd.date_range(start + pd.Timedelta(minutes=1),
                                                periods=data.shape[0], freq="T")
        dti.set_index('timestamp', inplace=True)
        dti['value'] = None
        res = pd.concat((data, dti)).sort_values(by='value')
        res = res[~res.index.duplicated(keep='first')]
        res = res.sort_index()
        return res
    return data


def preprocess_nab_labels(root_path):
    with open(root_path + '/datasets/NAB/labels/combined_labels.json') as json_file:
        labels = json.load(json_file)

    # with open(root_path + '/datasets/NAB/labels/combined_windows.json') as json_file:
    #     labels_window = json.load(json_file)

    # with open(root_path + '/datasets/NAB/labels/combined_windows_tiny.json') as json_file:
    #     labels_window_tiny = json.load(json_file)
    for attribute, value in labels.items():
        try:
            dataset = pd.read_csv('C:\\Users\\oxifl\\Documents\\uni\\NAB\\data\\' + attribute)
            dataset['is_anomaly'] = 0
            dataset.set_index('timestamp', inplace=True)
            for a_idx in value:
                dataset.loc[a_idx, 'is_anomaly'] = 1.0

            dataset.to_csv(root_path + '\\datasets\\NAB\\relevant\\' + attribute.split('/')[-1])
        except Exception as e:
            print(e)

    # for attribute, values in labels_window.items():
    #     try:
    #         dataset = pd.read_csv(root_path + '/datasets/NAB/relevant/' + attribute.split('/')[-1])
    #         dataset.set_index('timestamp', inplace=True)
    #         for value in values:
    #             start, end = value[0], value[1]
    #             bool_series = ((dataset.index >= start) & (dataset.index <= end))
    #             dataset.loc[bool_series, 'is_anomaly'] = 1.0
    #
    #         dataset.to_csv(root_path + '/datasets/NAB/relevant/' + attribute.split('/')[-1])
    #     except:
    #         pass

    # for attribute, values in labels_window_tiny.items():
    #     try:
    #         dataset = pd.read_csv(root_path + '/datasets/NAB/relevant/' + attribute.split('/')[-1])
    #         dataset.set_index('timestamp', inplace=True)
    #         for value in values:
    #             start, end = value[0], value[1]
    #             bool_series = ((dataset.index >= start) & (dataset.index <= end))
    #             dataset.loc[bool_series, 'is_anomaly'] = 1.0
    #
    #         dataset.to_csv(root_path + '/datasets/NAB/relevant/' + attribute.split('/')[-1])
    #     except:
    #         pass


def preprocess_telemanom_datatset(root_path_save):
    root_path = f'C:\\Users\\oxifl\\Documents\\uni\\telemanom\\data\\'
    for tp in ['train', 'test']:
        for filename in os.listdir(root_path + f'{tp}/'):
            f = os.path.join(root_path + f'{tp}\\', filename)
            if os.path.isfile(f):
                print(root_path + f'{tp}\\' + filename)
                data = np.load(root_path + f'{tp}\\' + filename)
                print(data)

                pd_data = pd.DataFrame(data)
                pd_data.to_csv(root_path_save + f'/datasets/telemanom/{tp}/' + filename.split('.')[0] + '.csv')


def KL(a, b):
    a = [0.00000001 if x == 0.0 else x for x in a]
    b = [0.00000001 if x == 0.0 else x for x in b]

    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


def intersection(a, b):
    d = 0.0
    for i, j in zip(a, b):
        d += min(i, j)

    return d


def fidelity(a, b):
    d = 0.0
    for i, j in zip(a, b):
        try:
            d += sqrt(i * j)
        except:
            pass

    return d


def sq_euclidian(a, b):
    d = 0.0
    for i, j in zip(a, b):
        d += (i - j) ** 2

    return d


def load_ucr_ts():
    data_path = 'datasets/ucr/'
    df = pd.read_csv('results/ts_properties/ucr_ts_features_c22.csv')
    idx = df.shape[0]
    for dirname in os.listdir(data_path):
        d = os.path.join(data_path, dirname)
        if not os.path.isfile(d):
            print(f"Processing ts {d}")
            for filename in os.listdir(d):
                print(filename)
                f = os.path.join(d, filename)
                if os.path.isfile(f) and '_TRAIN.txt' in str(filename):
                    my_file = open(f, "r")
                    content = my_file.read()
                    idx2 = 0
                    for ts in content.split('\n'):
                        data = [float(e) for e in ' '.join(ts.replace('\n', '').split(",")).split(" ") if e != '']
                        if data:
                            print(data)
                            res = catch22_all(data)
                            print(res)
                            df.loc[idx, 'ts'] = filename + str(idx2)
                            idx2 += 1
                            for name, val in zip(res['names'], res['values']):
                                df.loc[idx, name] = val
                            idx += 1
                            df.to_csv(f'results/ts_properties/ucr_ts_features_c22.csv')
                    my_file.close()


def drift_metrics(y_true, y_pred):
    fp = 0
    latency = 0.0
    misses = 0
    drift_now = False
    drift_detected = False
    drift_start_idx = None
    total_amount_drifts = 0

    for idx, (i_true, i_pred) in enumerate(zip(y_true, y_pred)):
        if i_true == 1:
            if drift_start_idx:
                misses += 1
            total_amount_drifts += 1
            drift_start_idx = idx
        if i_pred == 1:
            if not drift_start_idx:
                fp += 1
            else:
                try:
                    drift_stop_idx = y_true[drift_start_idx:].index(0)
                except:
                    drift_stop_idx = len(y_pred)
                drift_length = drift_stop_idx - drift_start_idx
                latency += (idx - drift_start_idx)  # / drift_length
                drift_start_idx = None

    if drift_start_idx:
        misses += 1

    if total_amount_drifts:
        latency = latency / total_amount_drifts
    return fp, latency, misses


def plot_general(model, dataset, type, name, data_test, y_pred_total, filename, drift_windows):
    try:
        confusion_visualization(data_test['timestamp'].tolist(), data_test['value'].tolist(),
                                data_test['is_anomaly'].tolist(), y_pred_total,
                                dataset, name, filename.replace('.csv', '') + '_point', type, drift_windows)
    except Exception as e:
        raise e

    try:
        if name in ['sarima', 'es']:
            model.plot(data_test[['timestamp', 'value']], dataset, type, filename, data_test, drift_windows)
        elif name in ['lstm']:
            model.plot(data_test['timestamp'].tolist(), dataset, type, filename, data_test, drift_windows)
        # elif name == 'ensemble':
        #     model.plot(dataset)
        elif name == 'sr':
            model.plot(dataset, type, filename, data_test, drift_windows)
            # if model.dynamic_threshold:
            #     model.plot_dynamic_threshold(data_test['timestamp'].tolist(), dataset, type, filename, data_test)
        # elif name == 'dbscan':
        #     plot_dbscan(all_labels, dataset, type, filename, data_test[['value', 'timestamp']], anomaly_window)
        # elif name == 'seasonal_decomp':
        #     model.plot(type, filename, )
    except Exception as e:
        raise e
        print(e)


def relable_yahoo():
    for ds in ['real']:
        data_path = f'datasets/yahoo/{ds}/'
        for filename in os.listdir(data_path):
            print(filename)
            f = os.path.join(data_path, filename)
            if os.path.isfile(f) and filename:
                df = pd.read_csv(f)
                labels = df['is_anomaly'].tolist()
                labels_new = []
                drift_start = False
                for idx in range(len(labels)):
                    if not drift_start and labels[idx] == 0:
                        labels_new.append(0)
                    elif not drift_start and labels[idx] == 1:
                        end = labels[idx:].index(0) if 0 in labels[idx:] else len(labels)
                        if end - idx <= 50:
                            drift_start = True
                            labels_new.append(1)
                        else:
                            labels_new.append(0)
                    elif drift_start and labels[idx] == 1:
                        labels_new.append(0)
                    elif drift_start and labels[idx] == 0:
                        labels_new = labels_new[:(idx - 1)] + [1, 0]
                        drift_start = False

                df['is_anomaly'] = labels_new
                df.to_csv(f'datasets/yahoo/point/{ds}/{filename}')