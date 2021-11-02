import json

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from skopt.callbacks import EarlyStopper


class Stopper(EarlyStopper):
    def __call__(self, result):
        ret = False
        if result.func_vals[-1] == 1.0:
            ret = True
        return ret


def create_dataset(X, y, time_steps=60):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i:(i + time_steps)].values)

    return np.array(Xs), np.array(ys)


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


def merge_model_performance():
    train_df = None
    for tp in ['real', 'synthetic']:
        dbscan = pd.read_csv(f'C:\\Users\\oxifl\\Desktop\\thesis_res\\2_opt_no_updt\yahoo\win_60\\real\\yahoo_{tp}_stats_dbscan.csv')
        lstm = pd.read_csv(f'C:\\Users\\oxifl\\Desktop\\thesis_res\\2_opt_no_updt\yahoo\win_60\\real\\yahoo_{tp}_stats_lstm.csv')
        sarima = pd.read_csv(f'C:\\Users\\oxifl\\Desktop\\thesis_res\\2_opt_no_updt\yahoo\win_60\\real\\yahoo_real_{tp}_sarima.csv')
        sr = pd.read_csv(
            f'C:\\Users\\oxifl\\Desktop\\thesis_res\\2_opt_no_updt\yahoo\win_60\\real\\yahoo_{tp}_stats_sr.csv')
        if not train_df:
            idx = 0
            train_df = dbscan[['autocorrelation', 'dataset', 'hurst', 'kurtosis', 'max_lyapunov_e',	'non-linearity',
                               'seasonality', 'skewness', 'trend']]
        else:
            idx = train_df.shape[0]
            train_df = pd.concat([train_df, dbscan[['autocorrelation', 'dataset', 'hurst', 'kurtosis', 'max_lyapunov_e',
                                                    'non-linearity', 'seasonality', 'skewness', 'trend']]])
        pos_f1_array = {
            0: 'dbscan',
            1: 'lstm',
            2: 'sarima',
            3: 'sr'
        }
        for f_dbscan, f_lstm, f_sarima, f_sr in zip(dbscan['f1'], lstm['f1'], sarima['f1'], sr['f1']):
            model_list = [f_dbscan, f_lstm, f_sarima, f_sr]
            best_model = pos_f1_array[model_list.index(max(model_list))]
            train_df.pos[idx, 'model_to_use'] = best_model
            idx += 1

    train_df.to_csv('ensemble_train_data.csv')


def preprocess_nab_labels(root_path):
    with open(root_path + '/datasets/NAB/labels/combined_labels.json') as json_file:
        labels = json.load(json_file)
    for attribute, value in labels.items():
        try:
            dataset = pd.read_csv(root_path + '/datasets/NAB/' + attribute)
            dataset['is_anomaly'] = 0
            dataset.set_index('timestamp', inplace=True)
            for a_idx in value:
                dataset.loc[a_idx, 'is_anomaly'] = 1.0

            dataset.to_csv(root_path + '/datasets/NAB/train/' + attribute.split('/')[-1])
        except:
            pass


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
