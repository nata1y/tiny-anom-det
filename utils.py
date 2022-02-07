import ast
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
                                dataset, name, filename.replace('.csv', ''), type, drift_windows)
    except Exception as e:
        raise e

    try:
        if name in ['sarima', 'es']:
            model.plot(data_test[['timestamp', 'value']], dataset, type, filename, data_test, drift_windows)
        elif name in ['lstm']:
            model.plot(data_test['timestamp'].tolist(), dataset, type, filename, data_test, drift_windows)
        elif name == 'dp':
            model.plot(data_test[['timestamp', 'value']], type, data_test)
        elif name == 'sr':
            model.plot(dataset, type, filename, data_test, drift_windows)
    except Exception as e:
        print(e)


def plot_change(score_array, anomaly_idxs, model, ts, dataset):
    helper = pd.DataFrame([])
    helper['value'] = score_array
    helper['ma'] = helper.rolling(window=5).mean()
    plt.plot(range(len(score_array)), score_array, marker='o', label='hamming distance')
    plt.plot(range(len(score_array)), helper['ma'].tolist(), marker='*', color='red', label='Rolling mean')
    for idx in anomaly_idxs:
        plt.axvline(x=idx, color='orange', linestyle='--')
    plt.xlabel('Batch')
    plt.ylabel('Hamming Distance')
    plt.legend()
    plt.savefig(f'results/imgs/change_performance/{dataset}/{model}_{ts}.png')
    plt.clf()


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


def merge_nab_drift_points():
    for ds in ['machine_metrics', 'non_machine_metrics']:
        data_path_drift = f'datasets/{ds}/nab_drift/'
        data_path_point = f'datasets/{ds}/nab_point/'

        for filename in os.listdir(data_path_drift):
            print(filename)
            f = os.path.join(data_path_drift, filename)
            try:
                f_p = os.path.join(data_path_point, filename)
            except:
                f_p = None
            if os.path.isfile(f) and filename:
                df = pd.read_csv(f)
                labels = df['is_anomaly'].tolist()
                labels_new = []
                drift_start = False

                # add drifts
                for idx in range(len(labels)):
                    if not drift_start and labels[idx] == 0:
                        labels_new.append(0)
                    elif not drift_start and labels[idx] == 1:
                        drift_start = True
                        labels_new.append(1)
                    elif drift_start and labels[idx] == 1:
                        labels_new.append(1)
                        drift_start = False
                    elif drift_start and labels[idx] == 0:
                        labels_new.append(1)

                # add points
                if f_p:
                    for idx in range(len(labels)):
                        if labels_new[idx] == 0 and labels[idx] == 1:
                            labels_new[idx] = 1

                df['is_anomaly'] = labels_new
                df.to_csv(f'datasets/NAB/normal_labels/{filename}')

        for filename in os.listdir(data_path_point):
            print(filename)
            f = os.path.join('datasets/NAB/normal_labels/', filename)
            if not os.path.isfile(f):
                df = pd.read_csv(data_path_point + filename)
                df.to_csv(f'datasets/NAB/normal_labels/{filename}')


anomaly_taxonomy = {
    # real_37, real_43, real_53, ec2_cpu_utilization_ac20cd, ec2_network_in_5abac7, speed_7578, 'Twitter_volume_KO'
    # Twitter_volume_UPS synthetic_2, synthetic_16 - ?
    '1a': ['real_1', 'real_2', 'real_3', 'real_4', 'real_6', 'real_10', 'real_11', 'real_12', 'real_13', 'real_15',
           'real_9', 'real_16', 'real_20', 'real_21', 'real_24', 'real_26', 'real_29', 'real_30', 'real_33', 'real_34',
           'real_38', 'real_39', 'real_41', 'real_45', 'real_50', 'real_51', 'real_52', 'real_60', 'real_61', 'real_62',
           'ambient_temperature_system_failure', 'ec2_cpu_utilization_5f5533', 'ec2_cpu_utilization_fe7f93',
           'ec2_disk_write_bytes_c0d644', 'ec2_request_latency_system_failure', 'elb_request_count_8c0756',
           'exchange-3_cpm', 'exchange-4_cpm', 'occupancy_6005', 'occupancy_t4013', 'speed_t4013', 'Twitter_volume_AAPL',
           'Twitter_volume_FB', 'synthetic_5', 'synthetic_7', 'synthetic_9', 'synthetic_10', 'synthetic_11',
           'synthetic_13', 'synthetic_14', 'synthetic_19', 'synthetic_21', 'synthetic_23', 'synthetic_25',
           'synthetic_27', 'synthetic_28', 'synthetic_30', 'synthetic_31', 'synthetic_35', 'synthetic_37',
           'synthetic_39', 'synthetic_41', 'synthetic_42', 'synthetic_43', 'synthetic_44', 'synthetic_47',
           'synthetic_48', 'synthetic_49', 'synthetic_51', 'synthetic_53', 'synthetic_55', 'synthetic_56',
           'synthetic_57', 'synthetic_58', 'synthetic_59', 'synthetic_60', 'synthetic_61', 'synthetic_62',
           'synthetic_63', 'synthetic_65', 'synthetic_67', 'synthetic_68', 'synthetic_69', 'synthetic_70',
           'synthetic_72', 'synthetic_73', 'synthetic_74', 'synthetic_75', 'synthetic_76', 'synthetic_77',
           'synthetic_79', 'synthetic_81', 'synthetic_83', 'synthetic_84', 'synthetic_85', 'synthetic_88',
           'synthetic_89', 'synthetic_90', 'synthetic_91', 'synthetic_93', 'synthetic_95', 'synthetic_97',
           'synthetic_98', 'synthetic_99', 'synthetic_100'],
    '1a_generated': ['art_daily_no_noise', 'art_daily_perfect_square_wave', 'art_daily_small_noise', 'art_flatline',
                     'art_noisy', 'ec2_cpu_utilization_c6585a', 'exchange-3_cpm_results', 'occupancy_6005',
                     'occupancy_t4013', 'rds_cpu_utilization_cc0c53', 'Twitter_volume_FB', 'real_1', 'real_10',
                     'real_12', 'real_16', 'real_18', 'real_21', 'real_24', 'real_25', 'real_3', 'real_30', 'real_33',
                     'real_35', 'real_36', 'real_4', 'real_41', 'real_42', 'real_45', 'real_49', 'real_5', 'real_50',
                     'real_53', 'real_54', 'real_58', 'real_59', 'real_62', 'real_64', 'real_67', 'real_8',
                     'synthetic_100', 'synthetic_11', 'synthetic_13', 'synthetic_14', 'synthetic_19',
                     'synthetic_21', 'synthetic_23', 'synthetic_25', 'synthetic_27', 'synthetic_28', 'synthetic_30',
                     'synthetic_31', 'synthetic_33', 'synthetic_34', 'synthetic_35', 'synthetic_37', 'synthetic_39',
                     'synthetic_41', 'synthetic_42', 'synthetic_43', 'synthetic_44', 'synthetic_46', 'synthetic_47',
                     'synthetic_48', 'synthetic_49', 'synthetic_5', 'synthetic_51', 'synthetic_53', 'synthetic_55',
                     'synthetic_56', 'synthetic_57', 'synthetic_58', 'synthetic_59', 'synthetic_60', 'synthetic_61',
                     'synthetic_62', 'synthetic_63', 'synthetic_65', 'synthetic_67', 'synthetic_69', 'synthetic_7',
                     'synthetic_70', 'synthetic_71', 'synthetic_72', 'synthetic_73', 'synthetic_74', 'synthetic_75',
                     'synthetic_76', 'synthetic_77', 'synthetic_79', 'synthetic_81', 'synthetic_83', 'synthetic_84',
                     'synthetic_85', 'synthetic_86', 'synthetic_87', 'synthetic_88', 'synthetic_89', 'synthetic_9',
                     'synthetic_90', 'synthetic_91', 'synthetic_93', 'synthetic_95', 'synthetic_97', 'synthetic_98',
                     'synthetic_99', 'A3Benchmark-TS64', 'A3Benchmark-TS66'],
    '1b': [],
    '4a': ['ec2_cpu_utilization_24ae8d', 'ec2_disk_write_bytes_1ef3de', 'nyc_taxi', 'synthetic_1', '',
           'synthetic_3', 'synthetic_4', 'synthetic_6', 'synthetic_8', 'synthetic_22', 'synthetic_15', 'synthetic_12',
           'synthetic_36', 'synthetic_40', 'synthetic_45', 'synthetic_50', 'synthetic_64', 'synthetic_66',
           'synthetic_78', 'synthetic_92', 'synthetic_96'],
    '7a': ['real_25', 'real_40', 'real_46', 'art_daily_flatmiddel', 'art_daily_jumpsdown', 'art_daily_jumpsup',
           'art_daily_nojumps'],
    '7b': ['real_8', 'ec2_cpu_utilization_53ea38'],
    '7c': ['ec2_cpu_utilization_ac20cd', 'grok_asg_anomaly', 'rds_cpu_utilization_cc0c53'],
    '7d': ['real_7', 'real_17', 'real_19', 'real_22', 'real_24', 'real_28', 'real_31', 'real_32', 'real_42', 'real_58',
           'real_63', 'real_66', 'real_67', 'cpu_utilization_asg_misconfiguration'],
    '7e': ['real_55', 'real_56'],
    '7f': ['real_65']
}


def score_per_anomaly_group():
    idx = 0
    correlation_df = pd.DataFrame([])
    for dataset, subsets in [('NAB', ['relevant']), ('yahoo', ['real', 'synthetic', 'A3Benchmark', 'A4Benchmark'])]:
        for tss in subsets:
            print(tss)
            sz = 60
            if dataset == 'kpi':
                sz = 1024

            sr = pd.read_csv(
                f'C:\\Users\\oxifl\\Desktop\\thesis_res\\2_opt_no_updt\\{dataset}\\'
                f'win_{sz}\\{tss}\\{dataset}_{tss}_stats_sr.csv')

            try:
                lstm = pd.read_csv(
                    f'C:\\Users\\oxifl\\Desktop\\thesis_res\\2_opt_no_updt\\{dataset}\\'
                    f'win_{sz}\\{tss}\\{dataset}_{tss}_stats_lstm.csv')
            except:
                try:
                    lstm = pd.read_csv(
                        f'C:\\Users\\oxifl\\Desktop\\thesis_res\\1_no_opt_no_updt\\{dataset}\\'
                        f'min_metrics\\win_{sz}\\{tss}\\{dataset}_{tss}_stats_lstm.csv')
                except:
                    lstm = pd.DataFrame([])
                    lstm['f1'] = [0 for i in range(sr.shape[0])]

            try:
                sarima = pd.read_csv(
                    f'C:\\Users\\oxifl\\Desktop\\thesis_res\\2_opt_no_updt\\{dataset}\\'
                    f'win_{sz}\\{tss}\\{dataset}_{tss}_stats_sarima.csv')
            except:
                try:
                    sarima = pd.read_csv(
                        f'C:\\Users\\oxifl\\Desktop\\thesis_res\\1_no_opt_no_updt\\{dataset}\\'
                        f'min_metrics\\win_{sz}\\{tss}\\{dataset}_{tss}_stats_sarima.csv')
                except Exception as e:
                    sarima = pd.DataFrame([])
                    sarima['f1'] = [0 for i in range(sr.shape[0])]

            for atype in ['1a_generated']:
                for f_lstm, f_sarima, f_sr, s_lstm, s_sarima, s_sr, ts in \
                        zip(lstm['f1'], sarima['f1'], sr['f1'], lstm['specificity'],
                            sarima['specificity'], sr['specificity'],
                            sr['dataset']):
                    print(ts)

                    if str(ts) != 'nan' and 'flatline' not in ts and ts.replace('.csv', '') in anomaly_taxonomy[atype]:
                        series = pd.read_csv('datasets/' + dataset + '/' + tss + '/' + ts + '.csv')
                        series.rename(columns={'timestamps': 'timestamp', 'anomaly': 'is_anomaly'}, inplace=True)
                        if dataset != 'kpi':
                            data_train, data_test = np.array_split(series, 2)
                            if data_test['is_anomaly'].tolist().count(1) == 0:
                                continue

                        correlation_df.loc[idx, 'dataset'] = dataset + '_' + tss
                        correlation_df.loc[idx, 'ts'] = ts

                        correlation_df.loc[idx, 'lstm_score'] = f_lstm
                        correlation_df.loc[idx, 'sarima_score'] = f_sarima
                        correlation_df.loc[idx, 'sr_score'] = f_sr

                        idx += 1

    print(correlation_df.head())
    if not correlation_df.empty:
        correlation_df.to_csv(f'results/ts_properties/1a_generated_anomaly_to_f1.csv')


def analyze_anomalies(root_path):
    type_one_anomalies = []
    for dataset, subsets in [('NAB', ['relevant']), ('yahoo', ['real', 'synthetic', 'A3Benchmark', 'A4Benchmark'])]:
        for tss in subsets:
            train_data_path = root_path + '/datasets/' + dataset + '/' + tss + '/'
            for filename in os.listdir(train_data_path):
                print(filename)
                f = os.path.join(train_data_path, filename)

                if os.path.isfile(f) and filename:
                    data = pd.read_csv(f)
                    data.rename(columns={'timestamps': 'timestamp', 'anomaly': 'is_anomaly'}, inplace=True)
                    data.drop_duplicates(subset=['timestamp'], keep=False, inplace=True)

                    anomalies = data[data['is_anomaly'] == 1]['value'].values
                    normals = data[data['is_anomaly'] == 0]['value'].values
                    normals_min, normals_max = np.min(normals), np.max(normals)

                    if not [a for a in anomalies if normals_min <= a <= normals_max]:
                        type_one_anomalies.append(filename.replace('.csv', ''))

    print(type_one_anomalies)


def avg_batch_f1():
    for dataset, subsets in [('NAB', ['relevant'])]:
        for tss in subsets:
                for model in ['sarima', 'sr', 'lstm']:
                    total = 0
                    avg_f1_before, avg_f1_after = 0.0, 0.0
                    try:
                        path = 'results/' + dataset + '_' + tss + '_stats_' + model + '_batched.csv'
                        df = pd.read_csv(path)
                        # print(df.head())
                        for idx, row in df.iterrows():
                            l = ast.literal_eval(row['batch_metrics'])
                            a = ast.literal_eval(row['anomaly_idxs'])
                            if not a:
                                continue
                            a1, al = a[0], a[-1]
                            try:
                                if not pd.isna(np.mean(l[:a1])) and not pd.isna(np.mean(l[al + 1:])):
                                    avg_f1_before += np.mean(l[:a1])
                                    avg_f1_after += np.mean(l[al + 1:])
                                    total += 1
                                    # print(pd.isna(np.mean(l[:a1])), np.mean(l[:a1]))
                            except Exception as e:
                                pass

                    except Exception as e:
                        raise e

                    if total > 0:
                        avg_f1_before /= total
                        avg_f1_after /= total
                        print(f'Benchmerk {dataset} with dataset {tss} has difference in avg f1 scores '
                              f'{avg_f1_after - avg_f1_before} for {model}')
