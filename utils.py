import ast
import json
from math import sqrt

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow
from catch22 import catch22_all
from matplotlib import cm
from scipy.stats import ks_2samp

from analysis.postanalysis import confusion_visualization


# groupings of datset time series into various categories ############################################################
from settings import entropy_params

kpi_ts_same_granularity = [
    '0efb375b-b902-3661-ab23-9a0bb799f4e3',
    '301c70d8-1630-35ac-8f96-bc1b6f4359ea',
    '54350a12-7a9d-3ca8-b81f-f886b9d156fd',
    'a8c06b47-cc41-3738-9110-12df0ee4c721',
    'ab216663-dcc2-3a24-b1ee-2c3e550e06c9',
    'c02607e8-7399-3dde-9d28-8a8da5e5d251',
    'e0747cad-8dc8-38a9-a9ab-855b61f5551d',

    'cpu_utilization_asg_misconfiguration',
    'ec2_cpu_utilization_24ae8d',
    'ec2_cpu_utilization_53ea38',
    'ec2_cpu_utilization_5f5533',
    'ec2_cpu_utilization_77c1ca',
    'ec2_cpu_utilization_825cc2',
    'ec2_cpu_utilization_ac20cd',
    'ec2_cpu_utilization_c6585a',
    'ec2_cpu_utilization_fe7f93',
    'ec2_disk_write_bytes_1ef3de',
    'ec2_disk_write_bytes_c0d644',
    'ec2_network_in_257a54',
    'ec2_network_in_5abac7',
    'ec2_request_latency_system_failure'
]


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
    '1a_generated': ['art_daily_no_noise.csv', 'art_daily_perfect_square_wave.csv', 'art_daily_small_noise.csv',
                     'art_flatline.csv', 'art_noisy.csv', 'ec2_cpu_utilization_c6585a.csv', 'exchange-3_cpm_results.csv',
                     'occupancy_6005.csv', 'occupancy_t4013.csv', 'rds_cpu_utilization_cc0c53.csv',
                     'Twitter_volume_FB.csv', 'real_1.csv', 'real_10.csv', 'real_12.csv', 'real_16.csv', 'real_18.csv',
                     'real_21.csv', 'real_24.csv', 'real_25.csv', 'real_3.csv', 'real_30.csv', 'real_33.csv',
                     'real_35.csv', 'real_36.csv', 'real_4.csv', 'real_41.csv', 'real_42.csv', 'real_45.csv',
                     'real_49.csv', 'real_5.csv', 'real_50.csv', 'real_53.csv', 'real_54.csv', 'real_58.csv',
                     'real_59.csv', 'real_62.csv', 'real_64.csv', 'real_67.csv', 'real_8.csv', 'synthetic_100.csv',
                     'synthetic_11.csv', 'synthetic_13.csv', 'synthetic_14.csv', 'synthetic_19.csv', 'synthetic_21.csv',
                     'synthetic_23.csv', 'synthetic_25.csv', 'synthetic_27.csv', 'synthetic_28.csv', 'synthetic_30.csv',
                     'synthetic_31.csv', 'synthetic_33.csv', 'synthetic_34.csv', 'synthetic_35.csv', 'synthetic_37.csv',
                     'synthetic_39.csv', 'synthetic_41.csv', 'synthetic_42.csv', 'synthetic_43.csv', 'synthetic_44.csv',
                     'synthetic_46.csv', 'synthetic_47.csv', 'synthetic_48.csv', 'synthetic_49.csv', 'synthetic_5.csv',
                     'synthetic_51.csv', 'synthetic_53.csv', 'synthetic_55.csv', 'synthetic_56.csv', 'synthetic_57.csv',
                     'synthetic_58.csv', 'synthetic_59.csv', 'synthetic_60.csv', 'synthetic_61.csv', 'synthetic_62.csv',
                     'synthetic_63.csv', 'synthetic_65.csv', 'synthetic_67.csv', 'synthetic_69.csv', 'synthetic_7.csv',
                     'synthetic_70.csv', 'synthetic_71.csv', 'synthetic_72.csv', 'synthetic_73.csv', 'synthetic_74.csv',
                     'synthetic_75.csv', 'synthetic_76.csv', 'synthetic_77.csv', 'synthetic_79.csv', 'synthetic_81.csv',
                     'synthetic_83.csv', 'synthetic_84.csv', 'synthetic_85.csv', 'synthetic_86.csv', 'synthetic_87.csv',
                     'synthetic_88.csv', 'synthetic_89.csv', 'synthetic_9.csv', 'synthetic_90.csv', 'synthetic_91.csv',
                     'synthetic_93.csv', 'synthetic_95.csv', 'synthetic_97.csv', 'synthetic_98.csv', 'synthetic_99.csv',
                     'A3Benchmark-TS64.csv', 'A3Benchmark-TS66.csv'],
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

nab_grouping = {
    'artificialNoAnomaly': [
        'art_daily_nojump',
        'art_daily_no_noise',
        'art_daily_small_noise',
        'art_flatline',
        'art_noisy'
    ],
    'artificialWithAnomaly': [
        'art_daily_flatmiddle',
        'art_daily_jumpsdown',
        'art_daily_jumpsup',
        'art_daily_perfect_square_wave',
        'art_increase_spike_density',
        'art_load_balancer_spikes'
    ],
    'realAWSCloudwatch': [
        'ec2_cpu_utilization_24ae8d',
        'ec2_cpu_utilization_53ea38',
        'ec2_cpu_utilization_5f5533',
        'ec2_cpu_utilization_77c1ca',
        'ec2_cpu_utilization_825cc2',
        'ec2_cpu_utilization_ac20cd',
        'ec2_cpu_utilization_c6585a',
        'ec2_cpu_utilization_fe7f93',
        'ec2_disk_write_bytes_1ef3de',
        'ec2_disk_write_bytes_c0d644',
        'ec2_network_in_257a54',
        'ec2_network_in_5abac7',
        'grok_asg_anomaly',
        'iio_us-east-1_i-a2eb1cd9_NetworkIn',
        'rds_cpu_utilization_cc0c53',
        'rds_cpu_utilization_e47b3b',
        'elb_request_count_8c0756'
    ],
    'realAdExchange': [
        'exchange-2_cpc_results',
        'exchange-2_cpm_results',
        'exchange-3_cpc_results',
        'exchange-3_cpm_results',
        'exchange-4_cpc_results',
        'exchange-4_cpm_results'
    ],
    'realKnownCause': [
        'ec2_request_latency_system_failure',
        'machine_temperature_system_failure',
        'nyc_taxi',
        'rogue_agent_key_hold',
        'rogue_agent_key_updown',
        'ambient_temperature_system_failure',
        'cpu_utilization_asg_misconfiguration',
    ],
    'realTraffic': [
        'speed_6005',
        'speed_7578',
        'speed_t4013',
        'TravelTime_387',
        'TravelTime_451',
        'occupancy_6005',
        'occupancy_t4013'
    ],
    'realTweets':[
        'Twitter_volume_AAPL',
        'Twitter_volume_AMZN',
        'Twitter_volume_CRM',
        'Twitter_volume_CVS',
        'Twitter_volume_FB',
        'Twitter_volume_GOOG',
        'Twitter_volume_IBM',
        'Twitter_volume_KO',
        'Twitter_volume_PFE',
        'Twitter_volume_UPS'
                      ]
}


# LSTM create dataset helpers
def create_dataset(X, y, time_steps=60):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i:(i + time_steps)].values)

    return np.array(Xs), np.array(ys)


def create_dataset_keras(X, y, time_steps=60):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i:(i + time_steps)].values)

    return tensorflow.convert_to_tensor(Xs), tensorflow.convert_to_tensor(ys)


# SARIMA adj threshold
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


# kpi preproccessing of hdf datafiles, not used anymore
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


# label nab dataset with their windows, not used anymore
def preprocess_nab_labels(root_path):
    with open(root_path + '/datasets/NAB/labels/combined_windows.json') as json_file:
        labels_window = json.load(json_file)

    for attribute, values in labels_window.items():
        try:
            dataset = pd.read_csv(root_path + '/datasets/NAB/relevant/' + attribute.split('/')[-1])
            dataset.set_index('timestamp', inplace=True)
            for value in values:
                start, end = value[0], value[1]
                bool_series = ((dataset.index >= start) & (dataset.index <= end))
                dataset.loc[bool_series, 'is_anomaly'] = 1.0

            dataset.to_csv(root_path + '/datasets/NAB/windows/' + attribute.split('/')[-1])
        except:
            pass


# Below are various distance measures between probability distributions, ddeprecated
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


# feature analysis of large UCR dataset,
# was used to check whether features of anomaly detection benchmarks
# came from the same 'general 'distributions
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


# evaluation metrics for drift detectors, deprecated
def drift_metrics(y_true, y_pred):
    fp = 0
    latency = 0.0
    misses = 0
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
                latency += (idx - drift_start_idx)
                drift_start_idx = None

    if drift_start_idx:
        misses += 1

    if total_amount_drifts:
        latency = latency / total_amount_drifts
    return fp, latency, misses


# general plotting function for anomaly detectors
def plot_general(model, dataset, type, name, data_test, y_pred_total, filename, drift_windows):
    try:
        confusion_visualization(data_test['timestamp'].tolist(), data_test['value'].tolist(),
                                data_test['is_anomaly'].tolist(), y_pred_total,
                                dataset, name, filename.replace('.csv', ''), type, drift_windows)
    except Exception as e:
        raise e

    try:
        if name == 'sarima':
            model.plot(data_test[['timestamp', 'value']], dataset, type, filename, data_test, drift_windows)
        elif name == 'lstm':
            model.plot(data_test['timestamp'].tolist(), dataset, type, filename, data_test, drift_windows)
        elif name == 'sr':
            model.plot(dataset, type, filename, data_test, drift_windows)
    except Exception as e:
        raise e


def avg_batch_f1():
    for dataset, subsets in [('kpi', ['train'])]:
        for tss in subsets:
            for suf in ['e', 'noe']:
                d = entropy_params[f'{dataset}_{tss}']['window']
                path = f'results\\{dataset}_{subsets}_stats_sr_batched.csv'
                df_sr = pd.read_csv(path)
                path = f'results\\{dataset}_{subsets}_stats_sarima_batched.csv'
                df_sarima = pd.read_csv(path)
                path = f'results\\{dataset}_{subsets}_stats_lstm_per_batched.csv'
                df_lstm = pd.read_csv(path)
                for score in ['f1-batched']:
                    for ((idx_sr, row_sr), (idx_sarima, row_sarima), (idx_lstm, row_lstm)) \
                            in zip(df_sr.iterrows(), df_sarima.iterrows(), df_lstm.iterrows()):
                        res = {}
                        res['sr-' + suf] = ast.literal_eval(row_sr[score + '-' + suf])
                        res['sarima-' + suf] = ast.literal_eval(row_sarima[score + '-' + suf])
                        res['lstm-' + suf] = ast.literal_eval(row_lstm[score + '-' + suf])
                        l = []
                        plot_change(l, [], 'all', row_sarima['dataset'], dataset, score, suf, res, suf, [])


def loss_behavior():
    df = pd.read_csv('results/kpi_train_stats_IDPSO_ELM_B_batched.csv')
    print(df.columns)
    for idx, row in df.iterrows():
        for add, tp in [('dynamic_entropy', 'f1-batched-my_threshold')]:
            losses = ast.literal_eval(row[tp])
            helper = pd.DataFrame([])
            helper['Prediction loss'] = losses
            # you may want to change rolling window size for shorter ts
            helper['MA'] = helper.rolling(window=100).mean()
            plt.plot(range(len(losses)), helper['MA'].tolist(), marker='*', label='Rolling mean ' + add)

            # average prediction accuracy in the beginning and in the end
            print(np.mean(helper['MA'].to_numpy()[-100:]) - np.mean(helper['MA'].to_numpy()[100:200]))

        plt.xlabel('Time')
        plt.ylabel('Monitoring metrics')
        plt.legend()
        plt.savefig(f'results/imgs/change_performance/kpi/idpso-elm_{row["dataset"]}.png')
        plt.clf()
