# Methods for analyzing TS features via different extractors: yahoo way, catch22 (fforma is in different project)
import os
import tsfresh
from statsmodels.tsa.seasonal import seasonal_decompose
import tsfresh.feature_extraction.feature_calculators as fc
import nolds
import pandas as pd
import numpy as np
from scipy import stats, fft, fftpack, signal
from scipy.special import boxcox1p
from catch22 import catch22_all


# Yahoo analysis
from analysis.preanalysis import periodicity_analysis


def series_analysis(data):
    period = periodicity_analysis(data)
    values = data['value']
    posdata = values[values > 0]
    bcdata, lam = stats.boxcox(posdata)
    valuest = boxcox1p(values, lam)

    decompose = seasonal_decompose(values.interpolate(), period=period)
    seasonality = 1 - (np.var(valuest - decompose.seasonal - decompose.trend) / np.var(valuest - decompose.trend))
    trend = 1 - (np.var(valuest - decompose.seasonal - decompose.trend) / np.var(valuest - decompose.seasonal))
    # Represents long-range dependence.
    autocrr = fc.autocorrelation(values, lag=period)
    # A non-linear time-series contains complex dynamics that are usually not represented by linear models.
    non_lin = fc.c3(values, lag=period)
    # Measures symmetry, or more precisely, the lack of symmetry.
    skewness = fc.skewness(values)
    # Measures if the data are peaked or flat, relative to a normal distribution
    kurtosis = fc.kurtosis(values)
    # A measure of long-term memory of time series
    hurst = nolds.hurst_rs(values)
    # chaos of a system
    lyapunov_e = nolds.lyap_r(values)

    if lyapunov_e > 0.0:
        lyapunov_e = 1.0
    elif lyapunov_e < 0.0:
        lyapunov_e = -1.0

    tf = tsfresh.extract_features(data, column_id='timestamp')
    # print(tf)

    return trend, seasonality, autocrr, non_lin, skewness, kurtosis, hurst, lyapunov_e, tf


# CATCH 22 analysis
def analyse_dataset_catch22(dataset, root_path):
    df = pd.DataFrame([])
    idx = 0
    data_path = root_path + '/datasets/' + dataset[0] + '/' + dataset[1] + '/'
    for filename in os.listdir(data_path):
        f = os.path.join(data_path, filename)
        if os.path.isfile(f):
            print(filename)
            data = pd.read_csv(f)
            res = catch22_all(data['value'].tolist())
            print(res)
            df.loc[idx, 'ts'] = filename
            for name, val in zip(res['names'], res['values']):
                df.loc[idx, name] = val
            idx += 1
            df.to_csv(f'results/ts_properties/{dataset[0]}_{dataset[1]}_features_c22.csv')

    # for filename in os.listdir(data_path):
    #     f = os.path.join(data_path, filename)
    #     if os.path.isfile(f):
    #         print(filename)
    #         data = pd.read_csv(f)
    #         if len(ts_set) != 0:
    #             ts_set.append([[x] for x in data['value'].to_list()][:sz])
    #             # y = np.append(y, data['is_anomaly'].to_numpy(), axis=1)
    #         else:
    #             ts_set = [[[x] for x in data['value'].to_list()[:sz]]]
    #             # y = np.array(data['is_anomaly'].to_numpy())
    #
    # ts_set = np.array(ts_set)
    # c22 = Catch22()
    # c22.fit(ts_set)
    # transformed_data = c22.transform(ts_set)
    #
    # print(transformed_data.head())
    # transformed_data.to_csv(f'results/ts_properties/{dataset[0]}_{dataset[1]}_features_c22.csv')


# extract influencial featuresfrom fforma/catch22 based on NAB mechanical performance
def extract_influencial_features_NAB():
    res_df = pd.DataFrame([])
    machine_data_path = f'results/ts_properties/f1_to_ts_machine_nab_corr.csv'
    df = pd.read_csv(machine_data_path)
    influencial_features = {}
    for feature in df.columns:
        if feature != 'model':
            if df[feature].max() >= 0.5:
                influencial_features[feature] = 'pos'
            elif df[feature].min() <= -0.5:
                influencial_features[feature] = 'neg'

    idx = 0
    for dataset, subsets in [('NAB', ['relevant'])]:
        #, ('yahoo', ['real', 'synthetic', 'A3Benchmark', 'A4Benchmark'])]:
        for tss in subsets:
            print(tss)
            ts_properties_c22 = pd.read_csv(f'results/ts_properties/{dataset}_{tss}_features_c22.csv').set_index('ts')
            ts_properties_fforma = pd.read_csv(f'results/ts_properties/{dataset}_{tss}_features_fforma.csv').set_index(
                'ts')
            sz = 60
            if dataset == 'kpi':
                sz = 1024
            dbscan = pd.read_csv(
                f'C:\\Users\\oxifl\\Desktop\\thesis_res\\2_opt_no_updt\\{dataset}\\'
                f'win_{sz}\\{tss}\\{dataset}_{tss}_stats_dbscan.csv')

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
                    lstm['f1'] = [0 for i in range(dbscan.shape[0])]

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
                    sarima['f1'] = [0 for i in range(dbscan.shape[0])]

            sr = pd.read_csv(
                f'C:\\Users\\oxifl\\Desktop\\thesis_res\\2_opt_no_updt\\{dataset}\\'
                f'win_{sz}\\{tss}\\{dataset}_{tss}_stats_sr.csv')

            machine_data_path = f'datasets/machine_metrics/{tss}/'
            # Collect all influencial features of TS from dataset
            for f_dbscan, f_lstm, f_sarima, f_sr, s_dbscan, s_lstm, s_sarima, s_sr, ts in \
                    zip(dbscan['f1'], lstm['f1'], sarima['f1'], sr['f1'],
                        dbscan['specificity'], lstm['specificity'],
                        sarima['specificity'], sr['specificity'],
                        dbscan['dataset']):

                rep_val = 'f1'
                # if str(ts) != 'nan' and 'flatline' not in ts:
                if str(ts) != 'nan' and ts + '.csv' in os.listdir(machine_data_path):

                    for col in ts_properties_c22.columns:
                        if col in influencial_features.keys():
                            res_df.loc[idx, col] = ts_properties_c22.loc[ts + '.csv', col]

                    for col in ts_properties_fforma.columns:
                        if col in influencial_features.keys():
                            res_df.loc[idx, col] = ts_properties_fforma.loc[ts + '.csv', col]

                    idx += 1

            mean_res_df = pd.DataFrame([])
            idx = 0
            # collect means and direction of influence of import features
            for col in influencial_features.keys():
                mean_res_df.loc[idx, 'feature'] = col
                mean_res_df.loc[idx, 'mean'] = res_df[col].mean()
                mean_res_df.loc[idx, 'std'] = np.std(res_df[col].tolist())
                mean_res_df.loc[idx, 'influence'] = influencial_features[col]
                idx += 1

            mean_res_df.set_index('feature', inplace=True)
            print(mean_res_df)
            # record performance on 'promising' TS via influencial features

            for ft in influencial_features.keys():
                final_df = pd.DataFrame([])
                idx = 0

                for f_dbscan, f_lstm, f_sarima, f_sr, s_dbscan, s_lstm, s_sarima, s_sr, ts in \
                        zip(dbscan['f1'], lstm['f1'], sarima['f1'], sr['f1'],
                            dbscan['specificity'], lstm['specificity'],
                            sarima['specificity'], sr['specificity'],
                            dbscan['dataset']):

                    rep_val = 'f1'
                    if str(ts) != 'nan' and 'flatline' not in ts:
                    # if str(ts) != 'nan' and ts + '.csv' in os.listdir(machine_data_path):
                        series = pd.read_csv('datasets/' + dataset + '/' + tss + '/' + ts + '.csv')
                        series.rename(columns={'timestamps': 'timestamp', 'anomaly': 'is_anomaly'}, inplace=True)
                        data_train, data_test = np.array_split(series, 2)
                        if data_test['is_anomaly'].tolist().count(1) == 0:
                            rep_val = 'specificity'

                        for col in influencial_features.keys():
                            if col in ts_properties_c22.columns:
                                ts_use = ts_properties_c22
                            else:
                                ts_use = ts_properties_fforma

                            if (mean_res_df.loc[col, 'influence'] == 'pos' and
                                mean_res_df.loc[col, 'mean'] + mean_res_df.loc[col, 'std'] <= ts_use.loc[ts + '.csv', col]) \
                                    or (mean_res_df.loc[col, 'influence'] == 'neg' and
                                        mean_res_df.loc[col, 'mean'] - mean_res_df.loc[col, 'std'] >= ts_use.loc[ts + '.csv', col]):
                                if rep_val == 'f1':
                                    final_df.loc[idx, 'dbscan_score'] = f_dbscan
                                    final_df.loc[idx, 'lstm_score'] = f_lstm
                                    final_df.loc[idx, 'sarima_score'] = f_sarima
                                    final_df.loc[idx, 'sr_score'] = f_sr
                                # else:
                                #     final_df.loc[idx, 'dbscan_score'] = s_dbscan
                                #     final_df.loc[idx, 'lstm_score'] = s_lstm
                                #     final_df.loc[idx, 'sarima_score'] = s_sarima
                                #     final_df.loc[idx, 'sr_score'] = s_sr

                                    final_df.loc[idx, 'ts'] = ts
                                    final_df.loc[idx, 'num_anomalies'] = data_test['is_anomaly'].tolist().count(1)

                                    idx += 1
                                    break

                final_df.to_csv(f'results/ts_properties/corrs/performance_to_{ft}_machine.csv')
