import copy
import itertools
import os
import random

import distance
import scipy
import tsfresh
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from scipy.interpolate import splrep
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor
from statsmodels.tsa.seasonal import seasonal_decompose
import tsfresh.feature_extraction.feature_calculators as fc
import nolds
import pandas as pd
import numpy as np
from scipy import stats, fft, fftpack, signal
from scipy.special import boxcox1p
from sktime.transformations.panel.catch22 import Catch22
from tsfresh import select_features
from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh.utilities.dataframe_functions import impute
from catch22 import catch22_all
import seaborn as sns
# from utils import handle_missing_values_kpi
from utils import KL, intersection, fidelity, sq_euclidian
import dictances


def visualize(data):
    print(data['value'].describe())
    pyplot.plot(data['timestamp'], data['value'])
    pyplot.show()
    quit()

    data['value'].hist()
    pyplot.show()

    autocorrelation_plot(data['value'])
    pyplot.show()

    # res = seasonal_decompose(data['value'].interpolate(),
    #                          freq=52)
    # res.plot()


# Yahoo analysis
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


def full_analysis(data, dataset, datatype):
    # series_analysis(data)
    periodicity_analysis(data, dataset, datatype)


def periodicity_analysis(data_, dataset='', datatype=''):
    pyplot.plot(data_['timestamp'], data_['value'])
    pyplot.plot()

    pyplot.savefig(f'results/imgs/preanalysis/{dataset}_{datatype}_train.png')
    pyplot.clf()

    data = copy.deepcopy(data_)
    freq = data['timestamp'].tolist()[1] - data['timestamp'].tolist()[0]

    data['value'] = signal.detrend(data.value.values)
    data.set_index('timestamp', inplace=True)

    ####################################################################################################################
    ft_vals = fftpack.fft(data['value'].tolist())[1:]
    frequencies = fftpack.fftfreq(data['value'].shape[0], freq)
    periods = 1 / frequencies[1:]

    pyplot.figure()
    pyplot.plot(periods, abs(ft_vals), 'o')
    pyplot.xlim(0, freq * data.shape[0])
    pyplot.xlabel('Period')
    pyplot.ylabel('FFT freq')

    pyplot.savefig(f'results/imgs/preanalysis/{dataset}_{datatype}_periods_fft.png')
    most_probable_period = int(abs(periods[np.argmax(ft_vals)]) / freq)
    print(f'Found most likely periodicity {most_probable_period}')
    pyplot.clf()

    return most_probable_period


def full_analyzis(data_train):
    # c22 = Catch22()
    # data_train['value'] = data_train['value'].apply(lambda x: np.array([x]))
    # c22.fit(data_train[['value']], data_train['is_anomaly'])
    # transformed_data = c22.transform(data_train[['value']])
    #
    # print(transformed_data.head())
    # transformed_data.to_csv(f'results/ts_properties/test_filtered_features_c22.csv')
    ####################################################################################################################

    transformer = TSFreshFeatureExtractor(default_fc_parameters="efficient")
    extracted_features = transformer.fit_transform(data_train[['value']].to_numpy().reshape((data_train.shape[0], 1, 1)))
    print(extracted_features.head())
    extracted_features.to_csv(f'results/ts_properties/test_filtered_features_tf.csv')

    classifier = make_pipeline(
        TSFreshFeatureExtractor(default_fc_parameters="efficient", show_warnings=False), IsolationForest()
    )
    res = classifier.fit_predict(data_train[['value']].to_numpy().reshape((data_train.shape[0], 1, 1)))
    print(res)
    ####################################################################################################################

    # data_train['id'] = data_train.index
    # extracted_features = tsfresh.extract_features(data_train[['id', 'value', 'timestamp']], column_id='id',
    #                                               column_sort='timestamp', column_value="value")
    # print(extracted_features)
    # impute(extracted_features)
    #
    # relevance_table = calculate_relevance_table(extracted_features, data_train['is_anomaly'])
    # relevance_table = relevance_table[relevance_table.relevant]
    # relevance_table.sort_values("p_value", inplace=True)
    # print(relevance_table)
    # print(relevance_table.columns)
    # print(relevance_table.sort_values('p_value', ascending=False)[:11])

    quit()


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


# Hyndman analysis
# def series_analysis_fforma(data):
#
#     period = periodicity_analysis(data)
#     values = data['value']
#
#     seasonality = seasonal_decompose(values.interpolate(), period=period).seasonal
#     X_train_df, y_train_df, X_test_df, y_test_df = prepare_m4_data(dataset, './data', 100)
#
#     y_holdout_train_df, y_val_df = temp_holdout(y_train_df, validation_periods)
#     features = tsfeatures(y_holdout_train_df, seasonality)
#
#     print(features)


def compare_dataset_properties():
    features = pd.read_csv(f'results/ts_properties/yahoo_real_features_fforma.csv').columns
    idx = 0
    for feature in features:
        print(feature)
        if feature not in ['ts', 'Unnamed: 0', 'arch_acf', 'garch_acf', 'arch_r2', 'garch_r2', 'hw_alpha', 'hw_beta',
                           'hw_gamma']:
            transform_full = pd.DataFrame([])

            for dataset in [('yahoo', 'real'), ('yahoo', 'synthetic'), ('yahoo', 'A3Benchmark'), ('yahoo', 'A4Benchmark'),
                        ('NAB', 'relevant'), ('kpi', 'train')]:

                data = pd.read_csv(f'results/ts_properties/{dataset[0]}_{dataset[1]}_features_fforma.csv')
                transform = pd.DataFrame([])
                transform[feature] = data[feature]
                transform['Dataset'] = dataset[0] + '_' + dataset[1] if dataset[0] != 'yahoo' else dataset[1]
                transform_full = pd.concat([transform_full, transform])

            sns.stripplot(x="Dataset", y=feature, data=transform_full, zorder=1)
            labels = [e.get_text() for e in pyplot.gca().get_xticklabels()]
            ticks = pyplot.gca().get_xticks()
            w = 0.1
            for idx, datas in enumerate(labels):
                idx = labels.index(datas)
                pyplot.hlines(transform_full[transform_full['Dataset'] == datas][feature].mean(), ticks[idx] - w,
                              ticks[idx] + w, color='k', linestyles='solid', linewidth=3.0, zorder=2)

            pyplot.savefig(f'results/ts_properties/imgs/{feature}_fforma.png')
            pyplot.clf()


def calculate_dists():
    random.seed(30)
    features = pd.read_csv(f'results/ts_properties/yahoo_real_features_c22.csv').columns
    df = pd.DataFrame([])
    idx = 0
    dfs = [('yahoo', 'real'), ('yahoo', 'synthetic'), ('yahoo', 'A3Benchmark'), ('yahoo', 'A4Benchmark'),
           ('NAB', 'relevant'), ('kpi', 'train')]
    exclude = ['ts', 'Unnamed: 0', 'arch_acf', 'garch_acf', 'arch_r2', 'garch_r2', 'hw_alpha', 'hw_beta', 'hw_gamma']

    for feature in features:
        if feature not in exclude:
            print(feature)
            for dataset1 in dfs:
                data1 = pd.read_csv(f'results/ts_properties/{dataset1[0]}_{dataset1[1]}_features_c22.csv')
                data1.fillna(0.0, inplace=True)
                for dataset2 in dfs[dfs.index(dataset1):]:
                    if dataset1 != dataset2:
                        data2 = pd.read_csv(f'results/ts_properties/{dataset2[0]}_{dataset2[1]}_features_c22.csv')
                        data2.fillna(0.0, inplace=True)

                        d1, d2 = data1[feature].to_numpy(), data2[feature].to_numpy()

                        scaler = MinMaxScaler()
                        d1 = list(scaler.fit_transform(d1.reshape(-1, 1)).flatten())
                        scaler = MinMaxScaler()
                        d2 = list(scaler.fit_transform(d2.reshape(-1, 1)).flatten())

                        d1, _ = np.histogram(d1, bins=100)
                        d2, _ = np.histogram(d2, bins=100)

                        wdist = wasserstein_distance(d1, d2)
                        edist = np.linalg.norm(d1 - d2)
                        sdist = distance.sorensen(d1, d2)
                        kldist = KL(d1, d2)
                        ipdist = np.inner(d1, d2)
                        fdist = fidelity(d1, d2)
                        sedist = sq_euclidian(d1, d2)
                        idist = intersection(d1, d1)

                        df.loc[idx, 'feature'] = feature
                        df.loc[idx, 'dataset1'] = dataset1[0] + '_' + dataset1[1]
                        df.loc[idx, 'dataset2'] = dataset2[0] + '_' + dataset2[1]
                        df.loc[idx, 'distance_wasserstein'] = wdist
                        df.loc[idx, 'distance_euclidian'] = edist
                        df.loc[idx, 'distance_sorensen'] = sdist
                        df.loc[idx, 'distance_kl'] = kldist
                        df.loc[idx, 'distance_inner_prod'] = ipdist
                        df.loc[idx, 'distance_fidelity'] = fdist
                        df.loc[idx, 'distance_intersection'] = idist
                        df.loc[idx, 'distance_squared_euclidian'] = sedist
                        idx += 1

    df.to_csv(f'results/ts_properties/features_was_dist_c22.csv')


def dist_between_sets():
    dists_p_f = pd.read_csv(f'results/ts_properties/features_was_dist_c22.csv')
    df = pd.DataFrame([])
    dfs = [('yahoo', 'real'), ('yahoo', 'synthetic'), ('yahoo', 'A3Benchmark'), ('yahoo', 'A4Benchmark'),
           ('NAB', 'relevant'), ('kpi', 'train')]
    dists = itertools.combinations(dfs, 2)

    idx = 0
    for s in list(dists):
        s1, s2 = s
        df.loc[idx, 'from'] = s1[0] + '_' + s1[1]
        df.loc[idx, 'to'] = s2[0] + '_' + s2[1]
        df.loc[idx, 'norm_sum_wasserstein'] = 0.0
        df.loc[idx, 'norm_sum_euclidian'] = 0.0
        df.loc[idx, 'norm_sum_sorensen'] = 0.0
        df.loc[idx, 'norm_sum_kl'] = 0.0
        df.loc[idx, 'norm_sum_inner_prod'] = 0.0
        df.loc[idx, 'norm_sum_fidelity'] = 0.0
        df.loc[idx, 'norm_sum_intersection'] = 0.0
        df.loc[idx, 'norm_sum_squared_euclidian'] = 0.0
        idx += 1

    for feature in dists_p_f['feature'].unique().tolist():
        print(feature)
        for dist in ['wasserstein', 'euclidian', 'sorensen', 'kl', 'inner_prod', 'fidelity',
                     'intersection', 'squared_euclidian']:
            snippest = dists_p_f[dists_p_f['feature'] == feature].reset_index()
            mval = np.max(snippest['distance_' + dist].tolist())
            if mval == 0.0:
                mval = 0.00000001
            idx = 0
            dists = itertools.combinations(dfs, 2)

            for s in dists:
                s1, s2 = s
                val = 0.0
                sub1 = snippest[snippest['dataset1'] == s1[0] + '_' + s1[1]]
                sub2 = snippest[snippest['dataset1'] == s2[0] + '_' + s2[1]]
                sub1 = sub1[sub1['dataset2'] == s2[0] + '_' + s2[1]]
                sub2 = sub2[sub2['dataset2'] == s1[0] + '_' + s1[1]]
                if sub1.shape[0] > 0:
                    val = sub1['distance_' + dist].tolist()[0]
                elif sub2.shape[0] > 0:
                    val = sub2['distance_' + dist].tolist()[0]

                df.loc[idx, 'norm_sum_' + dist] = df.loc[idx, 'norm_sum_' + dist] + val / mval
                idx += 1

    df.to_csv(f'results/ts_properties/dataset_was_dist_c22.csv')


# compare orderings via different distance measures
def compare_dataset_distances():
    for features in ['fforma', 'c22']:
        df = pd.DataFrame([], columns=['wasserstein', 'euclidian', 'sorensen', 'kl', 'inner_prod', 'fidelity',
                                       'intersection', 'squared_euclidian', 'distance_name'])
        df['distance_name'] = ['wasserstein', 'euclidian', 'sorensen', 'kl', 'inner_prod', 'fidelity',
                               'intersection', 'squared_euclidian']
        df.set_index('distance_name', inplace=True)
        data = pd.read_csv(f'results/ts_properties/dataset_was_dist_{features}.csv')
        for dist in ['wasserstein', 'euclidian', 'sorensen', 'kl', 'inner_prod', 'fidelity',
                     'intersection', 'squared_euclidian']:
            for dist2 in ['wasserstein', 'euclidian', 'sorensen', 'kl', 'inner_prod', 'fidelity',
                           'intersection', 'squared_euclidian']:
                print(data)
                tau, p_value = stats.kendalltau(data['norm_sum_' + dist], data['norm_sum_' + dist2])
                df.loc[dist, dist2] = tau

        df.to_csv(f'results/ts_properties/ranking_similarities_via_dists_{features}.csv')
