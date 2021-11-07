import copy

import scipy
import tsfresh
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from scipy.interpolate import splrep
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.pipeline import make_pipeline
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

# from utils import handle_missing_values_kpi


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


def series_analysis(data):
    values = data['value']
    posdata = values[values > 0]
    bcdata, lam = stats.boxcox(posdata)
    valuest = boxcox1p(values, lam)

    decompose = seasonal_decompose(values.interpolate(), period=12)
    seasonality = 1 - (np.var(valuest - decompose.seasonal - decompose.trend) / np.var(valuest - decompose.trend))
    trend = 1 - (np.var(valuest - decompose.seasonal - decompose.trend) / np.var(valuest - decompose.seasonal))
    # Represents long-range dependence.
    autocrr = fc.autocorrelation(values, lag=20)
    # A non-linear time-series contains complex dynamics that are usually not represented by linear models.
    non_lin = fc.c3(values, lag=20)
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
    c22 = Catch22()
    data_train['value'] = data_train['value'].apply(lambda x: np.array([x]))
    c22.fit(data_train[['value']], data_train['is_anomaly'])
    transformed_data = c22.transform(data_train[['value']])

    print(transformed_data.head())
    transformed_data.to_csv(f'results/ts_properties/test_filtered_features_c22.csv')
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
