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
