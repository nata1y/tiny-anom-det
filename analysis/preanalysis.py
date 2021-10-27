import scipy
import tsfresh
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from scipy.interpolate import splrep
from statsmodels.tsa.seasonal import seasonal_decompose
import tsfresh.feature_extraction.feature_calculators as fc
import nolds
import pandas as pd
import numpy as np
from scipy import stats, fft, fftpack
from scipy.special import boxcox1p


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

    return trend, seasonality, autocrr, non_lin, skewness, kurtosis, hurst, lyapunov_e


def full_analysis(data, dataset, datatype):
    # series_analysis(data)
    periodicity_analysis(data, dataset, datatype)


def periodicity_analysis(data, dataset, datatype):
    # result_mul = seasonal_decompose(data['value'], model='multiplicative', extrapolate_trend='freq', period=12)
    # deseasonalized = data.value.values / result_mul.seasonal
    # pyplot.plot(deseasonalized)
    # pyplot.plot()
    #
    # pyplot.savefig(f'results/imgs/preanalysis/{dataset}_{datatype}_deseasonalized_decompose.png')
    # pyplot.clf()

    freq = data['timestamp'].tolist()[1] - data['timestamp'].tolist()[0]

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

    print(most_probable_period)
    print(type(most_probable_period))
    result_mul = seasonal_decompose(data['value'], model='multiplicative', extrapolate_trend='freq', period=most_probable_period)
    deseasonalized = data.value.values / result_mul.seasonal
    pyplot.plot(deseasonalized)
    pyplot.plot()

    pyplot.savefig(f'results/imgs/preanalysis/{dataset}_{datatype}_deseasonalized_decompose2.png')
    pyplot.clf()

    return most_probable_period


