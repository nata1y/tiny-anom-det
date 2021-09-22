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
from scipy import stats
from scipy.special import boxcox1p


def visualize(data):
    print(data['value'].describe())
    pyplot.plot(data['timestamp'], data['value'])
    pyplot.show()

    data['value'].hist()
    pyplot.show()

    autocorrelation_plot(data['value'])
    pyplot.show()

    # res = seasonal_decompose(data['value'].interpolate(),
    #                          freq=52)
    # res.plot()


# def period(x, y):
#     # 3 knots
#     spl = splrep(x, y, t=3)


def series_analysis(data):
    values = data['value']
    posdata = values[values > 0]
    bcdata, lam = stats.boxcox(posdata)
    print(lam)
    valuest = boxcox1p(values, lam)
    # Frequency
    # periodicity = period(values, times)
    decompose = seasonal_decompose(values, period=2)
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
    # A measure of the rate of divergence of nearby trajectories.
    lyapunov = 0 #nolds.lyap_e(values)

    return trend, seasonality, autocrr, non_lin, skewness, kurtosis, hurst, lyapunov

