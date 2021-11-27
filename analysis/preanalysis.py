import copy
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
import numpy as np
from scipy import fftpack, signal


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
