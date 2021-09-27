from matplotlib import pyplot
from matplotlib.lines import Line2D
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import cross_val_score
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import matplotlib.pyplot as plt
import skopt
import numpy as np


def confusion_visualization(x, y, true_val, pred_val, dataset, name, filename, datatype):
    tp = [(x[i], y[i]) for i in range(len(true_val)) if true_val[i] == pred_val[i] == 1]
    tn = [(x[i], y[i]) for i in range(len(true_val)) if true_val[i] == pred_val[i] == 0]
    fp = [(x[i], y[i]) for i in range(len(true_val)) if true_val[i] == 0 and pred_val[i] == 1]
    fn = [(x[i], y[i]) for i in range(len(true_val)) if true_val[i] == 1 and pred_val[i] == 0]

    fig, ax = plt.subplots()
    ax.scatter([t[0] for t in tp], [t[1] for t in tp], color='k', s=1)
    ax.scatter([t[0] for t in tn], [t[1] for t in tn], color='k', s=1)
    ax.scatter([t[0] for t in fp], [t[1] for t in fp], color='r', s=5)
    ax.scatter([t[0] for t in fn], [t[1] for t in fn], color='y', s=5)

    legend_elements = [Line2D([0], [0], color='k', lw=2, label='Correct'),
                       Line2D([0], [0], marker='o', color='r', markersize=5, label='FP'),
                       Line2D([0], [0], marker='o', color='y', markersize=5, label='FN')]

    ax.legend(handles=legend_elements, loc='best')

    pyplot.savefig(f'results/imgs/{dataset}/{datatype}/{name}/{name}_{filename}.png')
    pyplot.clf()
    plt.close('all')

    # errors = pd.Series(v for _, v in sorted(fp + fn, key=lambda x: x[0]))
    # is_whitenoise(errors)


def is_whitenoise(serie):
    print(serie)
    serie.hist()
    pyplot.show()

    autocorrelation_plot(serie)
    pyplot.show()
    pyplot.clf()
