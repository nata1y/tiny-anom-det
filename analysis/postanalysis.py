from matplotlib import pyplot
from matplotlib.lines import Line2D
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_log_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import matplotlib.pyplot as plt


def confusion_visualization(x, y, true_val, pred_val, dataset, name, filename, datatype):
    tp = [(x[i], y[i]) for i in range(len(true_val)) if true_val[i] == pred_val[i] == 1]
    tn = [(x[i], y[i]) for i in range(len(true_val)) if true_val[i] == pred_val[i] == 0]
    fp = [(x[i], y[i]) for i in range(len(true_val)) if true_val[i] == 0 and pred_val[i] == 1]
    fn = [(x[i], y[i]) for i in range(len(true_val)) if true_val[i] == 1 and pred_val[i] == 0]

    fig, ax = pyplot.subplots()
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

    # errors = pd.Series(v for _, v in sorted(fp + fn, key=lambda x: x[0]))
    # is_whitenoise(errors)


def is_whitenoise(serie):
    print(serie)
    serie.hist()
    pyplot.show()

    autocorrelation_plot(serie)
    pyplot.show()
    pyplot.clf()


def plotSARIMAX(full_pred, y, dataset, datatype, filename, full_test_data):
    # 0 1970-01-01 00:00:01  0.000000
    #
    ax = y['1970':].plot(label='observed')

    idx = 1
    idx2 = 0
    for _, pred in enumerate(full_pred):
        pred_ci = pred.conf_int()
        pred.predicted_mean.plot(ax=ax, label=f'Window {idx} forecast', alpha=.7, figsize=(14, 7))
        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='k', alpha=.2)

        for tm, row in pred_ci.iterrows():
            if (row['lower value'] > y['value'].tolist()[idx2] or \
                    y['value'].tolist()[idx2] > row['upper value']) and full_test_data['is_anomaly'].tolist()[idx2] == 0:
                ax.scatter(tm, y.loc[tm, 'value'], color='r')
            if (row['lower value'] <= y['value'].tolist()[idx2] <= row['upper value']) \
                    and full_test_data['is_anomaly'].tolist()[idx2] == 1:
                ax.scatter(tm, y.loc[tm, 'value'], color='darkmagenta')
            idx2 += 1

        idx += 1

    ax.set_xlabel('Date')
    ax.set_ylabel('Values')
    plt.legend()
    plt.savefig(f'results/imgs/{dataset}/{datatype}/sarima/sarima_{filename.replace(".csv", "")}_full.png')

    return ax
