# methods for scoring and ploting of predictor performances
from matplotlib import pyplot
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import datetime


def confusion_visualization(x, y, true_val, pred_val, dataset, name, filename, datatype, drift_windows):
    try:
        x = [datetime.datetime.strptime(x, '%m/%d/%Y %H:%M') for x in x]
    except:
        try:
            x = [datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in x]
        except:
            pass

    fp = [(x[i], y[i]) for i in range(len(true_val)) if true_val[i] == 0 and pred_val[i] == 1]
    fn = [(x[i], y[i]) for i in range(len(true_val)) if true_val[i] == 1 and pred_val[i] == 0]

    fig, ax = plt.subplots()
    ax.plot(x, y, color='grey', lw=0.5, zorder=0)
    ax.scatter([t[0] for t in fp], [t[1] for t in fp], color='r', s=5, zorder=5)
    ax.scatter([t[0] for t in fn], [t[1] for t in fn], color='y', s=5, zorder=5)

    legend_elements = [Line2D([0], [0], color='k', lw=2, label='Correct'),
                       Line2D([0], [0], marker='o', color='r', markersize=5, label='FP'),
                       Line2D([0], [0], marker='o', color='y', markersize=5, label='FN')]

    for wd in drift_windows:
        ax.axvspan(wd[0], wd[1], alpha=0.3, color='red')
    ax.legend(handles=legend_elements, loc='best')

    pyplot.savefig(f'results/imgs/{dataset}/{datatype}/{name}_{filename}.png')
    pyplot.clf()
    pyplot.close('all')
    plt.close('all')
    del fig
    del ax
