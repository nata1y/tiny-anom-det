# methods for scoring and ploting of predictor performances
import scipy
from matplotlib import pyplot
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import datetime
import scipy.stats as stats


def confusion_visualization(data_test, pred_val, dataset, datatype, filename, modelname):
    data_test = data_test[:7000]
    x, y, true_val = data_test['timestamp'].tolist(), data_test['value'].tolist(), data_test['is_anomaly'].tolist()
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
    # ax.scatter([t[0] for t in fp], [t[1] for t in fp], color='r', s=5, zorder=5)
    # ax.scatter([t[0] for t in fn], [t[1] for t in fn], color='y', s=5, zorder=5)

    # legend_elements = [Line2D([0], [0], color='k', lw=2, label='Correct'),
    #                    Line2D([0], [0], marker='o', color='r', markersize=5, label='FP'),
    #                    Line2D([0], [0], marker='o', color='y', markersize=5, label='FN')]

    # ax.legend(handles=legend_elements, loc='best')

    pyplot.savefig(f'results/imgs/{dataset}/{datatype}/{modelname}_{filename}.png')
    pyplot.clf()
    pyplot.close('all')
    plt.close('all')
    del fig
    del ax


def test_statistics(data, dataset, model):
    print(dataset, model)
    # A p-value larger than a chosen threshold (e.g. 5% or 1%) indicates that our observation is not so
    # unlikely to have occurred by chance. Therefore, we do not reject the null hypothesis of equal population means.
    # plt.hist(data['f1-noe'])
    # plt.show()
    _, pvalue = scipy.stats.wilcoxon(x=data['f1-noe'], y=data['f1-e'], mode='approx')
    print(pvalue)
