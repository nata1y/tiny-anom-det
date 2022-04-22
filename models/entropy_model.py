import copy
import math

import ordpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import antropy as ant
import scipy.interpolate
from sklearn.metrics import f1_score, hamming_loss, confusion_matrix


def entropy_modelling():
    q = np.arange(0.01, 100, 0.01).tolist()
    interval = 100
    # f1_scores = pd.read_csv(f'results/ts_properties/entropies/entropy_vs_window_w_factor_non_nan.csv')
    # f1_scores = f1_scores[f1_scores['dataset'] == 'NAB']
    # # f1_scores['factor'] = f1_scores['factor'].apply(lambda x: round(x, 1))
    # # f1_scores.drop_duplicates(inplace=True, subset=['factor', 'dataset', 'window'], keep='last')
    # x = sorted(f1_scores['window'].unique())
    # y = sorted(f1_scores['factor'].unique())
    # z = f1_scores['f1-score'].to_numpy()
    #
    # # Z = []
    # # for wdw in sorted(f1_scores['window'].unique().tolist()):
    # #     if wdw > 5:
    # #         Z.append(f1_scores[f1_scores['window'] == wdw].sort_values(by='factor', ascending=True)['f1-score'].tolist())
    # #
    # # plt.imshow(np.array(Z), cmap=plt.cm.magma, interpolation='bilinear', origin="lower")
    # plt.scatter(f1_scores['window'].tolist(), f1_scores['factor'].tolist(), c=z, cmap=plt.cm.magma)
    # plt.xlabel('Window')
    # plt.xticks(x)
    # plt.yticks(y)
    # plt.ylabel('Factor')
    # plt.colorbar()
    # plt.savefig(f'results/ts_properties/imgs/entropy_analysis/entropy_colormap_NAB.png')
    # quit()
    entropies_no_anomalies = []
    # step = 100

    # for entropy_name in ['spectral_entropy', 'value_decomposition_entropy', 'approximate_entropy',
    #                      'sample_entropy', 'permutation_entropy', 'renyi', 'tsallis']:
    for wdw in range(100, 5, -5):
        for entropy_name in ['value_decomposition_entropy']:
            # approximate & value decompose
            for factor in np.arange(0.5, 4.1, 0.1):
                factor = round(factor, 1)
                print(f'Doing entropy {wdw}, {factor}')
                for dataset, subsets in [('yahoo', 'real'), ('yahoo', 'synthetic'),
                                         ('yahoo', 'A4Benchmark'), ('yahoo', 'A3Benchmark')]:
                    # print(f1_scores[(f1_scores['factor'] == factor) & (f1_scores['window'] == wdw) & (f1_scores['dataset'] == dataset)])
                    if not f1_scores[(f1_scores['factor'] == factor) & (f1_scores['window'] == wdw) & (f1_scores['dataset'] == dataset)].empty:
                    #     continue
                    # for tss in subsets:
                        hcs = pd.DataFrame([])
                        train_data_path = 'datasets/' + dataset + '/' + subsets + '/'

                        res = pd.DataFrame([])
                        for filename in os.listdir(train_data_path):
                            print(filename)
                            f = os.path.join(train_data_path, filename)
                            ts = pd.read_csv(f)

                            ts2 = None
                            if dataset == 'kpi':
                                ts2 = pd.read_csv('datasets/' + dataset + '/test/' + filename)
                            ts.rename(columns={'timestamps': 'timestamp', 'anomaly': 'is_anomaly'}, inplace=True)

                            max_qs, max_cs = [], []
                            batches_anomalies = []
                            entropies = []
                            entropy_differences = []
                            boundaries_bottom, boundaries_up = [], []
                            collected_entropies = []
                            mean_entropy, std_entropy, boundary_up, boundary_bottom = None, None, None, None
                            y_predicted = []
                            y_true = []

                            if dataset != 'kpi':
                                ts1, ts2 = np.array_split(copy.deepcopy(ts), 2)
                            else:
                                ts1 = ts

                            step = wdw
                            try:
                                for start in range(0, ts1.shape[0], step):
                                    window = ts1.iloc[start:start+step]
                                    # curve = ordpy.tsallis_complexity_entropy(window['value'].to_numpy(), dx=4, q=q)
                                    # max_q, max_c = np.max([a[0] for a in curve]), np.max([a[1] for a in curve])
                                    # max_cs.append(max_c)
                                    # max_qs.append(max_q)

                                    if True:
                                        if entropy_name == 'spectral_entropy':
                                            collected_entropies.append(ant.spectral_entropy(window['value'].to_numpy(),
                                                                                  sf=100, method='welch',
                                                                                  normalize=True))
                                        elif entropy_name == 'value_decomposition_entropy':
                                            collected_entropies.append(
                                                ant.svd_entropy(window['value'].to_numpy(), normalize=True))
                                        elif entropy_name == 'approximate_entropy':
                                            collected_entropies.append(ant.app_entropy(window['value'].to_numpy()))
                                        elif entropy_name == 'sample_entropy':
                                            collected_entropies.append(ant.sample_entropy(window['value'].to_numpy()))
                                        elif entropy_name == 'hjorth_entropy':
                                            collected_entropies.append(ant.hjorth_params(window['value'].to_numpy()))
                                        elif entropy_name == 'permutation_entropy':
                                            collected_entropies.append(ant.perm_entropy(window['value'].to_numpy(), normalize=True))
                                        elif entropy_name == 'tsallis':
                                            collected_entropies.append(
                                                ordpy.tsallis_entropy(window['value'].to_numpy()))
                                        elif entropy_name == 'renyi':
                                            collected_entropies.append(
                                                ordpy.renyi_entropy(window['value'].to_numpy()))

                                        if len(collected_entropies) > 1:
                                            entropy_differences.append(
                                                abs(collected_entropies[-1] - collected_entropies[-2]))

                                        if window['is_anomaly'].tolist().count(1) > 0:
                                            batches_anomalies.append(start // step)
                                        else:
                                            entropies_no_anomalies.append(collected_entropies[-1])

                                entropies = copy.deepcopy(collected_entropies)
                                mean_entropy = np.mean(np.array([v for v in collected_entropies if pd.notna(v)]))
                                std_entropy = np.std(np.array([v for v in collected_entropies if pd.notna(v)]))
                                boundary_bottom = mean_entropy - factor * std_entropy
                                boundary_up = mean_entropy + factor * std_entropy

                                # accepted_difference = np.mean(entropy_differences) + np.std(entropy_differences)

                                # boundary_bottom = np.min(entropies_no_anomalies) # * 0.9
                                # boundary_up = np.max(entropies_no_anomalies) # * 1.1

                                for start in range(0, ts2.shape[0], step):
                                    window = ts2.iloc[start:start + step]
                                    if window.shape[0] == step:
                                        if entropy_name == 'spectral_entropy':
                                            entropies.append(ant.spectral_entropy(window['value'].to_numpy(),
                                                                                  sf=100, method='welch', normalize=True))
                                        elif entropy_name == 'value_decomposition_entropy':
                                            entropies.append(ant.svd_entropy(window['value'].to_numpy(), normalize=True))
                                        elif entropy_name == 'approximate_entropy':
                                            entropies.append(ant.app_entropy(window['value'].to_numpy()))
                                        elif entropy_name == 'sample_entropy':
                                            entropies.append(ant.sample_entropy(window['value'].to_numpy()))
                                        elif entropy_name == 'hjorth_entropy':
                                            entropies.append(ant.hjorth_params(window['value'].to_numpy()))
                                        elif entropy_name == 'permutation_entropy':
                                            entropies.append(
                                                ant.perm_entropy(window['value'].to_numpy(), normalize=True))
                                        elif entropy_name == 'tsallis':
                                            entropies.append(
                                                ordpy.tsallis_entropy(window['value'].to_numpy()))
                                        elif entropy_name == 'renyi':
                                            entropies.append(
                                                ordpy.renyi_entropy(window['value'].to_numpy()))

                                        if window['is_anomaly'].tolist().count(1) > 0:
                                            batches_anomalies.append(len(collected_entropies) + (start // step))

                                        if mean_entropy:
                                            # rolling = pd.DataFrame(entropies[:-1]).rolling(5)
                                            # rolling_mean = rolling.mean()
                                            # rolling_std = rolling.std()
                                            #
                                            # boundary_bottom = (rolling_mean - 3 * rolling_std)[0].tolist()[-1]
                                            # boundary_up = (rolling_mean + 3 * rolling_std)[0].tolist()[-1]
                                            #
                                            # boundaries_bottom.append(boundary_bottom)
                                            # boundaries_up.append(boundary_up)

                                            if boundary_bottom <= entropies[-1] <= boundary_up:
                                                y_predicted.append(0)
                                            else:
                                                y_predicted.append(1)
                                            #
                                            # if abs(entropies[-1] - entropies[-2]) <= accepted_difference:
                                            #     y_predicted.append(0)
                                            #     entropies_no_anomalies.append(entropies[-1])
                                            # else:
                                            #     y_predicted.append(1)
                                            #     entropies_no_anomalies.append(entropies[-1])

                                            if window['is_anomaly'].tolist().count(1) > 0:
                                                y_true.append(1)
                                            else:
                                                y_true.append(0)

                                # color = ['red' if i in batches_anomalies else 'blue' for i in range(len(entropies))]

                                # rolling = pd.DataFrame(entropies).rolling(3)
                                # rolling_mean = rolling.mean()
                                # rolling_std = rolling.std()

                                # plt.bar(list(range(len(entropies))), entropies, color=color)
                                # plt.fill_between(list(range(1, len(entropies))),
                                #                  list(map(lambda x: x-accepted_difference, entropies[:-1])),
                                #                  list(map(lambda x: x+accepted_difference, entropies[:-1])),
                                #                  color='b', alpha=.5)
                                # plt.fill_between(list(range(len(collected_entropies), len(entropies))),
                                #                  boundaries_bottom,
                                #                  boundaries_up,
                                #                  color='b', alpha=.5)
                                # plt.axvline(x=int(len(entropies) // 2), color='orange',
                                #             linestyle='--', label='train-test separation')
                                # plt.axhline(y=boundary_bottom, color='y', linestyle='--')
                                # plt.axhline(y=boundary_up, color='y', linestyle='--')

                                # plt.plot(range(len(entropies) - len(boundaries_bottom), len(entropies)),
                                #          boundaries_bottom, color='y')
                                # plt.plot(range(len(entropies) - len(boundaries_up), len(entropies)),
                                #          boundaries_up, color='y')

                                # plt.plot(list(range(len(entropies))), rolling_mean[0].tolist(),
                                #          label='rolling mean', color='crimson')
                                # plt.plot(list(range(len(entropies))), (rolling_mean + rolling_std)[0].tolist(),
                                #          label='upper_bound', color='y')
                                # plt.plot(list(range(len(entropies))), (rolling_mean - rolling_std)[0].tolist(),
                                #          label='lower bound', color='y')
                                #
                                # plt.legend()
                                # plt.ylabel(entropy_name)
                                # plt.xlabel('Batch')

                                # fig, axs = plt.subplots(2)
                                # axs[0].bar(list(range(len(max_cs))), max_qs, color=color)
                                # axs[0].axvline(x=int(ts.shape[0] // (2 * step)), color='k', linestyle='--')
                                #
                                # axs[0].set_ylabel('q_H*')
                                #
                                # axs[1].bar(list(range(len(max_cs))), max_cs, color=color)
                                # axs[1].axvline(x=int(ts.shape[0] // (2 * step)), color='k', linestyle='--')
                                #
                                # axs[1].set_ylabel('q_C*')
                                # axs[1].set_xlabel('Batch')

                                # plt.savefig(f'results/ts_properties/imgs/entropy_analysis/{entropy_name}_{dataset}_{tss}_{filename.replace(".csv", "")}_step_{step}_std.png')
                                # plt.clf()

                                try:
                                    tn, fp, fn, tp = confusion_matrix(y_true, y_predicted).ravel()
                                except:
                                    tn, fp, fn, tp = len([1 for i in range(len(y_true)) if y_true[i] == y_predicted[i] == 0]),\
                                                     len([1 for i in range(len(y_true)) if y_true[i] + 1 == y_predicted[i]]), \
                                                     0, 0

                                res = res.append({
                                    'ts': filename.replace('.csv', ''),
                                    'y_pred': y_predicted,
                                    'y_true': y_true,
                                    'f1': f1_score(y_true, y_predicted, average='binary'),
                                    'f1-macro': f1_score(y_true, y_predicted, average='macro'),
                                    'f1-micro': f1_score(y_true, y_predicted, average='micro'),
                                    'hamming': hamming_loss(y_true, y_predicted),
                                    'step': step,
                                    'entropy': entropy_name,
                                    'total_anomalies': y_true.count(1),
                                    'fp': fp,
                                    'fn': fn,
                                    'tp': tp,
                                    'tn': tn
                                }, ignore_index=True)
                            except Exception as e:
                                pass

                        # res.to_csv(f'results/ts_properties/entropies/{entropy_name}_{dataset}_{tss}_step_{wdw}.csv')
                        f1_scores = f1_scores.append({
                            'window': wdw,
                            'f1-score': np.mean(res['f1'].to_numpy()),
                            'f1-score-macro': np.mean(res['f1-macro'].to_numpy()),
                            'f1-score-micro': np.mean(res['f1-micro'].to_numpy()),
                            'dataset': dataset,
                            'factor': factor
                        }, ignore_index=True)
                        f1_scores.to_csv(f'results/ts_properties/entropies/entropy_vs_window_{dataset}_{subsets}.csv')
                # hcs.to_csv(f'results/ts_properties/permutation_analysis_{dataset}_{tss}.csv')
