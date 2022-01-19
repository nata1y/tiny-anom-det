# analyze extracted TS properties to model performance correlations
import copy

import nolds
import pandas as pd
import os
import ordpy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import f1_score, hamming_loss
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import antropy as ant


def machine_ts_to_features_correlation():
    correlation_df = pd.DataFrame([])
    idx = 0
    exclude = ['ts', 'Unnamed: 0', 'arch_acf', 'garch_acf', 'arch_r2', 'garch_r2', 'hw_alpha', 'hw_beta', 'hw_gamma']
    for dataset, subsets in [('yahoo', ['real', 'synthetic', 'A3Benchmark', 'A4Benchmark']), ('NAB', ['relevant'])]:
        #, ('kpi', ['train'])]:
        #, ('yahoo', ['real', 'synthetic', 'A3Benchmark', 'A4Benchmark'])]:
        for tss in subsets:
            print(tss)
            ts_properties_c22 = pd.read_csv(f'results/ts_properties/{dataset}_{tss}_features_c22.csv').set_index('ts')
            ts_properties_fforma = pd.read_csv(f'results/ts_properties/{dataset}_{tss}_features_fforma.csv').set_index('ts')
            sz = 60
            if dataset == 'kpi':
                sz = 1024
            dbscan = pd.read_csv(
                f'C:\\Users\\oxifl\\Desktop\\thesis_res\\2_opt_no_updt\\{dataset}\\'
                f'win_{sz}\\{tss}\\{dataset}_{tss}_stats_dbscan.csv')

            try:
                lstm = pd.read_csv(
                    f'C:\\Users\\oxifl\\Desktop\\thesis_res\\2_opt_no_updt\\{dataset}\\'
                    f'win_{sz}\\{tss}\\{dataset}_{tss}_stats_lstm.csv')
            except:
                try:
                    lstm = pd.read_csv(
                        f'C:\\Users\\oxifl\\Desktop\\thesis_res\\1_no_opt_no_updt\\{dataset}\\'
                        f'min_metrics\\win_{sz}\\{tss}\\{dataset}_{tss}_stats_lstm.csv')
                except:
                    lstm = pd.DataFrame([])
                    lstm['f1'] = [0 for i in range(dbscan.shape[0])]

            try:
                sarima = pd.read_csv(
                    f'C:\\Users\\oxifl\\Desktop\\thesis_res\\2_opt_no_updt\\{dataset}\\'
                    f'win_{sz}\\{tss}\\{dataset}_{tss}_stats_sarima.csv')
            except:
                try:
                    sarima = pd.read_csv(
                        f'C:\\Users\\oxifl\\Desktop\\thesis_res\\1_no_opt_no_updt\\{dataset}\\'
                        f'min_metrics\\win_{sz}\\{tss}\\{dataset}_{tss}_stats_sarima.csv')
                except Exception as e:
                    sarima = pd.DataFrame([])
                    sarima['f1'] = [0 for i in range(dbscan.shape[0])]

            sr = pd.read_csv(
                f'C:\\Users\\oxifl\\Desktop\\thesis_res\\2_opt_no_updt\\{dataset}\\'
                f'win_{sz}\\{tss}\\{dataset}_{tss}_stats_sr.csv')

            machine_data_path = f'datasets/machine_metrics/{tss}/'
            for f_dbscan, f_lstm, f_sarima, f_sr, s_dbscan, s_lstm, s_sarima, s_sr, ts in \
                    zip(dbscan['f1'], lstm['f1'], sarima['f1'], sr['f1'],
                                                            dbscan['specificity'], lstm['specificity'],
                                                            sarima['specificity'], sr['specificity'],
                                                            dbscan['dataset']):

                rep_val = 'f1'
                if str(ts) != 'nan' and 'flatline' not in ts:
                # if str(ts) != 'nan' and ts + '.csv' in os.listdir(machine_data_path):
                    series = pd.read_csv('datasets/' + dataset + '/' + tss + '/' + ts + '.csv')
                    series.rename(columns={'timestamps': 'timestamp', 'anomaly': 'is_anomaly'}, inplace=True)
                    if dataset != 'kpi':
                        data_train, data_test = np.array_split(series, 2)
                        if data_test['is_anomaly'].tolist().count(1) == 0:
                            rep_val = 'specificity'
                    correlation_df.loc[idx, 'dataset'] = dataset + '_' + tss
                    correlation_df.loc[idx, 'ts'] = ts

                    if rep_val == 'f1':
                        correlation_df.loc[idx, 'dbscan_score'] = f_dbscan
                        correlation_df.loc[idx, 'lstm_score'] = f_lstm
                        correlation_df.loc[idx, 'sarima_score'] = f_sarima
                        correlation_df.loc[idx, 'sr_score'] = f_sr
                    # else:
                    #     correlation_df.loc[idx, 'dbscan_score'] = s_dbscan
                    #     correlation_df.loc[idx, 'lstm_score'] = s_lstm
                    #     correlation_df.loc[idx, 'sarima_score'] = s_sarima
                    #     correlation_df.loc[idx, 'sr_score'] = s_sr

                    for col in ts_properties_c22.columns:
                        if col not in exclude:
                            correlation_df.loc[idx, col] = ts_properties_c22.loc[ts + '.csv', col]

                    for col in ts_properties_fforma.columns:
                        if col not in exclude:
                            correlation_df.loc[idx, col] = ts_properties_fforma.loc[ts + '.csv', col]
                    idx += 1

    correlation_df.to_csv(f'results/ts_properties/f1_to_yahoo_nab.csv')
    res = correlation_df.corr(method='pearson')
    res.to_csv(f'results/ts_properties/f1_to_yahoo_nab_corr.csv')


def ts_properties_to_accuracy():
    correlation_df = pd.DataFrame([])
    idx = 0
    exclude = ['ts', 'Unnamed: 0', 'arch_acf', 'garch_acf', 'arch_r2', 'garch_r2', 'hw_alpha', 'hw_beta', 'hw_gamma']
    for dataset, subsets in [('kpi', ['train']), ('yahoo', ['real', 'synthetic', 'A3Benchmark', 'A4Benchmark']),
                             ('NAB', ['relevant'])]:
        for tss in subsets:
            ts_chaos = pd.read_csv(f'results/ts_properties/permutation_analysis_{dataset}_{tss}.csv').set_index('ts')

            try:
                sr = pd.read_csv(
                    f'C:\\Users\\oxifl\\Desktop\\batched_metrics\\{dataset}_{tss}_stats_sr_batched.csv')
                sarima = pd.read_csv(
                    f'C:\\Users\\oxifl\\Desktop\\batched_metrics\\{dataset}_{tss}_stats_sarima_batched.csv')
                lstm = pd.read_csv(
                    f'C:\\Users\\oxifl\\Desktop\\batched_metrics\\{dataset}_{tss}_stats_lstm_batched.csv')

                for f_lstm, f_sarima, f_sr, s_lstm, s_sarima, s_sr, ts in \
                        zip(lstm['f1'], sarima['f1'], sr['f1'], lstm['specificity'],
                                                                sarima['specificity'], sr['specificity'],
                                                                sr['dataset']):

                    rep_val = 'f1'
                    if str(ts) != 'nan' and 'flatline' not in ts:
                        series = pd.read_csv('datasets/' + dataset + '/' + tss + '/' + ts + '.csv')
                        series.rename(columns={'timestamps': 'timestamp', 'anomaly': 'is_anomaly'}, inplace=True)
                        if dataset != 'kpi':
                            data_train, data_test = np.array_split(series, 2)
                            if data_test['is_anomaly'].tolist().count(1) == 0:
                                continue

                        try:
                            correlation_df.loc[idx, 'dataset'] = dataset + '_' + tss
                            correlation_df.loc[idx, 'ts'] = ts

                            if rep_val == 'f1':
                                correlation_df.loc[idx, 'lstm_score'] = f_lstm
                                correlation_df.loc[idx, 'sarima_score'] = f_sarima
                                correlation_df.loc[idx, 'sr_score'] = f_sr
                            else:
                                correlation_df.loc[idx, 'lstm_score'] = s_lstm
                                correlation_df.loc[idx, 'sarima_score'] = s_sarima
                                correlation_df.loc[idx, 'sr_score'] = s_sr

                            for entrops in ['permutation_entropy', 'statistical_complexity', 'lyapunov_exp_max',
                                            'spectral_entropy', 'value_decomposition_entropy', 'approximate_entropy',
                                            'sample_entropy', 'hjorth_entropy']:
                                correlation_df.loc[idx, entrops] = ts_chaos.loc[ts, entrops]
                            idx += 1
                        except Exception as e:
                            print(e)
            except Exception as e:
                print(e)

    correlation_df.to_csv(f'results/ts_properties/chaos_to_f1.csv')
    mutual_info_of_features_to_f1()


def mutual_info_of_features_to_f1():
    from sklearn.feature_selection import mutual_info_regression
    df = pd.read_csv(f'results/ts_properties/chaos_to_f1.csv')
    df.dropna(axis=1, inplace=True)
    res = pd.DataFrame([])
    idx = 0
    cols = ['permutation_entropy', 'statistical_complexity', 'lyapunov_exp_max',
                                        'spectral_entropy', 'value_decomposition_entropy', 'approximate_entropy',
                                        'sample_entropy']
    # cols = ['DN_HistogramMode_5', 'DN_HistogramMode_10', 'CO_f1ecac', 'CO_FirstMin_ac',
    #                                         'CO_HistogramAMI_even_2_5',	'CO_trev_1_num', 'MD_hrv_classic_pnn40',
    #                                         'SB_BinaryStats_mean_longstretch1',	'SB_TransitionMatrix_3ac_sumdiagcov',
    #                                         'PD_PeriodicityWang_th0_01', 'CO_Embed2_Dist_tau_d_expfit_meandiff',
    #                                         'IN_AutoMutualInfoStats_40_gaussian_fmmi', 'FC_LocalSimple_mean1_tauresrat',
    #                                         'DN_OutlierInclude_p_001_mdrmd', 'DN_OutlierInclude_n_001_mdrmd',
    #                                         'SP_Summaries_welch_rect_area_5_1',	'SB_BinaryStats_diff_longstretch0',
    #                                         'SB_MotifThree_quantile_hh', 'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1',
    #                                         'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1', 'SP_Summaries_welch_rect_centroid',
    #                                         'FC_LocalSimple_mean3_stderr', 'unitroot_pp',
    #                                         'unitroot_kpss', 'stability', 'seasonal_period', 'trend',
    #                                         'spike', 'linearity', 'curvature', 'e_acf1', 'e_acf10',	'x_pacf5',
    #                                         'diff1x_pacf5',	'diff2x_pacf5',	'nonlinearity',	'lumpiness', 'alpha', 'beta',
    #                                         'flat_spots', 'crossing_points',	'arch_lm', 'x_acf1', 'x_acf10',
    #                                         'diff1_acf1', 'diff1_acf10', 'diff2_acf1', 'diff2_acf10']

    for model in ['lstm', 'sr', 'sarima']:
        try:
            mis = mutual_info_regression(df[cols], df[f'{model}_score'])

            for mi, f in zip(mis, cols):
                res.loc[idx, 'model'] = model
                res.loc[idx, 'feature'] = f
                res.loc[idx, 'mutual_information'] = mi
                idx += 1
        except Exception as e:
            print(e)

    res.to_csv(f'results/ts_properties/mutual_info_chaos.csv')


def find_relationship():
    from sklearn.inspection import plot_partial_dependence
    import matplotlib.pyplot as plt
    df = pd.read_csv(f'results/ts_properties/features_to_f1_corr.csv')
    df.dropna(axis=1, inplace=True)

    df2 = pd.read_csv(f'results/ts_properties/mutual_info.csv')
    df2['mutual_information'] = df2['mutual_information'].astype('float64')

    cols = ['DN_HistogramMode_5', 'DN_HistogramMode_10', 'CO_f1ecac', 'CO_FirstMin_ac',
                                            'CO_HistogramAMI_even_2_5',	'CO_trev_1_num', 'MD_hrv_classic_pnn40',
                                            'SB_BinaryStats_mean_longstretch1',	'SB_TransitionMatrix_3ac_sumdiagcov',
                                            'PD_PeriodicityWang_th0_01', 'CO_Embed2_Dist_tau_d_expfit_meandiff',
                                            'IN_AutoMutualInfoStats_40_gaussian_fmmi', 'FC_LocalSimple_mean1_tauresrat',
                                            'DN_OutlierInclude_p_001_mdrmd', 'DN_OutlierInclude_n_001_mdrmd',
                                            'SP_Summaries_welch_rect_area_5_1',	'SB_BinaryStats_diff_longstretch0',
                                            'SB_MotifThree_quantile_hh', 'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1',
                                            'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1', 'SP_Summaries_welch_rect_centroid',
                                            'FC_LocalSimple_mean3_stderr',	'unitroot_pp',
                                            'unitroot_kpss', 'stability', 'seasonal_period', 'trend',
                                            'spike', 'linearity', 'curvature', 'e_acf1', 'e_acf10',	'x_pacf5',
                                            'diff1x_pacf5',	'diff2x_pacf5',	'nonlinearity',	'lumpiness', 'alpha', 'beta',
                                            'flat_spots', 'crossing_points', 'arch_lm', 'x_acf1', 'x_acf10',
                                            'diff1_acf1', 'diff1_acf10', 'diff2_acf1', 'diff2_acf10']

    for model in ['lstm', 'sr', 'sarima']:
        for modeller in ['mlp', 'gbr']:
            # cols_filtered = df2[df2['model'] == model][df2['mutual_information'] > 0.25]['feature'].tolist()
            cols_filtered = df2[df2['model'] == model].sort_values('mutual_information', ascending=False)['feature'].tolist()[:2]
            cols_filtered += [copy.deepcopy(cols_filtered)]

            print(f'Filtered columns for {model} are: {cols_filtered}')
            X = df[cols]
            y = df[f'{model}_score']

            gbr = GradientBoostingRegressor()
            gbr.fit(X, y)

            mlp = make_pipeline(StandardScaler(),
                                MLPRegressor(hidden_layer_sizes=(100, 100),
                                             tol=1e-2, max_iter=500, random_state=0))
            mlp.fit(X, y)

            fig, ax = plt.subplots(figsize=(12, 10))
            gbr_disp = plot_partial_dependence(gbr, X, cols_filtered, ax=ax)

            fig, ax = plt.subplots(figsize=(12, 10))
            mlp_disp = plot_partial_dependence(mlp, X, cols_filtered, ax=ax, line_kw={"color": "red"})

            plt.title(f'{model.upper()}: 2 most influential features and their joint effect PDP')
            fig, axs = plt.subplots(1, len(cols_filtered), figsize=(20, 10))
            if modeller == 'mlp':
                mlp_disp.plot(ax=axs, line_kw={"label": "Multi-layer Perceptron", "color": "red"})
            else:
                gbr_disp.plot(ax=axs, line_kw={"label": "Gradient Boosting Regressor"})

            plt.savefig(f'results/imgs/dependence/{model}_all_{modeller}_max.png')

            plt.clf()


def permutation_analysis():
    q = np.arange(0.01, 100, 0.01).tolist()

    for entropy_name in ['spectral_entropy', 'value_decomposition_entropy', 'approximate_entropy',
                         'sample_entropy', 'permutation_entropy', 'renyi', 'tsallis']:
        # approximate & value decompose
        print(f'Doing entropy {entropy_name}')
        for dataset, subsets in [('NAB', ['relevant']), ('kpi', ['train'])]:
                for tss in subsets:
                    hcs = pd.DataFrame([])
                    train_data_path = 'datasets/' + dataset + '/' + tss + '/'

                    res = pd.DataFrame([])
                    for filename in os.listdir(train_data_path):
                        print(filename)
                        f = os.path.join(train_data_path, filename)
                        ts = pd.read_csv(f)
                        ts.rename(columns={'timestamps': 'timestamp', 'anomaly': 'is_anomaly'}, inplace=True)

                        # lyapunov_e = nolds.lyap_r(ts['value'].to_numpy())
                    #
                    # if lyapunov_e > 0.0:
                    #     lyapunov_e = 1.0
                    # elif lyapunov_e < 0.0:
                    #     lyapunov_e = -1.0
                    #
                    # hc = ordpy.complexity_entropy(ts['value'].to_numpy(), dx=4)
                    #
                    # hcs = hcs.append({
                    #     'ts': filename.replace('.csv', ''),
                    #     'permutation_entropy': hc[0],
                    #     'statistical_complexity': hc[1],
                    #     'lyapunov_exp_max': lyapunov_e,
                    #     'spectral_entropy': ant.spectral_entropy(ts['value'].to_numpy(), sf=100, method='welch', normalize=True),
                    #     'value_decomposition_entropy': ant.svd_entropy(ts['value'].to_numpy(), normalize=True),
                    #     'approximate_entropy': ant.app_entropy(ts['value'].to_numpy()),
                    #     'sample_entropy': ant.sample_entropy(ts['value'].to_numpy()),
                    #     'hjorth_entropy': ant.hjorth_params(ts['value'].to_numpy())
                    # }, ignore_index=True)

                    # continue
                    #
                    # if hc[1] > 0.15:
                    #     print(filename)
                    #     print(hc)
                    #     # plt.plot(ts['timestamp'], ts['value'])
                    #     # plt.show()
                        max_qs, max_cs = [], []
                        batches_anomalies = []
                        entropies = []
                        interval = 100
                        step = 100
                        collected_entropies = []
                        mean_entropy, std_entropy, boundary_up, boundary_bottom = None, None, None, None
                        y_predicted = []
                        y_true = []

                        try:
                            for start in range(0, ts.shape[0], step):
                                window = ts.iloc[start:start+step]
                                if window.shape[0] == step:
                                    # curve = ordpy.tsallis_complexity_entropy(window['value'].to_numpy(), dx=4, q=q)
                                    # max_q, max_c = np.max([a[0] for a in curve]), np.max([a[1] for a in curve])
                                    # max_cs.append(max_c)
                                    # max_qs.append(max_q)

                                    if start < ts.shape[0] // 2 and window['is_anomaly'].tolist().count(1) == 0:
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

                                    elif start >= ts.shape[0] // 2 and not mean_entropy:
                                        mean_entropy = np.mean(collected_entropies)
                                        std_entropy = np.std(collected_entropies)
                                        # boundary_bottom = mean_entropy - 1.25 * std_entropy
                                        # boundary_up = mean_entropy + 1.25 * std_entropy
                                        boundary_bottom = np.min(collected_entropies) * 0.9
                                        boundary_up = np.max(collected_entropies) * 1.1

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
                                        batches_anomalies.append(start // step)

                                    if mean_entropy:
                                        if boundary_bottom <= entropies[-1] <= boundary_up:
                                            y_predicted.append(0)
                                        else:
                                            y_predicted.append(1)

                                        if window['is_anomaly'].tolist().count(1) > 0:
                                            y_true.append(1)
                                        else:
                                            y_true.append(0)

                            color = ['red' if i in batches_anomalies else 'blue' for i in range(len(entropies))]

                            plt.bar(list(range(len(entropies))), entropies, color=color)
                            plt.axvline(x=int(ts.shape[0] // (2 * step)), color='orange', linestyle='--', label='train-test separation')
                            plt.axhline(y=boundary_bottom, color='gray', linestyle='--')
                            plt.axhline(y=boundary_up, color='gray', linestyle='--')

                            plt.legend()
                            plt.ylabel(entropy_name)
                            plt.xlabel('Batch')

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

                            plt.savefig(f'results/ts_properties/imgs/entropy_analysis/{entropy_name}_{dataset}_{tss}_{filename.replace(".csv", "")}_step_100.png')
                            plt.clf()

                            res = res.append({
                                'ts': filename.replace('.csv', ''),
                                'y_pred': y_predicted,
                                'y_true': y_true,
                                'f1': f1_score(y_true, y_predicted, average='binary'),
                                'hamming': hamming_loss(y_true, y_predicted),
                                'step': 100,
                                'entropy': entropy_name,
                                'total_anomalies': y_true.count(1)

                            }, ignore_index=True)
                        except Exception as e:
                            raise e
                    res.to_csv(f'results/ts_properties/entropies/{entropy_name}_{dataset}_{tss}.csv')
            # hcs.to_csv(f'results/ts_properties/permutation_analysis_{dataset}_{tss}.csv')

            # f, ax = plt.subplots(figsize=(8.19, 6.3))
            #
            # for idx, row in hcs.iterrows():
            #     ax.scatter(x=row['permutation_entropy'], y=row['statistical_complexity'], s=100)
            #
            # ax.set_xlabel('Permutation entropy, $H$')
            # ax.set_ylabel('Statistical complexity, $C$')
            #
            # f.savefig(f'results/ts_properties/imgs/permutation_analysis_{dataset}_{tss}.png')


def performance():
    for dataset, subsets in [('NAB', ['relevant']), ('yahoo', ['real', 'synthetic'])]:
        for tss in subsets:
            train_data_path = 'datasets/' + dataset + '/' + tss + '/'
            chaos = pd.read_csv(f'results/ts_properties/permutation_analysis_{dataset}_{tss}.csv').set_index('ts')
            for model in ['sr', 'sarima', 'lstm']:
                f1 = []
                performance = f'C:\\Users\\oxifl\\Desktop\\batched_metrics\\{dataset}_{tss}_stats_{model}_batched.csv'
                performance = pd.read_csv(performance).set_index('dataset')
                for filename in os.listdir(train_data_path):
                    try:
                        laypunov = chaos.loc[filename, 'lyapunov_exp_max']
                        if laypunov <= 0:
                            f1.append(performance.loc[filename.replace('.csv', ''), 'f1'])
                    except:
                        pass

                print(f"For model {model}, dataset {dataset} of part {tss}, avg f1 is {np.mean(f1)}")
