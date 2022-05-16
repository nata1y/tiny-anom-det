# analyze extracted TS properties to model performance correlations
import copy
import pandas as pd
import os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.feature_selection import mutual_info_classif
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def machine_ts_to_features_correlation():
    correlation_df = pd.DataFrame([])
    idx = 0
    exclude = ['ts', 'Unnamed: 0', 'arch_acf', 'garch_acf', 'arch_r2', 'garch_r2', 'hw_alpha', 'hw_beta', 'hw_gamma']
    for dataset, subsets in [('yahoo', ['real', 'synthetic', 'A3Benchmark', 'A4Benchmark']), ('NAB', ['relevant'])]:
        for tss in subsets:
            print(tss)
            ts_properties_c22 = pd.read_csv(f'results/ts_properties/{dataset}_{tss}_features_c22.csv').set_index('ts')
            ts_properties_fforma = pd.read_csv(f'results/ts_properties/{dataset}_{tss}_features_fforma.csv').set_index('ts')
            dbscan = pd.read_csv(
                f'results/{dataset}_{tss}_stats_dbscan.csv')

            try:
                lstm = pd.read_csv(
                    f'results/{dataset}_{tss}_stats_lstm.csv')
            except:
                try:
                    lstm = pd.read_csv(
                        f'results/{dataset}_{tss}_stats_lstm.csv')
                except:
                    lstm = pd.DataFrame([])
                    lstm['f1'] = [0 for i in range(dbscan.shape[0])]

            try:
                sarima = pd.read_csv(
                    f'results/{dataset}_{tss}_stats_sarima.csv')
            except:
                try:
                    sarima = pd.read_csv(
                        f'results/{dataset}_{tss}_stats_sarima.csv')
                except Exception as e:
                    sarima = pd.DataFrame([])
                    sarima['f1'] = [0 for i in range(dbscan.shape[0])]

            sr = pd.read_csv(
                f'results/{dataset}_{tss}_stats_sr.csv')

            for f_dbscan, f_lstm, f_sarima, f_sr, s_dbscan, s_lstm, s_sarima, s_sr, ts in \
                    zip(dbscan['f1'], lstm['f1'], sarima['f1'], sr['f1'],
                                                            dbscan['specificity'], lstm['specificity'],
                                                            sarima['specificity'], sr['specificity'],
                                                            dbscan['dataset']):

                rep_val = 'f1'
                if str(ts) != 'nan' and 'flatline' not in ts:
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
                    else:
                        correlation_df.loc[idx, 'dbscan_score'] = s_dbscan
                        correlation_df.loc[idx, 'lstm_score'] = s_lstm
                        correlation_df.loc[idx, 'sarima_score'] = s_sarima
                        correlation_df.loc[idx, 'sr_score'] = s_sr

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
    for dataset, subsets in [('kpi', ['train']), ('yahoo', ['real', 'synthetic', 'A3Benchmark', 'A4Benchmark']),
                             ('NAB', ['relevant'])]:
        for tss in subsets:
            ts_chaos = pd.read_csv(f'results/ts_properties/permutation_analysis_{dataset}_{tss}.csv').set_index('ts')

            try:
                sr = pd.read_csv(
                    f'results/{dataset}_{tss}_stats_sr_batched.csv')
                sarima = pd.read_csv(
                    f'results/{dataset}_{tss}_stats_sarima_batched.csv')
                lstm = pd.read_csv(
                    f'results/{dataset}_{tss}_stats_lstm_batched.csv')

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


def performance():
    df = pd.DataFrame([])
    for dataset, subsets in [('NAB', ['relevant'])]:
        for tss in subsets:
            try:
                train_data_path = 'datasets/' + dataset + '/' + tss + '/'
                chaos = pd.read_csv(f'results/ts_properties/permutation_analysis_{dataset}_{tss}.csv').dropna(subset=['ts'])
                for model in ['sr', 'sarima', 'lstm']:
                    performance = f'results/{dataset}_{tss}_stats_{model}_per_batch.csv'
                    performance = pd.read_csv(performance).dropna(subset=['ts'])
                    chaos = chaos.merge(performance, how='inner', on='ts').set_index('ts')
                    performance = performance.set_index('ts')

                    mis = mutual_info_classif(performance[['f1']], chaos['lyapunov_exp_max'])
                    df = df.append({
                        'dataset': dataset,
                        'tss': tss,
                        'model': model,
                        'mutual_info': mis
                    }, ignore_index=True)
                    for le in ['chaotic', 'periodic', 'steady']:
                        f1 = []
                        for filename in os.listdir(train_data_path):
                            try:
                                laypunov = chaos.loc[filename.replace('.csv', ''), 'lyapunov_exp_max']
                                if laypunov == le and pd.notna(performance.loc[filename.replace('.csv', ''), 'f1']):
                                    f1.append(performance.loc[filename.replace('.csv', ''), 'f1'])
                            except:
                                pass

                        print(f"For model {model}, dataset {dataset} of part {tss}, avg f1 is {np.mean(f1)} on {le} ts's")
            except Exception as e:
                print(e)

    df.to_csv('results/ts_properties/discrete_mutual_info_f1_per_dataset.csv')


def complexity_entropy_analysis():
    import matplotlib.pyplot as plt
    for dataset, subsets in [('NAB', ['relevant']),
                             ('yahoo', ['real', 'synthetic', 'A3Benchmark', 'A4Benchmark'])]:
        for tss in subsets:
            df = pd.read_csv(f'results/ts_properties/permutation_analysis_{dataset}_{tss}.csv')

            f, ax = plt.subplots(figsize=(8.19, 6.3))

            for idx, row in df.iterrows():
                lyapunov_e = row['lyapunov_exp_max']

                if lyapunov_e == 'chaotic':
                    # chaos
                    color = 'red'
                elif lyapunov_e == 'steady':
                    # steady
                    color = 'blue'
                else:
                    # periodic
                    color = 'green'

                ax.scatter(x=row['permutation_entropy'], y=row['statistical_complexity'], s=100, color=color)

            ax.set_xlabel('Permutation entropy, $H$')
            ax.set_ylabel('Statistical complexity, $C$')
            ax.set_title(f'{dataset}: {tss}')
            df.to_csv(f'results/ts_properties/permutation_analysis_{dataset}_{tss}.csv')
            f.savefig(f'results/ts_properties/imgs/permutation_analysis_{dataset}_{tss}_lyapunove.png')
