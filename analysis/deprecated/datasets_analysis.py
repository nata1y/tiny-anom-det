# analyze benchmark differences via dustances between their feature distributions etc
import itertools
import random

import distance
from matplotlib import pyplot
from scipy.stats import wasserstein_distance, kstest
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
from utils import KL, intersection, fidelity, sq_euclidian


def compare_dataset_properties():
    features = pd.read_csv(f'results/ts_properties/yahoo_real_features_c22.csv').columns
    idxs = {
        'real': 0,
        'synthetic': 1,
        'A3Benchmark': 2,
        'A4Benchmark': 3,
        'NAB': 4,
        'kpi': 5
    }
    for feature in features:
        print(feature)
        if feature not in ['ts', 'Unnamed: 0', 'arch_acf', 'garch_acf', 'arch_r2', 'garch_r2', 'hw_alpha', 'hw_beta',
                           'hw_gamma']:
            transform_full = pd.DataFrame([])
            max_ucr = pd.read_csv(f'results/ts_properties/ucr_ts_features_c22.csv')[feature].max()
            print(max_ucr)
            for dataset in [('yahoo', 'real'), ('yahoo', 'synthetic'), ('yahoo', 'A3Benchmark'), ('yahoo', 'A4Benchmark'),
                        ('NAB', 'relevant'), ('kpi', 'fit'), ('ucr', 'ts')]:

                data = pd.read_csv(f'results/ts_properties/{dataset[0]}_{dataset[1]}_features_c22.csv')
                transform = pd.DataFrame([])
                transform[feature] = data[feature]
                transform['Dataset'] = dataset[0] if dataset[0] != 'yahoo' else dataset[1]
                transform['ts'] = data['ts']
                transform_full = pd.concat([transform_full, transform])

            ax = sns.stripplot(x="Dataset", y=feature, data=transform_full, zorder=2)
            labels = [e.get_text() for e in pyplot.gca().get_xticklabels()]
            ticks = pyplot.gca().get_xticks()
            w = 0.1
            for idx, datas in enumerate(labels):
                idx = labels.index(datas)
                pyplot.hlines(transform_full[transform_full['Dataset'] == datas][feature].mean(), ticks[idx] - w,
                              ticks[idx] + w, color='k', linestyles='solid', linewidth=3.0, zorder=3)

            outliers = transform_full[transform_full[feature] > max_ucr]
            for ds in ['real', 'synthetic', 'A3Benchmark', 'A4Benchmark', 'NAB', 'kpi']:
                for idx, row in list(outliers[outliers['Dataset'] == ds].sort_values(feature, ascending=False).iterrows())[:2]:
                    ax.text(idxs[row['Dataset']] - 1, row[feature] + .5, row['ts'].split('.')[0], size='xx-small')

            pyplot.savefig(f'results/ts_properties/imgs/{feature}_c22_ucr_outliers.png')
            pyplot.clf()


def calculate_dists():
    random.seed(30)
    features = pd.read_csv(f'results/ts_properties/yahoo_real_features_c22.csv').columns
    df = pd.DataFrame([])
    idx = 0
    dfs = [('yahoo', 'real'), ('yahoo', 'synthetic'), ('yahoo', 'A3Benchmark'), ('yahoo', 'A4Benchmark'),
           ('NAB', 'relevant'), ('kpi', 'fit')]
    exclude = ['ts', 'Unnamed: 0', 'arch_acf', 'garch_acf', 'arch_r2', 'garch_r2', 'hw_alpha', 'hw_beta', 'hw_gamma']

    for feature in features:
        if feature not in exclude:
            print(feature)
            for dataset1 in dfs:
                data1 = pd.read_csv(f'results/ts_properties/{dataset1[0]}_{dataset1[1]}_features_c22.csv')
                data1.fillna(0.0, inplace=True)
                for dataset2 in dfs[dfs.index(dataset1):]:
                    if dataset1 != dataset2:
                        data2 = pd.read_csv(f'results/ts_properties/{dataset2[0]}_{dataset2[1]}_features_c22.csv')
                        data2.fillna(0.0, inplace=True)

                        d1, d2 = data1[feature].to_numpy(), data2[feature].to_numpy()

                        scaler = MinMaxScaler()
                        d1 = list(scaler.fit_transform(d1.reshape(-1, 1)).flatten())
                        scaler = MinMaxScaler()
                        d2 = list(scaler.fit_transform(d2.reshape(-1, 1)).flatten())

                        d1, _ = np.histogram(d1, bins=100)
                        d2, _ = np.histogram(d2, bins=100)

                        wdist = wasserstein_distance(d1, d2)
                        edist = np.linalg.norm(d1 - d2)
                        sdist = distance.sorensen(d1, d2)
                        kldist = KL(d1, d2)
                        ipdist = np.inner(d1, d2)
                        fdist = fidelity(d1, d2)
                        sedist = sq_euclidian(d1, d2)
                        idist = intersection(d1, d1)

                        df.loc[idx, 'feature'] = feature
                        df.loc[idx, 'dataset1'] = dataset1[0] + '_' + dataset1[1]
                        df.loc[idx, 'dataset2'] = dataset2[0] + '_' + dataset2[1]
                        df.loc[idx, 'distance_wasserstein'] = wdist
                        df.loc[idx, 'distance_euclidian'] = edist
                        df.loc[idx, 'distance_sorensen'] = sdist
                        df.loc[idx, 'distance_kl'] = kldist
                        df.loc[idx, 'distance_inner_prod'] = ipdist
                        df.loc[idx, 'distance_fidelity'] = fdist
                        df.loc[idx, 'distance_intersection'] = idist
                        df.loc[idx, 'distance_squared_euclidian'] = sedist
                        idx += 1

    df.to_csv(f'results/ts_properties/features_was_dist_c22.csv')


def dist_between_sets():
    dists_p_f = pd.read_csv(f'results/ts_properties/features_was_dist_c22.csv')
    df = pd.DataFrame([])
    dfs = [('yahoo', 'real'), ('yahoo', 'synthetic'), ('yahoo', 'A3Benchmark'), ('yahoo', 'A4Benchmark'),
           ('NAB', 'relevant'), ('kpi', 'fit')]
    dists = itertools.combinations(dfs, 2)

    idx = 0
    for s in list(dists):
        s1, s2 = s
        df.loc[idx, 'from'] = s1[0] + '_' + s1[1]
        df.loc[idx, 'to'] = s2[0] + '_' + s2[1]
        df.loc[idx, 'norm_sum_wasserstein'] = 0.0
        df.loc[idx, 'norm_sum_euclidian'] = 0.0
        df.loc[idx, 'norm_sum_sorensen'] = 0.0
        df.loc[idx, 'norm_sum_kl'] = 0.0
        df.loc[idx, 'norm_sum_inner_prod'] = 0.0
        df.loc[idx, 'norm_sum_fidelity'] = 0.0
        df.loc[idx, 'norm_sum_intersection'] = 0.0
        df.loc[idx, 'norm_sum_squared_euclidian'] = 0.0
        idx += 1

    for feature in dists_p_f['feature'].unique().tolist():
        print(feature)
        for dist in ['wasserstein', 'euclidian', 'sorensen', 'kl', 'inner_prod', 'fidelity',
                     'intersection', 'squared_euclidian']:
            snippest = dists_p_f[dists_p_f['feature'] == feature].reset_index()
            mval = np.max(snippest['distance_' + dist].tolist())
            if mval == 0.0:
                mval = 0.00000001
            idx = 0
            dists = itertools.combinations(dfs, 2)

            for s in dists:
                s1, s2 = s
                val = 0.0
                sub1 = snippest[snippest['dataset1'] == s1[0] + '_' + s1[1]]
                sub2 = snippest[snippest['dataset1'] == s2[0] + '_' + s2[1]]
                sub1 = sub1[sub1['dataset2'] == s2[0] + '_' + s2[1]]
                sub2 = sub2[sub2['dataset2'] == s1[0] + '_' + s1[1]]
                if sub1.shape[0] > 0:
                    val = sub1['distance_' + dist].tolist()[0]
                elif sub2.shape[0] > 0:
                    val = sub2['distance_' + dist].tolist()[0]

                df.loc[idx, 'norm_sum_' + dist] = df.loc[idx, 'norm_sum_' + dist] + val / mval
                idx += 1

    df.to_csv(f'results/ts_properties/dataset_was_dist_c22.csv')


# compare orderings via different distance measures
def compare_dataset_distances():
    for features in ['c22']:
        df = pd.DataFrame([], columns=['wasserstein', 'euclidian', 'sorensen', 'kl', 'inner_prod', 'fidelity',
                                       'intersection', 'squared_euclidian', 'distance_name'])
        df['distance_name'] = ['wasserstein', 'euclidian', 'sorensen', 'kl', 'inner_prod', 'fidelity',
                               'intersection', 'squared_euclidian']
        df.set_index('distance_name', inplace=True)
        data = pd.read_csv(f'results/ts_properties/dataset_was_dist_{features}.csv')
        for dist in ['wasserstein', 'euclidian', 'sorensen', 'kl', 'inner_prod', 'fidelity',
                     'intersection', 'squared_euclidian']:
            for dist2 in ['wasserstein', 'euclidian', 'sorensen', 'kl', 'inner_prod', 'fidelity',
                           'intersection', 'squared_euclidian']:
                print(data)
                tau, p_value = stats.kendalltau([sorted(data['norm_sum_' + dist]).index(x) for x in data['norm_sum_' + dist]],
                                                [sorted(data['norm_sum_' + dist2]).index(x) for x in data['norm_sum_' + dist2]])
                df.loc[dist, dist2] = tau

        df.to_csv(f'results/ts_properties/ranking_similarities_via_dists_{features}.csv')


# Compare per-feature distributions to each other in different datsetse - are they drown fom the same distribution?
def compare_feature_samples_from_same_dist():
    random.seed(30)
    features = pd.read_csv(f'results/ts_properties/yahoo_real_features_c22.csv').columns
    df = pd.DataFrame([])
    idx = 0
    dfs = [('yahoo', 'real'), ('yahoo', 'synthetic'), ('yahoo', 'A3Benchmark'), ('yahoo', 'A4Benchmark'),
           ('NAB', 'relevant'), ('kpi', 'fit')]
    exclude = ['ts', 'Unnamed: 0', 'arch_acf', 'garch_acf', 'arch_r2', 'garch_r2', 'hw_alpha', 'hw_beta', 'hw_gamma']

    for feature in features:
        if feature not in exclude:
            print(feature)
            for dataset1 in dfs:
                data1 = pd.read_csv(f'results/ts_properties/{dataset1[0]}_{dataset1[1]}_features_c22.csv')
                data1.fillna(0.0, inplace=True)
                for dataset2 in dfs[dfs.index(dataset1):]:
                    if dataset1 != dataset2:
                        data2 = pd.read_csv(f'results/ts_properties/{dataset2[0]}_{dataset2[1]}_features_c22.csv')
                        data2.fillna(0.0, inplace=True)

                        d1, d2 = data1[feature].to_numpy(), data2[feature].to_numpy()

                        ks_stats = kstest(d1, d2)

                        df.loc[idx, 'feature'] = feature
                        df.loc[idx, 'dataset1'] = dataset1[0] + '_' + dataset1[1]
                        df.loc[idx, 'dataset2'] = dataset2[0] + '_' + dataset2[1]
                        df.loc[idx, 'KS_p_val'] = ks_stats[0]
                        if df.loc[idx, 'KS_p_val'] < 0.05:
                            df.loc[idx, 'from_same_dist'] = False
                        else:
                            df.loc[idx, 'from_same_dist'] = True
                        idx += 1

    df.to_csv(f'results/ts_properties/datasets_features_KS_stats.csv')


# Compare per-feature distributions to ones in UCR (large) dataset - are they drown fom the same distribution?
def check_dist_sample():
    df = pd.read_csv(f'results/ts_properties/ucr_ts_features_c22.csv')
    transform = pd.DataFrame([])
    idx = 0
    for dataset in [('yahoo', 'real'), ('yahoo', 'synthetic'), ('yahoo', 'A3Benchmark'),
                    ('yahoo', 'A4Benchmark'),
                    ('NAB', 'relevant'), ('kpi', 'fit')]:
        for feature in df.columns:
            if feature not in ['ts', 'Unnamed: 0', 'Unnamed: 0.1']:
                transform.loc[idx, 'Dataset'] = dataset[0] + '_' + dataset[1] if dataset[0] != 'yahoo' else dataset[1]
                transform.loc[idx, 'feature'] = feature
                data = pd.read_csv(f'results/ts_properties/{dataset[0]}_{dataset[1]}_features_c22.csv')
                transform.loc[idx, 'KS_p_val'] = kstest(df[feature].to_numpy(), data[feature].to_numpy())[0]
                # Here we reject null hypothesis that 2 samples came from the same distribution if p val < 0.05
                if transform.loc[idx, 'KS_p_val'] < 0.05:
                    transform.loc[idx, 'from_same_dist'] = False
                else:
                    transform.loc[idx, 'from_same_dist'] = True
                idx += 1

        transform.to_csv(f'results/ts_properties/datasets_to_ucr_c22_features_KS_stats.csv')


# For every dataset, we take its per- TS features and see their variance.
# Catch22 assumes that all features are important and are PC, so if we have features with low variance within dataset
# it might indicate that it is not representative from the nab_point of view of that feature
def check_low_variance_features():
    random.seed(30)
    res = pd.DataFrame([])
    features = pd.read_csv(f'results/ts_properties/yahoo_real_features_c22.csv').columns
    df = pd.DataFrame([])
    idx = 0
    dfs = [('yahoo', 'real'), ('yahoo', 'synthetic'), ('yahoo', 'A3Benchmark'), ('yahoo', 'A4Benchmark'),
           ('NAB', 'relevant'), ('kpi', 'fit'), ('ucr', 'ts')]
    exclude = ['ts', 'Unnamed: 0', 'arch_acf', 'garch_acf', 'arch_r2', 'garch_r2', 'hw_alpha', 'hw_beta', 'hw_gamma']

    for dataset1 in dfs:
        data1 = pd.read_csv(f'results/ts_properties/{dataset1[0]}_{dataset1[1]}_features_c22.csv')
        data1.fillna(0.0, inplace=True)
        fs = [f for f in data1.columns if f not in exclude]
        features = data1[fs]
        for f in fs:
            print(f)
            scaler = MinMaxScaler()
            features_norm = scaler.fit_transform(features[[f]].to_numpy())

            # we need variance of ALL features
            vt = VarianceThreshold(threshold=-.1)
            vt.fit(features_norm)
            print(f'{f} has variance of {vt.variances_[0]}')
            res.loc[idx, 'dataset'] = dataset1[0] + '_' + dataset1[1]
            res.loc[idx, 'feature'] = f
            res.loc[idx, 'variance'] = vt.variances_[0]
            idx += 1

        res.to_csv(f'results/ts_properties/variances_of_c22_features_withinn_datasets.csv')


# compare dataset distances via features to those via model f-scores
def compare_dataset_model_distances():
    for features in ['c22']:
        df = pd.DataFrame([], columns=['wasserstein', 'euclidian', 'sorensen', 'kl', 'inner_prod', 'fidelity',
                                       'intersection', 'squared_euclidian', 'distance_name'])
        df['distance_name'] = ['wasserstein', 'euclidian', 'sorensen', 'kl', 'inner_prod', 'fidelity',
                               'intersection', 'squared_euclidian']
        df.set_index('distance_name', inplace=True)
        data = pd.read_csv(f'results/ts_properties/dataset_was_dist_{features}.csv')
        models = pd.read_csv(f'results/ts_properties/model_performance_per_dataset.csv')
        models.set_index('model', inplace=True)
        for dist in ['wasserstein', 'euclidian', 'sorensen', 'kl', 'inner_prod', 'fidelity',
                     'intersection', 'squared_euclidian']:
            for dist2 in models.columns:
                print(data)
                tau, p_value = stats.kendalltau([sorted(data['norm_sum_' + dist]).index(x) for x in data['norm_sum_' + dist]],
                                                [sorted(data[dist2]).index(x) for x in data[dist2]])
                df.loc[dist, dist2] = tau

        df.to_csv(f'results/ts_properties/ranking_similarities_via_dists_{features}.csv')


def split_ensemble_stats():
    df = pd.read_csv('results/ensemble_train_data.csv')
    df = df[df['dataset'] != 'NAB_relevant']
    df1 = df[df['model_f1'] >= 0.7]
    df1.to_csv('results/all_series_can_predict_nonab.csv')
    df2 = df[df['model_f1'] < 0.7]
    df2.to_csv('results/all_series_cannot_predict_nonab.csv')


def analyze_ensemble_stats():
    import seaborn as sns
    cor = pd.read_csv('results/all_series_can_predict_nonab.csv')
    fail = pd.read_csv('results/all_series_cannot_predict_nonab.csv')
    for feature in cor.columns:
        if feature not in ['model_to_use', 'model_f1', 'dataset', 'ts']:
            data1 = pd.DataFrame([])
            data1[feature] = cor[feature]
            data1['Predictability'] = 'reasonable'
            data2 = pd.DataFrame([])
            data2[feature] = fail[feature]
            data2['Predictability'] = 'poor'
            data_full = pd.concat([data1, data2], ignore_index=True)

            ax = sns.stripplot(x="Predictability", y=feature, data=data_full, zorder=2)
            labels = [e.get_text() for e in pyplot.gca().get_xticklabels()]
            ticks = pyplot.gca().get_xticks()
            w = 0.1
            for idx, datas in enumerate(labels):
                idx = labels.index(datas)
                pyplot.hlines(data_full[data_full['Predictability'] == datas][feature].mean(), ticks[idx] - w,
                              ticks[idx] + w, color='k', linestyles='solid', linewidth=3.0, zorder=3)

            pyplot.savefig(f'results/ts_properties/imgs/{feature}_predictability_dists_nonab.png')
            pyplot.clf()
