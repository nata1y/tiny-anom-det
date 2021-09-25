import pandas as pd


def analyse_series_properties(dataset, type, name):
    df = pd.read_csv(f'results/{dataset}_{type}_stats_{name}.csv')

    for model in df.model.unique().tolist():
        print(f'Analyzing time series stats versus {model} predictions')
        data = df[df.model == model].reset_index()
        print(data.corr(method='pearson'))
        data.corr(method='pearson').to_csv(f'results/correlation_{dataset}_{type}_{model}.csv')
