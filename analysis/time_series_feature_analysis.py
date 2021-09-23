import pandas as pd


def analyse_series_properties(df):

    for model in df.model.unique().tolist():
        print(f'Analyzing time series stats versus {model} predictions')
        data = df[df.model == model].reset_index()
        print(data.corr(method='pearson'))
        data.corr(method='pearson').to_csv(f'results/correlation_yahoo_real_{model}.csv')
