import numpy as np
import os
import pandas as pd


def create_dataset(X, y, time_steps=60):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i:(i + time_steps)].values)

    return np.array(Xs), np.array(ys)


def adjust_range(val, oper, factor):
    if val < 0:
        sign = -1.0
        val *= -1.0
        if oper == 'div':
            val = val * factor * sign
        else:
            val = sign * val / factor

        return val

    if oper == 'div':
        val = val / factor
    else:
        val = val * factor

    return val


def preprocess_kpi(cwd):
    dataset, type = 'kpi', 'synthetic'

    for t in ['train', 'test']:
        train_data_path = cwd + '\\' + dataset + '\\' + t + '\\'
        for filename in os.listdir(train_data_path):
            f = os.path.join(train_data_path, filename)
            if os.path.isfile(f):
                try:
                    data = pd.read_csv(f)
                except:
                    data = pd.read_hdf(f)

                for kpi_id in data['KPI ID'].unique().tolist():
                    df = data[data['KPI ID'] == kpi_id].reset_index(drop=True)[['timestamp', 'value', 'label']]
                    df.rename(columns={"label": "is_anomaly"}, inplace=True)
                    df.to_csv(cwd + '\\' + dataset + '\\' + t + '\\' + kpi_id + '.csv')

