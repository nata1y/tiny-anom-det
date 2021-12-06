import numpy as np
import pandas as pd


class NaiveDetector:
    def __init__(self, k, u, c, b, useabs):
        self.k = k
        self.u = u
        self.c = c
        self.b = b
        self.useabs = useabs
        self.movdatas = []

    def predict(self, ts):
        prex = np.diff(ts['value'].tolist())
        if self.useabs:
            prex = np.abs(prex)

        x = pd.DataFrame([])
        x['value'] = prex
        mn = x.rolling(window=self.k).mean()
        x['std'] = x.rolling(window=self.k).std()
        x['mean'] = mn
        x.fillna(0.0, inplace=True)
        movdata = [self.b for _ in range(x.shape[0])] + self.u * x['mean'].to_numpy() + self.c * x['std'].to_numpy()
        self.movdatas.append(movdata)
        if x['value'].tolist()[-1] > movdata[-1]:
            return 0
        else:
            return 1
