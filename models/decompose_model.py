import copy
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt

from analysis.preanalysis import periodicity_analysis
import plotly.graph_objects as go


class DecomposeResidual:
    model = None
    dataset = ''

    def __init__(self, train_size=20000):
        self.total_residuals = pd.Series([])
        self.train_size = train_size
        self.latest_train_snippest = pd.DataFrame([])
        self.end_train_time = 0

    def _get_time_index(self, data):
        if self.dataset in ['yahoo']:
            data.loc[:, 'timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
        if self.dataset in ['kpi']:
            data.loc[:, 'timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
        data.set_index('timestamp', inplace=True)
        if self.dataset in ['kpi']:
            data = data.asfreq('T')
            if self.model:
                dti = pd.DataFrame([])
                dti.loc[:, 'timestamp'] = pd.date_range(self.model.fittedvalues.index.max() + pd.Timedelta(minutes=1),
                                                        periods=data.shape[0], freq="T")
                dti.set_index('timestamp', inplace=True)
                dti['value'] = None
                res = pd.concat((data, dti)).sort_values(by='value')
                res = res[~res.index.duplicated(keep='first')]
                res = res.sort_index()
                return res
            # data.dropna(inplace=True)
        return data

    def fit(self, data, dataset):
        self.period = periodicity_analysis(data, self.dataset)
        self.latest_train_snippest = data[-self.train_size:]

        if dataset != 'retrain':
            self.dataset = dataset
            data = self._get_time_index(data)
        ############################################# Seasaonl decomposing #############################################
        res = seasonal_decompose(data.interpolate(), model='additive', freq=self.period)
        res.plot()
        plt.savefig(f'results/imgs/preanalysis/{dataset}_add_decomp.png')

        data['value'] = res.resid
        dres = [r for r in res.resid.tolist() if str(r) != 'nan']
        self.threshold = st.t.interval(alpha=0.999, df=len(dres)-1,
                                                   loc=np.mean(dres),
                                                   scale=st.sem(dres))
        self.threshold = (self.threshold[0] * 500, self.threshold[1] * 500)
        print(self.threshold)
        self.end_train_time = data.index.max()

    def predict(self, newdf):
        # print(newdf)
        self.latest_train_snippest = pd.concat([self.latest_train_snippest, newdf])[-self.train_size:]

        data = self._get_time_index(copy.deepcopy(self.latest_train_snippest)).dropna()
        print(data)
        ############################################# Seasaonl decomposing #############################################
        res = seasonal_decompose(data.interpolate(), model='additive', freq=self.period)
        res.plot()
        plt.savefig(f'results/imgs/preanalysis/{self.dataset}_add_decomp2.png')
        y_pred = []
        for p in res.resid:
            if p > self.threshold[1] or p < self.threshold[0]:
                y_pred.append(1)
            else:
                y_pred.append(0)

        self.total_residuals = pd.concat([self.total_residuals, res.resid.dropna()[-newdf.shape[0]:]])
        return y_pred[-newdf.shape[0]:]

    def plot(self, datatype, filename, time):
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=self.total_residuals.index, y=self.total_residuals,
                                 name='Decomposed residuals'))
        fig.add_trace(
            go.Scatter(x=self.total_residuals.index, y=[self.threshold[0] for _ in range(len(self.total_residuals))],
                       name='Threshold Bottom'))
        fig.add_trace(
            go.Scatter(x=self.total_residuals.index, y=[self.threshold[1] for _ in range(len(self.total_residuals))],
                       name='Threshold Up'))
        fig.update_layout(showlegend=True, title='Residuals vs. Threshold')
        fig.write_image(
            f'results/imgs/{self.dataset}/{datatype}/seasonal_decomp/seasonal_decomp_{filename.replace(".csv", "")}_residuals.png')

        fig.data = []

        # if self.dataset == 'kpi!':
        #     dti = pd.DataFrame([])
        #     dti.loc[:, 'timestamp'] = pd.date_range(start=self.end_train_time + pd.Timedelta(minutes=1),
        #                                             periods=time.shape[0],
        #                                             freq="T")
        #     dti.set_index('timestamp', inplace=True)
        #     dti['value'] = 0.0
        #     # print(self.total_residuals)
        #     res = pd.concat((self.total_residuals, dti)).sort_values(by='value')
        #     res = res[~res.index.duplicated(keep='first')]
        #     res = res.sort_index()
        #     res = res[res.index > self.end_train_time]['value']
        #     print(res.shape)
        #     print(res.index.min(), res.index.max())
        #     res = res.tolist()
        #
        # else:
        #     print(self.total_residuals.shape)
        #     print(time.shape)
        #     print(time.index.min(), time.index.max())
        #     self.total_residuals = self.total_residuals[~self.total_residuals.index.duplicated(keep='first')]
        #     print(self.total_residuals.shape)
        #
        #     print(self.total_residuals.index.min(), self.total_residuals.index.max())
        #     res = self.total_residuals[self.total_residuals.index > self.end_train_time]
        #     print(res.shape)
        #     print(res.index.min(), res.index.max())
        #     res = res.tolist()
        res = self.total_residuals.tolist()

        return [0 if self.threshold[1] > r > self.threshold[0] else 1 for r in res]
