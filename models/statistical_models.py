# from https://github.com/RuoyunCarina-D/Anomly-Detection-SARIMA/blob/master/SARIMA.py
import copy
import itertools
import math

import numpy as np
from sklearn.metrics import mean_squared_log_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import matplotlib.pyplot as plt

from utils import adjust_range


class SARIMA:
    model = None
    dataset = ''

    def __init__(self, conf=1.5):
        self.full_pred = []
        self.conf = conf

    def _get_time_index(self, data):
        if self.dataset in ['yahoo', 'kpi']:
            data.loc[:, 'timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
        data.set_index('timestamp', inplace=True)
        return data

    def fit(self, data, dataset):
        self.dataset = dataset
        data = self._get_time_index(data)
        ############################################# Seasaonl decomposing #############################################
        # res = seasonal_decompose(data.interpolate(), model='additive')
        # res.plot()

        ############################################# Model Selection###################################################

        # define the p, d and q parameters to take any value between 0 and 2
        p = d = q = range(0, 2)

        # generate all different combinations of p, d and q triplets
        pdq = list(itertools.product(p, d, q))

        # generate all different combinations of seasonal p, q and q triplets
        seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

        best_aic = np.inf
        best_pdq = None
        best_seasonal_pdq = None

        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    tmp_mdl = SARIMAX(data,
                                      order=param,
                                      seasonal_order=param_seasonal,
                                      enforce_stationarity=True,
                                      enforce_invertibility=False)

                    res = tmp_mdl.fit()
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_pdq = param
                        best_seasonal_pdq = param_seasonal
                        self.model = tmp_mdl

                except Exception as e:
                    print(e)

        print("Best SARIMAX{}x{}12 model - AIC:{}".format(best_pdq, best_seasonal_pdq, best_aic))
        self.model = self.model.fit()

    def predict(self, newdf):
        y_pred = []
        newdf = self._get_time_index(newdf)

        # add new observation
        self.model = self.model.append(newdf)

        pred = self.model.get_prediction(start=newdf.index.min(), end=newdf.index.max(), dynamic=False, alpha=0.05)
        self.full_pred.append(pred)
        pred_ci = pred.conf_int()

        for idx, row in pred_ci.iterrows():
            if adjust_range(row['lower value'], 'div', self.conf) <= newdf.loc[idx, 'value'] \
                    <= adjust_range(row['upper value'], 'mult', self.conf):
                y_pred.append(0)
            else:
                y_pred.append(1)

        return y_pred

    def plot(self, y, dataset, datatype, filename, full_test_data):
        y = self._get_time_index(y)
        full_test_data = self._get_time_index(full_test_data)
        ax = y['value'].plot(label='observed')

        idx = 1
        for _, pred in enumerate(self.full_pred):
            pred_ci = pred.conf_int()

            pred.predicted_mean.plot(ax=ax, label=f'Window {idx} forecast', alpha=.7, figsize=(14, 7))
            ax.fill_between(pred_ci.index,
                            pred_ci.iloc[:, 0].apply(lambda x: adjust_range(x, 'div', self.conf)),
                            pred_ci.iloc[:, 1].apply(lambda x: adjust_range(x, 'mult', self.conf)),
                            color='k', alpha=.2)

            for tm, row in pred_ci.iterrows():
                if (adjust_range(row['lower value'], 'div', self.conf) > y.loc[tm, 'value'] or
                    y.loc[tm, 'value'] > adjust_range(row['upper value'], 'mult', self.conf)) and \
                        full_test_data.loc[tm, 'is_anomaly'] == 0:
                    ax.scatter(tm, y.loc[tm, 'value'], color='r')
                if (adjust_range(row['lower value'], 'div', self.conf) <= y.loc[tm, 'value'] <=
                    adjust_range(row['upper value'], 'mult', self.conf)) \
                        and full_test_data.loc[tm, 'is_anomaly'] == 1:
                    ax.scatter(tm, y.loc[tm, 'value'], color='darkmagenta')

            idx += 1

        ax.set_xlabel('Date')
        ax.set_ylabel('Values')
        plt.legend()
        plt.savefig(f'results/imgs/{dataset}/{datatype}/sarima/sarima_{filename.replace(".csv", "")}_full.png')

        plt.close('all')
        plt.clf()
        self.full_pred = []


class ExpSmoothing:
    model = None
    dataset = ''
    full_pred = []

    def __init__(self, sims=100):
        self.full_pred = []
        self.sims = sims

    def _get_time_index(self, data_):
        data = copy.deepcopy(data_)
        if self.dataset in ['yahoo', 'kpi']:
            data.loc[:, 'timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
        return data.set_index('timestamp')

    def fit(self, data, dataset):
        self.dataset = dataset
        data = self._get_time_index(data)
        self.model = ExponentialSmoothing(data,
                                          seasonal_periods=12,
                                          trend="add",
                                          seasonal="add",
                                          damped_trend=True,
                                          initialization_method="estimated",
                                          ).fit()

    def predict(self, newdf, window):
        y_pred = []
        newdf = self._get_time_index(newdf)

        pred = self.model.forecast(min(window, newdf.shape[0]))
        simulations = self.model.simulate(min(window, newdf.shape[0]), repetitions=self.sims, error="add")
        simulations['upper value'] = simulations.max(axis=1)
        simulations['lower value'] = simulations.min(axis=1)
        simulations['pred'] = pred
        self.full_pred.append(simulations[['upper value', 'lower value', 'pred']])
        simulations['timestamp'] = newdf.index
        simulations = self._get_time_index(simulations)

        for idx, row in simulations.iterrows():
            try:
                if row['lower value'] <= newdf.loc[idx, 'value'] <= row['upper value']:
                    y_pred.append(0)
                else:
                    y_pred.append(1)
            except Exception as e:
                print(idx)
                print(simulations.shape)
                print(newdf.shape)
                raise e

        return y_pred

    def plot(self, y, dataset, datatype, filename, full_test_data):
        full_test_data = self._get_time_index(full_test_data)
        ax = full_test_data['value'].plot(label='observed')

        for w, fpred in enumerate(self.full_pred):
            ax.fill_between(fpred.index,
                            fpred['lower value'],
                            fpred['upper value'], color='k', alpha=.2)

            fpred['pred'].plot(label=f'Window {w} forecast', alpha=.7, figsize=(14, 7))
            for idx, pred in fpred.iterrows():
                try:
                    if (pred['lower value'] > full_test_data.loc[idx, 'value'] or
                        full_test_data.loc[idx, 'value'] > pred['upper value']) and \
                            full_test_data.loc[idx, 'is_anomaly'] == 0:
                        ax.scatter(idx, full_test_data.loc[idx, 'value'], color='r')
                    if (pred['lower value'] <= full_test_data.loc[idx, 'value'] <= pred['upper value']) \
                            and full_test_data.loc[idx, 'is_anomaly'] == 1:
                        ax.scatter(idx, full_test_data.loc[idx, 'value'], color='darkmagenta')
                except Exception as e:
                    print('==========================================================')
                    print(self.full_pred)
                    print(pred)
                    print(idx)
                    raise e

        ax.set_xlabel('Date')
        ax.set_ylabel('Values')
        plt.legend()
        plt.savefig(f'results/imgs/{dataset}/{datatype}/es/es_{filename.replace(".csv", "")}_full.png')

        plt.close('all')
        plt.clf()
        self.full_pred = []
