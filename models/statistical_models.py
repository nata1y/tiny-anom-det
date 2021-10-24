# from https://github.com/RuoyunCarina-D/Anomly-Detection-SARIMA/blob/master/SARIMA.py
import copy
import itertools
import math
from collections import Counter

import funcy
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import mean_squared_log_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import matplotlib.pyplot as plt
from tslearn.metrics import dtw

from utils import adjust_range
import plotly.graph_objects as go


class SARIMA:
    model = None
    dataset = ''

    def __init__(self, conf_top=1.5, conf_botton=1.5, train_size=3000):
        self.full_pred = []
        self.conf_top = conf_top
        self.conf_botton = conf_botton
        self.threshold_modelling = False
        self.train_size = train_size
        self.latest_train_snippest = pd.DataFrame([])

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
        if dataset != 'retrain':
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
        self.latest_train_snippest = data

    def predict(self, newdf, retrain=False):
        y_pred = []
        # print(newdf)
        newdf = self._get_time_index(newdf)
        # newdf_ = newdf.dropna(subset=['value'])

        # # add new observation
        # refit=True
        print(self.model.fittedvalues)
        print(newdf)
        print('%%%%')
        ################################################################################################################
        # old_model = copy.deepcopy(self.model)
        ################################################################################################################
        self.model = self.model.append(newdf)
        pred = self.model.get_prediction(start=newdf.index.min(), end=newdf.index.max(),
                                         dynamic=False, alpha=0.01)

        self.full_pred.append(pred)
        pred_ci = pred.conf_int()
        deanomalized_window = pd.DataFrame([])

        # play around with lower value of threshold
        # if self.threshold_modelling:
        #     pred_ci['lower value'] = 0.0
        has_anomalies = False
        anomaly_idxs = []

        for idx, row in pred_ci.iterrows():
            value = None
            if str(newdf.loc[idx, 'value']).lower() not in ['nan', 'none', '']:
                if adjust_range(row['lower value'], 'div', self.conf_botton) <= newdf.loc[idx, 'value'] \
                        <= adjust_range(row['upper value'], 'mult', self.conf_top):
                    y_pred.append(0)
                    value = newdf.loc[idx, 'value']
                else:
                    has_anomalies = True
                    anomaly_idxs.append(idx)
                    y_pred.append(1)
            deanomalized_window.loc[idx, 'value'] = value

        ################################################################################################################
        # if has_anomalies or True:
        #     model = DBSCAN(eps=abs(np.max(pred_ci['upper value']) - np.min(pred_ci['lower value'])),
        #                    min_samples=1, metric=dtw)
        #     y_pred = model.fit_predict(newdf_)
        #     common = Counter(funcy.lflatten(y_pred)).most_common(1)[0][0]
        #     for pred, (idx, row) in zip(y_pred, newdf.iterrows()):
        #         if idx not in newdf_.index or pred != common:
        #             deanomalized_window.loc[idx, 'value'] = None
        #         else:
        #             deanomalized_window.loc[idx, 'value'] = row['value']
        ################################################################################################################

        self.model = old_model.append(deanomalized_window)

        # retrain on anomaly but throw away anomalies from dataset
        self.latest_train_snippest = pd.concat([self.latest_train_snippest, deanomalized_window])[-self.train_size:]

        if retrain:
            if y_pred.count(1) > 0:
                self.fit(self.latest_train_snippest, 'retrain')

        return y_pred

    def plot_threshold(self, y, dataset, datatype, filename, full_test_data, model):
        print(y)
        y = y[y['timestamp'] > 1500200000]
        y = y[y['timestamp'] < 1500300000]

        y = self._get_time_index(y)
        full_test_data = self._get_time_index(full_test_data)
        y = y.dropna(subset=['value'])
        print(y)
        print('====================================================')
        ax = y['value'].plot(label='observed')

        idx = 1
        for _, pred in enumerate(self.full_pred):
            pred_ci = pred.conf_int()

            # play around with lower value of threshold
            if self.threshold_modelling:
                pred_ci['lower value'] = 0.0

            print(pred_ci)
            if pred_ci.index.max() in y.index or pred_ci.index.min() in y.index:

                pred.predicted_mean.plot(ax=ax, label=f'Window {idx} forecast', alpha=.7, figsize=(14, 7))
                ax.fill_between(pred_ci.index,
                                pred_ci.iloc[:, 0].apply(lambda x: adjust_range(x, 'div', self.conf_botton)),
                                pred_ci.iloc[:, 1].apply(lambda x: adjust_range(x, 'mult', self.conf_top)),
                                color='k', alpha=.2)

            for tm, row in pred_ci.iterrows():
                if tm in y.index:
                    if (adjust_range(row['lower value'], 'div', self.conf_botton) > y.loc[tm, 'value'] or
                        y.loc[tm, 'value'] > adjust_range(row['upper value'], 'mult', self.conf_top)) and \
                            full_test_data.loc[tm, 'is_anomaly'] == 0:
                        ax.scatter(tm, y.loc[tm, 'value'], color='r')
                    if (adjust_range(row['lower value'], 'div', self.conf_botton) <= y.loc[tm, 'value'] <=
                        adjust_range(row['upper value'], 'mult', self.conf_top)) \
                            and full_test_data.loc[tm, 'is_anomaly'] == 1:
                        ax.scatter(tm, y.loc[tm, 'value'], color='darkmagenta')

            idx += 1

        ax.set_xlabel('Date')
        ax.set_ylabel('Values')
        plt.legend()
        plt.savefig(f'results/imgs/{dataset}/{datatype}/{model}/{model}_{filename.replace(".csv", "")}_threshold_via_sarima_snippest.png')

        plt.close('all')
        plt.clf()
        self.full_pred = []
        return y

    def plot(self, y, dataset, datatype, filename, full_test_data):
        y = y[y['timestamp'] > 1500200000]
        y = y[y['timestamp'] < 1500300000]

        y = self._get_time_index(y)
        full_test_data = self._get_time_index(full_test_data)
        y = y.dropna(subset=['value'])
        print(y)
        print('====================================================')
        ax = y['value'].plot(label='observed')

        idx = 1
        for _, pred in enumerate(self.full_pred):
            pred_ci = pred.conf_int()

            # play around with lower value of threshold
            if pred_ci.index.max() in y.index or pred_ci.index.min() in y.index:
                pred.predicted_mean.plot(ax=ax, label=f'Window {idx} forecast', alpha=.7, figsize=(14, 7))
                ax.fill_between(pred_ci.index,
                                pred_ci.iloc[:, 0].apply(lambda x: adjust_range(x, 'div', self.conf_botton)),
                                pred_ci.iloc[:, 1].apply(lambda x: adjust_range(x, 'mult', self.conf_top)),
                                color='k', alpha=.2)

            for tm, row in pred_ci.iterrows():
                if tm in y.index:
                    if (adjust_range(row['lower value'], 'div', self.conf_botton) > y.loc[tm, 'value'] or
                        y.loc[tm, 'value'] > adjust_range(row['upper value'], 'mult', self.conf_top)) and \
                            full_test_data.loc[tm, 'is_anomaly'] == 0:
                        ax.scatter(tm, y.loc[tm, 'value'], color='r')
                    if (adjust_range(row['lower value'], 'div', self.conf_botton) <= y.loc[tm, 'value'] <=
                        adjust_range(row['upper value'], 'mult', self.conf_top)) \
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

    def plot(self, y, dataset, datatype, filename, full_test_data_):
        full_test_data = self._get_time_index(full_test_data_)
        ax = full_test_data['value'].plot(label='observed')

        for w, fpred in enumerate(self.full_pred):
            ax.fill_between(fpred.index,
                            fpred['lower value'],
                            fpred['upper value'], color='k', alpha=.2)

            fpred['pred'].plot(label=f'Window {w} forecast', alpha=.7, figsize=(14, 7))
            for idx, pred in fpred.iterrows():
                try:
                    if idx in full_test_data.index:
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
        plt.savefig(f'results/imgs/{dataset}/{datatype}/es/es_{filename.replace(".csv", "")}_train.png')

        plt.close('all')
        plt.clf()
        self.full_pred = []
