# from https://github.com/RuoyunCarina-D/Anomly-Detection-SARIMA/blob/master/SARIMA.py
import copy
import datetime
import itertools
import math
import sys
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
import antropy as ant
from tslearn.metrics import dtw

from analysis.preanalysis import periodicity_analysis
from utils import adjust_range
import plotly.graph_objects as go


class SARIMA:
    model = None
    dataset = ''

    def __init__(self, conf_top=1.5, conf_botton=1.5, train_size=3000):
        self.full_pred = []
        self.mean_fluctuation = []
        self.monitoring_pred = []
        self.conf_top = conf_top
        self.conf_botton = conf_botton
        self.threshold_modelling = False
        self.train_size = train_size
        self.latest_train_snippest = pd.DataFrame([])
        self.result_memory_size = 500
        self.memory_threshold = 500
        self.freq = None

    def _check_freq(self, idx):
        if self.freq:
            return self.freq
        my_freq = idx[1].to_pydatetime() - idx[0].to_pydatetime()
        return my_freq

    def _get_time_index(self, data):
        if self.dataset == 'yahoo':
            data.loc[:, 'timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
            data.set_index('timestamp', inplace=True)
            self.freq = data.index.inferred_freq
            return data
        if self.dataset in ['NAB']:
            data.loc[:, 'timestamp'] = pd.to_datetime(data['timestamp'], infer_datetime_format=True)
            # preprocessing depending on TS.....
            data.set_index('timestamp', inplace=True)
            data = data[~data.index.duplicated(keep='first')]

            self.freq = self._check_freq(data.index)
            data = data.asfreq(self.freq)

            return data
        if self.dataset in ['kpi']:
            data.loc[:, 'timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
            data.set_index('timestamp', inplace=True)
            data = data[~data.index.duplicated(keep='first')]

            self.freq = self._check_freq(data.index)
            data = data.asfreq(self.freq)
            return data

    def fit(self, data, dataset):
        period = 12
        # period = periodicity_analysis(data, self.dataset)

        if dataset != 'retrain':
            self.dataset = dataset
            data = self._get_time_index(data)

        # define the p, d and q parameters to take any value between 0 and 2
        p = d = q = range(0, 2)

        # generate all different combinations of p, d and q triplets
        pdq = list(itertools.product(p, d, q))

        # generate all different combinations of seasonal p, q and q triplets
        seasonal_pdq = [(x[0], x[1], x[2], period) for x in list(itertools.product(p, d, q))]

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
                    print('error 222')

        # print("Best SARIMAX{}x{}12 model - AIC:{}".format(best_pdq, best_seasonal_pdq, best_aic))
        self.model = self.model.fit()
        # print(self.model.summary())
        self.latest_train_snippest = data
        self.form_entropy(data)

    def form_entropy(self, data):
        collected_entropies = []
        entropy_differences = []
        entropies_no_anomalies = []

        for start in range(0, data.shape[0], 100):
            window = data.iloc[start:start + 100]

            try:
                collected_entropies.append(
                    ant.svd_entropy(window['value'].to_numpy(), normalize=True))
            except Exception as e:
                pass

            # if len(collected_entropies) > 1:
            #     entropy_differences.append(
            #         abs(collected_entropies[-1] - collected_entropies[-2]))

            # if window['is_anomaly'].tolist().count(1) == 0:
            #     entropies_no_anomalies.append(collected_entropies[-1])

        self.entropy_boundary_bottom = np.mean(collected_entropies) - np.std(collected_entropies)
        self.entropy_boundary_up = np.mean(collected_entropies) + np.std(collected_entropies)

    def predict(self, newdf, optimization=False, in_dp=False):
        y_pred = []
        newdf = self._get_time_index(newdf)
        print('=====================================')
        print(self.model.fittedvalues.tail())
        print(self.model.fittedvalues.shape)
        print(newdf.tail())
        print(optimization)

        try:
            if not optimization:
                if self.model.fittedvalues.shape[0] > self.memory_threshold:
                    latest_obs = pd.DataFrame([])
                    latest_obs['value'] = self.model.fittedvalues.values[-self.result_memory_size:]
                    latest_obs.index = self.model.fittedvalues.index[-self.result_memory_size:]
                    latest_obs = latest_obs.asfreq(self.freq)

                    if in_dp:
                        newdf = newdf[0:1]
                    joined = pd.concat([latest_obs, newdf])
                    joined = joined.asfreq(self.freq)
                    print('1~~~~~~~~')
                    print(joined)
                    print(joined.shape)
                    self.model = self.model.apply(joined, refit=False)
                else:
                    if in_dp:
                        newdf = newdf[0:1]
                    print('2==============')
                    print(newdf)
                    print(newdf.shape)
                    self.model = self.model.append(newdf)
                    # self.model = self.model.apply(newdf, refit=False)

        except ValueError as e:
            if not optimization:
                stuffed_value = pd.DataFrame([])
                stuffed_value['timestamp'] = pd.date_range(start=self.model.fittedvalues.index.max(), end=newdf.index.min(),
                                                           freq=self.freq)
                stuffed_value['value'] = None
                stuffed_value.set_index('timestamp', inplace=True)
                stuffed_value = stuffed_value[1:-1]
                newdf = pd.concat([stuffed_value, newdf])
                if in_dp:
                    newdf = newdf[0:1]
                newdf = newdf.asfreq(self.freq)
                print('3==============')
                print(newdf)
                print(newdf.shape)
                self.model = self.model.append(newdf.astype(float))

        print('4==============')
        print(newdf)
        print(newdf.shape)
        print(self.model.fittedvalues.shape)
        pred = self.model.get_prediction(start=newdf.index.min(), end=newdf.index.max(),
                                         dynamic=False, alpha=0.01)

        if not in_dp:
            print('not in dp')
            print(pred)
            self.full_pred.append(pred)
        else:
            try:
                pred_point = self.model.get_prediction(start=newdf.index.min(), end=newdf.index.tolist()[1],
                                                       dynamic=False, alpha=0.01)
            except:
                pred_point = self.model.get_prediction(start=newdf.index.min(), end=newdf.index.max(),
                                                       dynamic=False, alpha=0.01)

            self.full_pred.append(pred_point)

        pred_ci = pred.conf_int()
        deanomalized_window = pd.DataFrame([])

        anomalies_count = 0
        anomaly_idxs = []

        print('calculating anomalies')

        for idx, row in pred_ci.iterrows():
            value = None
            if str(newdf.loc[idx, 'value']).lower() not in ['nan', 'none', '']:
                if adjust_range(row['lower value'], 'div', self.conf_botton) <= newdf.loc[idx, 'value'] \
                        <= adjust_range(row['upper value'], 'mult', self.conf_top):
                    y_pred.append(0)
                    value = newdf.loc[idx, 'value']
                else:
                    anomalies_count += 1
                    anomaly_idxs.append(idx)
                    y_pred.append(1)
                    newdf.loc[idx, 'value'] = None

            deanomalized_window.loc[idx, 'value'] = value

        self.latest_train_snippest = pd.concat([self.latest_train_snippest, deanomalized_window])[-self.train_size:]

        try:
            current_entropy = ant.svd_entropy(newdf['value'].to_numpy(), normalize=True)
            entropy_identifier = self.entropy_boundary_bottom <= current_entropy <= self.entropy_boundary_up
        except:
            entropy_identifier = True

        print('round done')

        return y_pred, entropy_identifier

    def get_pred_mean(self):
        pred_thr = pd.DataFrame([])
        for _, pred in enumerate(self.full_pred):
            pred_thr = pd.concat([pred_thr, pred.conf_int()])

        return pred_thr[['lower value', 'upper value']]

    def plot_ensemble(self, y, dataset, datatype, filename, full_test_data, drift_windows, detected_anomalies):
        detected_anomalies = pd.to_datetime(pd.Series(detected_anomalies), unit='s').tolist()
        full_test_data = self._get_time_index(full_test_data)

        y = self._get_time_index(y)
        y = y.dropna(subset=['value'])
        ax = y['value'].plot(label='observed', zorder=1)

        pred_ci = pd.concat([x.conf_int() for x in self.full_pred], axis=0)
        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0].apply(lambda x: adjust_range(x, 'div', self.conf_botton)),
                        pred_ci.iloc[:, 1].apply(lambda x: adjust_range(x, 'mult', self.conf_top)),
                        color='b', alpha=.2, zorder=0)

        for tm in detected_anomalies:
            if full_test_data.loc[tm, 'is_anomaly'] == 0:
                ax.scatter(tm, y.loc[tm, 'value'], color='r')
        for tm, row in full_test_data[full_test_data['is_anomaly'] == 1].iterrows():
            if tm not in detected_anomalies:
                ax.scatter(tm, y.loc[tm, 'value'], color='darkmagenta')

        for wd in drift_windows:
            ax.axvspan(wd[0], wd[1], alpha=0.3, color='red')
        ax.set_xlabel('Date')
        ax.set_ylabel('Values')
        plt.legend()
        plt.savefig(f'results/imgs/{dataset}/{datatype}/dp/sarima_{filename.replace(".csv", "")}_full.png')

        plt.close('all')
        plt.clf()

        self.full_pred = []

    def plot(self, y, dataset, datatype, filename, full_test_data, drift_windows):
        print('plotting....')
        y = self._get_time_index(y)
        full_test_data = self._get_time_index(full_test_data)
        y = y.dropna(subset=['value'])
        ax = y['value'].plot(label='observed')

        idx = 1
        for _, pred in enumerate(self.full_pred):
            pred_ci = pred.conf_int()
            if pred_ci.index.max() in y.index or pred_ci.index.min() in y.index:
                pred.predicted_mean.plot(ax=ax, label=f'Window {idx} forecast', alpha=.7, figsize=(14, 7))
                ax.fill_between(pred_ci.index,
                                pred_ci.iloc[:, 0].apply(lambda x: adjust_range(x, 'div', self.conf_botton)),
                                pred_ci.iloc[:, 1].apply(lambda x: adjust_range(x, 'mult', self.conf_top)),
                                color='b', alpha=.2)

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

        for wd in drift_windows:
            ax.axvspan(wd[0], wd[1], alpha=0.3, color='red')
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

    def __init__(self, sims=10):
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
        simulations['timestamp'] = newdf.index
        simulations = self._get_time_index(simulations)
        self.full_pred.append(simulations[['upper value', 'lower value', 'pred']])

        for idx, row in simulations.iterrows():
            try:
                if row['lower value'] <= newdf.loc[idx, 'value'] <= row['upper value']:
                    y_pred.append(0)
                else:
                    y_pred.append(1)
            except Exception as e:
                raise e

        return y_pred

    def plot(self, y, dataset, datatype, filename, full_test_data_, drift_windows):
        full_test_data = self._get_time_index(full_test_data_)
        ax = full_test_data['value'].plot(label='observed')

        for w, fpred in enumerate(self.full_pred):
            ax.fill_between(fpred.index,
                            fpred['lower value'],
                            fpred['upper value'], color='k', alpha=.2)

            print(fpred)
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

        for wd in drift_windows:
            ax.axvspan(wd[0], wd[1], alpha=0.3, color='red')

        ax.set_xlabel('Date')
        ax.set_ylabel('Values')
        plt.legend()
        plt.savefig(f'results/imgs/{dataset}/{datatype}/es/es_{filename.replace(".csv", "")}_smooth.png')

        plt.close('all')
        plt.clf()
        self.full_pred = []
