# twist on https://github.com/RuoyunCarina-D/Anomly-Detection-SARIMA/blob/master/SARIMA.py
import copy
import datetime
import itertools
import time

import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import matplotlib.pyplot as plt
import antropy as ant

from drift_detectors.ecdd import ECDD
from settings import entropy_params
from utils import adjust_range


class SARIMA:

    def __init__(self, dataset, datatype, filename, conf_top=1.5, conf_bottom=1.5,
                 train_size=3000, w=0.25, c=0.25, drift_count_limit=10):
        self.full_pred = []
        self.mean_fluctuation = []
        self.monitoring_pred = []
        self.conf_top = conf_top
        self.conf_bottom = conf_bottom
        self.threshold_modelling = False
        self.train_size = train_size
        self.latest_train_snippest = pd.DataFrame([])
        self.result_memory_size = 500
        self.memory_threshold = 500
        self.freq = None
        self.datatype = datatype
        self.dataset = dataset
        self.filename = filename.replace(".csv", "")
        self.entropy_factor = entropy_params[f'{dataset}_{datatype}']['factor']
        self.entropy_window = entropy_params[f'{dataset}_{datatype}']['window']
        self.w = w
        self.c = c
        self.drift_detector = ECDD(0.2, w, c)
        self.is_drift = False
        self.drift_alerting_cts = 0
        self.drift_count_limit = drift_count_limit
        self.use_drift_adaptation = True

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
            data = data.asfreq(self.freq, method='ffill')
            return data
        if self.dataset in ['kpi']:
            try:
                data['timestamp'] = data['timestamp'].apply(lambda x:
                                                            datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
            except:
                pass
            data.loc[:, 'timestamp'] = pd.to_datetime(data['timestamp'], unit='T')
            data.set_index('timestamp', inplace=True)
            data = data[~data.index.duplicated(keep='first')]
            self.freq = self._check_freq(data.index)
            data = data.asfreq(self.freq)
            return data

    def fit(self, data_train, dataset=None):
        if dataset != 'retrain':
            data = copy.deepcopy(data_train[['timestamp', 'value']])
            self.form_entropy(data)
            data = self._get_time_index(data)
        else:
            data = data_train

        data = data[-5000:]
        period = 12

        # define the p, d and q parameters to take any value between 0 and 2
        p = d = q = range(0, 2)

        # generate all different combinations of p, d and q triplets
        pdq = list(itertools.product(p, d, q))

        # generate all different combinations of seasonal p, q and q triplets
        seasonal_pdq = [(x[0], x[1], x[2], period) for x in list(itertools.product(p, d, q))]

        best_aic = np.inf

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
                        self.model = tmp_mdl

                except Exception as e:
                    print(e)

        self.model = self.model.fit()
        loss = [abs(x - y) for x, y in zip(data['value'].tolist(), self.model.get_prediction().predicted_mean)]
        self.drift_detector.record(np.mean(loss), np.std(loss))
        print(data.tail())
        print(self.model.fittedvalues.tail())
        print('=======')

    def form_entropy(self, data):
        collected_entropies = []

        for start in range(0, data.shape[0], self.entropy_window):
            window = data.iloc[start:start + self.entropy_window]

            try:
                collected_entropies.append(
                    ant.svd_entropy(window['value'].to_numpy(), normalize=True))
            except Exception as e:
                pass

        self.mean_entropy = np.mean([v for v in collected_entropies if pd.notna(v)])
        self.collected_entropies = collected_entropies
        self.entropy_boundary_bottom = self.mean_entropy - \
                                       self.entropy_factor * \
                                       np.std([v for v in collected_entropies if pd.notna(v)])
        self.entropy_boundary_up = self.mean_entropy + \
                                   self.entropy_factor * \
                                   np.std([v for v in collected_entropies if pd.notna(v)])

    def predict(self, newdf, optimization=False, in_dp=False):
        newdf = newdf[['timestamp', 'value']]
        y_pred = []
        y_pred_e = []
        newdf = self._get_time_index(newdf)
        print(newdf.head())
        print('))))))))))))))))))))))))')

        try:
            current_entropy = ant.svd_entropy(newdf['value'].to_numpy(), normalize=True)
            entropy_identifier = self.entropy_boundary_bottom <= current_entropy <= self.entropy_boundary_up
            extent = stats.percentileofscore(self.collected_entropies, current_entropy) / 100.0
            extent = 1.5 - max(extent, 1.0 - extent)
            conf_bottom_e = adjust_range(self.conf_bottom, 'div', extent)
            conf_top_e = adjust_range(self.conf_top, 'mult', extent)
        except:
            entropy_identifier = True
            conf_bottom_e = self.conf_bottom
            conf_top_e = self.conf_top

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
                    self.model = self.model.apply(joined, refit=False)
                else:
                    if in_dp:
                        newdf = newdf[0:1]
                    self.model = self.model.append(newdf)

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
                self.model = self.model.append(newdf.astype(float))
        print(newdf.head())
        print(self.model.fittedvalues.tail())
        print('*****************************')
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

            #self.full_pred.append(pred_point)

        pred_ci = pred.conf_int()

        print('calculating anomalies')

        for idx, row in pred_ci.iterrows():
            if str(newdf.loc[idx, 'value']).lower() not in ['nan', 'none', '']:
                print(idx)
                print(type(idx))
                try:
                    t = datetime.datetime.strptime(str(idx), '%Y-%m-%d %H:%M:%S').timestamp()
                except:
                    t = idx
                error = abs(newdf.loc[idx, 'value'] - pred.predicted_mean.loc[idx])
                self.drift_detector.update_ewma(error=error, t=t)
                response = self.drift_detector.monitor()
                print('**', error, response)
                if response == self.drift_detector.drift:
                    self.drift_alerting_cts += 1

                if adjust_range(row['lower value'], 'div', self.conf_bottom) <= newdf.loc[idx, 'value'] \
                        <= adjust_range(row['upper value'], 'mult', self.conf_top):
                    y_pred.append(0)
                    if entropy_identifier:
                        y_pred_e.append(0)
                else:
                    y_pred.append(1)
                    if entropy_identifier:
                        y_pred_e.append(1)

                if not entropy_identifier:
                    if adjust_range(row['lower value'], 'div', conf_bottom_e) <= newdf.loc[idx, 'value'] \
                            <= adjust_range(row['upper value'], 'mult', conf_top_e):
                        y_pred_e.append(0)
                    else:
                        y_pred_e.append(1)

        # self.latest_train_snippest = pd.concat([self.latest_train_snippest, newdf])[-self.train_size:]
        print('round done')
        print('++', self.drift_alerting_cts, self.drift_count_limit)
        if self.use_drift_adaptation:
            if self.drift_alerting_cts >= self.drift_count_limit:
                print('Drift detected: retraining')
                start_time = time.time()
                self.drift_alerting_cts = 0
                self.drift_detector = ECDD(0.2, self.w, self.c)
                self.drift_detector.record(np.mean(newdf['value'].tolist()), np.std(newdf['value'].tolist()))
                self.fit(newdf, 'retrain')
                end_time = time.time()
                diff = end_time - start_time
                print(f"Trained sarima for {diff}")

        return y_pred, y_pred_e

    def get_pred_mean(self):
        pred_thr = pd.DataFrame([])
        for _, pred in enumerate(self.full_pred):
            pred_thr = pd.concat([pred_thr, pred.conf_int()])

        return pred_thr[['lower value', 'upper value']]

    def plot(self, datatest):
        full_test_data = copy.deepcopy(datatest)
        y = self._get_time_index(datatest[['timestamp', 'value']])
        full_test_data = self._get_time_index(full_test_data)
        y = y.dropna(subset=['value'])
        ax = y['value'].plot(label='observed')

        idx = 1
        for _, pred in enumerate(self.full_pred):
            pred_ci = pred.conf_int()
            if pred_ci.index.max() in y.index or pred_ci.index.min() in y.index:
                pred.predicted_mean.plot(ax=ax, label=f'Window {idx} forecast', alpha=.7, figsize=(14, 7))
                ax.fill_between(pred_ci.index.tolist(),
                                pred_ci['lower value'].apply(lambda x: adjust_range(x, 'div', self.conf_bottom)).tolist(),
                                pred_ci['upper value'].apply(lambda x: adjust_range(x, 'mult', self.conf_top)).tolist(),
                                color='b', alpha=.2)

            for tm, row in pred_ci.iterrows():
                if tm in y.index:
                    if (adjust_range(row['lower value'], 'div', self.conf_bottom) > y.loc[tm, 'value'] or
                        y.loc[tm, 'value'] > adjust_range(row['upper value'], 'mult', self.conf_top)) and \
                            full_test_data.loc[tm, 'is_anomaly'] == 0:
                        ax.scatter(tm, y.loc[tm, 'value'], color='r')
                    if (adjust_range(row['lower value'], 'div', self.conf_bottom) <= y.loc[tm, 'value'] <=
                        adjust_range(row['upper value'], 'mult', self.conf_top)) \
                            and full_test_data.loc[tm, 'is_anomaly'] == 1:
                        ax.scatter(tm, y.loc[tm, 'value'], color='darkmagenta')

            idx += 1

        ax.set_xlabel('Date')
        ax.set_ylabel('Values')
        plt.legend()
        plt.savefig(f'results/imgs/{self.dataset}/{self.datatype}/sarima_{self.filename}_forecast.png')

        plt.close('all')
        plt.clf()

        self.full_pred = []
