# from https://github.com/RuoyunCarina-D/Anomly-Detection-SARIMA/blob/master/SARIMA.py

import itertools
import numpy as np
from sklearn.metrics import mean_squared_log_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import matplotlib.pyplot as plt


class SARIMA:
    model = None
    dataset = ''
    full_pred = []

    def _get_time_index(self, data):
        if self.dataset == 'yahoo':
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
        pred = self.model.get_prediction(start=newdf.index.min(), dynamic=False, alpha=0.001)
        self.full_pred.append(pred)
        pred_ci = pred.conf_int()
        for idx, row in pred_ci.iterrows():
            if row['lower value'] <= newdf.loc[idx, 'value'] <= row['upper value']:
                y_pred.append(0)
            else:
                y_pred.append(1)

        return y_pred

    def plot(self, y, dataset, datatype, filename, full_test_data):
        y = self._get_time_index(y)
        # 0 1970-01-01 00:00:01  0.000000
        # yahoo dataset only!!!
        # TODO: fix
        ax = y['1970':].plot(label='observed')

        idx = 1
        idx2 = 0
        for _, pred in enumerate(self.full_pred):
            pred_ci = pred.conf_int()
            pred.predicted_mean.plot(ax=ax, label=f'Window {idx} forecast', alpha=.7, figsize=(14, 7))
            ax.fill_between(pred_ci.index,
                            pred_ci.iloc[:, 0],
                            pred_ci.iloc[:, 1], color='k', alpha=.2)

            for tm, row in pred_ci.iterrows():
                if (row['lower value'] > y['value'].tolist()[idx2] or
                    y['value'].tolist()[idx2] > row['upper value']) and \
                        full_test_data['is_anomaly'].tolist()[idx2] == 0:
                    ax.scatter(tm, y.loc[tm, 'value'], color='r')
                if (row['lower value'] <= y['value'].tolist()[idx2] <= row['upper value']) \
                        and full_test_data['is_anomaly'].tolist()[idx2] == 1:
                    ax.scatter(tm, y.loc[tm, 'value'], color='darkmagenta')
                idx2 += 1

            idx += 1

        ax.set_xlabel('Date')
        ax.set_ylabel('Values')
        plt.legend()
        plt.savefig(f'results/imgs/{dataset}/{datatype}/sarima/sarima_{filename.replace(".csv", "")}_full.png')

        plt.close('all')


class ExpSmoothing:
    model = None
    dataset = ''
    full_pred = pd.DataFrame([], columns=['upper value', 'lower value', 'pred'])

    def _get_time_index(self, data):
        if self.dataset == 'yahoo':
            data.loc[:, 'timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
        data.set_index('timestamp', inplace=True)
        return data

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

        pred = self.model.forecast(window)
        simulations = self.model.simulate(min(window, newdf.shape[0]), repetitions=100, error="add")
        simulations['upper value'] = simulations.max(axis=1)
        simulations['lower value'] = simulations.min(axis=1)
        simulations['pred'] = pred
        simulations['timestamp'] = newdf.index
        simulations = self._get_time_index(simulations)
        self.full_pred = pd.concat([self.full_pred, simulations[['upper value', 'lower value', 'pred']]])
        for idx, row in simulations.iterrows():
            try:
                if row['lower value'] <= newdf.loc[idx, 'value'] <= row['upper value']:
                    y_pred.append(0)
                else:
                    y_pred.append(1)
            except:
                pass

        return y_pred

    def plot(self, y, dataset, datatype, filename, full_test_data):
        y = self._get_time_index(y)
        full_test_data = self._get_time_index(full_test_data)
        # 0 1970-01-01 00:00:01  0.000000
        # yahoo dataset only!!!
        # TODO: fix
        ax = y['1970':].plot(label='observed')
        ax.fill_between(self.full_pred.index,
                        self.full_pred['lower value'],
                        self.full_pred['upper value'], color='k', alpha=.2)

        for idx, pred in self.full_pred.iterrows():
            ax.plot(ax=pred['pred'], label=f'Window {idx} forecast', alpha=.7, figsize=(14, 7))

            print(full_test_data)
            if (pred['lower value'] > y.loc[idx, 'value'] or
                y.loc[idx, 'value'] > pred['upper value']) and \
                    full_test_data.loc[idx, 'is_anomaly'] == 0:
                ax.scatter(idx, y.loc[idx, 'value'], color='r')
            if (pred['lower value'] <= y.loc[idx, 'value'] <= pred['upper value']) \
                    and full_test_data.loc[idx, 'is_anomaly'] == 1:
                ax.scatter(idx, y.loc[idx, 'value'], color='darkmagenta')

        ax.set_xlabel('Date')
        ax.set_ylabel('Values')
        plt.legend()
        plt.savefig(f'results/imgs/{dataset}/{datatype}/es/es_{filename.replace(".csv", "")}_full.png')

        plt.close('all')
