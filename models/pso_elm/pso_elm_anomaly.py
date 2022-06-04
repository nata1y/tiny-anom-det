# -*- coding: utf-8 -*-
'''
Created on 6 de fev de 2017
By Gustavo Oliveira
Universidade Federal de Pernambuco, Recife, Brasil
E-mail: ghfmo@cin.ufpe.br
OLIVEIRA, Gustavo HFM et al. Time series forecasting in the presence of concept drift: A algoritmos_online-based approach.
In: 2017 IEEE 29th International Conference on Tools with Artificial Intelligence (ICTAI). 
IEEE, 2017. p. 239-246.
https://ieeexplore.ieee.org/document/8371949
IDPSO-ELM-S is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>
'''
import operator

from models.pso_elm.utils.moving_window import MowingWindow
from models.pso_elm.regressors.idpso_elm import IDPSO_ELM
import matplotlib.pyplot as plt
from models.pso_elm.b import B
import time
import pandas as pd
import numpy as np
import antropy as ant
import scipy.stats as st
from scipy import stats
from sklearn.metrics import mean_absolute_error
from settings import inercia_inicial, c1, c2, crit, inercia_final, xmax, split_dataset, it


class PSO_ELM_anomaly:
    def __init__(self, dataset, datatype, filename, drift_detector,
                 n=500, lags=5, qtd_neurons=10, num_particles=10, limit=10, w=0.25, c=0.25,
                 magnitude=5, entropy_window=100, error_threshold=0.1):
        '''
        :param n: fit/retrain size
        :param lags: amount of lags in input
        :param num_particles: num particles in swarm
        :param limit: counter before alerting drift
        :param entropy_window: window for calculating entropy
        :param magnitude: magnitude of allowed deviation for entropy
        :param error_threshold: error threshold
        '''

        self.n = n
        self.lags = lags
        self.qtd_neurons = qtd_neurons
        self.num_particles = num_particles

        self.limit = limit
        self.w = w
        self.c = c
        self.error_threshold = error_threshold

        self.predictions_df = pd.DataFrame([])
        self.magnitude = magnitude
        self.svd_entropies = []
        self.entropy_window = entropy_window
        self.results = {'fixed': [],
                        'dynamic': []
                        }
        self.anomalous_batches = []
        self.deteccoes = []
        self.error_stream = 0
        self.loss_thresholds = {'fixed': [],
                                'dynamic': []
                                }
        self.datatype = datatype
        self.dataset = dataset
        self.filename = filename.replace(".csv", "")
        self.use_drift_adaptation = True
        self.drift_detector = drift_detector

    def fit(self, data_train):
        data_train = data_train['value'].to_numpy()
        self.per_batch_metrics = []
        self.y_true_batch = []

        start_time = time.time()
        # Particle Swarm creation
        self.swarm = IDPSO_ELM(data_train, split_dataset, self.lags, self.qtd_neurons)
        self.swarm.set_params(it, self.num_particles, inercia_inicial, inercia_final, c1, c2, xmax, crit)
        self.swarm.train()

        # adjust prediction window
        self.window_prediction = MowingWindow()
        self.window_prediction.adjust(self.swarm.dataset[0][(len(self.swarm.dataset[0]) - 1):])
        self.prediction = self.swarm.predict(self.window_prediction.data)

        # window that records current concept
        self.window_params = MowingWindow()
        self.window_params.adjust(data_train)

        self.b = B(self.limit, self.w, self.c, self.drift_detector)
        self.b.record(self.window_params.data, self.lags, self.swarm)
        end_time = time.time()

        self.traintime = end_time - start_time
        print(self.traintime)

        self.predictions_df['predictions'] = []
        self.predictions_df['errors'] = []

        self.window_train_loss = MowingWindow()
        self.window_train_loss.adjust(self.swarm.dataset[0][:1])

        for start in range(0, len(data_train), self.entropy_window):
            try:
                self.svd_entropies.append(
                    ant.svd_entropy(data_train[start:start + self.entropy_window],
                                    normalize=True))
            except:
                pass

        prediction_train = []
        for d in data_train:
            self.window_prediction.add_window(d)
            prediction_train.append(self.swarm.predict(self.window_train_loss.data)[0])

        loss = np.abs(np.subtract(prediction_train, data_train.tolist()))
        self.error_threshold = self.error_threshold * st.t.interval(alpha=0.99, df=len(loss)-1,
                                                                    loc=np.mean(loss),
                                                                    scale=st.sem(loss))[1]
        factor = self.magnitude
        self.boundary_bottom = np.mean([v for v in self.svd_entropies if pd.notna(v)]) - \
                          factor * np.std([v for v in self.svd_entropies if pd.notna(v)])
        self.boundary_up = np.mean([v for v in self.svd_entropies if pd.notna(v)]) + \
                      factor * np.std([v for v in self.svd_entropies if pd.notna(v)])
        self.entropies = [v for v in self.svd_entropies if pd.notna(v)]
        self.entropy_mean = np.mean([v for v in self.svd_entropies if pd.notna(v)])

        # identify no drift for the next iter
        self.drift_detected = False

    def predict(self, stream):
        stream = stream['value'].to_numpy()
        try:
            current_entropy = ant.svd_entropy(stream, normalize=True)
            extent = stats.percentileofscore(self.svd_entropies, current_entropy) / 100.0
            extent = 1.5 - max(extent, 1.0 - extent)
            threshold_adapted = self.error_threshold * extent
        except:
            threshold_adapted = self.error_threshold

        offset = self.predictions_df.shape[0]

        for j in range(len(stream)):
            loss = mean_absolute_error(stream[j:j + 1], self.prediction)
            self.error_stream += loss

            self.window_prediction.add_window(stream[j])
            prediction = self.swarm.predict(self.window_prediction.data)

            self.predictions_df = self.predictions_df.append({
                'predictions': prediction,
                'errors': loss
            }, ignore_index=True)

            if self.predictions_df[f'errors'].tolist()[offset+j] > self.error_threshold:
                self.results['fixed'].append(1)
            else:
                self.results['fixed'].append(0)

            if self.predictions_df[f'errors'].tolist()[offset+j] > threshold_adapted:
                self.results['dynamic'].append(1)
            else:
                self.results['dynamic'].append(0)

            self.loss_thresholds['fixed'].append(self.error_threshold)
            self.loss_thresholds['dynamic'].append(threshold_adapted)

            if self.use_drift_adaptation:
                if not self.drift_detected:
                    drift = self.b.monitor(self.window_prediction.data, stream[j:j + 1], self.swarm, j)

                    if drift:
                        self.deteccoes.append(j)
                        self.window_params.nullify()
                        self.drift_detected = True

                else:

                    if len(self.window_params.data) < self.n:
                        self.window_params.increment(stream[j])

                    else:
                        self.swarm = IDPSO_ELM(self.window_params.data,
                                                split_dataset, self.lags,
                                                self.qtd_neurons)
                        self.swarm.set_params(it, self.num_particles,
                                               inercia_inicial, inercia_final,
                                               c1, c2, xmax, crit)
                        self.swarm.train()

                        self.window_prediction = MowingWindow()
                        self.window_prediction.adjust(self.swarm.dataset[0][(len(self.swarm.dataset[0]) - 1):])
                        self.prediction = self.swarm.predict(self.window_prediction.data)

                        self.b = B(self.limit, self.w, self.c, self.drift_detector)
                        self.b.record(self.window_params.data, self.lags, self.swarm)

                        self.drift_detected = False

        return self.results['fixed'][-len(stream):], self.results['dynamic'][-len(stream):]

    def plot(self, datatest, threshold_type='dynamic'):
        stream, labels = datatest['value'].tolist(), datatest['is_anomaly'].tolist()
        MAE = self.error_stream / len(stream)
        wd = 2

        legend_place = (1.13, 0.85)

        eixox = [0, len(stream)]
        erro_eixoy = [0, 0.2]
        x_intervalos = range(eixox[0], eixox[1], 900)

        figura = plt.figure()
        figura.suptitle('PSO-ELM', fontsize=11, fontweight='bold')

        grafico1 = figura.add_subplot(2, 10, (1, 9))
        grafico1.plot(stream, label='Original Serie', color='blue')
        grafico1.plot(self.predictions_df['predictions'], label='Forecast', color='red')

        if not MAE:
            grafico1.set_title("Real dataset and forecast")
        else:
            grafico1.set_title("Real dataset and forecast | MAE: %.3f |" % (MAE))

        grafico1.axvline(-1000, linewidth=wd,
                         linestyle='dashed', label='Anomaly Batch', color='orange')
        for i in range(len(self.anomalous_batches)):
            counter = self.anomalous_batches[i]
            grafico1.axvline(counter,
                             linewidth=wd, linestyle='dashed', color='green')

        plt.ylabel('Observations')
        plt.xlabel('Time')
        grafico1.axis([eixox[0], eixox[1],
                       min(np.min(stream), np.min(self.predictions_df['predictions'])) - .05,
                       max(np.max(stream), np.max(self.predictions_df['predictions'])) + .05])
        offset = len(stream) - len(labels)
        grafico1.legend(loc='best', bbox_to_anchor=legend_place, ncol=1,
                        fancybox=True, shadow=True)
        plt.xticks(x_intervalos, rotation=45)

        grafico2 = figura.add_subplot(3, 10, (21, 29))
        grafico2.plot(self.predictions_df['errors'],
                      label='Forecasting Error', color='blue', zorder=1)
        grafico2.plot(list(range(len(self.loss_thresholds[threshold_type]))),
                      self.loss_thresholds[threshold_type], linewidth=wd,
                      linestyle='dashed', color='m', label='Loss threshold')
        grafico2.set_title("Forecasting Error")

        grafico2.axvline(-1000, linewidth=wd,
                         linestyle='dashed', label='Anomaly Batch', color='green')
        for i in range(len(self.anomalous_batches)):
            counter = self.anomalous_batches[i]
            grafico2.axvline(counter, linewidth=wd, linestyle='dashed', color='green')

        fp_idx = [offset + x for x in range(len(labels))
                  if labels[x] == 0 and self.results[threshold_type][x] == 1]
        fn_idx = [offset + x for x in range(len(labels))
                  if labels[x] == 1 and self.results[threshold_type][x] == 0]

        grafico2.scatter(x=fp_idx, y=[self.predictions_df['errors'][i]
                                      for i in fp_idx], label='FP', color='crimson', s=15,
                         zorder=10)
        grafico2.scatter(x=fn_idx, y=[self.predictions_df['errors'][i]
                                      for i in fn_idx], label='FN', color='y', s=15, zorder=10)

        plt.ylabel('MAE')
        plt.xlabel('Time')
        grafico2.legend(loc='best',
                        ncol=1, fancybox=True, shadow=True)
        grafico2.axis([eixox[0], eixox[1], erro_eixoy[0],
                       max(np.max(self.predictions_df['errors']) + 0.05,
                           np.max(self.loss_thresholds[threshold_type]) + 0.05)])
        plt.xticks(x_intervalos, rotation=45)

        plt.savefig(f'results/imgs/{self.dataset}/{self.datatype}/pso-elm_{self.filename}_forecasting_loss.png')

