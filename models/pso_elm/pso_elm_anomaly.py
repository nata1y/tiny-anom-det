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
from models.pso_elm.utils.moving_window import MowingWindow
from models.pso_elm.regressors.IDPSO_ELM import IDPSO_ELM
import matplotlib.pyplot as plt
from drift_detectors.B import B
import time
import pandas as pd
import numpy as np
import antropy as ant
from scipy import stats
from sklearn.metrics import mean_absolute_error


#parametros IDPSO
it = 50 
inercia_inicial = 0.8
inercia_final = 0.4
xmax = 1
c1 = 2
c2 = 2
crit_parada = 2
divisao_dataset = [0.8, 0.2, 0]


class PSO_ELM_anomaly():
    def __init__(self, n=500, lags=5, qtd_neuronios=10, num_particles=10, limite=10, w=0.25, c=0.25,
                 magnitude=5, entropy_window=100, error_threshold=0.1):
        '''
        construtor do algoritmo que detecta a mudanca de ambiente por meio do comportamento das particulas
        :param dataset: serie temporal que o algoritmo vai executar
        :param qtd_train_inicial: quantidade de exemplos para o treinamento inicial
        :param tamanho_janela: tamanho da janela de caracteristicas para identificar a mudanca
        :param n: tamanho do n para reavaliar o metodo de deteccao
        :param lags: quantidade de lags para modelar as entradas da RNA
        :param qtd_neuronios: quantidade de neuronios escondidos da RNA
        :param num_particles: numero de particulas para serem usadas no IDPSO
        :param n_particulas_comportamento: numero de particulas para serem monitoradas na detecccao de mudanca
        :param limite: contador para verificar a mudanca
        '''

        self.n = n
        self.lags = lags
        self.qtd_neuronios = qtd_neuronios
        self.num_particles = num_particles

        self.limite = limite
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
        self.erro_stream = 0
        self.loss_thresholds = {'fixed': [],
                                'dynamic': []
                                }

    def train(self, data_train):
        data_train = data_train['value'].to_numpy()
        self.per_batch_metrics = []
        self.y_true_batch = []

        start_time = time.time()
        # criando e treinando um enxame_vigente para realizar as previsões
        self.enxame = IDPSO_ELM(data_train, divisao_dataset, self.lags, self.qtd_neuronios)
        self.enxame.Parametros_IDPSO(it, self.num_particles, inercia_inicial, inercia_final, c1, c2, xmax, crit_parada)
        self.enxame.Treinar()

        # ajustando com os dados finais do treinamento a janela de predicao
        self.janela_predicao = MowingWindow()
        self.janela_predicao.adjust(self.enxame.dataset[0][(len(self.enxame.dataset[0]) - 1):])
        self.predicao = self.enxame.Predizer(self.janela_predicao.data)

        # janela com o atual conceito, tambem utilizada para armazenar os dados de retreinamento
        self.janela_caracteristicas = MowingWindow()
        self.janela_caracteristicas.adjust(data_train)

        # ativando o sensor de comportamento de acordo com a
        # primeira janela de caracteristicas para media e desvio padrão
        self.b = B(self.limite, self.w, self.c)
        self.b.record(self.janela_caracteristicas.data, self.lags, self.enxame)
        end_time = time.time()

        self.traintime = end_time - start_time
        print(self.traintime)

        self.predictions_df['predictions'] = []
        self.predictions_df['errors'] = []

        self.janela_train_loss = MowingWindow()
        self.janela_train_loss.adjust(self.enxame.dataset[0][:1])

        for start in range(0, len(data_train), self.entropy_window):
            try:
                self.svd_entropies.append(
                    ant.svd_entropy(data_train[start:start + self.entropy_window],
                                    normalize=True))
            except:
                pass

        factor = self.magnitude
        self.boundary_bottom = np.mean([v for v in self.svd_entropies if pd.notna(v)]) - \
                          factor * np.std([v for v in self.svd_entropies if pd.notna(v)])
        self.boundary_up = np.mean([v for v in self.svd_entropies if pd.notna(v)]) + \
                      factor * np.std([v for v in self.svd_entropies if pd.notna(v)])
        self.entropies = [v for v in self.svd_entropies if pd.notna(v)]
        self.entropy_mean = np.mean([v for v in self.svd_entropies if pd.notna(v)])

        # identify no drift for the next iter
        self.mudanca_ocorreu = False

    def predict(self, stream):
        '''
        Metodo para executar o procedimento do algoritmo
        :param grafico: variavel booleana para ativar ou desativar o grafico
        :return: retorna 5 variaveis: [falsos_alarmes, atrasos, falta_deteccao, MAPE, tempo_execucao]
        '''
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
            loss = mean_absolute_error(stream[j:j + 1], self.predicao)
            self.erro_stream += loss

            # adicionando o novo dado a janela de predicao
            self.janela_predicao.Add_janela(stream[j])

            # realizando a nova predicao com a nova janela de predicao
            predicao = self.enxame.Predizer(self.janela_predicao.data)

            self.predictions_df = self.predictions_df.append({
                'predictions': predicao,
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

            if not self.mudanca_ocorreu:

                #computando o comportamento para a janela de predicao, para somente uma instancia - media e desvio padrão
                mudou = self.b.monitor(self.janela_predicao.data, stream[j:j + 1], self.enxame, j)

                if mudou:
                    self.deteccoes.append(j)

                    #zerando a janela de treinamento
                    self.janela_caracteristicas.Zerar_Janela()

                    #variavel para alterar o fluxo, ir para o periodo de retreinamento
                    self.mudanca_ocorreu = True

            else:

                if len(self.janela_caracteristicas.data) < self.n:
                    #adicionando a nova instancia na janela de caracteristicas
                    self.janela_caracteristicas.Increment_Add(stream[j])

                else:
                    #atualizando o enxame_vigente preditivo
                    self.enxame = IDPSO_ELM(self.janela_caracteristicas.data,
                                            divisao_dataset, self.lags,
                                            self.qtd_neuronios)
                    self.enxame.Parametros_IDPSO(it, self.num_particles,
                                                 inercia_inicial, inercia_final,
                                                 c1, c2, xmax, crit_parada)
                    self.enxame.Treinar()

                    #ajustando com os dados finais do treinamento a janela de predicao
                    self.janela_predicao = MowingWindow()
                    self.janela_predicao.adjust(self.enxame.dataset[0][(len(self.enxame.dataset[0]) - 1):])
                    self.predicao = self.enxame.Predizer(self.janela_predicao.data)

                    # atualizando o conceito para a caracteristica de comportamento
                    self.b = B(self.limite, self.w, self.c)
                    self.b.record(self.janela_caracteristicas.data, self.lags, self.enxame)

                    #variavel para voltar para o loop principal
                    self.mudanca_ocorreu = False

        return self.results['fixed'][-len(stream):], self.results['dynamic'][-len(stream):]

    def plot(self, stream, labels, ts, dataset, datatype, threshold_type='dynamic'):
        MAE = self.erro_stream / len(stream)
        largura_deteccoes = 2

        localizacao_legenda = (1.13, 0.85)

        eixox = [0, len(stream)]
        erro_eixoy = [0, 0.2]
        x_intervalos = range(eixox[0], eixox[1], 900)

        # criando uma figura
        figura = plt.figure()
        figura.suptitle('PSO-ELM', fontsize=11, fontweight='bold')

        # definindo a figura 1 com a previsao e os dados reais
        grafico1 = figura.add_subplot(2, 10, (1, 9))
        grafico1.plot(stream, label='Original Serie', color='blue')
        grafico1.plot(self.predictions_df['predictions'], label='Forecast', color='red')

        if not MAE:
            grafico1.set_title("Real dataset and forecast")
        else:
            grafico1.set_title("Real dataset and forecast | MAE: %.3f |" % (MAE))

        grafico1.axvline(-1000, linewidth=largura_deteccoes,
                         linestyle='dashed', label='Anomaly Batch', color='orange')
        for i in range(len(self.anomalous_batches)):
            contador = self.anomalous_batches[i]
            grafico1.axvline(contador,
                             linewidth=largura_deteccoes, linestyle='dashed', color='green')

        # colocando legenda e definindo os eixos do grafico
        plt.ylabel('Observations')
        plt.xlabel('Time')
        grafico1.axis([eixox[0], eixox[1],
                       min(np.min(stream), np.min(self.predictions_df['predictions'])) - .05,
                       max(np.max(stream), np.max(self.predictions_df['predictions'])) + .05])
        offset = len(stream) - len(labels)
        grafico1.legend(loc='best', bbox_to_anchor=localizacao_legenda, ncol=1,
                        fancybox=True, shadow=True)
        plt.xticks(x_intervalos, rotation=45)

        # definindo a figura 2 com o erro de previsao
        grafico2 = figura.add_subplot(3, 10, (21, 29))
        grafico2.plot(self.predictions_df['errors'],
                      label='Forecasting Error', color='blue', zorder=1)
        grafico2.plot(list(range(len(self.loss_thresholds[threshold_type]))),
                      self.loss_thresholds[threshold_type], linewidth=largura_deteccoes,
                      linestyle='dashed', color='m', label='Loss threshold')
        grafico2.set_title("Forecasting Error")

        grafico2.axvline(-1000, linewidth=largura_deteccoes,
                         linestyle='dashed', label='Anomaly Batch', color='green')
        for i in range(len(self.anomalous_batches)):
            contador = self.anomalous_batches[i]
            grafico2.axvline(contador, linewidth=largura_deteccoes, linestyle='dashed', color='green')

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

        plt.savefig(f'results/imgs/{dataset}/{datatype}/pso-elm_{ts.replace(".csv", "")}_full.png')

