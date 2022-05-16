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
from models.pso_elem.ferramentas.Janela_deslizante import Janela
from models.pso_elem.graficos.Graficos_execucao import Grafico
from models.pso_elem.regressores.IDPSO_ELM import IDPSO_ELM
from scipy.stats import percentileofscore
from models.pso_elem.detectores.B import B
import time
import pandas as pd
import numpy as np
import antropy as ant
from scipy import stats
import random
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
    def __init__(self, n=500, lags=5, qtd_neuronios=10, numero_particulas=10, limite=10, w=0.25, c=0.25,
                 magnitude=5, entropy_window=100, error_threshold=0.1):
        '''
        construtor do algoritmo que detecta a mudanca de ambiente por meio do comportamento das particulas
        :param dataset: serie temporal que o algoritmo vai executar
        :param qtd_train_inicial: quantidade de exemplos para o treinamento inicial
        :param tamanho_janela: tamanho da janela de caracteristicas para identificar a mudanca
        :param n: tamanho do n para reavaliar o metodo de deteccao
        :param lags: quantidade de lags para modelar as entradas da RNA
        :param qtd_neuronios: quantidade de neuronios escondidos da RNA
        :param numero_particulas: numero de particulas para serem usadas no IDPSO
        :param n_particulas_comportamento: numero de particulas para serem monitoradas na detecccao de mudanca
        :param limite: contador para verificar a mudanca
        '''

        self.n = n
        self.lags = lags
        self.qtd_neuronios = qtd_neuronios
        self.numero_particulas = numero_particulas

        self.limite = limite
        self.w = w
        self.c = c
        self.error_threshold = error_threshold

        self.predictions_df = pd.DataFrame([])
        self.anomaly_predictions = []
        self.runtime = 0
        self.magnitude = magnitude
        self.svd_entropies = []
        self.entropy_window = entropy_window
        self.entropy_threshold = 0
        self.entropy_differences = []
        self.drift_occured = False
        self.results = {'fixed': [],
                        'dynamic': []
                        }
        self.loaded_loss = None
        self.anomalous_batches = []
        self.deteccoes = []
        self.erro_stream = 0

    def train(self, data_train):
        self.per_batch_metrics = []
        self.y_true_batch = []

        start_time = time.time()
        # criando e treinando um enxame_vigente para realizar as previsões
        self.enxame = IDPSO_ELM(data_train, divisao_dataset, self.lags, self.qtd_neuronios)
        self.enxame.Parametros_IDPSO(it, self.numero_particulas, inercia_inicial, inercia_final, c1, c2, xmax, crit_parada)
        self.enxame.Treinar()

        # ajustando com os dados finais do treinamento a janela de predicao
        self.janela_predicao = Janela()
        self.janela_predicao.Ajustar(self.enxame.dataset[0][(len(self.enxame.dataset[0]) - 1):])
        self.predicao = self.enxame.Predizer(self.janela_predicao.dados)

        # janela com o atual conceito, tambem utilizada para armazenar os dados de retreinamento
        self.janela_caracteristicas = Janela()
        self.janela_caracteristicas.Ajustar(data_train)

        # ativando o sensor de comportamento de acordo com a
        # primeira janela de caracteristicas para media e desvio padrão
        self.b = B(self.limite, self.w, self.c)
        self.b.armazenar_conceito(self.janela_caracteristicas.dados, self.lags, self.enxame)
        end_time = time.time()

        self.traintime = end_time - start_time
        print(self.traintime)

        self.predictions_df['predictions'] = []
        self.predictions_df['errors-abs'] = []
        self.predictions_df['errors-pinball'] = []

        self.janela_train_loss = Janela()
        self.janela_train_loss.Ajustar(self.enxame.dataset[0][:1])

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

    def test(self, stream):
        '''
        Metodo para executar o procedimento do algoritmo
        :param grafico: variavel booleana para ativar ou desativar o grafico
        :return: retorna 5 variaveis: [falsos_alarmes, atrasos, falta_deteccao, MAPE, tempo_execucao]
        '''

        current_entropy = ant.svd_entropy(stream, normalize=True)
        extent = stats.percentileofscore(self.svd_entropies, current_entropy) / 100.0
        extent = 1.5 - max(extent, 1.0 - extent)
        threshold_adapted = self.error_threshold * extent
        offset = self.predictions_df.shape[0]

        for j in range(len(stream)):
            loss = mean_absolute_error(stream[j:j + 1], self.predicao)
            self.erro_stream += loss

            # adicionando o novo dado a janela de predicao
            self.janela_predicao.Add_janela(stream[j])

            # realizando a nova predicao com a nova janela de predicao
            predicao = self.enxame.Predizer(self.janela_predicao.dados)

            self.predictions_df = self.predictions_df.append({
                'predictions': predicao,
                'errors': loss
            }, ignore_index=True)

            if self.predictions_df[f'errors'].tolist()[offset+j] > self.error_threshold:
                self.results[f'fixed'].append(1)
            else:
                self.results[f'fixed'].append(0)

            if self.predictions_df[f'errors'].tolist()[offset+j] > threshold_adapted:
                self.results[f'dynamic'].append(1)
            else:
                self.results[f'dynamic'].append(0)

            if self.mudanca_ocorreu is False:

                #computando o comportamento para a janela de predicao, para somente uma instancia - media e desvio padrão
                mudou = self.b.monitorar(self.janela_predicao.dados, stream[j:j+1], self.enxame, j)

                if mudou:
                    self.deteccoes.append(j)

                    #zerando a janela de treinamento
                    self.janela_caracteristicas.Zerar_Janela()

                    #variavel para alterar o fluxo, ir para o periodo de retreinamento
                    self.mudanca_ocorreu = True
                    self.drift_occured = True

            else:

                if len(self.janela_caracteristicas.dados) < self.n:
                    #adicionando a nova instancia na janela de caracteristicas
                    self.janela_caracteristicas.Increment_Add(stream[j])

                else:
                    #atualizando o enxame_vigente preditivo
                    self.enxame = IDPSO_ELM(self.janela_caracteristicas.dados,
                                            divisao_dataset, self.lags,
                                            self.qtd_neuronios)
                    self.enxame.Parametros_IDPSO(it, self.numero_particulas,
                                                 inercia_inicial, inercia_final,
                                                 c1, c2, xmax, crit_parada)
                    self.enxame.Treinar()

                    #ajustando com os dados finais do treinamento a janela de predicao
                    self.janela_predicao = Janela()
                    self.janela_predicao.Ajustar(self.enxame.dataset[0][(len(self.enxame.dataset[0])-1):])
                    self.predicao = self.enxame.Predizer(self.janela_predicao.dados)

                    # atualizando o conceito para a caracteristica de comportamento
                    self.b = B(self.limite, self.w, self.c)
                    self.b.armazenar_conceito(self.janela_caracteristicas.dados, self.lags, self.enxame)

                    #variavel para voltar para o loop principal
                    self.mudanca_ocorreu = False

    # def plot(self):
    #     g = Grafico()
    #     loss_threshold = 0
    #     print(self.results)
    #     g.Plotar_graficos(stream[:len(self.results['0.01-0.1-abs'])], self.labels[:len(self.results['0.01-0.1-abs'])],
    #                       self.results['0.01-0.1-abs'],
    #                       self.predictions_df['predictions'].tolist(),
    #                       deteccoes, alarmes, self.predictions_df['errors-pinball'].tolist(), self.n, atrasos,
    #                       falsos_alarmes, tempo_execucao, loss_threshold, loss_thresholds, MAE, nome=self.tecnica,
    #                       ts=ts, anomalous_batches=self.anomalous_batches)

