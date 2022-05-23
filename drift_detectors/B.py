#-*- coding: utf-8 -*-
'''
Created on 10 de mar de 2017
@author: gusta
'''

from models.pso_elm.utils.partition_series import Particionar_series
from drift_detectors.ECDD import ECDD
from sklearn.metrics import mean_absolute_error
import numpy as np


class B:
    def __init__(self, limite, w, c):
        '''
        Método para criar um modelo do ECDD
        :param Lambda: float com o valor de lambda
        :param w: float com o nivel de alarme
        :param l: float com o nivel de deteccao
        '''

        self.limite = limite
        self.w = w
        self.c = c        
        self.mean_zero = 0
        self.deviation_zero = 0
        self.contador = 0
        self.ecdd = ECDD(0.2, w, c)

    def record(self, data, lags, enxame):
        behavior = self.update_behavior(data, lags, enxame)
        self.mean_zero = behavior[0]
        self.deviation_zero = behavior[1]
        self.ecdd.record(self.mean_zero, self.deviation_zero)

    def atualizar_ewma(self, MI0, i):
        self.ecdd.update_ewma(MI0, i)

    def monitor(self, data, real, swarm, i):
        '''
        This method aims to monitor an error to know if it has changed distribution
        '''
        comportamento_atual = self.compute_current_behavior(data, real, swarm)

        #atualizando o ewma
        self.atualizar_ewma(comportamento_atual[0], i)

        #monitorando o modulo ecdd
        string_ecdd = self.ecdd.monitor()

        #somando o contador
        if(string_ecdd == self.ecdd.drift):
            self.contador = self.contador + 1
        elif((string_ecdd == self.ecdd.alert) or (string_ecdd == self.ecdd.nodrift)):
            self.contador = 0

        #procedimento pos mudanca
        if self.contador == self.limite:
            self.contador = 0 
            return True
        else:
            return False

    def update_behavior(self, characteristics, lags, swarm):
        '''
        Method to compute the detection of change in the time series through the behavior of particles
        '''

        #particionando o vetor de caracteristicas para usar para treinar 
        particle = Particionar_series(characteristics, [1, 0, 0], lags)
        [caracteristicas_entrada, caracteristicas_saida] = particle.Part_train()

        #variavel para salvar as medias das predicoes
        medias = []

        #realizando as previsoes e armazenando as acuracias 
        for i in range(swarm.num_particles):
            prediction = swarm.sensores[i].Predizer(caracteristicas_entrada)
            MAE = mean_absolute_error(caracteristicas_saida, prediction)
            medias.append(MAE)          

        #salvando a media e desvio padrao das acuracias
        comportamento = [0] * 2
        comportamento[0] = np.mean(medias)
        comportamento[1] = np.std(medias)

        return comportamento

    def compute_current_behavior(self, data, real, swarm):
        '''
        Method to compute behavior for current data
        '''
        measures = []

        for i in range(swarm.num_particles):
            prediction = swarm.sensores[i].Predizer(data)
            MAE = mean_absolute_error(real, prediction)
            measures.append(MAE)

        behavior = [0] * 2
        behavior[0] = np.mean(measures)
        behavior[1] = np.std(measures)

        return behavior
