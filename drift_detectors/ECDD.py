'''
Created on 10 de mar de 2017
@author: gusta
Modified 23.05.2022
@au
'''
import numpy as np

class ECDD:
    def __init__(self, Lambda, w, l):
        '''
        Método para criar um modelo do ECDD
        :param Lambda: float com o valor de lambda
        :param w: float com o nivel de alarme
        :param l: float com o nivel de deteccao
        '''
        self.lmbd = Lambda
        self.w = w
        self.l = l
        self.mean_zero = 0
        self.deviation_zero = 0
        self.deviation_z = 0
        self.zt = 0
        self.nodrift = "Nada"
        self.alert = "Alerta"
        self.drift = "Mudanca"
        self.sensor_drift = True

    def record(self, MI0, SIGMA0):
        '''
        Este método tem por objetivo armazenar um conceito de um erro
        :param MI0: média dos erros
        :param SIGMA0: desvio dos erros
        '''
        self.mean_zero = MI0
        self.deviation_zero = SIGMA0

    def update_ewma(self, error, t):
        '''
        método para atualizar o ewma conforme o erro no tempo t
        :param erro: double com o erro para ser verificado
        :param t: instante de tempo
        '''

        #calculando a média movel
        if t == 0:
            self.zt = (1 - self.lmbd) * self.mean_zero + self.lmbd * error
        elif self.sensor_drift:
            self.sensor_drift = False
            self.zt = (1 - self.lmbd) * self.mean_zero + self.lmbd * error
        else:
            self.zt = (1 - self.lmbd) * self.zt + self.lmbd * error

        #calculando o desvio da média movel
        part1 = (self.lmbd / (2 - self.lmbd))
        part2 = (1 - self.lmbd)
        part3 = (2*t)
        part4 = (1 - (part2**part3))
        part5 = (part1 * part4 * self.deviation_zero)
        self.deviation_z = np.sqrt(part5)

    def monitor(self):
        '''
        método para monitor a condicao do detector
        '''

        if self.zt > self.mean_zero + (self.l * self.deviation_z):
            self.sensor_drift = True
            return self.drift
        elif self.zt > self.mean_zero + (self.w * self.deviation_z):
            return self.alert
        else:
            return self.nodrift
