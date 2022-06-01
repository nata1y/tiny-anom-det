'''
Created on 10 de mar de 2017
@author: gusta
Modified 23.05.2022
@author nata1y
'''
import numpy as np


class ECDD:
    def __init__(self, Lambda, w, l):
        self.lmbd = Lambda
        self.w = w
        self.l = l
        self.mean_zero = 0
        self.deviation_zero = 0
        self.deviation_z = 0
        self.zt = 0
        self.nodrift = "No"
        self.alert = "Alert"
        self.drift = "Drift"
        self.sensor_drift = False

    def record(self, data):
        '''
        Record error concept for a single error measure
        '''
        MI0 = np.mean(data)
        SIGMA0 = np.std(data)
        self.mean_zero = MI0
        self.deviation_zero = SIGMA0

    def record_emwa(self, MI0, SIGMA0):
        '''
        Record error concept for a swarm of particles
        '''
        self.mean_zero = MI0
        self.deviation_zero = SIGMA0

    def update(self, error, t):
        '''
        method to update ewma with error at time t
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
        method to monitor the condition of the detector
        '''

        if self.zt > self.mean_zero + (self.l * self.deviation_z):
            self.sensor_drift = True
            return self.drift
        elif self.zt > self.mean_zero + (self.w * self.deviation_z):
            return self.alert
        else:
            return self.nodrift

    def reset(self):
        self.mean_zero = 0
        self.deviation_zero = 0
        self.deviation_z = 0
        self.zt = 0
        self.sensor_drift = False
