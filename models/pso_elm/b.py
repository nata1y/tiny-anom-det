#-*- coding: utf-8 -*-
'''
Created on 10 de mar de 2017
@author: gusta
Modified 23.05.2022
@author nata1y
'''

from models.pso_elm.utils.partition_series import SeriesPreprocessor
from drift_detectors.ecdd import ECDD
from sklearn.metrics import mean_absolute_error
import numpy as np


class B:
    def __init__(self, limit, w, c, drift_detector):
        self.limit = limit
        self.w = w
        self.c = c        
        self.mean_zero = 0
        self.deviation_zero = 0
        self.counter = 0
        self.drift_detector = drift_detector

    def record(self, data, lags, swarm):
        behavior = self.update_behavior(data, lags, swarm)
        self.mean_zero = behavior[0]
        self.deviation_zero = behavior[1]
        self.drift_detector.record_emwa(self.mean_zero, self.deviation_zero)

    def update_ewma(self, MI0, i):
        self.drift_detector.update(MI0, i)

    def monitor(self, data, real, swarm, i):
        '''
        This method aims to monitor an error to know if it has changed distribution
        '''
        current_behavior = self.compute_current_behavior(data, real, swarm)
        self.update_ewma(current_behavior[0], i)
        string_ecdd = self.drift_detector.monitor()

        if string_ecdd == self.drift_detector.drift:
            self.counter = self.counter + 1
        elif (string_ecdd == self.drift_detector.alert) or (string_ecdd == self.drift_detector.nodrift):
            self.counter = 0

        if self.counter == self.limit:
            self.counter = 0
            return True
        else:
            return False

    def update_behavior(self, characteristics, lags, swarm):
        '''
        Method to compute the detection of change in the time series through the behavior of particles
        '''
        particle = SeriesPreprocessor(characteristics, [1, 0, 0], lags)
        [start, end] = particle.part_train()

        measures = []

        for i in range(swarm.num_particles):
            prediction = swarm.sensors[i].predict(start)
            MAE = mean_absolute_error(end, prediction)
            measures.append(MAE)

        behavior = (np.mean(measures), np.std(measures))
        return behavior

    def compute_current_behavior(self, data, real, swarm):
        '''
        Method to compute behavior for current data
        '''
        measures = []

        for i in range(swarm.num_particles):
            prediction = swarm.sensors[i].predict(data)
            MAE = mean_absolute_error(real, prediction)
            measures.append(MAE)

        behavior = [0] * 2
        behavior[0] = np.mean(measures)
        behavior[1] = np.std(measures)

        return behavior
