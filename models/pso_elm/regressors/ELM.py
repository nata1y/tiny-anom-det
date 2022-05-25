#-*- coding: utf-8 -*-
'''
Created on 31 de jan de 2017

@author: gusta
Modified 23.05.2022
@author nata1y
'''

import numpy as np
from sklearn.metrics import mean_absolute_error
from models.pso_elm.utils.partition_series import SeriesPreprocessor
import matplotlib.pyplot as plt


class ELMRegressor:
    def __init__(self, hidden_neurons=None):
        self.hidden_neurons = hidden_neurons
        
        self.train_start = []
        self.train_end = []
        self.val_start = []
        self.val_end = []
        self.test_start = []
        self.test_end = []

    def fit(self, start_info, end_info, weights=None):
        '''
        Method for training ELM through pseudo-inverse
        :param start_info: inputs to network training, this data is an array with a defined lag amount
        :param end_info: outputs for network training, this data is a vector with the outputs corresponding to the inputs
        :return: returns output weights of the net
        '''
        # stacking two arrays of the same dimension
        # if it is an array, the first column is given by the values of the array,
        # while the second is filled in with 1s
        start_info = np.column_stack([start_info, np.ones([start_info.shape[0], 1])])

        # setting the starting weights randomly
        # creates an array with random numbers of Row x column size
        # initial weights corresponding to input
        if not np.any(weights):
            self.initial_weights = np.random.randn(start_info.shape[1], self.hidden_neurons)
        else:
            self.initial_weights = weights
        
        # computing the activation neurons with the hyperbolic function
        G = np.tanh(start_info.dot(self.initial_weights))
        
        # computes the prediction for the input data
        self.end_weights = np.linalg.pinv(G).dot(end_info)
      
    def predict(self, start_info):
        '''
        perform predictions
        '''
        start_info = np.column_stack([start_info, np.ones([start_info.shape[0], 1])])

        # computed by entry neurons
        G = np.tanh(start_info.dot(self.initial_weights))
            
        # computed by output neurons
        return G.dot(self.end_weights)
    
    def adjust_phase(self, start_info):
        # initial predictions
        prediction = self.predict(start_info)
        
        # phase adjustment
        adjustment = np.column_stack([start_info[:, 1:], prediction])
        
        # final forecast with adjustment
        return self.predict(adjustment)
    
    def optimize_network(self, max_neurons, optimization_data):
        '''
        Method to optimize ELM
        :param max_neurons: maximum amount of neurons that is tuned
        :param optimization_data: this parameter is a list with the following data
        [info_start, info_end, val_start, val_end]]
        '''
        MAE_TEST_MINS = []
        n_min = None

        for M in range(1, max_neurons, 1):
            MAES_TEST = []
            
            print("Training with %s neurons..."%M)

            for i in range(10):
                ELM = ELMRegressor(M)
                ELM.fit(optimization_data[0], optimization_data[1])
                prediction = ELM.predict(optimization_data[2])
                MAES_TEST.append(mean_absolute_error(optimization_data[3], prediction))

            MAE_TEST_MINS.append(np.mean(MAES_TEST))
            n_min = min(MAE_TEST_MINS)
            n_pos = MAE_TEST_MINS.index(n_min)
            self.hidden_neurons = n_pos

        print("Minimum MAE ELM =", n_min)

    def handle_data(self, serie, split, lags):
        partition = SeriesPreprocessor(serie, split, lags)

        [train_start, train_end] = partition.part_train()
        [val_start, val_end] = partition.part_val()
        [test_start, test_end] = partition.part_test()

        self.train_start = np.asarray(train_start)
        self.train_end = np.asarray(train_end)
        self.val_start = np.asarray(val_start)
        self.val_end = np.asarray(val_end)
        self.test_start = np.asarray(test_start)
        self.test_end = np.asarray(test_end)
