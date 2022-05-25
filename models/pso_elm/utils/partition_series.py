#-*- coding: utf-8 -*-
import numpy as np
import copy


def window_time(serie, window):
    sz_mat = len(serie) - window
    
    input_mat = []
    for i in range(sz_mat):
        input_mat.append([0.0] * window)
    
    output_vector = []
    for i in range(len(input_mat)):
        input_mat[i] = serie[i:i+window]
        output_vector.append(serie[i+window])
        
    return np.asarray(input_mat), np.asarray(output_vector)

    
class SeriesPreprocessor:
    def __init__(self, serie=[0], split=[0, 0, 0], window=5, norm=False):
        self.min = 0
        self.max = 0
        self.serie = serie
        
        if norm:
            self.serie = self.normalize(serie)
        
        self.window = window
            
        self.pct_train = split[0]
        self.pct_val = split[1]
        self.pct_test = split[2]
       
        self.index_train = 0
        self.index_val = 0
        
        if len(self.serie) > 1:
            self.input_mat, self.output_vector = window_time(self.serie, self.window)
        
    def part_train(self):
        '''
        method that returns only the training part of the time series
        '''
        
        self.index_train = self.pct_train * len(self.output_vector)
        self.index_train = int(round(self.index_train))
        
        input_train = np.asarray(self.input_mat[:self.index_train])
        output_train = np.asarray(self.output_vector[:self.index_train])
        
        return input_train, output_train
    
    def part_val(self):
        self.index_val = self.pct_val * len(self.output_vector)
        self.index_val = int(round(self.index_val))
        self.index_val = self.index_train + self.index_val
        
        input_val = np.asarray(self.input_mat[self.index_train:self.index_val])
        output_val = np.asarray(self.output_vector[self.index_train:self.index_val])
        
        return input_val, output_val
    
    def part_test(self):
        input_teste = np.asarray(self.input_mat[self.index_val:])
        output_teste = np.asarray(self.output_vector[self.index_val:])
        
        return input_teste, output_teste
    
    def normalize(self, serie):
        self.min = copy.deepcopy(np.min(serie))
        self.max = copy.deepcopy(np.max(serie))
        
        serie_norm = []
        for e in serie:
            val = (e - self.min)/(self.max - self.min)
            serie_norm.append(val)
        
        return serie_norm  
     
    def denormalize(self, serie):
        serie_norm = []
        for e in serie:
            val = e * (self.max - self.min) + self.min
            serie_norm.append(val)

        return serie_norm  
