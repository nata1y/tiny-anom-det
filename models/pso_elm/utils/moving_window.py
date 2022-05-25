#-*- coding: utf-8 -*-
'''
Created on 6 de fev de 2017

@author: gusta
'''
import numpy as np


class MowingWindow:
    def __init__(self):
        self.data = []
        self.rest_data = []
        
    def adjust(self, values):
        '''
        Method to adjust the size of the sliding wdw
        '''
        
        self.data = values
        self.rest_data = np.append([0], values)
    
    def add_window(self, val):
        self.data = self.queue(self.data, val)
        self.rest_data = self.queue(self.rest_data, val)
        
    def queue(self, input, val):
        
        if len(input) == 1:
            aux2 = len(input[0])
            aux = [0] * aux2
            aux[len(input[0]) - 1] = val
            aux[:len(aux)-1] = input[0][1:]
            input[0] = aux
            input[0] = np.asarray(input[0])
            input[0] = np.column_stack(input[0])
            
            return input
            
        else:
            aux2 = len(input)
            aux = [0] * aux2
            aux[len(input) - 1] = val
            aux[:len(aux)-1] = input[1:]
            input = aux
            input = np.asarray(input)
            input = np.column_stack(input)
            
            return input
        
    def increment(self, val):
        '''
        Add data ro sliding wdw
        '''
        if len(self.data) > 0:
            aux = [0] * (len(self.data) + 1)
            aux[:len(self.data)] = self.data
            aux[len(self.data)] = val
            self.data = aux
        else:
            self.data.append(val)
    
    def nullify(self):
        self.data = []
