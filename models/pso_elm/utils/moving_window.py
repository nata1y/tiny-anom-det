#-*- coding: utf-8 -*-
'''
Created on 6 de fev de 2017

@author: gusta
'''
import numpy as np


class MowingWindow:
    def __init__(self):
        '''
        Classe para instanciar a janela deslizante 
        '''
        self.data = []
        self.rest_data = []
        
    def adjust(self, values):
        '''
        Metodo para ajustar o tamanho da jenela deslizante 
        :param values: valores para serem inseridos na janela
        '''
        
        self.data = values
        self.rest_data = np.append([0], values)
    
    def Add_janela(self, valor):
        '''
        Metodo para inserir na janela deslizante, o valor mais antigo sera excluido 
        :param valor: valor de entrada
        '''
        self.data = self.Fila(self.data, valor)
        self.rest_data = self.Fila(self.rest_data, valor)
        
    def Fila(self, lista, valor):
        '''
        metodo para adicionar um novo valor a um ndarray
        :param: lista: lista que será acrescida
        :param: valor: valor a ser adicionado
        :return: retorna a lista com o valor acrescido
        '''
        
        if(len(lista) == 1):
            aux2 = len(lista[0])
            aux = [0] * aux2
            aux[len(lista[0])-1] = valor
            aux[:len(aux)-1] = lista[0][1:]
            lista[0] = aux
            lista[0] = np.asarray(lista[0])
            lista[0] = np.column_stack(lista[0])
            
            return lista
            
        else:
            aux2 = len(lista)
            aux = [0] * aux2
            aux[len(lista)-1] = valor
            aux[:len(aux)-1] = lista[1:]
            lista = aux
            lista = np.asarray(lista)
            lista = np.column_stack(lista)
            
            return lista
        
    def Increment_Add(self, valor):
        '''
        Metodo para inserir mais dados na janela deslizante 
        :param valor: valor de entrada
        '''
        
        if len(self.data) > 0:
            aux = np.asarray(self.data)
            aux = [0] * (len(self.data) + 1)
            aux[:len(self.data)] = self.data
            aux[len(self.data)] = valor
            self.data = aux
        else:
            self.data.append(valor)
    
    def Zerar_Janela(self):
        self.data = []
