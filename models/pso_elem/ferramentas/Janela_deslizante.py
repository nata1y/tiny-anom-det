#-*- coding: utf-8 -*-
'''
Created on 6 de fev de 2017

@author: gusta
'''
import numpy as np


class Janela():
    def __init__(self):
        '''
        Classe para instanciar a janela deslizante 
        '''
        self.dados = []
        self.dados_mais = []
        
    def Ajustar(self, valores):
        '''
        Metodo para ajustar o tamanho da jenela deslizante 
        :param valores: valores para serem inseridos na janela
        '''
        
        self.dados = valores
        self.dados_mais = np.append([0], valores)
    
    def Add_janela(self, valor):
        '''
        Metodo para inserir na janela deslizante, o valor mais antigo sera excluido 
        :param valor: valor de entrada
        '''
        self.dados = self.Fila(self.dados, valor)
        self.dados_mais = self.Fila(self.dados_mais, valor)
        
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
        
        if(len(self.dados) > 0):
            aux = np.asarray(self.dados)
            aux = [0] * (len(self.dados)+1)
            aux[:len(self.dados)] = self.dados
            aux[len(self.dados)] = valor
            self.dados = aux
        else:
            self.dados.append(valor)
    
    def Zerar_Janela(self):
        self.dados = []
