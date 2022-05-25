#-*- coding: utf-8 -*-
import random
import numpy as np
import copy
from numpy import array
from models.pso_elm.utils.partition_series import SeriesPreprocessor
from models.pso_elm.regressors.ELM import ELMRegressor
from sklearn.metrics import mean_absolute_error
from settings import mi

#auxs vars
counter = 0
fitness = 0
mse = []


class Particle:
    def __init__(self):
        self.position = None


class IDPSO_ELM:
    def __init__(self, serie, split, window, qtd_neurons):
        dataset = self.prepare_data(serie, split, window)
        
        self.dataset = dataset
        self.qtd_neurons = qtd_neurons
        self.best_elm = []
        
        #default IDPSO
        self.lines = self.dataset[0].shape[1] + 1
        self.dimentions = self.lines * qtd_neurons
        
        self.iters = 1000
        self.num_particles = 30
        self.inercia = 0.5
        self.inercia_final = 0.3
        self.c1 = 2.4
        self.c2 = 1.4
        self.crit = 50
        self.particles = []
        self.gbest = []
        
        self.ordered_particles = [0] * self.num_particles
        self.sensors = [0] * self.num_particles
        
        self.tx_spread = 0
        
    def set_params(self, iterations, num_particles, inercia, inercia_final, c1, c2, Xmax, crit):
        '''
        Method to change basic IDPSO parameters
        :param iterations: number of training iterations
        :param num_particles: amount of particles used for training
        :param inercia: initial inertial for training
        :param inercia_final: final inertia_final for variation
        :param c1: cognitive coefficient
        :param c2: personal coefficient
        :param crit: stop criteria to limit repetition not improvement of gbest
        '''
        
        self.iters = iterations
        self.num_particles = num_particles
        self.inercia_inicial = inercia
        self.inercia_final = inercia_final
        self.c1 = c1
        self.c2 = c2
        self.crit = crit
        
        self.ordered_particles = [0] * self.num_particles
        self.sensors = [0] * self.num_particles
        
        self.xmax = Xmax
        self.xmin = -Xmax
        self.posMax = Xmax
        self.posMin = self.xmin
    
    def prepare_data(self, serie, split, window):
        '''
        Method to divide the time series into training and validation
        '''
        partition = SeriesPreprocessor(serie, split, window)

        [train_start, train_end] = partition.part_train()
        [val_start, val_end] = partition.part_val()
        [test_start, test_end] = partition.part_test()

        data = []
        data.append(train_start)
        data.append(train_end)
        data.append(val_start)
        data.append(val_end)
        data.append(test_start)
        data.append(test_end)

        return data
      
    def create_particles(self):
        global counter, fitness, mse
        counter = 0
        fitness = 0
        mse = []
        
        for i in range(self.num_particles):
            p = Particle()
            p.position = np.random.randn(1, self.dimentions)
            p.position = p.position[0]
            p.fitness = self.objective(p.position)
            p.velocity = array([0.0 for i in range(self.dimentions)])
            p.best = p.position
            p.fit_best = p.fitness
            p.c1 = self.c1
            p.c2 = self.c2
            p.inercia = self.inercia
            p.phi = 0
            self.particles.append(p)
        
        self.gbest = self.particles[0]
        
    def objective(self, position):
        '''
        Method to calculate the objective function of the IDPSO,
        in this case the function is the prediction of an ELM
        '''
        ELM = ELMRegressor(self.qtd_neurons)
        position = position.reshape(self.lines, self.qtd_neurons)
        ELM.fit(self.dataset[0], self.dataset[1], position)
        prediction_val = ELM.predict(self.dataset[2])
        MAE_val = mean_absolute_error(self.dataset[3], prediction_val)

        return MAE_val
    
    def fitness(self):
        '''
        Methods to calculate fitness for particles
        '''
        for i in self.particles:
            i.fitness = self.objective(i.position)
        
    def velocity(self):
        '''
        Calculate velocity of particles
        '''
        
        for i in self.particles:
            for j in range(len(i.position)):
                value_c1 = (i.best[j] - i.position[j])
                value_c2 = (self.gbest.position[j] - i.position[j])
                
                inercia = (i.inercia * i.velocidade[j])
                cognitive = ((i.c1 * random.random()) * value_c1)
                group = ((i.c2 * random.random()) * value_c2)
              
                i.velocidade[j] = inercia + cognitive + group
                
                if i.velocity[j] >= self.xmax:
                    i.velocity[j] = self.xmax
                elif i.velocity[j] <= self.xmin:
                    i.velocity[j] = self.xmin
              
    def update_particles(self):
        for i in self.particles:
            for j in range(len(i.position)):
                i.position[j] = i.position[j] + i.velocidade[j]
                
                if i.position[j] >= self.posMax:
                    i.position[j] = self.posMax
                elif i.position[j] <= self.posMin:
                    i.position[j] = self.posMin

    def update_params(self, iteration):
        '''
        Update inercia, c1, c2
        '''
        
        for i in self.particles:
            part1 = 0
            part2 = 0
            
            for j in range(len(i.position)):
                part1 = part1 + self.gbest.position[j] - i.position[j]
                part2 = part2 + i.best[j] - i.position[j]
                
                if part1 == 0:
                    part1 = 1
                if part2 == 0:
                    part2 = 1
                    
            i.phi = abs(part1/part2)
            
        for i in self.particles:
            ln = np.log(i.phi)
            value = i.phi * (iteration - ((1 + ln) * self.iters) / mi)
            i.inercia = ((self.inercia - self.inercia_final) / (1 + np.exp(value))) + self.inercia_final
            i.c1 = self.c1 * (i.phi ** (-1))
            i.c2 = self.c2 * i.phi
       
    def Pbest(self):
        '''
        Calculates particle's personal best
        '''
        for i in self.particles:
            if i.fit_best >= i.fitness:
                i.best = i.position
                i.fit_best = i.fitness

    def Gbest(self):
        '''
        Calculates group best
        '''
        for i in self.particles:
            if i.fitness <= self.gbest.fitness:
                self.gbest = copy.deepcopy(i)
    
    def stop_criteria(self, i):
        '''
        Method to compute the stop criteria, both for the GL5 and
        for no more improvement of the best solution
        '''
        
        global counter, fitness, mse
        
        if i == 0:
            fitness = copy.deepcopy(self.gbest.fitness)
            return i
        else:
            
            if counter == self.crit:
                return self.iters
            elif fitness == self.gbest.fitness:
                counter += 1
                return i
            else:
                fitness = copy.deepcopy(self.gbest.fitness)
                counter = 0
                return i
            
    def predict(self, input, num_sensor=None, output=None):
        '''
        Method to perform the prediction with the best particle (ELM) of the Swarm
        '''
        
        # if the sensor number is not passed then the prediction is made with gbest
        if num_sensor is None:
            if output is None:
                prediction = self.best_elm.predict(input)
                return prediction
            else:
                prediction = self.best_elm.predict(input)
                MSE = mean_absolute_error(output, prediction)
                print('\n MSE: %.2f' %MSE)
                
                return MSE
        else:
            prediction = self.sensors[num_sensor].predict(input)
            return prediction
        
    def forecast(self, input):
        '''
        Prediction performed be the bast particle
        '''

        return self.best_elm.predict(input)
    
    def order_particles(self):
        '''
        Metodo para ordenar as particulas por menor fitness  
        '''
        
        self.ordered_particles = copy.deepcopy(self.particles)
        
        for i in range(0, len(self.ordered_particles) - 1):
            imin = i
            for j in range(i+1, len(self.ordered_particles)):
                if self.ordered_particles[j].fitness < self.ordered_particles[imin].fitness:
                    imin = j
            aux = self.ordered_particles[imin]
            self.ordered_particles[imin] = self.ordered_particles[i]
            self.ordered_particles[i] = aux
             
    def get_sensors(self):
        self.order_particles()
        
        for x, i in enumerate(self.ordered_particles):
            ELM = ELMRegressor(self.qtd_neurons)
            position = i.position.reshape(self.lines, self.qtd_neurons)
            ELM.fit(self.dataset[0], self.dataset[1], position)
            self.sensors[x] = ELM
            
        self.best_elm = self.sensors[0]
        
    def train(self):
        self.create_particles()
        
        i = 0
        while i < self.iters:
            i += 1
            
            self.fitness()
            self.Gbest()
            self.Pbest()
            self.velocity()
            self.update_params(i)
            self.update_particles()
            i = self.stop_criteria(i)
        
        self.get_sensors()
        self.determine_best_elm()
        
    def determine_best_elm(self):
        ELM = ELMRegressor(self.qtd_neurons)
        position = self.gbest.position.reshape(self.lines, self.qtd_neurons)
        ELM.fit(self.dataset[0], self.dataset[1], position)
            
        self.best_elm = ELM
        
    def spread(self):
        '''
        method to erase swarm part information
        '''
        
        qtd = len(self.particles)
        tx = int(qtd * self.tx_spread)
        choices = []
        
        for i in range(tx):
            
            j = self.generate(qtd - 1, choices)
            choices.append(j)
        
            self.particles[j].position = np.random.randn(1, self.dimentions)
            self.particles[j].position = self.particles[j].position[0]
            self.particles[j].fitness = self.objective(self.particles[j].position)
            self.particles[j].velocidade = array([0.0 for i in range(self.dimentions)])
            self.particles[j].best = self.particles[j].position
            self.particles[j].fit_best = self.particles[j].fitness
            self.particles[j].c1 = self.c1
            self.particles[j].c2 = self.c2
            self.particles[j].inercia = self.inercia
            self.particles[j].phi = 0
            
        global counter
        counter = 0
    
    def generate(self, qtd, choices):
        '''
        Random num generator
        '''
        j = np.random.randint(0, qtd)
        if j in choices:
            
            return self.generate(qtd, choices)
            
        else:
            return j
    
    def retrain(self):
        self.spread()
        
        i = 0
        while i < self.iters:
            i += 1
            
            self.fitness()
            self.Gbest()
            self.Pbest()
            self.velocity()
            self.update_params(i)
            self.update_particles()
            i = self.stop_criteria(i)

        self.get_sensors()
    
    def update_best(self, new_data):
        self.best_elm = new_data
