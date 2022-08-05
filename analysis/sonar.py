#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import neural.neural as n
import neural.validation as v
import pandas as pd
import numpy as np
import random as r

#%% import data
pasta = '/home/machina/Documentos/SEM_8/machine_learning/perceptron/data/'
sonar = pd.read_csv(pasta + 'sonar_csv.csv')

attributes  = np.array(sonar)[:,:-1]
classes     = np.array(sonar)[:, -1]
results     = np.array([[0,1] if term=='Mine' else [1,0] for term in classes])

rand_1_3        = r.sample(range(len(attributes)), int(len(attributes)/3))
rand_1_3.sort()
rand_2_3        = [i for i,_ in enumerate(attributes)]
for val in rand_1_3:
    rand_2_3.remove(val) 

attributes_1_3  = np.array([attributes[i] for i in rand_1_3])
results_1_3     = np.array([results[i] for i in rand_1_3])

attributes_2_3  = np.array([attributes[i] for i in rand_2_3])
results_2_3     = np.array([results[i] for i in rand_2_3])

#%% função de ativação e taxa de aprendizado
def deg(x):
    if  x>0:
        return 1
    else:
        return 0
    
def tanh(v):
    return np.tanh(v/2)

def sigmoid(v):
    return 1/(1+np.exp(-v))

#%% treinamento
SR              = [[np.array([1,0]), 'Rock'],
                   [np.array([0,1]), 'Mine']]
neuron_layer    = n.layer(2, 61, deg, SR)
neuron_layer.weight_randomize()
error           = v.training_layer(neuron_layer, attributes_2_3, results_2_3, eta = lambda a: 0.01)

#%% Plotagem
plt.plot(range(len(error)), error)
plt.show()

#%% Avaliação
correct = 0 
for i,val in enumerate(attributes_1_3):
    if all(results_1_3[i] == np.array([1 if round(i) > 0 else 0 for i in neuron_layer.aplicate(val)])):
        print(f'{i}/{len(results_1_3)} : Passou')
        print(neuron_layer.aplicate(val))
        correct+=1
    else:
        print(f'{i}/{len(results_1_3)} : Não Passou')
        print(neuron_layer.aplicate(val))
    print('---------------------------')
print('---------------------------')
print(f'Resultado: {round(correct/len(attributes_1_3)*100,2)}%')




