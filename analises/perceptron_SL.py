#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:21:43 2022

@author: widmark
"""

import matplotlib.pyplot as plt
import numpy as np
import network.perceptron as p
import network.practice as tr

#%% Seed
np.random.seed(1)

#%% Funções




#%% Gera dados linearmente Separáveis




#%% berries
inp = np.loadtxt(f'{tr.path_data}/input_berries.txt')
out = np.loadtxt(f'{tr.path_data}/output_berries.txt')


plt.plot(inp[:7,0],inp[:7,1],    'bo', label = 'mertilo')
plt.plot(inp[7:16,0],inp[7:16,1],'ro', label = 'framboesa')
plt.plot(inp[16:,0],inp[16:,1]  ,'go', label = 'açai')

plt.xlabel('Pesos')
plt.ylabel('PH')
plt.grid()
plt.legend()
plt.show()
#%% funções taxa de aprendizado
def eta(ssns):
    x0 = 5.13e-2
    xf = 2.3026
    x  = (xf-x0)/(100_000) * ssns + x0
    return np.exp(-x)
#%% Criando camada
model = p.single_layer(3,2)
#%% Treinamento e Desempenho
training_function = lambda database, data_out: tr.training_layer(model, 
                                                                 database, 
                                                                 data_out, 
                                                                 number_of_epoca = 60_000)
tr.bootstrap_singlelayer(model,training_function,inp, out)













