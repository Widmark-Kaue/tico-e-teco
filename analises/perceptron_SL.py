#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:21:43 2022

@author: widmark
"""

import matplotlib.pyplot as plt
import network.perceptron as pe
import network.practice as tr
import numpy as np

#%% import data

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
def f(ssns):
    x0 = 5.13e-2
    xf = 2.3026
    x  = (xf-x0)/(100_000) * ssns + x0
    return np.exp(-x)
#%% Treinamento

model = pe.single_layer(3,2)
model.weight_random_init()












