#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import matplotlib.pyplot as plt
import neural.neural as n
import numpy as np

#%% função de ativação
def deg(x):
    if  x>0:
        return 1
    else:
        return 0

def tanh(v):
    return np.tanh(v/2)

def sigmoid(v):
    return 1/(1 + np.exp(-v))
#%% import dados
  
pasta  = '/home/widmark/Documentos/Code/machine_learning/meu_unico_neuronio/data/'

Input  = np.loadtxt(pasta + 'input.txt')
Output = np.zeros([len(Input), 2])
with open(pasta + 'output.txt') as file:
    out_text = [i for i in file]
    for i,text in enumerate(out_text):
        text = text.replace('\n', '')
        if text == 'laranja':
            Output[i,0] = 1 
        elif text == 'maca':
            Output[i,1] = 1
    
    
#%% taxa de aprendizado
def f(ssns):
    x0 = 5.13e-2
    xf = 2.3026
    x  = (xf-x0)/(100_000) * ssns + x0
    return np.exp(-x)

#%% treinamento
SR = [[np.array([0,1]), 'Maçã'],
      [np.array([1,0]), 'Laranja']]

neuron_layer = n.layer(2, 3, deg, syntax_resp=SR)
error = n.training_layer(neuron_layer, Input, Output, eta = lambda a: 0.01)

#%% Plotagem das retas
# w0_m = neuron_layer.neurons[0].weight[0]
# w1_m = neuron_layer.neurons[0].weight[1]
# w2_m = neuron_layer.neurons[0].weight[2]

# Ph_maca = lambda P:(w1_m*P + w0_m)/-w2_m

# w0_l = neuron_layer.neurons[1].weight[0]
# w1_l = neuron_layer.neurons[1].weight[1]
# w2_l = neuron_layer.neurons[1].weight[2]

# Ph_laranja = lambda P:(w1_l*P + w0_l)/-w2_l

# for i,YX in enumerate(Input):
#     if all(Output[i] == [0,1]):
#         plt.plot(YX[0], YX[1], 'ro')
#     else:
#         plt.plot(YX[0], YX[1], 'yx')
        
# X = np.linspace(95,125,5)
# plt.plot(X,Ph_maca(X),'r') 
# plt.plot(X,Ph_laranja(X),'y') 
# plt.show()

#%% plot error
ite = np.arange(0, len(error))
plt.plot(ite,error[:, 0],'r')
plt.show()
  