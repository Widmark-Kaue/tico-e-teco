#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 21:06:07 2022

@author: widmark
"""
import numpy as np
#%% Função de Ativação

def deg(v):
    if  v>=0:
        return 1
    else:
        return 0

def tanh(v):
    return np.tanh(v/2)

def sigmoid(v):
    return 1/(1 + np.exp(-v))

def dsigmoid(v):
    return sigmoid(v)*(1 - sigmoid(v))

#%%
class single_layer:
    def __init__(self, number_of_neurons:int, number_of_atributes:int,
                 func_activation:any = sigmoid):
        self.number_of_atributes    = number_of_atributes
        self.number_of_inputs       = self.number_of_atributes + 1
        self.number_of_neurons      = number_of_neurons
        self.weight_matriz          = np.zeros([self.number_of_neurons, self.number_of_inputs])
        self.phi                    = func_activation
    
    def __repr__(self):
        rep  = f'Camada:{self.number_of_neurons} neurônios \nEntradas:{self.number_of_atributes} atributos mais o bias\n\n'
        rep += 'Pesos:\n'
        for i in range(self.number_of_neurons):
            rep += f'\nNeurônio {i+1}\n'
            rep += '-'*16 + '\n'
            for j in range(self.number_of_inputs):
                x = self.weight_matriz[i,j]
                if x >= 0:
                    sin = '+'
                else:
                    sin = '-'
                rep += f'|w{j} =' + sin + f'{abs(x):.3e}' + '|\n'
            rep += '-'*16 + '\n'
        return rep
        
    def weight_random_init(self, value_max:int = 1, value_min:int = -1):
        assert value_max >= value_min, "Valor máximo dever ser maior ou igual que valor mínimo"
        a = (value_max - value_min)*value_max
        self.weight_matriz = a*np.random.random(self.weight_matriz.shape) + value_min
        
    def weight_aplicate(self, new_weight:np.array):
        assert self.weight_matriz.shape == new_weight.shape, "Matriz de peso incoerente com a camada"
        self.weight_matriz = new_weight.copy()
    
    def field_ind(self, data_input:np.array):
        assert len(data_input) == self.number_of_atributes, "Dados incoerentes com o número de atributos"
        X = np.r_[1, data_input]
        v = np.dot(X, self.weight_matriz.T)
        return v
        
    def apply(self, data_input:np.array):
        v = self.field_ind(data_input)
        phi_v = [self.phi(i) for i in v]
        return np.array(phi_v)
    

    
class multi_layer:
    def __init__(self, layer_list:list, derivate_func_activation:any = dsigmoid):
        self.layer_list       = layer_list
        self.number_of_layers = len(self.layer_list)
        self.dphi             = derivate_func_activation
        
        for i in range(len(layer_list)-1):
            aux = layer_list[i].number_of_neurons == layer_list[i+1].number_of_atributes
            assert aux,f'Camada {i+1} não possui entradas o suficiente'
        
    def _foward_propagation_(self, data:np.array):
        entrada = data.copy()
        phi_v   = []
        v       = [] 
        for layer_i in self.layer_list:
            v.append(layer_i.field_ind(entrada))
            entrada = layer_i.apply(entrada)
            phi_v.append(entrada.copy())
        dphi_v = [self.dphi(i) for i in v]
        return phi_v, dphi_v
    
    def apply(self, data:np.array):
        phi_v,_ = self._foward_propagation_(data)
        return phi_v[-1]
    
    def weight_list(self,):
        return [layer.weight_matriz for layer in self.layer_list]
    
    def weight_random_init(self, value_max:int = 1, value_min:int = -1):
        for layer_i in self.layer_list:
            layer_i.weight_random_init(value_max, value_min)
            
#%% Função de criação do mlp

def begin_mlp(neurons_per_layer:list, inputs:int, outputs:int, random_init:bool = True):
    assert neurons_per_layer[-1] == outputs,"Número de neurônios na última camada deve ser igual ao número de classes"
    aux = []
    for k,i in enumerate(neurons_per_layer):
        if k == 0:
            aux.append(single_layer(i, inputs))
        else:
            aux.append(single_layer(i, neurons_per_layer[k-1]))
    mlp = multi_layer(aux)
    if random_init: mlp.weight_random_init()
    return mlp
    
    
