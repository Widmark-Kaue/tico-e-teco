#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 14:35:43 2022

@author: widmark
"""
import perceptron as p
import numpy as np

#%% Treinamento
def training_layer(layer:p.single_layer, database:np.array, data_out:np.array, 
                         eta:any = lambda x: 0.1, number_of_epoca:int = 80_000, 
                         show_per_epoca:int = 1000, random:bool = True, 
                         tol:float = 1e-5):
    
    rows, cols = database.shape
    row_vector = np.arange(0, rows)
    
    for epoca in range(number_of_epoca):
        erro_ac = 0.0
        for row in row_vector:
            phi_v   = layer.apply(database[row])
            erro    = (data_out[row] - phi_v).reshape(layer.number_of_neurons, 1)
            erro_ac += float(sum(0.5*erro**2))
            X       = np.r_[1, database[row]].reshape(1, cols+1)
            w       = layer.weight_matriz + np.dot(erro,X)*eta(epoca)
            layer.weight_aplicate(w)            
        
        if random:
            np.random.shuffle(row_vector)
        if erro_ac <= tol:
            print('-'*10+ f'Época de treinamento {epoca}'+'-'*10)
            print("Perceptron Convergiu")
            print(f'Erro na última época = {erro_ac}')
            break
        if epoca%show_per_epoca == 0:
            print('-'*10+ f'Época de treinamento {epoca}'+'-'*10)
            print(f'Erro na época {epoca} = {erro_ac}')

        
def training_multi_layer(mlp:p.multi_layer, database:np.array, data_out:np.array, 
                         number_of_epoca:int = 80_000, show_per_epoca:int = 1000, 
                         eta:any = lambda x: 0.1, random:bool = True, 
                         tol:float = 1e-5):
    
    
    rows,_     = database.shape
    row_vector = np.arange(0, rows)
        
    for epoca in range(number_of_epoca):
        for row in row_vector:
          phi_v, dphi_v = mlp._foward_propagation_(database[row])
          state_weight  = mlp.weight_list().copy()
          
          phi_v.append(data_out[row])
          phi_v.insert(0, database[row])
          
          
          delta_k       = [0,0]
          w_k           = []
          erro_n        = 0.0
          erro_N        = 0.0  
          aux           = mlp.layer_list.copy()
          
          aux.reverse()
          for i, layer in enumerate(aux):      
              k = -(i+1)
              if i == 0:                                    #Camada de saída
                  e_k = (phi_v[k] - phi_v[k-1])
              else:                                         #Camadas escondidas
                  e_k = np.dot(delta_k[0].T, state_weight[k+1])
                  e_k = e_k[0,1:]                           #descartanto o bias
              
              erro_n += sum(e_k**2/2)
              Yi  = np.r_[1, phi_v[k-2]].reshape(1, len(phi_v[k-2]) + 1)
              aux2 = e_k*dphi_v[k]
              delta_k[i%2] = (aux2).reshape(layer.number_of_neurons, 1)
              
              Delta_k = eta(epoca)*np.dot(delta_k[i],Yi)
              w_k     = layer.weight_matriz + Delta_k
              layer.weight_aplicate(w_k)
          
        erro_N += erro_n/epoca
        
        if random:
            np.random.shuffle(row_vector)
        if erro_N <= tol:
            print('-'*10+ f'Época de treinamento {epoca}'+'-'*10)
            print("Perceptron Convergiu")
            print(f'Erro na última época = {erro_N}')
            break
        if epoca%show_per_epoca == 0:
            print('-'*10+ f'Época de treinamento {epoca}'+'-'*10)
            print(f'Erro médio na época {epoca} = {erro_N}')  
            print(f'Pesos  = {mlp.weight_list()}')
#%% Desempenho

# =============================================================================
# def bootstrap:
# def crossvalidation:
# =============================================================================

#%% 
pasta = '/home/widmark/Documentos/Code/machine_learning/tico-e-teco/dados/'

#%%teste uma camada
inp = np.loadtxt(pasta + 'input.txt')
out = np.loadtxt(pasta + 'output.txt')

camada = p.single_layer(2,2)

training_layer(camada, inp, out)
#%% teste mlp
inp = np.loadtxt(pasta + 'input_berries.txt')
out = np.loadtxt(pasta + 'output_berries.txt')

lista1 = [p.single_layer(3,2), p.single_layer(2,3)]

mlp = p.begin_mlp([2,3], 2, 3)

#%%
training_multi_layer(mlp, inp, out)






