#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 14:35:43 2022

@author: widmark
"""
import perceptron as p
import numpy as np

#%% Treinamento
def training_layer(layer:p.layer, database:np.array, data_out:np.array, 
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
          phi_v, dphi_v = mlp.foward_propagation(database[row])
          state_weight  = mlp.weight_list.copy()
          
          phi_v.append(data_out[row])
          phi_v.insert(0, database[row])
          
          phi_v.reverse()
          dphi_v.reverse()
          state_weight.reverse()
          
          delta_k       = []
          w_k           = []
          erro_ac       = 0.0
 
          for k, layer in enumerate(mlp.layer_list.reverse()):      
              if k == 0:                                    #Camada de saída
                  e_k = (phi_v[k] - phi_v[k+1])
              else:                                         #Camadas escondidas
                  e_k = np.dot(delta_k[k-1].T, state_weight[k-1]) 
              
              
              Yi      = np.r_[1, phi_v[k+1]].reshape(1, len(phi_v[k+1]) + 1)
              delta_k.append(e_k*dphi_v[k].reshape(layer.number_of_neurons,1))
              
              Delta_k = eta(epoca)*np.dot(delta_k[k],Yi)
              w_k     = layer.weight_matriz + Delta_k
              layer.weight_aplicate(w_k)
          
          erro_ac += float(sum(0.5*e_k**2))
          mlp.layer_list.reverse()
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
         
    return 0
    
#%% Desempenho

# teste = [1,2,3]
# mlp = p.multi_layer(lista1,dsigmoid)
#%%
# entrada = teste.copy()
# phi_v     = []
# v         = [] 
# i = 1
# for layer_i in lista1:
#     print(f'Ite = {i}')
#     i +=1
#     entrada = layer_i.apply(entrada)
#     print(f'TE = {len(entrada)}')
#     v.append(layer_i.field_ind(entrada))
#     phi_v.append(entrada.copy())
# dphi_v = [dsigmoid(i) for i in v]

#%% 
pasta = '/home/widmark/Documentos/Code/machine_learning/tico-e-teco/dados/'

#%%teste uma camada
inp = np.loadtxt(pasta + 'input.txt')
out = np.loadtxt(pasta + 'output.txt')

camada = p.layer(2,2)

training_layer(camada, inp, out)
#%% teste 2
inp = np.loadtxt(pasta + 'input_berries.txt')
out = np.loadtxt(pasta + 'output_berries.txt')

lista1 = [p.layer(3,2), p.layer(2,3)]

mlp = p.multi_layer(lista1)
training_multi_layer(mlp, inp, out)






