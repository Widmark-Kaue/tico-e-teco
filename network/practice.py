#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 14:35:43 2022

@author: widmark
"""

from sklearn.utils import resample
import network.perceptron as p
from os import getcwd
import numpy as np

#%% Pastas de databases
path_data   = f"{getcwd().replace('analises', '')}/dados"

#%% Funções

def gera_dados(raio:list or np.array, xy:list, n_ponto:int = None,
               rho:float = None, s:int = None):

    assert len(raio) == len(xy), "Quantidade de raios deve ser igual a quanditade de centros."
    assert rho != None or n_ponto != None, "É necessário passar a quantide de pontos ou a densidade da região"
    
    if s != None: np.random.seed(s)
    
    C    = []
    raio = np.array(raio) 
    if n_ponto == None: n_ponto = np.pi*raio**2*rho
    for i in range(len(raio)):
        xy_a        = np.array(xy[i])
        r           = raio[i] 
        
        if n_ponto == None: 
            n = np.pi*r**2*rho
        else:
            n = n_ponto
        print(xy_a)
        value_max   = xy_a + r
        value_min   = xy_a - r
        a           = (value_max - value_min)*value_max
        print(a)
        aux = [a[i]*np.random.random(n)+value_min[i] for i in range(len(a))]
        C.append(aux)
    return C



    
#%% Treinamento
def training_layer(layer:p.single_layer, database:np.array, data_out:np.array, 
                         eta:any = lambda x: 0.1, number_of_epoca:int = 80_000, 
                         show_per_epoca:int = 1000, random:bool = True, 
                         tol:float = 1e-5):
    
    rows, cols = database.shape
    row_vector = np.arange(0, rows)
    erro_n     = [] 
    
    for epoca in range(1,number_of_epoca+1):
        erro_ac = 0.0
        for row in row_vector:
            phi_v   = layer.apply(database[row])
            erro    = (data_out[row] - phi_v).reshape(layer.number_of_neurons, 1)
            erro_ac += float(sum(0.5*erro**2))
            X       = np.r_[1, database[row]].reshape(1, cols+1)
            w       = layer.weight_matriz + np.dot(erro,X)*eta(epoca)
            layer.weight_aplicate(w)            
        
        erro_n.append(erro_ac/rows)
        if random:
            np.random.shuffle(row_vector)
        if erro_n[-1] <= tol:
            print('-'*10+ f'Época de treinamento {epoca}'+'-'*10)
            print("Perceptron Convergiu")
            print(f'Erro na última época = {erro_n[-1]}')
            break
        if epoca%show_per_epoca == 0:
            print('-'*10+ f'Época de treinamento {epoca}'+'-'*10)
            print(f'Erro na época {epoca} = {erro_n[-1]/rows}')

        
def training_multi_layer(mlp:p.multi_layer, database:np.array, data_out:np.array, 
                         number_of_epoca:int = 80_000, show_per_epoca:int = 1000, 
                         eta:any = lambda x: 0.1, random:bool = True, 
                         tol:float = 1e-5):
    
    
    rows,_     = database.shape
    row_vector = np.arange(0, rows)
    erro_N     = 0

    for epoca in range(1,number_of_epoca + 1):
        erro_n = 0.0
        for row in row_vector:
          phi_v, dphi_v = mlp._foward_propagation_(database[row])
         
          phi_v.insert(0, database[row])
          
          delta_k       = []
          aux           = np.arange(0,len(mlp.layer_list))
          
          #======================BackPropagation=========================#
          for k in reversed(aux):      
              layer = mlp.layer_list[k]
              if k == len(mlp.layer_list) - 1:                  #Output Layer
                  e_k   = (data_out[row] - phi_v[-1])
                  erro_n+= sum(e_k**2/2)
              else:                                             #Hidden Layer
                  bfr_layer = mlp.layer_list[k+1]             
                  e_k   = np.dot(delta_k[-1].T, bfr_layer.weight_matriz[:,1:])
                  
              aux2      = e_k*dphi_v[k]
              delta_k.append((aux2).reshape(layer.number_of_neurons, 1))
              
          #=====================Ajuste de Pesos==========================#
          delta_k.reverse()
          for k in reversed(aux): 
              layer = mlp.layer_list[k]
              Yi        = np.r_[1, phi_v[k]].reshape(1, len(phi_v[k]) + 1)
              Delta_k   = eta(epoca)*np.dot(delta_k[k],Yi)             
              w_k       = layer.weight_matriz + Delta_k
              layer.weight_aplicate(w_k)
          
        erro_N += erro_n/rows
        
        if random:
            np.random.shuffle(row_vector)
        if erro_N/epoca <= tol:
            print('-'*10+ f'Época de treinamento {epoca}'+'-'*10)
            print("Perceptron Convergiu")
            print(f'Erro na última época = {erro_N/epoca}')
            break
        if epoca%show_per_epoca == 0:
            print('-'*10+ f'Época de treinamento {epoca}'+'-'*10)
            print(f'Erro médio = {erro_N}')
            print(f'Erro médio por época = {erro_N/epoca}')  
            print(f'Saída = {phi_v[-1]}')
            
#%% Desempenho
def bootstrap_singlelayer (model:p.single_layer,training_function:any,
                           database:np.array, data_out:np.array,N:int=10):
    erro = []
    for foo in range(N):
        print(f'======================= BOOTSTRAP {foo} =======================')
        model.weight_random_init()
        train_index = resample(list(range(len(database))))
        test_index  = list(filter(lambda x: x!=-1,
                      [i if not i in train_index else -1 for i in range(len(database))]))
        
        data_train      = np.array([database[i] for i in train_index])
        results_train   = np.array([data_out[i] for i in train_index])
        
        data_test       = np.array([database[i] for i in test_index])
        results_test    = np.array([data_out[i] for i in test_index])

        training_function(data_train, results_train)
        erro_iteration  = 0
        for i,res in enumerate(data_test):
            if not all(model.apply(res) == results_test[i]):
                erro_iteration+=1
        erro.append(erro_iteration/len(results_test))
        print(f'--> erro bootstrap: {erro_iteration/len(results_test)}')
    return np.average(erro), np.std(erro), np.array(erro)

#%%teste uma camada

#%% teste mlp
# =============================================================================
# inp = np.loadtxt(pasta + 'input_berries.txt')
# out = np.loadtxt(pasta + 'output_berries.txt')
# 
# lista1 = [p.single_layer(3,2), p.single_layer(2,3)]
# 
# mlp = p.begin_mlp([3,2,3], 2, 3, random_init=False)
# 
# 
# training_multi_layer(mlp, inp, out, number_of_epoca=10000)
# =============================================================================






