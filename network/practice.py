#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 14:35:43 2022

@author: widmark
"""

from sklearn.utils import resample
import network.perceptron as p
from numpy import random as rd
from os import getcwd
import numpy as np

#%% Pastas de databases
path_data   = f"{getcwd().replace('analises', '')}/dados"
path_save   = f"{path_data.replace('dados','')}/resultados" 
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
        value_max   = xy_a + r
        value_min   = xy_a - r
        C.append([rd.random(n)+rd.randint(value_min[i],
                                       value_max[i], size = n) for i in range(len(xy_a))])
    return C
    
#%% Treinamento
def training_layer(layer:p.single_layer, database:np.array, data_out:np.array, 
                         eta:any = lambda x: 0.1, number_of_epoca:int = 80_000, 
                         show_per_epoca:int = 1000, random:bool = True, 
                         tol:float = 1e-3):
    _,col = data_out.shape
    
    assert layer.number_of_neurons >= col, "Número de neurônios dever maior ou igual ao número de classes"
    
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
        if show_per_epoca != None and epoca%show_per_epoca == 0:
            print('-'*10+ f'Época de treinamento {epoca}'+'-'*10)
            print(f'Erro na época {epoca} = {erro_n[-1]}')
    
    return np.array(erro_n)

        
def training_multi_layer(mlp:p.multi_layer, database:np.array, data_out:np.array, 
                         number_of_epoca:int = 80_000, show_per_epoca:int = 1000, 
                         eta:any = lambda x: 0.1, random:bool = True, 
                         tol:float = 1e-3):
    
    
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
        if show_per_epoca != None and epoca%show_per_epoca == 0:
            print('-'*10+ f'Época de treinamento {epoca}'+'-'*10)
            print(f'Erro médio = {erro_N}')
            print(f'Erro médio por época = {erro_N/epoca}')  
            print(f'Saída = {phi_v[-1]}')
            
#%% Avaliação de performance

def bootstrap_singlelayer (model:p.single_layer, training_function:any,
                           database:np.array, dataout:np.array, Bs:int = 12):
    erro = []
    f = lambda x: x != -3
    for boot in range(1,Bs+1):
        print(f'======================= BOOTSTRAP {boot} =======================')
        model.weight_random_init()
        ind         = np.arange(0, len(database)) 
        training    = resample(ind)
        avaliation  = [ -3 if j in training else j for j in ind]
        avaliation  = list(filter(f, avaliation))
      
        
        database_tr = np.array([database[i] for i in training])
        dataout_tr  = np.array([dataout[i]  for i in training])
        database_tt = np.array([database[i] for i in avaliation])
        dataout_tt  = np.array([dataout[i]  for i in avaliation])

        training_function(database_tr, dataout_tr)
        Tx_erro     = 0
        for i, dtt in enumerate(database_tt):
            if not all(model.apply_abs(dtt) == dataout_tt[i]):
                Tx_erro+=1
        erro.append(Tx_erro/len(dataout_tt))
        print(f'>>> Erro Bootstraps: {erro[-1]} <<<')
    return np.std(erro), np.average(erro) 





