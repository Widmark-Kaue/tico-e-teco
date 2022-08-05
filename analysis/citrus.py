#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 23:19:18 2022

@author: widmark
"""

import pandas as pd
import network.practice as v
import network.perceptron as n
import numpy as np

path = f'{v.path_data}/citrus_1.txt' 

data_hans = np.array(pd.read_csv(path))
data = data_hans[:,:-1]
data = np.array([[float(data[i,0]), float(data[i, 1])] for i,_ in enumerate(data)])
results = np.array([[1,0] if val == 'maca' else [0,1] for val in data_hans[:,-1]])
def sigm(v):
    val = []
    for i,k in enumerate(v):
        val.append(n.sigmoid(k))
    return np.array(val)

#%%
neuron = n.single_layer(2, 2)
v.bootstrap_singlelayer(neuron, 
                        lambda train, results: v.training_layer(neuron, train, results, 
                                                                              show_per_epoca=150_000), 
                        data, results)