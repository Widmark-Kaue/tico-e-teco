#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:21:43 2022

@author: widmark
"""

import matplotlib.pyplot as plt
import network.perceptron as p
import network.practice as tr
import pandas as pd
import numpy as np

#%% Seed
np.random.seed(2524)

save = False
#%%
# =============================================================================
#                           ANÁLISE SINTETIC DATA
#               Base linearmente separável gerada pelo autor
# =============================================================================

#%% Gera dados linearmente Separáveis

#2 Classes
C2 = tr.gera_dados([20, 20],[[100,3], [170.5, 35]], n_ponto = 100)

C = C2

labels = ['classe 1', 'classe 2']
cor    = ['b','r', 'g'] 
arq = open(f'{tr.path_data}/sintetic.txt','w')
for i in range (len(C)):
    x,y = C[i]
    arq.write(f"#Classe {i+1}\n")
    vet = np.zeros(len(C))
    vet[i] = 1
    for j in range(len(x)):
        arq.write(f'{x[j]} {y[j]} {int(vet[0])} {int(vet[1])}\n')
    plt.plot(x,y,f'{cor[i]}o', label = f'{labels[i]}')
arq.close()

plt.title('Base de Dados Sintética')
plt.xlabel('Atributo 1')
plt.ylabel('Atributo 2')
plt.legend()
plt.grid()

if save:
    plt.savefig(f'{tr.path_save}/imagens/sintetica',dpi = 720) 
plt.show()

#%% Import Sintetic data 

sint        = np.loadtxt(f'{tr.path_data}/sintetic.txt')
sint_input  = sint[:, 0:2]
sint_output = sint[:,  2:]  

#%% Sintetic - Treinamento com função sigmoid e degrau
print(10*"=="+"Função Sigmoid"+10*"==")
model1      = p.single_layer(2,2)
erro_sig    = tr.training_layer(model1, sint_input, sint_output)

print(10*"=="+"Função degrau"+10*"==")
model2      = p.single_layer(2,2, func_activation =  p.deg)
erro_deg    = tr.training_layer(model2, sint_input, sint_output)

np.savetxt(f'{tr.path_save}/arquivos/erro_sig.txt', erro_sig)
np.savetxt(f'{tr.path_save}/arquivos/erro_deg.txt', erro_deg)

#%% Sintetic - Plot Erro RMS
epoca_sig = np.arange(1, len(erro_sig)+1)
epoca_deg = np.arange(1, len(erro_deg)+1)
plt.plot(epoca_sig, [erro_sig[i-1] for i in epoca_sig], 'purple', label = "Sigmoid")
plt.plot(epoca_deg, [erro_deg[i-1] for i in epoca_deg], 'orange', alpha = 0.5 ,label = "Degrau")

plt.title("Comparação entre Funções de Ativação")
plt.xlabel('Épocas')
plt.ylabel('Erro RMS')

plt.legend()
plt.grid()
if save:
    plt.savefig(f'{tr.path_save}/imagens/Erro_RMS',dpi = 720) 
plt.show()

#%% Sintetic - Plot da reta
func = []

for j in range(model1.number_of_neurons):
    w = model1.weight_matriz[j]
    y = lambda x: -(w[1]*x + w[0])/w[2]
    func.append(y)

x_a = np.linspace(min(sint_input[:,0]), max(sint_input[:,0]))
for i in range (len(C)):
    x,y = C[i]
    plt.plot(x,y,f'{cor[i]}o', label = f'{labels[i]}')
    plt.plot(x_a, func[i](x_a), f'{cor[i]}')
    
plt.title('Separação das Classes')
plt.xlabel('Atributo 1')
plt.ylabel('Atributo 2')
plt.legend()
plt.grid()
if save:
    plt.savefig(f'{tr.path_save}/imagens/reta',dpi = 720) 
plt.show()    

#%% Sintetic - Experimentos com etas Fixos
eta_val = [0.5, 0.3, 0.15, 0.1, 0.05, 0.01]
eta_dict = {}
model3 = p.single_layer(2,2)
for i in eta_val:
    model3.weight_random_init(0)
    erro = tr.training_layer(model3, sint_input, sint_output, eta = lambda x: i, show_per_epoca = None)
    eta_dict[str(i)] = erro.copy()
#%% Plot etas fixos

for i in eta_val:
    temp = eta_dict[str(i)]
    epoca = np.arange(1,len(temp)+1)
    plt.plot(epoca, temp, alpha = i + 0.5, label = f'$\eta$ = {i}')

plt.title("Comparação entre Taxas de aprendizagens Fixas")
plt.xlabel('Épocas')
plt.ylabel('Erro RMS')

plt.legend()
plt.grid()
if save:
    plt.savefig(f'{tr.path_save}/imagens/eta_fixo',dpi = 720) 
plt.show()
#%% Experimentos com etas variáveis
def exp_eta(epoca:int, epoca_final:int, eta0:float = 0.5, etaf:float = 0.1): 
    x0 = -np.log(eta0)
    xf = -np.log(etaf)
    x  = (xf-x0)/(epoca_final -1) * (epoca-1) + x0
    return np.exp(-x)

def rampa_eta(epoca:int, epoca_final:int, eta0:float = 0.5, etaf:float = 0.05):
    return ((etaf - eta0)/(epoca_final- 1))*(epoca-1) + eta0

rampa_eta_f = lambda x: rampa_eta(x, 5_000)
exp_eta_f   = lambda x: exp_eta(x, 5_000) 


model3.weight_random_init(0)
erro = tr.training_layer(model3, sint_input, sint_output, eta = rampa_eta_f)
eta_dict["rampa_eta"] = erro.copy()

model3.weight_random_init(0)
erro = tr.training_layer(model3, sint_input, sint_output, eta = exp_eta_f)
eta_dict["exp_eta"] = erro.copy()
#%% Plot eta variável

eta_rampa = eta_dict['rampa_eta']
eta_exp   = eta_dict['exp_eta']

epoca_rampa = np.arange(1, len(eta_rampa) +1)
epoca_exp   = np.arange(1, len(eta_exp)   +1)

plt.plot(epoca_rampa,   eta_rampa,    label = 'Rampa')
plt.plot(epoca_exp,     eta_exp,      label = 'Exp'  )


plt.title("Comparação entre Taxas de aprendizagens Variáveis")
plt.xlabel('Épocas')
plt.ylabel('Erro RMS')

plt.legend()
plt.grid()
if save:
    plt.savefig(f'{tr.path_save}/imagens/eta_variável',dpi = 720) 
plt.show()


#%% Sintetic - Desempenho
model4  = p.single_layer(2,2)
model41 = p.single_layer(2,2)
training_function = lambda database, data_out: tr.training_layer(model4, 
                                                                  database, 
                                                                  data_out,
                                                                  eta = lambda x: 0.3,
                                                                  number_of_epoca = 10_000,
                                                                  show_per_epoca = None)

training_function2 = lambda database, data_out: tr.training_layer(model4, 
                                                                  database, 
                                                                  data_out,
                                                                  eta = exp_eta_f,
                                                                  number_of_epoca = 10_000,
                                                                  show_per_epoca = None)

boot11 = tr.bootstrap_singlelayer(model4,training_function,sint_input, sint_output, Bs = 10)
boot22 = tr.bootstrap_singlelayer(model4,training_function2,sint_input, sint_output, Bs = 10)

np.savetxt(f"{tr.path_save}/arquivos/boot_0.3", boot11)
np.savetxt(f"{tr.path_save}/arquivos/boot_exp", boot22)

# Resultados bootstrap
print("Eta constante - 0.3")
print(f'Média = {round(boot11[1]*100,3)}%')
print(f'Desvio padrão = {round(boot11[0]*100,3)}%\n')
print("Eta curva exponencial")
print(f'Média = {round(boot22[1]*100,3)}%')
print(f'Desvio padrão = {round(boot22[0]*100,3)}%')

#%%
# =============================================================================
#                           ANÁLISES IRIS.DATA
#                       Base não linearmente separável
# =============================================================================

#%% Import Iris-data 

iris_data   = np.array(pd.read_csv(f'{tr.path_data}/iris.data'))
row, col    = iris_data.shape
iris_input  = np.zeros([row,col-1]) 
iris_output = np.zeros([row,3])

for i in range(len(iris_data)):
    iris_input[i] = np.float_(iris_data[i,:-1])
    if iris_data[i,-1] == 'Iris-setosa':
        iris_output[i] = np.array([1,0,0])
    elif iris_data[i,-1] == 'Iris-versicolor':
        iris_output[i] = np.array([0,1,0])
    else:
        iris_output[i] = np.array([0,0,1])

#%% Iris - Treinamento  com função sigmoid 
print(10*"=="+"Função Sigmoid"+10*"==")
model5      = p.single_layer(3,col-1)
erro_sig    = tr.training_layer(model5, iris_input, iris_output, show_per_epoca = 10_000)

np.savetxt(f"{tr.path_save}/arquivos/erro_sig_Iris.txt", erro_sig)


#%% Iris - Plot Erro RMS 
epoca_sig = np.arange(1, len(erro_sig)+1)
plt.plot(epoca_sig, [erro_sig[i-1] for i in epoca_sig], 'purple', label = "Sigmoid")

plt.title("Comparação entre Funções de Ativação")
plt.xlabel('Épocas')
plt.ylabel('Erro RMS')

plt.legend(loc = 1)
plt.grid()
if save:
    plt.savefig(f'{tr.path_save}/imagens/Erro_RMS_Iris',dpi = 720)    
plt.show()
#
#%% Iris - Desempenho

model7 = p.single_layer(3,col-1)
training_function = lambda database, data_out: tr.training_layer(model7, 
                                                                  database, 
                                                                  data_out, 
                                                                  number_of_epoca = 60_000,
                                                                  show_per_epoca =  60_000)

boot2 = tr.bootstrap_singlelayer(model7,training_function,iris_input, iris_output)
np.savetxt(f"{tr.path_save}/arquivos/boot2.txt", boot2)
# Resultados bootstrap
print(f'Média = {round(boot2[1]*100,3)}%')
print(f'Desvio padrão = {round(boot2[0]*100,3)}%')
#%%







