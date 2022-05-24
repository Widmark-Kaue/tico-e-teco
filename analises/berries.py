#%%
import matplotlib.pyplot as plt
import network.perceptron as pe
import network.practice as tr
import numpy as np


#%% Gera dados linearmente separáveis



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

#%% treinamento
SR = [[np.array([1,0,0]), 'Mertilo'  ],
      [np.array([0,1,0]), 'Framboesa'],
      [np.array([0,0,1]), 'Açaí'     ]]

neuron_layer = n.layer(number_of_neurons=3, number_of_elements=3, activation_function=deg, syntax_resp=SR)
error        = n.training_layer(neuron_layer, inp_berries, out_berries, eta = lambda a: 0.1)
print(neuron_layer.neurons)

#%% plot error
ite = np.arange(0, len(error))
plt.plot(ite,error[:, 0],'r')
plt.show()

