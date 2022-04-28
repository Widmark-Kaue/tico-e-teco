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
  
pasta  = '/home/machina/Documentos/SEM_8/machine_learning/perceptron/data/'

inp_berries  = np.loadtxt(pasta + 'input_berries.txt')
out_berries = np.zeros([len(inp_berries), 3])
with open(pasta + 'output_berries.txt') as file:
    out_text = [i for i in file]
    for i,text in enumerate(out_text):
        text = text.replace('\n', '')
        if text == 'mertilo':
            out_berries[i,0] = 1 
        elif text == 'framboesa':
            out_berries[i,1] = 1
        else:
            out_berries[i,2] = 1

# print(f'Input Berries = {inp_berries}') 
# print(f'Output Berries = {out_berries}')  
    
#%% taxa de aprendizado
def f(ssns):
    x0 = 5.13e-2
    xf = 2.3026
    x  = (xf-x0)/(100_000) * ssns + x0
    return np.exp(-x)

plt.plot(inp_berries[ 0: 7,0], inp_berries[ 0: 7,1], 'bo', label = 'mertilo')
plt.plot(inp_berries[ 7:16,0], inp_berries[ 7:16,1], 'ro', label = 'framboesa')
plt.plot(inp_berries[ 16: ,0], inp_berries[ 16: ,1], 'co', label = 'açaí')
plt.legend()
plt.grid()
plt.show()  
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

