import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

setorosa    = [[],[],[],[]]
versicolor  = [[],[],[],[]]
virginica   = [[],[],[],[]]
with open('iris.data','r') as archive:
    for line in archive:
        # line = line.replace('\n', '')
        print("line = ", line)
        if line != '':
            line = line .split(',')
            if line[-1] == 'Iris-setosa':
                setorosa[0]  .append(float(line[0]))
                setorosa[1]  .append(float(line[1]))
                setorosa[2]  .append(float(line[2]))
                setorosa[3]  .append(float(line[3]))
            elif line[-1] == 'Iris-versicolor':
                versicolor[0]  .append(float(line[0]))
                versicolor[1]  .append(float(line[1]))
                versicolor[2]  .append(float(line[2]))
                versicolor[3]  .append(float(line[3]))
            elif line[-1] == 'Iris-virginica':
                virginica[0]  .append(float(line[0]))
                virginica[1]  .append(float(line[1]))
                virginica[2]  .append(float(line[2]))
                virginica[3]  .append(float(line[3]))

def detect_outlier(data:list) -> list:
    q1, q3      = np.percentile(data, [25, 75])
    iqr         = q3-q1
    upper_bound = q3 + 1.5*iqr 
    lower_bound = q1 - 1.5*iqr
    
    outlier = [data.index(value) for value in data if value > upper_bound or value < lower_bound]
    return outlier

# def remove_data(element:list) -> list:
A,B,C,D = setorosa[:]

outlier = detect_outlier(A)
for c,i in enumerate(outlier):
    A.remove(A[i-c])
    B.remove(B[i-c])
    
outlier = detect_outlier(B)
for c,i in enumerate(outlier):
    A.remove(A[i-c])
    B.remove(B[i-c])
    
outlier = detect_outlier(C)
for c,i in enumerate(outlier):
    C.remove(C[i-c])
    D.remove(D[i-c])
    
outlier = detect_outlier(D)
for c,i in enumerate(outlier):
    C.remove(C[i-c])
    D.remove(D[i-c])
        
plt.boxplot(setorosa[0])
plt.boxplot(setorosa[1])
plt.boxplot(setorosa[2])
plt.boxplot(setorosa[3])
plt.show()