import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import *
from scipy.linalg import norm, pinv


def Q(weights, kernels):
    return sum(np.multiply(weights,kernels))
    

def _basisfunc(beta, c, d):
    return exp(-beta * norm(c-d)**2)
     
def _calcAct(X, centers, numCenters):
    # calculate activations of RBFs
    G = zeros((X.shape[0], numCenters), float)
    for ci, c in enumerate(centers):
        for xi, x in enumerate(X):
                G[xi,ci] = _basisfunc(0.08, c, x)
    return G / sum(G) #normalizing Neuron activations


centers = [1,2,3,4,5,6]

a = [-1,1]
        
Y = []
XL = []
Q_value = []
epsilon = 0.7

weightsL = [0]*len(centers)
weightsL = np.asarray(weightsL)
weightsR = [0]*len(centers)
weightsR = np.asarray(weightsR)

Lrate = 0.7
gamma = 0.5

states = [2,3,4,5]

for i in range(10000):
    s1 = random.choice(states,1)
    while s1 >1.5 and s1<5.5:
        
        G1 = _calcAct(np.asarray([s1]), centers, len(centers))
        
        QL = Q(weightsL, G1)
        QR = Q(weightsR, G1)

        #define action
        if  np.random.uniform() < epsilon:
            action = int(np.random.choice(a,1))
        else:      
            if QL > QR:
                action = -1
            else:
                action = 1

        s2 = s1 + action + np.random.normal(0,0.01)

        if s2 < 1.5:
            R = 1
        elif s2 > 5.5:
            R = 5
        else:
            R = 0

        G2 = _calcAct(np.asarray([s2]), centers, len(centers))
        QL_new = Q(weightsL, G2)
        QR_new = Q(weightsR, G2)
        
        if action == -1:    
            weightsL = weightsL + Lrate * (R + gamma*max(QL_new, QR_new) - QL) * G1
        if action == 1:
            weightsR = weightsR + Lrate * (R + gamma*max(QL_new, QR_new) - QR) * G1
        s1 = s2


x = np.arange(1,6,0.01)
yL = []
yR = []
for index,i in enumerate(x):
    yL.append(Q(weightsL, _calcAct(np.asarray([i]), centers, len(centers)))/np.sum(_calcAct(np.asarray([i]), centers, len(centers))))
    yR.append(Q(weightsR, _calcAct(np.asarray([i]), centers, len(centers)))/np.sum(_calcAct(np.asarray([i]), centers, len(centers))))

line_1,= plt.plot(x,yL, 'b-')
line_2,= plt.plot(x,yR,'r-')
plt.xlabel('states')
plt.ylabel('Q Value')
plt.legend([line_1, line_2], ['Q values on the left', 'Q values on the right'])
plt.show()


