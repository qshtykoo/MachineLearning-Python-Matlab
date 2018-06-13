import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import *
from scipy.linalg import norm, pinv

'''Linear Approximation for Q'''


def Q(weights, kernels):
    return sum(np.multiply(weights, kernels))  # Q(s,a)=θ_0*1+θ_1*ϕ1(s,a)+⋯+θ_n*ϕn(s,a)=θ^Tϕ(s,a)


'''Radial Basic Function(RBF)'''


def _basisfunc(beta, c, d):
    return exp(-beta * norm(c - d) ** 2)  # ϕ(c,d)=e^[-(beta*(c-d))^2]


'''Calculate Activations of RBFs'''


def _calcAct(X, centers, numCenters):
    G = zeros((X.shape[0], numCenters), float)
    for ci, c in enumerate(centers):
        for xi, x in enumerate(X):
            G[xi, ci] = _basisfunc(12, c, x)
    return G / sum(G)  # normalizing Neuron activations


centers = [1,2,3,4,5,6]
a = [-1, 1]

weightsL = [0, 0, 0, 0, 0, 0]
weightsL = np.asarray(weightsL)  # weightsL^T
weightsR = [0, 0, 0, 0, 0, 0]
weightsR = np.asarray(weightsR)

Lrate = 0.5
epsilon = 0.5
gamma = 0.5
Action = 1

for i in range(10000):
    s1 = random.uniform(1.5, 5.5)
    while s1 > 1.5 and s1 < 5.5:
        G1 = _calcAct(np.asarray([s1]), centers, 6)  # calculate the RBF of sl(=origin) to each center

        # calculate Q for left and right action
        QL = Q(weightsL, G1)
        QR = Q(weightsR, G1)

        # define action
        if np.random.uniform() <= epsilon:
            action = int(np.random.choice(a, 1))  # randomly choose left or right
        else:
            if QL > QR:
                action = -1
            else:
                action = 1
        s2 = s1 + action + np.random.normal(0, 0.01)  # update state

        if s2 < 1.5:
            R = 1
        elif s2 > 5.5:
            R = 5
        else:
            R = 0


        G2 = _calcAct(np.asarray([s2]), centers, 6)
        QL_new = Q(weightsL, G2)
        QR_new = Q(weightsR, G2)
        # define action
        if np.random.uniform() <= epsilon:
            Action = int(np.random.choice(a, 1))  # randomly choose left or right
        else:
            if QL_new > QR_new:
                Action = -1
            else:
                Action = 1

        # update weights
        if (action == -1 and Action == -1):
            weightsL = weightsL + Lrate * (R + gamma * QL_new - QL) * G1
        if (action == -1 and Action == 1):
            weightsL = weightsL + Lrate * (R + gamma * QR_new - QL) * G1
        if (action == 1 and Action == -1):
            weightsR = weightsR + Lrate * (R + gamma * QL_new - QR) * G1
        if (action == 1 and Action == 1):
            weightsR = weightsR + Lrate * (R + gamma * QR_new - QR) * G1
        s1 = s2

x = np.arange(1.0, 6.0, 0.005)
yL = []
yR = []
for index, i in enumerate(x):
    yL.append(Q(weightsL, _calcAct(np.asarray([i]), centers, 6)))
    yR.append(Q(weightsR, _calcAct(np.asarray([i]), centers, 6)))

line_1, = plt.plot(x, yL, 'y-')
line_2, = plt.plot(x, yR, 'g-')
plt.xlabel('State')
plt.ylabel('Q Value')
plt.legend([line_1, line_2],
           ['Q-left', 'Q-right'])
plt.show()
