import numpy as np
import pylab as plt
import random




MATRIX_SIZE = 6 #equal to the number of states


#learning parameters:
gamma = 0.5
#epsilon = 0.9 #probability / epsilon greedy
alpha = 0.01


def _reward(next_state):
    if next_state == 0:
        R = 1
    elif next_state == 5:
        R = 5
    else:
        R = 0
    return R
    

action_space = [-1,1]
from numpy import linalg as LA
iteration_number = 3000
#alpha_vec = [0.01, 0.02, 0.1]
epsilon_vec = [0.1, 0.2, 1.0]

Q_true = np.matrix([[0,0],[0.8235,0.3679],[0.3929,0.6981],[0.4987,1.6955],[1.2111,4.1176],[0,0]])

DifferenceTotal = np.zeros(shape=(5,iteration_number))
for index,i in enumerate(epsilon_vec):

    Q = np.matrix(np.zeros([MATRIX_SIZE, 2]))
    Difference = [] #to store the norm-2 difference values over the number of iteration
    for i in range(iteration_number):
        current_state = np.random.randint(1, int(Q.shape[0])-1) #cut out the state 0 and state 5
        while current_state > 0 and current_state < 5:
            #select action first
            if random.random()<epsilon_vec[index]:
                action = int(np.random.choice(action_space,1))
                next_state = current_state + action
                R = _reward(next_state)         
            else: #act greedy
                if Q[current_state,0] > Q[current_state,1]:
                    action = -1
                    next_state = current_state + action
                else:
                    action = 1
                    next_state = current_state + action
                R = _reward(next_state)
            #implement the probability of taking the selected action
            if random.random()<0.3:
                next_state = current_state
                R = 0
                if action == -1:
                    Q[current_state,0] = (1-alpha) * Q[current_state,0] + alpha * (R + gamma*max(Q[next_state,0],Q[next_state,1]))
                else:
                    Q[current_state,1] = (1-alpha) * Q[current_state,1] + alpha * (R + gamma*max(Q[next_state,0],Q[next_state,1]))
            else:
                if action == -1:
                    Q[current_state,0] = (1-alpha) * Q[current_state,0] + alpha * (R + gamma*max(Q[next_state,0],Q[next_state,1]))
                else:
                    Q[current_state,1] = (1-alpha) * Q[current_state,1] + alpha * (R + gamma*max(Q[next_state,0],Q[next_state,1]))
            current_state = next_state

        Q_difference = Q - Q_true
        Difference.append(LA.norm(Q_difference))

    DifferenceTotal[index][0:iteration_number] = Difference



plt.figure()
line_1,=plt.plot(np.arange(iteration_number), DifferenceTotal[0][0:iteration_number], label='alpha = 0.01')
line_2,=plt.plot(np.arange(iteration_number), DifferenceTotal[1][0:iteration_number], label='alpha = 0.02')
line_3,=plt.plot(np.arange(iteration_number), DifferenceTotal[2][0:iteration_number], label='alpha = 0.1')
#line_6,=plt.plot(np.arange(iteration_number), DifferenceTotal[5][0:iteration_number], label='alpha = 0.01')
plt.legend([line_1, line_2, line_3],['epsilon = 0.1', 'epsilon = 0.2', 'epsilon = 1.0'])
plt.show()

