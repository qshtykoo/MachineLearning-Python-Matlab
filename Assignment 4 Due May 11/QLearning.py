import numpy as np
import pylab as plt

# map cell to cell, add circular cell to goal points
points_list = [(0,1), (1,2), (2,3), (3,4), (4,5)]

goal_one = 0
goal_two = 5

import networkx as nx

G = nx.Graph()
G.add_edges_from(points_list)
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G,pos)
nx.draw_networkx_edges(G,pos)
nx.draw_networkx_labels(G,pos)
#plt.show()


MATRIX_SIZE = 6 #equal to the number of states

#create matrix R

R = np.matrix(np.ones(shape=(MATRIX_SIZE, MATRIX_SIZE)))
R *= -1

#assign 0 to existing paths and 5 or 1 to state 1 (charger) and state 6 (trash bin)

for point in points_list:
    print(point)
    if point[1] == goal_two:
        R[point] = 5
    else:
        R[point] = 0

    if point[0] == goal_one:
        # reverse of point (reverse of coordinates of point)
        R[point[::-1]] = 1
    else:
        R[point[::-1]] = 0

R[0][0] = np.ones([1,6])*(-1)
R[5][0] = np.ones([1,6])*(-1)
        

#learning parameters:
gamma = 0.5
epsilon = 0.9 #probability / epsilon greedy
#alpha = 0.8

import random

#available actions for current state
def available_actions(state):
    
    current_state_row = R[state,]
    av_act = np.where(current_state_row >= 0)[1]
    return av_act


#randomly choose the next action a:
def sample_next_action(state, available_actions_range, epsilon):
    if  np.random.uniform() < epsilon or len(available_actions_range) < 2:
        next_action = int(np.random.choice(available_actions_range,1)) #1 here indicates the size of output
    else:
        if  Q[state, available_actions_range[0]] > Q[state, available_actions_range[1]]:
            next_action = available_actions_range[0]#act greedy
        else:
            next_action = available_actions_range[1]
    return next_action


def update(current_state, action, gamma, alpha):
    
  max_index = np.where(Q[action,] == np.max(Q[action,]))[1]
  
  if max_index.shape[0] > 1:
      max_index = int(np.random.choice(max_index, size = 1))
  else:
      max_index = int(max_index)
  max_value = Q[action, max_index]
  
  Q[current_state, action] = (1-alpha) * Q[current_state, action] + alpha * (R[current_state, action] + gamma * max_value)



# Training
from numpy import linalg as LA
#iteration_number = [0, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700]
iteration_number = 350
alpha_vec = [0.1, 0.2, 0.5, 0.8, 1.0]
#epsilon_vec = [0.1, 0.2, 0.5, 0.8, 1.0]

Q_true = np.matrix([[0,0,0,0,0,0], [1.0,0,0.625,0,0,0], [0,0.5,0,1.25,0,0], [0,0,0.625,0,2.5,0], [0,0,0,1.25,0,5.0], [0,0,0,0,0,0]])

DifferenceTotal = np.zeros(shape=(5,iteration_number))
for index,i in enumerate(alpha_vec):

    Q = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE]))
    scores = []
    Difference = [] #to store the norm-2 difference values over the number of iteration
    for i in range(iteration_number):
        current_state = np.random.randint(1, int(Q.shape[0])-1) #cut out the state 0 and state 5
        while current_state > 0 and current_state < 5:
            available_act = available_actions(current_state)
            action = sample_next_action(current_state, available_act, epsilon)
            score = update(current_state, action, gamma, alpha_vec[index])
            scores.append(score)
            current_state = action
            #print ('Score:', str(score))
            
        #print("Trained Q matrix:")
        #print(Q/np.max(Q))

        Q_difference = Q - Q_true
        Difference.append(LA.norm(Q_difference))

    DifferenceTotal[index][0:iteration_number] = Difference



plt.figure()
line_1,=plt.plot(np.arange(iteration_number), DifferenceTotal[0][0:iteration_number], label='alpha = 0.1')
line_2,=plt.plot(np.arange(iteration_number), DifferenceTotal[1][0:iteration_number], label='alpha = 0.2')
line_3,=plt.plot(np.arange(iteration_number), DifferenceTotal[2][0:iteration_number], label='alpha = 0.5')
line_4,=plt.plot(np.arange(iteration_number), DifferenceTotal[3][0:iteration_number], label='alpha = 0.8')
line_5,=plt.plot(np.arange(iteration_number), DifferenceTotal[4][0:iteration_number], label='alpha = 1.0')
#line_6,=plt.plot(np.arange(iteration_number), DifferenceTotal[5][0:iteration_number], label='alpha = 0.01')
plt.legend([line_1, line_2, line_3, line_4, line_5],['alpha = 0.1', 'alpha = 0.2', 'alpha = 0.5', 'alpha = 0.8', 'alpha = 1.0'])
plt.show()

