import numpy as np
import pylab as plt
import random


MATRIX_SIZE = 8 #equal to the number of states

action_nodes = np.matrix(np.ones(shape=(4, MATRIX_SIZE)))
action_nodes *= -1
action_nodes[:,0] = np.array([-1, 1, -1, -1]).reshape(4,1)
action_nodes[:,1] = np.array([0, 3, -1, 2]).reshape(4,1)
action_nodes[:,2] = np.array([-1, 4, 1, 7]).reshape(4,1)
action_nodes[:,3] = np.array([1, -1, 5, 4]).reshape(4,1)
action_nodes[:,4] = np.array([2, -1, 3, -1]).reshape(4,1)
action_nodes[:,5] = np.array([6, -1, -1, 3]).reshape(4,1)
action_nodes[:,6] = np.array([-1, 5, -1, -1]).reshape(4,1)
action_nodes[:,7] = np.array([7, 7, 2, 7]).reshape(4,1)




#create matrix R
R = np.matrix(np.ones(shape=(4, MATRIX_SIZE)))
R *= -1

R[:,0] = np.array([-1, 0, -1, -1]).reshape(4,1)
R[:,1] = np.array([0, 0, -1, 0]).reshape(4,1)
R[:,2] = np.array([-1, 0, 0, 100]).reshape(4,1)
R[:,3] = np.array([0, -1, 0, 0]).reshape(4,1)
R[:,4] = np.array([0, -1, 0, -1]).reshape(4,1)
R[:,5] = np.array([0, -1, -1, 0]).reshape(4,1)
R[:,6] = np.array([-1, 0, -1, -1]).reshape(4,1)
R[:,7] = np.array([100, 100, 0, 100]).reshape(4,1)

#initialize matrix Q
Q = np.matrix(np.zeros(shape=(4, MATRIX_SIZE)))
        
#learning parameters:
gamma = 0.5
epsilon = 0.9 #probability / epsilon greedy
alpha = 0.8


#available actions for current state
def available_actions(state):
    
    current_state_col = R[:,state]
    av_act = np.where(current_state_col != -1)[0]   #get the index, i.e., the actions
    return av_act

#randomly choose the next action a:
def sample_next_action(state, available_actions_range, epsilon=0.9):
    if  np.random.uniform() < epsilon or len(available_actions_range) < 2:
        next_action = int(np.random.choice(available_actions_range,1)) #randomly select next action
    else:      #action greedy
        max_value = 0
        max_ind = 0
        for i in available_actions_range:
            if Q[i, state] > max_value:
                max_value = Q[i, state]
                max_ind = i
            
        next_action = max_ind

    return next_action

def get_max_index(max_index):
    if max_index.shape[0] > 1:
      max_index = int(np.random.choice(max_index, size = 1))
    else:
      max_index = int(max_index)

    return max_index
    

iteration_number = 200
for i in range(iteration_number):
        current_state = int(np.random.randint(7)) #cut out the state 0 and state 5
        while current_state < 7:
            available_act = available_actions(current_state)
            action = sample_next_action(current_state, available_act, epsilon)
            next_state = int(action_nodes[action, current_state])

            max_Q_value = np.max(Q[:,next_state])

            Q[action, current_state] = (1-alpha) * Q[action, current_state] + alpha * (R[action, current_state] + gamma * max_Q_value)
            #Q[action, current_state] = R[action, current_state] + alpha * max_Q_value
            current_state = next_state

print(Q)

