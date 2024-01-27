# td_prediction.py

import numpy as np
from environment import Env 

gamma = 0.9
alpha = 5e-3

def get_state_index(state_space, state):
    for i_s, s, in enumerate(state_space):
        if (s == state).all():
            return i_s
    assert False, "Couldn't find the state from the state space"
    
def td_value_prediction(env, policy):
    value_vector = np.zeros([len(env.state_space)])
    
    # repeat policy evaluation
    for loop_count in range(10000):
        done = False
        step_count = 0
        s = env.reset()
        # generate an episode
        while not done:
            i_s = get_state_index(env.state_space, s)
            pi_s = policy[i_s]
            a = np.random.choice(env.action_space, p=pi_s)
            r, s_next, done = env.step(a)
            
            i_s_next = get_state_index(env.state_space, s_next)
            td = r + gamma * value_vector[i_s_next] - value_vector[i_s]
            value_vector[i_s] = value_vector[i_s] + alpha * td
            
            if done:
                value_vector[i_s_next] = 0
            
            step_count += 1
            s = s_next
        
        if (loop_count + 1) % 100 == 0:
            print(f"[{loop_count}] value_vector: \n{value_vector}")
            
    return value_vector

def td_action_value_prediction(env, policy):
    action_value_matrix = np.zeros([len(env.state_space), len(env.action_space)])
    
    # repeat policy evaluation
    for loop_count in range(10000):
        done = False
        step_count = 0
        s = env.reset()
        
        i_s = get_state_index(env.state_space, s)
        pi_s = policy[i_s]
        a = np.random.choice(env.action_space, p=pi_s)
        
        # generate an episode
        while not done:
            r, s_next, done = env.step(a)
            i_s_next = get_state_index(env.state_space, s_next)
            pi_s_next = policy[i_s_next]
            a_next = np.random.choice(env.action_space, p=pi_s_next)
            
            td = r + gamma * action_value_matrix[i_s_next][a_next] - action_value_matrix[i_s][a]
            action_value_matrix[i_s][a] = action_value_matrix[i_s][a] + 5 * alpha * td      # for acceleration we multiply 5 to the alpha
            
            if done:
                action_value_matrix[i_s_next] = 0
            
            step_count += 1
            s = s_next
            i_s = i_s_next
            a = a_next
        
        if (loop_count + 1) % 100 == 0:
            print(f"[{loop_count}] action_value_matrix: \n{action_value_matrix}")
            
    return action_value_matrix

if __name__ == "__main__":
    np.set_printoptions(formatter={'flaot': '{:0.3f}'.format})
    
    env = Env()
    
    # policy 1 : random selection
    policy1 = list()
    for i_s, s in enumerate(env.state_space):
        pi = np.array([0.25, 0.25, 0.25, 0.25])
        policy1.append(pi)
    policy1 = np.array(policy1)     # (|S|, |A|)
    
    value_vector1 = td_value_prediction(env, policy1)
    value_table1 = value_vector1.reshape(4, 4)
    
    action_value_matrix1 = td_action_value_prediction(env, policy1)    # (|S|, |A|)
    value_vector_temp1 = np.sum(policy1 * action_value_matrix1, axis=-1)        # matrix multiplication with policy and action value matrix. it is for comparing
    
    # policy 2 : right and down are preferrable
    policy2 = list()
    for i_s, s in enumerate(env.state_space):
        pi = np.array([0.1, 0.4, 0.4, 0.1])
        policy2.append(pi)
    policy2 = np.array(policy2)
    
    value_vector2 = td_value_prediction(env, policy2)
    value_table2 = value_vector2.reshape(4, 4)
    
    action_value_matrix2 = td_action_value_prediction(env, policy2)     # (|S|, |A|)
    value_vector_temp2 = np.sum(policy2 * action_value_matrix2, axis=-1)
    
    print(f"value_table1: \n{value_table1}")
    print(f"action_value_matrix1: \n{action_value_matrix1}")
    print(f"value_vector1: \n{value_vector1}")
    print(f"value_vector_temp1: \n{value_vector_temp1}")
    
    print(f"value_table2: \n{value_table2}")
    print(f"action_value_matrix2: \n{action_value_matrix2}")
    print(f"value_vector2: \n{value_vector2}")
    print(f"value_vector_temp2: \n{value_vector_temp2}")