# value_iteration.py

import numpy as np
from environment import Env

def value_iteration(env):
    
    gamma = 0.9
    delta = 1e-3
    Delta = delta
    value_vector = np.zeros([len(env.state_space)])
    
    loop_count = 0
    while Delta >= delta:
        Delta = 0
        new_value_vector = np.zeros([len(env.state_space)])
        for i_s, s in enumerate(env.state_space):
            # sum over action
            action_values = np.zeros([len(env.action_space)])
            for a in env.action_space:
                # sum over next state
                action_value = 0
                for i_s_next, s_next in enumerate(env.state_space):
                    p_s_next = env.transition_probability(s, a, s_next)
                    reward = env.reward(s, a, s_next)
                    action_value = action_value + p_s_next * (reward + gamma * value_vector[i_s_next])
                action_values[a] = action_value
            
            new_value_vector[i_s] = np.max(action_values)
            value_delta = abs(new_value_vector[i_s] - value_vector[i_s])
            
            Delta = max(Delta, value_delta)
        value_vector = new_value_vector
        loop_count += 1
        
        print(f"[{loop_count}] Delta: {Delta}")
        print(value_vector)
        
    policy = list()
    for i_s, s in enumerate(env.state_space):
        # sum over action
        action_values = np.zeros([len(env.action_space)])
        for a in env.action_space:
            # sum over next state
            for i_s_next, s_next in enumerate(env.state_space):
                p_s_next = env.transition_probability(s, a, s_next)
                reward = env.reward(s, a, s_next)
                action_value = action_value + p_s_next * (reward + gamma * value_vector[i_s_next])
            action_values[a] = action_value
            
        # get an argument which maximizes the action value
        a_max = np.argmax(action_values)
        
        policy_s = np.zeros([len(env.action_space)])
        # maximized action for a specific state (policy) is set to 1
        policy_s[a_max] = 1    
        policy.append(policy_s)           
            
            
    return value_vector, policy

if __name__ == "__main__":
    env = Env()
    value_vector, policy = value_iteration(env)
    value_table = np.zeros([3, 4])
    value_table[0, 0] = value_vector[0]
    value_table[0, 3] = value_vector[1]
    value_table[1, :] = value_vector[2:6]
    value_table[2, :] = value_vector[6:10]
    print(f"value_table: \n{value_table}")