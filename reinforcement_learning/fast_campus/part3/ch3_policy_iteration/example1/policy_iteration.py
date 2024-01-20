# policy_iteration.py

import numpy as np
from environment import Env

gamma = 0.9
delta = 1e-3

def policy_evaluation(env, value_vector, policy):
    Delta = delta
    
    loop_count = 0
    while Delta >= delta:
        Delta = 0
        new_value_vector = np.zeros([len(env.state_space)])
        for i_s, s in enumerate(env.state_space):
            v_s = 0
            # sum over action
            for a in env.action_space:
                # sum over next state
                for i_s_next, s_next in enumerate(env.state_space):
                    pi_a = policy[i_s][a]
                    p_s_next = env.transition_probability(s, a, s_next)
                    reward = env.reward(s, a, s_next)
                    
                    v_s = v_s + p_s_next * pi_a * (reward + gamma * value_vector[i_s_next])
            new_value_vector[i_s] = v_s
            value_delta = abs(new_value_vector[i_s] - value_vector[i_s])
            
            Delta = max(Delta, value_delta)
        value_vector = new_value_vector
        loop_count += 1
    
    print(f"value_vector: \n{value_vector}")
    print(f"value_table: \n{value_vector.reshape(4, 4)}")
    
    return value_vector

def policy_improvement(env, value_vector, policy):
    for i_s, s in enumerate(env.state_space):
        action_values = np.zeros(len(env.action_space))
        for a in env.action_space:
            action_value = 0
            for i_s_next, s_next in enumerate(env.state_space):
                p_s_next = env.transition_probability(s, a, s_next)
                action_value += p_s_next * (env.reward(s, a, s_next) +gamma * value_vector[i_s_next])
            action_values[a] = action_value
        a_max = action_values.argmax()
        policy[i_s][:] = 0
        policy[i_s][a_max] = 1
        
if __name__ == "__main__":
    env = Env()
    
    # initialized value function and policy
    value_vector = np.zeros([len(env.state_space)])
    policy = list()
    for s in env.state_space:
        _policy = np.array([0.25, 0.25, 0.25, 0.25])    # up, right, down, left
        policy.append(_policy)
    
    # policy iteration
    while True:
        value_vector_new = policy_evaluation(env, value_vector, policy) 
        
        Delta = 0
        for i_s, s in enumerate(env.state_space):
            Delta = max(Delta, abs(value_vector_new[i_s] - value_vector[i_s]))
            
        if Delta >= delta:
            value_vector = value_vector_new
            policy_improvement(env, value_vector, policy)
        else:
            break   
            
                    
                    