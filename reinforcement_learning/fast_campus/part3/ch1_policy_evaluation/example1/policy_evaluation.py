import numpy as np
from environment import Env

def policy_evaluation(env, policy):
    
    gamma = 0.9
    delta = 1e-3
    Delta = delta
    # create 0 value vector
    value_vector = np.zeros([len(env.state_space)])
    
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
    
        print(f"[{loop_count}] Delta: {delta}")
    return value_vector

if __name__ == "__main__":
    env = Env()
    
    policy1 = list()
    for s in env.state_space:
        _policy = np.array([0.25, 0.25, 0.25, 0.25])    # up, right, down, left
        policy1.append(_policy)
    
    value_vector1 = policy_evaluation(env, policy1)     # [16]
    table = value_vector1.reshape(4, 4)     # [4, 4]
    print(f"value_vector1: \n{value_vector1}")
    print(f"table: \n{table}")
    
    policy2 = list()
    for s in env.state_space:
        _policy = np.array([0.1, 0.4, 0.4, 0.1])        # up, right, down, left
        policy2.append(_policy)
    
    value_vector2 = policy_evaluation(env, policy2)
    table = value_vector2.reshape(4, 4)
    print(f"value_vector2: \n{value_vector2}")
    print(f"table: \n{table}")
