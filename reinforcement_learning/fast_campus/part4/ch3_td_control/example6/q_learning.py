# q_learning.py

import numpy as np
from environment import Env 

gamma = 0.95
k_alpha = 2e-2
k_eps = 1e-4

def get_state_index(state_space, state):
    for i_s, s in enumerate(state_space):
        if (s == state).all():
            return i_s
    assert False, "Couldn't find the state from the state space"
    
def q_learning(env):
    action_value_matrix = np.zeros([len(env.state_space), len(env.action_space)])
    
    def sample_action(eps, action_Value):
        a_max = action_Value.argmax()
        pi = np.zeros([len(env.action_space)])
        pi[:] = eps / len(env.action_space)
        pi[a_max] = pi[a_max] + 1 - eps
        a = np.random.choice(env.action_space, p=pi)
        return a
    
    def get_eps(total_step_count):
        return 1 / (1 + k_eps * total_step_count)
    
    # repeat q-learning loop
    total_step_count = 0
    for loop_count in range(10000):
        done = False
        step_count = 0
        
        s = env.reset()
        i_s = get_state_index(env.state_space, s)
        
        # generate an episode
        while not done:
            action_value = action_value_matrix[i_s]
            eps = get_eps(total_step_count)
            a = sample_action(eps, action_value)
            r, s_next, done = env.step(a)
            
            i_s_next = get_state_index(env.state_space, s_next)
            alpha = 1 / (1 + k_alpha * loop_count)
            td = r + gamma * action_value_matrix[i_s_next].max() - action_value_matrix[i_s][a] 
            action_value_matrix[i_s][a] = action_value_matrix[i_s][a] + alpha * td
            
            if done:
                action_value_matrix[i_s_next] = 0
                
            step_count += 1
            total_step_count += 1
            
            s = s_next
            i_s = i_s_next
            
        if (loop_count + 1) % 100 == 0:
            print(
                f"[{loop_count}] action_value_matrix: \n{action_value_matrix} "
                + f"eps: {get_eps(total_step_count):.4f} "
                + f"alpha: {alpha:.4f}"
            )
            
    # generate optimal policy from the action value function
    policy = np.zeros([len(env.state_space), len(env.action_space)])
    state_indexes = np.arange(len(env.state_space))
    argmax_actions = action_value_matrix.argmax(axis=-1)
    policy[state_indexes, argmax_actions] = 1.0
    
    return action_value_matrix, policy

if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{:0.3f}'.format})
    env = Env()
    action_value_matrix, policy = q_learning(env)
    
    argmax_actions = action_value_matrix.argmax(axis=-1)
    value_vector = np.sum(policy * action_value_matrix, axis=-1)
    
    value_table = value_vector.reshape(4, 4)
    argmax_actions_table = argmax_actions.reshape(4, 4)
    
    print(
        f"value_table: \n{value_table}\n"
        + f"argmax_actions: \n{argmax_actions.reshape(4, 4)}"
    )