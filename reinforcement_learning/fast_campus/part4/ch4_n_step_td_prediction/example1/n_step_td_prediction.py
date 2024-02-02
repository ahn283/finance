# n_step_td_prediction.py

import numpy as np 
import matplotlib.pyplot as plt
from environment import Env 

gamma = 0.9
alpha_init = 2e-1
k_alpha = 2.5e-1

def get_state_index(state_space, state):
    for i_s, s in enumerate(state_space):
        if (s == state).all():
            return i_s
    assert False, "Couldn't find the state from the state space"
    
def calc_return(gamma, rewards):
    n = len(rewards)
    rewards = np.array(rewards)
    gammas = gamma * np.ones([n])
    powers = np.arange(n)
    
    power_of_gammas = np.power(gammas, powers)
    discounted_rewards = rewards * power_of_gammas  # [r, gamma * r, gamma^2 * r, ...]
    g = np.sum(discounted_rewards)
    
    return g

def n_step_td_value_prediction(env, policy, n, color):
    value_vector = np.zeros([len(env.state_space)])
    plot_buffer = {'x': [], 'y': []}    # buffer for plotting graphs
    
    # repeat policy evaluation
    for loop_count in range(1000):
        trajectory = {
            'states': list(),
            'actions': list(),
            'rewards': list()
        }
        done = False
        step_count = 0
        s = env.reset()
        trajectory['states'].append(s)
        # generate a trajectory
        while not done:
            i_s = get_state_index(env.state_space, s)
            pi_s = policy[i_s]
            a = np.random.choice(env.action_space, p=pi_s)
            r, s_next, done = env.step(a)
            
            trajectory['states'].append(s_next)
            trajectory['actions'].append(a)
            trajectory['rewards'].append(r)
            
            s = s_next
            step_count += 1
            
            if step_count >= n + 1:
                # remove updated transition
                trajectory['states'].pop(0)
                trajectory['actions'].pop(0)
                trajectory['rewards'].pop(0)
                
            if step_count >= n:
                assert len(trajectory['rewards']) == n, f"Trajectory length should be n={n} but {len(trajectory['rewards'])}"
                s_t_sub_n = trajectory['states'][0]
                i_s_t_sub_n = get_state_index(env.state_space, s_t_sub_n)
                s_t = trajectory['states'][-1]
                i_s_t = get_state_index(env.state_space, s_t)
                
                alpha = alpha_init / (1 + k_alpha * loop_count)
                discounted_rewards = calc_return(gamma, trajectory['rewards'])
                td = discounted_rewards + (gamma ** n) * value_vector[i_s_t] - value_vector[i_s_t_sub_n]
                value_vector[i_s_t_sub_n] = value_vector[i_s_t_sub_n] + alpha * td
                
            if done:
                k_min = min(step_count, n - 1)
                s_t = trajectory['states'][-1]
                i_s_t = get_state_index(env.state_space, s_t)
                
                alpha = alpha_init / (1 + k_alpha * loop_count)
                for i in range(1, k_min + 1):
                    s_t_sub_i = trajectory['states'][-i-1]
                    i_s_t_sub_i = get_state_index(env.state_space, s_t_sub_i)
                    
                    discounted_rewards = calc_return(gamma, trajectory['rewards'][-i:])
                    td = discounted_rewards + (gamma ** k_min) * value_vector[i_s_t] - value_vector[i_s_t_sub_i]
                    value_vector[i_s_t_sub_i] = value_vector[i_s_t_sub_i] + alpha * td
                value_vector[i_s_t] = 0     # set V(s_T) = 0 for the terminal state
        
        if (loop_count + 1) % 100 == 0:
            print(
                f"[{loop_count}] value_vector: \n{value_vector}"
                + f"[{loop_count}] value_vector: \n{value_vector}"
                + f"\nalpha: {alpha:.4f}"
            )
        
        # add new point
        plot_buffer['x'].append(loop_count)
        plot_buffer['y'].append(value_vector[0])
        if loop_count > 0:
            # draw a new line
            plt.plot(plot_buffer['x'], plot_buffer['y'], color=color)
            # remove drawed point
            plot_buffer['x'].pop(0)
            plot_buffer['y'].pop(0)
    
    # add legend for plotting
    plt.plot(0, 0, color=color, label=f"n={n}")
    
    return value_vector

if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{:0.3f}'.format})
    
    env = Env()
    policy = list()
    for i_s, s in enumerate(env.state_space):
        pi = np.array([0.25, 0.25, 0.25, 0.25])
        policy.append(pi)
    policy = np.array(policy)
    
    value_vector_1 = n_step_td_value_prediction(env, policy, 1, color='red')
    value_vector_3 = n_step_td_value_prediction(env, policy, 3, color='green')
    value_vector_5 = n_step_td_value_prediction(env, policy, 5, color='blue')
    
    print(f"n: {1}, value_vector_1: \n{value_vector_1}")
    print(f"n: {3}, value_vector_3: \n{value_vector_3}")
    print(f"n: {5}, value_vector: \n{value_vector_5}")
    
    plt.legend()
    plt.show()