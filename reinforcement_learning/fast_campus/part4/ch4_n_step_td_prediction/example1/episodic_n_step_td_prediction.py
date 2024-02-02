# episodic_n_step_td_prediction.py

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
    discounted_rewards = rewards * power_of_gammas      # [r, gamma * r, gamma^2 * r, ...]
    g = np.sum(discounted_rewards)
    
    return g

def episodic_n_step_td_value_prediction(env, policy, n, color):
    value_vector = np.zeros([len(env.state_space)])
    plot_buffer = {'x': [], 'y': []}
    
    # repeat policy evaluation
    for loop_count in range(2000):
        episode = {
            'states': list(),
            'actions': list(),
            'rewards': list()
        }
        done = False
        step_count = 0
        s = env.reset()
        # generate an episode
        while not done:
            i_s = get_state_index(env.state_space, s)
            pi_s = policy[i_s]
            a = np.random.choice(env.action_space, p=pi_s)
            r, s_next, done = env.step(a)
            
            episode['states'].append(s)
            episode['actions'].append(a)
            episode['rewards'].append(r)
            
            step_count += 1
            s = s_next
        episode['states'].append(s)     # append s_T (the termination state)
        
        for t in range(step_count):
            alpha = alpha_init / (1 + k_alpha * loop_count)
            k_min = min(n, step_count - t)
            
            s_t = episode['states'][t]
            i_s_t = get_state_index(env.state_space, s_t)
            s_t_k = episode['states'][t + k_min]
            i_s_t_k = get_state_index(env.state_space, s_t_k)
            discounted_rewards = calc_return(gamma, episode['rewards'][t:t + k_min])
            td = discounted_rewards + (gamma ** k_min) * value_vector[i_s_t_k] - value_vector[i_s_t]
            value_vector[i_s_t] = value_vector[i_s_t] + alpha * td
            
        if (loop_count + 1) % 100 == 0:
            print(
                f"[{loop_count}] value_vector: \n{value_vector}"
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
    
    value_vector_1 = episodic_n_step_td_value_prediction(env, policy, 1, color='red')
    value_vector_3 = episodic_n_step_td_value_prediction(env, policy, 3, color='green')
    value_vector_5 = episodic_n_step_td_value_prediction(env, policy, 5, color='blue')
    
    print(f"n: {1}, value_vector: \n{value_vector_1}")
    print(f"n: {3}, value_vector: \n{value_vector_3}")
    print(f"n: {5}, value_vector: \n{value_vector_5}")
    
    plt.legend()
    plt.show()