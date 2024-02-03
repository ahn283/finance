# td_lambda.py

import numpy as np
import matplotlib.pyplot as plt
from environment import Env

gamma = 0.9
alpha_init =2e-1
k_alpha = 2.5e-1

def get_state_index(state_space, state):
    for i_s, s in enumerate(state_space):
        if (s == state).all():
            return i_s
    assert False, "Couldn't find the state from eht state space"
    
def calc_return(gamma, rewards):
    n = len(rewards)
    rewards = np.array(rewards)
    gammas = gamma * np.ones([n])
    powers = np.arange(n)
    
    power_of_gammas = np.power(gammas, powers)
    discounted_rewards = rewards * power_of_gammas  # [r, gamma * r, gamma^2 * r, ...]
    g = np.sum(discounted_rewards)
    
    return g


def td_lambda(env, policy, lam, color):
    value_vector = np.zeros([len(env.state_space)])
    plot_buffer = {'x': [], 'y': []}
    
    # repeat policy evaluation
    for loop_count in range(1500):
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
        
        # calculate TD-lambda and update value function
        for t in range(step_count):
            G_t_lambda = 0
            n = step_count - t
            
            # calculate TD(lambda)
            for k in range(n):
                # calculate G_t^(k + 1)
                s_td = episode['states'][t + k + 1]
                i_s_td = get_state_index(env.state_space, s_td)
                
                discounted_rewards = calc_return(gamma, episode['rewards'][t:t + k + 1])
                G_t = discounted_rewards + (gamma ** (k + 1)) * value_vector[i_s_td]
                if k == n - 1:
                    # adjusting actual weight to the last one
                    G_t = G_t / (1 - lam)
                G_t_lambda = G_t_lambda + (lam ** k) * G_t
            G_t_lambda = (1 - lam) * G_t_lambda
            
            s_t = episode['states'][t]
            i_s_t = get_state_index(env.state_space, s_t)
            
            alpha = alpha_init / (1 + k_alpha * loop_count)
            td_lambda = G_t_lambda - value_vector[i_s_t]
            value_vector[i_s_t] = value_vector[i_s_t] + alpha * td_lambda
            
        s_T = episode['states'][-1]
        i_s_T = get_state_index(env.state_space, s_T)
        
        value_vector[i_s_T] = 0
        
        if (loop_count + 1) % 10 == 0:
            print(
                f"[{loop_count}] value_vector: \n{value_vector}"
                + f"\nalpha: {alpha:.4f}"
            )
        
        # add new point
        plot_buffer['x'].append(loop_count)
        plot_buffer['y'].append(value_vector[0])
        if loop_count > 0:
            # draw a new linw
            plt.plot(plot_buffer['x'], plot_buffer['y'], color=color)
            # remove drawed point
            plot_buffer['x'].pop(0)
            plot_buffer['y'].pop(0)
    
    # add legend for plotting
    plt.plot(0, 0, color=color, label=f"lambda={lam}")
    
    return value_vector

if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{:0.3f}'.format})
    
    env = Env()
    policy = list()
    for i_s, s in enumerate(env.state_space):
        pi = np.array([0.25, 0.25, 0.25, 0.25])
        policy.append(pi)
    policy = np.array(policy)
    
    value_vector1 = td_lambda(env, policy, 0.3, color='red')
    value_vector2 = td_lambda(env, policy, 0.6, color='green')
    value_vector3 = td_lambda(env, policy, 0.9, color='blue')
    
    print(f"lam: {0.3}, value_vector: \n{value_vector1}")
    print(f"lam: {0.6}, value_vector: \n{value_vector2}")
    print(f"lam: {0.9}, value_vector: \n{value_vector3}")
    
    plt.legend()
    plt.show()
    