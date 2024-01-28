# mc_control.py

import numpy as np
from environment import Env


gamma = 0.9
eps = 0.05


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
    discounted_rewards = rewards * power_of_gammas
    g = np.sum(discounted_rewards)

    return g


def mc_control_es(env, policy):
    action_value_matrix = np.zeros([len(env.state_space), len(env.action_space)])
    returns = [[{'n':0, 'avg':0} for a in env.action_space] for s in env.state_space]
    
    # Repeat policy evaluation
    for loop_count in range(10000):
        episode = {
            'states': list(),
            'actions': list(),
            'rewards': list(),
        }
        done = False
        step_count = 0
        s = env.exploring_reset()
        # Generate an episode
        while not done:
            i_s = get_state_index(env.state_space, s)
            if step_count == 0:  # Exploring start
                a = np.random.choice(env.action_space, p=[0.25, 0.25, 0.25, 0.25])
            else:  # Normal policy
                pi_s = policy[i_s]
                a = np.random.choice(env.action_space, p=pi_s)
            r, s_next, done = env.step(a)
            
            episode['states'].append(s)
            episode['actions'].append(a)
            episode['rewards'].append(r)

            step_count += 1            
            s = s_next

            is_dead_lock = False
            if step_count > 1000: 
                is_dead_lock = True
                break

        if is_dead_lock:
            continue

        episode['states'].append(s) # append s_T (the termination state)

        for t in range(step_count):
            s_t = episode['states'][t]
            a_t = episode['actions'][t]
            i_s_t = get_state_index(env.state_space, s_t)
            i_a_t = env.action_space.index(a_t)
            g_t = calc_return(gamma, episode['rewards'][t:])

            n_prev, avg_prev = returns[i_s_t][i_a_t]['n'], returns[i_s_t][i_a_t]['avg']
            returns[i_s_t][i_a_t]['avg'] = (avg_prev * n_prev + g_t) / (n_prev + 1)
            returns[i_s_t][i_a_t]['n'] = n_prev + 1
            action_value_matrix[i_s_t][i_a_t] = returns[i_s_t][i_a_t]['avg']

        for t in range(step_count):
            s_t = episode['states'][t]
            i_s_t = get_state_index(env.state_space, s_t)

            a_max = action_value_matrix[i_s_t].argmax()
            policy[i_s_t][:] = 0
            policy[i_s_t][a_max] = 1

        if (loop_count + 1) % 100 == 0:
            print(f"[{loop_count}] action_value_matrix: \n{action_value_matrix}")

    return policy, action_value_matrix


def mc_control_epsilon_soft(env, policy):
    action_value_matrix = np.zeros([len(env.state_space), len(env.action_space)])
    returns = [[{'n':0, 'avg':0} for a in env.action_space] for s in env.state_space]
    
    # Repeat policy evaluation
    for loop_count in range(20000):
        episode = {
            'states': list(),
            'actions': list(),
            'rewards': list(),
        }
        done = False
        step_count = 0
        s = env.reset()
        # Generate an episode
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

            is_dead_lock = False
            if step_count > 1000: 
                is_dead_lock = True
                break

        if is_dead_lock:
            continue

        episode['states'].append(s)  # append s_T (the termination state)

        for t in range(step_count):
            s_t = episode['states'][t]
            a_t = episode['actions'][t]
            i_s_t = get_state_index(env.state_space, s_t)
            i_a_t = env.action_space.index(a_t)
            g_t = calc_return(gamma, episode['rewards'][t:])

            n_prev, avg_prev = returns[i_s_t][i_a_t]['n'], returns[i_s_t][i_a_t]['avg']
            returns[i_s_t][i_a_t]['avg'] = (avg_prev * n_prev + g_t) / (n_prev + 1)
            returns[i_s_t][i_a_t]['n'] = n_prev + 1
            action_value_matrix[i_s_t][i_a_t] = returns[i_s_t][i_a_t]['avg']

        for t in range(step_count):
            s_t = episode['states'][t]
            i_s_t = get_state_index(env.state_space, s_t)
            
            a_max = action_value_matrix[i_s_t].argmax()
            policy[i_s_t][:] = eps / len(env.action_space)
            policy[i_s_t][a_max] += 1 - eps

        if (loop_count + 1) % 100 == 0:
            print(f"[{loop_count}]action_value_matrix: \n{action_value_matrix}")

    return policy, action_value_matrix


if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    env = Env()
    policy = list()
    for i_s, s in enumerate(env.state_space):
        pi = np.array([0.25, 0.25, 0.25, 0.25])
        policy.append(pi)
    policy = np.array(policy)  # (|S|, |A|)
    
    policy, action_value_matrix = mc_control_es(env, policy)

    policy_eps_soft = list()
    for i_s, s in enumerate(env.state_space):
        pi = np.array([0.25, 0.25, 0.25, 0.25])
        policy_eps_soft.append(pi)
    policy_eps_soft = np.array(policy_eps_soft)  # (|S|, |A|)
    policy_eps_soft, action_value_matrix_eps_soft = mc_control_epsilon_soft(env, policy_eps_soft)

    value_vector = np.sum(policy * action_value_matrix, axis=-1)  # (|S|)
    value_table = np.zeros([3, 4])
    value_table[0, 0] = value_vector[0]
    value_table[0, 3] = value_vector[1]
    value_table[1:] = value_vector[2:6]
    value_table[2:] = value_vector[6:10]
    print(f"value_table: \n{value_table}")

    value_vector_eps_soft = np.sum(policy_eps_soft * action_value_matrix_eps_soft, axis=-1)
    value_table_eps_soft = np.zeros([3, 4])
    value_table_eps_soft[0, 0] = value_vector_eps_soft[0]
    value_table_eps_soft[0, 3] = value_vector_eps_soft[1]
    value_table_eps_soft[1:] = value_vector_eps_soft[2:6]
    value_table_eps_soft[2:] = value_vector_eps_soft[6:10]
    print(f"value_table_eps_soft: \n{value_table_eps_soft}")
    
    
    
