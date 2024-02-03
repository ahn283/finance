# environment.py

import numpy as np


class Env:
    def __init__(self):
        '''
        state_space: 4x4 grid info using numpy
        value ot the agent location: 1
        value of the workplace location: 2
        value of the home location: 3
        value of the park: -1

        action_space: {0, 1, 2, 3}
        0: up
        1: right
        2: down
        3: left
        '''
        self.agent_pos = {'y': 0, 'x': 0}
        self.workplace_pos = {'y': 0, 'x': 3}
        self.home_pos = {'y': 3, 'x': 3}
        self.park_area = {'y':[1, 2], 'x':[1, 2]}
        self.y_min, self.x_min, self.y_max, self.x_max = 0, 0, 3, 3

        
        # set up state
        self.state = self.set_state(self.agent_pos['y'], self.agent_pos['x'])

        self.state_space = list()
        for y in range(4):
            for x in range(4):
                state = self.set_state(y, x)
                self.state_space.append(state)

        self.action_space = [0, 1, 2, 3]

    def set_state(self, y_agent, x_agent):
        state = np.zeros([4,4])
        state[self.workplace_pos['y'], self.workplace_pos['x']] = 2
        state[self.home_pos['y'], self.home_pos['x']] = 3
        state[self.park_area['y'], self.park_area['x']] = -1
        state[y_agent, x_agent] = 1
        return state

    def reset(self):
        self.agent_pos = {'y': 0, 'x': 0}
        self.state = self.set_state(self.agent_pos['y'], self.agent_pos['x'])
        
        return self.state

    def exploring_reset(self):
        while True:
            y_random = np.random.randint(4)
            x_random = np.random.randint(4)
            self.agent_pos['y'] = y_random
            self.agent_pos['x'] = x_random
            self.state = self.set_state(self.agent_pos['y'], self.agent_pos['x'])
            if not (
                    self.agent_pos == self.home_pos or
                    self.agent_pos == self.workplace_pos
                ):
                break
        return self.state

    def step(self, action):
        # Update environmental variables 
        is_random_action = np.random.choice([0, 1], p=[0.7, 0.3])

        if is_random_action:
            random_action_set = list(self.action_space)
            random_action_set.remove(action)
            action = np.random.choice(random_action_set)

        if action == 0:
            # 'y' should be decreased by 1 or stay the same when it is at the top row
            self.agent_pos['y'] = max(
                self.agent_pos['y'] - 1, 
                self.y_min
            )
        elif action == 1:
            # 'x' should be increased by 1 or stay the same when it is at the most right column
            self.agent_pos['x'] = min(
                self.agent_pos['x'] + 1, 
                self.x_max
            )
        elif action == 2:
            # 'y' should be increased by 1 or stay the same when it is at the bottom row
            self.agent_pos['y'] = min(
                self.agent_pos['y'] + 1, 
                self.y_max
            )
        elif action == 3:
            # 'x' should be decreased by 1 or stay the same when it is at the most left column
            self.agent_pos['x'] = max(
                self.agent_pos['x'] - 1, 
                self.x_min
            )
        else:
            assert False, "Invalid action value was fed to step."

        # Make a next state after transition
        prev_state = self.state
        self.state = self.set_state(self.agent_pos['y'], self.agent_pos['x'])

        done = False
        if (
                self.agent_pos == self.workplace_pos or
                self.agent_pos == self.home_pos
            ):
            done = True

        reward = self.reward(prev_state, action, self.state)

        return reward, self.state, done

    def reward(self, s, a, s_next):
        reward = -0.5
        y, x = np.where(s == 1)
        y_next, x_next = np.where(s_next == 1)

        was_at_workplace = (
            y == self.workplace_pos['y'] and 
            x == self.workplace_pos['x']
        )
        is_at_workplace = (
            y_next == self.workplace_pos['y'] and 
            x_next == self.workplace_pos['x']
        )

        was_at_home = (
            y == self.home_pos['y'] and 
            x == self.home_pos['x']
        )
        is_at_home = (
            y_next == self.home_pos['y'] and 
            x_next == self.home_pos['x']
        )

        is_in_park = (
            y_next in self.park_area['y'] and 
            x_next in self.park_area['x']
        )
        
        if was_at_workplace and is_at_workplace:
            reward = 0 
        elif (
                not was_at_workplace and
                is_at_workplace
            ):  # Reached the workplace
            reward = 5
        
        if was_at_home and is_at_home:
            reward = 0 
        if (
                not was_at_home and
                is_at_home
            ): # Reached the home
            reward = 10
        
        if is_in_park:
            reward = -1.0
            
        return reward

    def is_correct_deterministic_transition(self, s, a, s_next):
        y, x = np.where(s == 1)  # get agent pos from s
        y_next, x_next = np.where(s_next == 1)  # get agent pos from next_s
        
        was_at_workplace = (
            y == self.workplace_pos['y'] and 
            x == self.workplace_pos['x']
        )

        was_at_home = (
            y == self.home_pos['y'] and 
            x == self.home_pos['x']
        )

        # Already reached goal
        if was_at_workplace:
            y_next_temp, x_next_temp = self.workplace_pos['y'], self.workplace_pos['x']
        elif was_at_home:
            y_next_temp, x_next_temp = self.home_pos['y'], self.home_pos['x']
        # upward movement
        elif a == 0: 
            y_next_temp, x_next_temp = max(y - 1, self.y_min), x
        # right movement
        elif a == 1: 
            y_next_temp, x_next_temp = y, min(x + 1, self.x_max)
        # downward movement
        elif a == 2: 
            y_next_temp, x_next_temp = min(y + 1, self.y_max), x
        # left movement
        elif a == 3: 
            y_next_temp, x_next_temp = y, max(x - 1, self.x_min)
        else:
            assert False, "Invalid action value was fed to step."

        is_correct_transition = (
            y_next_temp == y_next and 
            x_next_temp == x_next
        )

        if is_correct_transition:
            return 1.0
        else:
            return 0.0

    def transition_probability(self, s, a, s_next):
        p = 0
        if self.is_correct_deterministic_transition(s, a, s_next):
            p += 0.7
        
        random_action_set = list(self.action_space)
        random_action_set.remove(a)

        for action in random_action_set:
            if self.is_correct_deterministic_transition(s, action, s_next):
                p += 0.1
        return p



if __name__ == "__main__":
    import random
    from IPython import embed

    env = Env()
    s = env.reset()
    transition_list = list()

    for i in range(10000):
        a = np.random.randint(len(env.action_space))
        r, s_next, done = env.step(a)
        transition = (s, a, s_next)
        transition_list.append(transition)
        
        s = s_next
        if done:
            s = env.reset()

    for transition in transition_list:
        s, a, s_next = transition
        check1 = env.transition_probability(s, a, s_next) == 1.0
        

        s_next_ = s_next
        while (s_next_ == s_next).all():
            s_next_ = random.sample(env.state_space, 1)[0]
        check2 = env.transition_probability(s, a, s_next_) == 0.0
        
        check = check1 and check2 == True

        if not check:
            embed()
            exit()










