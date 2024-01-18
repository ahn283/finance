# environment.py

import numpy as np

class Env:
    def __init__(self):
        '''
        state space : 4x4 grid info using numpy
        value of the agent location : 1
        value of the goal location : -1
        
        action_space : {0, 1, 2, 3}
        0: up
        1: right
        2: down
        3: left
        '''
        self.agent_pos = {'y': 0, 'x': 0}
        self.goal_pos = {'y': 3, 'x': 3}
        self.y_min, self.x_min, self.y_max, self.x_max = 0, 0, 3, 3
        
        # set up state
        self.state = np.zeros([4, 4])
        self.state[self.goal_pos['y'], self.goal_pos['x']] = -1
        self.state[self.agent_pos['y'], self.agent_pos['y']] = 1
        
        # make state space list
        self.state_space = list()
        for y in range(4):
            for x in range(4):
                state = np.zeros([4, 4])
                state[self.goal_pos['y'], self.goal_pos['x']] = -1
                state[y, x] = 1
                self.state_space.append(state)
        print(self.state_space)
        self.action_space = [0, 1, 2, 3] 
        

if __name__ == "__main__":
    import random
    
    env = Env()
           
                