# mc_prediction.py

import numpy as np
from environment import Env 

gamma = 0.9

def get_state_index(state_space, state):
    