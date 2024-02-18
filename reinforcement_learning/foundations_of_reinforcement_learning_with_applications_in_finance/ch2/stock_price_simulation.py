from dataclasses import dataclass
from typing import Optional, Mapping, Sequence, Tuple
from collections import Counter
import numpy as np
from numpy.random import binomial
import itertools
from operator import itemgetter
from utils.common_funcs import get_logistic_func, get_unit_sigmoid_func

@dataclass
class Process1:
    @dataclass
    class State:
        price: int
        
    level_param: int    # level to which price mean-reverts
    alpha1: float = 0.25 # strength of mean-reversion (non-negative value)
    
    def up_prob(self, state: State) -> float:
        return get_logistic_func(self.alpha1)(self.level_param - state.price)
    
    def next_state(self, state: State) -> State:
        up_move: int = binomial(1, self.up_prob(state), 1)[0]
        return Process1.State(price=state.price + up_move * 2 - 1)
    
handy_map : Mapping[Optional[bool], int] = {True: -1, False: 1, None: 0}



