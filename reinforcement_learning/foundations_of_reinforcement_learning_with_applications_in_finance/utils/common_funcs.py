from typing import Callable
import numpy as np
VSML = 1e-8

def get_logistic_func(alpha: float) -> Callable[[float], float]:
    return lambda x: 1. / (1 + np.exp(-alpha * x))

def get_unit_sigmoid_func(alpha: float) -> Callable[[float], float]:
    return lambda x: 1. / (1+ (1 / np.where(x == 0, VSML, x) - 1) ** alpha)