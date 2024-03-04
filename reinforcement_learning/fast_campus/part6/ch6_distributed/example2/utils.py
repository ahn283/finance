# utils.py
import gym
import numpy as np
import multiprocessing
from multiprocessing import get_context
from multiprocessing.queues import Queue


def create_env(config):
    env = gym.make("CartPole-v1")
    env = RewardShapingWrapper(env)
    env = TimeStepAppendWrapper(env)
    
    return env


class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env):
        self.env = env

    def step(self, action):
        s_next, r, done, info = self.env.step(action)
        r = 0.01 * r
        return s_next, r, done, info


class TimeStepAppendWrapper(gym.Wrapper):
    def __init__(self, env):
        self.env = env
        self.step_count = 0
        self.observation_space = gym.spaces.Box(
            np.concatenate([self.env.observation_space.low, [0]], axis=-1),
            np.concatenate([self.env.observation_space.high, [5]], axis=-1),
            (self.env.observation_space.shape[0] + 1, ),
            np.float32
        )

    def reset(self):
        self.step_count = 0
        s = self.env.reset()
        s = np.concatenate([s, [0.01 * self.step_count]], axis=-1)
        return s

    def step(self, action):
        s_next, r, done, info = self.env.step(action)
        self.step_count += 1
        s_next = np.concatenate([s_next, [0.01 * self.step_count]], axis=-1)
        return s_next, r, done, info


class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattribute__(self, item):
        if item in self:
            return self[item]
        else:
            return super().__getattribute__(item)

    @classmethod
    def from_nested_dics(cls, data):
        if not isinstance(data, dict):
            return data
        else:
            return cls({key:cls.from_nested_dics(data[key]) for key in data})


class SharedCounter(object):
    def __init__(self, n=0):
        self.count = multiprocessing.Value('i', n)

    def increment(self, n=1):
        with self.count.get_lock():
            self.count.value += n

    @property
    def value(self):
        return self.count.value


class MPQueue(Queue):
    def __init__(self, *args, **kwargs):
        super(MPQueue, self).__init__(*args, ctx=get_context(), **kwargs)
        self.size = SharedCounter(0)

    def __getstate__(self):
        return (super(MPQueue, self).__getstate__(), self.size)

    def __setstate__(self, state):
        super(MPQueue, self).__setstate__(state[0])
        self.size = state[1]

    def put(self, *args, **kwargs):
        self.size.increment(1)
        super(MPQueue, self).put(*args, **kwargs)

    def get(self, *args, **kwargs):
        self.size.increment(-1)  # uncomment this for error
        return super(MPQueue, self).get(*args, **kwargs)

    def qsize(self):
        return self.size.value

    def empty(self):
        return not self.qsize()

    def clear(self):
        while not self.empty():
            self.get()