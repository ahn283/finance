# utils.py

import gym
import numpy as np

def create_env(config):
    env = gym.make("CartPole-v1")
    env = RewardShapingWrapper(env)
    env = TimeStepAppendWrapper(env)
    
    return env


class RewardShapingWrapper(gym.Wrapper):
    # 원래 리워드에 0.1을 곱하는 wrapper
    def __init__(self, env):
        self.env = env
        
    def step(self, action):
        # 첫번째 값이 1/(1-gamma)=100이라는 큰수를 가질 수 있으므로 학습의 안정성을 위해 1/10 적용
        s_next, r, done, info = self.env.step(action)
        r = 0.1 * r
        return s_next, r, done, info
    
class TimeStepAppendWrapper(gym.Wrapper):
    # 기존 상태 return에 timestep 추가해서 5개의 값을 return해주도록 변경
    # agent의 환경을 더 잘 이해하기 위해 timestep을 추가
    def __init__(self, env):
        self.env = env
        self.step_count = 0
        self.observation_space = gym.spaces.Box(
            np.concatenate([self.env.observation_space.low, [0]], axis=-1),  # 상태 최소값 = 0
            np.concatenate([self.env.observation_space.high, [5]], axis=-1),  # 상태 최대값 = 500이지만 step_count에 0.01을 곱함
            (self.env.observation_space.shape[0] + 1, ),    # dimension 기존 4 + 1
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
    # attribute를 설정하면 dict의 key, value 값 설정
    __setattr__ = dict.__setitem__
    
    def __getattribute__(self, item):
        # attribute 가져오는 함수
        # attribute가 key(item)로 존재하면 해당 key return
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