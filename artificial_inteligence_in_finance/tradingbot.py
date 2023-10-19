#
# Financial Q-Learning Agent
#

import os
import random
import logging
import numpy as np
from collections import deque
import tensorflow as tf 
from tensorflow import keras

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

from keras.layers import Dense, Dropout
from keras.models import Sequential

def set_seeds(seed=100):
    '''
    Function to set seeds for all
    random nuber generator
    '''
    
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
class TradingBot:
    def __init__(self, hidden_units, learning_rate, learn_env,
                 valid_env=None, val=True, dropout=False):
        self.learn_env = learn_env
        self.valid_env = valid_env
        self.val = val
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99
        self.learning_rate = learning_rate
        self.gamma = 0.5
        self.batch_size = 128
        self.max_treward = 0
        self.averages = list()
        self.trewards = []
        self.performances = list()
        self.aperformances = list()
        self.vperformances = list()
        self.memory = deque(maxlen=2000)
        self.model = self._build_model(hidden_units, learning_rate, dropout)
        
    def _build_model(self, hu, lr, dropout):
        '''
        Method to create DNN model.
        '''
        
        model = Sequential()
        model.add(Dense(hu, input_shape=(self.learn_env.lags, self.learn_env.n_features), activation='relu'))
        if dropout:
           model.add(Dropout(0.3, seed=100))
        model.add(Dense(hu, activation='relu'))
        if dropout:
            model.add(Dropout(0.3, seed=100))
        model.add(Dense(2, activation='linear'))
        model.compile(
            loss='mse',
            optimizer=keras.optimizers.legacy.RMSprop(learning_rate=lr)
        )
        return model
    
    def act(self, state):
        '''
        method for taking action based on
        a) exploration
        b) exploitation
        '''
        if random.random() <= self.epsilon:
            return self.learn_env.action_space.sample()
        action = self.model.predict(state)[0, 0]
        return np.argmax(action)
    
    def replay(self):
        ''''''