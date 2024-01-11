import datetime
import numpy as np
import gym
import ray
from actor import Actor
from replay import ReplayBuffer
from learner import Learner
from parameter_server import ParameterServer
import tensorflow as tf
tf.get_logger().setLevel('WARNING')

def get_env_parameters(config):
    env = gym.make(config['env'])
    config['obs_shape'] = env.observation_space.shape
    config['n_actions'] = env.action_space.n
    
def main(config, max_samples):
    get_env_parameters(config)
    log_dir = 'logs/scalars/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
    file_writer.set_as_default()
    config['log_dir'] = log_dir
    ray.init()
    parameter_server = ParameterServer.remote(config)
    replay_buffer = ReplayBuffer.remote(config)
    learner = Learner.remote(config, replay_buffer, parameter_server)
    training_actor_ids = []
    eval_actor_ids = []
    
    learner.start_learning.remote()
    
    # create training actors
    for i in range(config['num_workers']):
        eps = config['max_eps'] * i / config['num_workers']
        actor = Actor.remote('train-'+str(i),
                             replay_buffer,
                             parameter_server,
                             config,
                             eps)
        actor.sample.remote()
        training_actor_ids.append(actor)
        
    # create eval actors
    for i in range(config['eval_num_workers']):
        eps = 0
        actor = Actor.remote('eval' + str(i),
                             replay_buffer,
                             parameter_server,
                             config,
                             eps,
                             True)
        eval_actor_ids.append(actor)
    
    total_samples = 0
    best_eval_mean_reward = np.NINF
    eval_mean_rewards = []
    
