# configuration.py

from utils import AttrDict

config = AttrDict(
    gamma=0.99,
    lr=5e-5,
    batch_size=64,
    hidden_size=128,
    replay_capacity=10000,
    replay_init_ratio=0.3,
    train_env_steps=200000,
    target_update_period=100,
    eps_init=1.0,
    eps_final=0.05,
    eps_decrease_step=10000,
    num_eval_episode=20,
    eval_period=500,
)