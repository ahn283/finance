# configuration.py

from utils import AttrDict

config = AttrDict(
    gamma=0.99,
    lr=5e-4,
    c1=1,
    c2=1,
    c3=0,
    batch_size=8,
    hidden_size=128,
    train_env_steps=2000000,
    num_eval_episode=100,
)