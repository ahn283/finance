# configuration.py

from utils import AttrDict

config = AttrDict(
    gamma=0.9,
    lr=1e-4,
    c1=1,
    c2=0.5,
    c3=1e-4,
    num_env=4,
    seq_length=32,
    batch_size=16,
    hidden_size=128,
    train_env_steps=5000000,
    num_eval_episode=100,
)