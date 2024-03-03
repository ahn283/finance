# configuration.py

from utils import AttrDict

config = AttrDict(
    gamma=0.99,
    lam=0.5,
    lr=1e-4,
    c1=1,
    c2=0.5,
    c3=2e-4,
    rho_trunc=1,
    c_trunc=1,
    num_env=8,
    replay_capacity=500,
    replay_ratio=0.5,
    num_replay=4,
    seq_length=16,
    batch_size=16,
    hidden_size=128,
    train_env_steps=5000000,
    num_eval_episode=100,
)