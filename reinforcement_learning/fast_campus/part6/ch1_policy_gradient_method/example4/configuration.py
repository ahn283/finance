# configuration.py

from utils import AttrDict

config = AttrDict(
    gamma=0.99,
    lr=1e-4,
    c1=1,
    c2=0.5,
    c3=5e-3,
    use_n_step_advantage=True,
    num_env=8,
    seq_length=16,
    batch_size=16,
    hidden_size=128,
    train_env_steps=5000000,
    num_eval_episode=100,
)