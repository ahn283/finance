# configuration.py
from utils import AttrDict

config = AttrDict(
    gamma=0.99,
    lr=1e-4,
    batch_size=128,
    hidden_size=512,
    replay_capacity=50000,
    replay_init_ratio=0.5,
    traiin_env_steps=500000,
    target_update_period=2000,
    eps_init=1.0,
    eps_fianl=0.1,
    eps_decrease_step=50000,
    num_eval_episode=20,
    eval_start_step=0,
    eval_period=500,
    action_repeat=4,
    sampling_strategy='proportional',
    alpha=0.6,
    beta_init=0.4,
    eps_replay=1e-2
)