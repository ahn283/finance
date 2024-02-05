# configuration.py

from utils import AttrDict    # dict를 attribute로 호출할 수 있는 함수

config = AttrDict(
    gamma=0.99,
    lr=5e-5,
    batch_size=64,
    hidden_size=128,
    replay_capacity=10000,
    replay_init_ratio=0.3,    # replay를 얼만큼 채우고 시작할지
    train_env_steps=200000,
    target_update_period=100,
    eps_init=1.0,
    eps_final=0.05,
    eps_decrease_step=100000,
    num_val_episode=20,   # evaluation 횟수
    eval_period=500,    # evaluation 주기
)