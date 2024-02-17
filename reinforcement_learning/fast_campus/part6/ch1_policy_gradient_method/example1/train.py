# train.py

import torch
import datetime

import numpy as np

from utils import create_env
from agent import Agent
from configuration import config
from collections import deque
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    env = create_env(config)
    agent = Agent(env, config)
    agent.set_optimizer()
    
    dt_now = datetime.datetime.now()
    logdir = f"logdir/{dt_now.strftime('%y-%m-%d_%H-%M-%S')}"
    writer = SummaryWriter(logdir)
    
    score_que = deque([], maxlen=config.num_eval_episode)
    count_step_que = deque([], maxlen=config.num_eval_episode)
    
    score = 0
    count_step = 0
    s = env.reset()
    for train_env_step in range(config.train_env_steps):
        a = agent.action(s)
        s_next, r, done, info = env.step(a)
        agent.add_to_batch(s, a, r, s_next, done)
        
        score += r
        count_step += 1
        s = s_next
        if done:
            s = env.reset()
            score_que.append(score)
            count_step_que.append(count_step)
            
            score = 0
            count_step = 0
            
            if len(agent.batch) == config.batch_size:
                loss = agent.train()
                writer.add_scalar('Train/loss', loss, train_env_step)
                
            if len(score_que) == config.num_eval_episode:
                score_avg = np.mean(score_que)
                count_step_avg = np.mean(count_step_que)
                writer.add_scalar('Env/score_avg', score_avg, train_env_step)
                writer.add_scalar('Env/count_step_avg', score_avg, train_env_step)
                
                print(f"[{train_env_step}] score_avg: {score_avg:.3f} count_step_avg: {count_step_avg:.3f}")
                score_que.clear()
                count_step_que.clear()
    
    torch.save(agent.state_dict(), f"{logdir}/state_dict.pth")
                
        