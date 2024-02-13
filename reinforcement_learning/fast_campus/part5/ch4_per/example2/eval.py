# eval.py
import gym
import time
import torch
import argparse

from utils import create_env
from per_agent import DQNAgent, config 

def eval_agent_with_rendering(config, env, agent):
    score_sum = 0
    step_count_sum = 0
    for _ in range(config.num_eval_episode):
        s = env.reset()
        step_count = 0
        done = False
        score = 0
        while not done:
            with torch.no_grad():
                a = agent.get_argmax_action(s)
            s_next, r, done, info = env.step(a)
            step_count += 1
            
            score += 1
            s = s_next
        
        score_sum += score
        step_count_sum += step_count
        
    score_avg = score_sum / config.num_eval_episode
    step_count_avg = step_count_sum / config.num_eval_episode
    
    print(f"score avg: {score_avg} step_count_avg: {step_count_avg}")
    return score_avg, step_count_avg

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_eval', type=int, default=1, help='Numver of episode to be evaluated')
    parser.add_argument('--model_path', type=str, help='Specity the apth to your model to be evaluted')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    config.num_eval_episode = args.num_eval
    env = create_env(config, render_mode='human')
    agent = DQNAgent(env, config)
    
    if args.model_path:
        state_dict = torch.load(args.model_path)
        agent.load_state_dict(state_dict)
        
    eval_agent_with_rendering(config, env, agent)