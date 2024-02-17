# agent.py

import torch

import numpy as np

from torch import nn 
from collections import deque
from IPython import embed

def calc_return_seq(gamma, reward_seq):
    # calculate (g_0, g_1, g_2...) using (r_0, r_1, r_2, ...) and gamma
    n = reward_seq.shape[0]
    gamma_seq = gamma * np.ones([n])
    return_seq = np.zeros(n)    # initialize return sequence
    
    for t in range(n):
        gamma_seq_from_t = gamma_seq[t:]
        powers = np.arange(n - t)
        gamma_power_seq_from_t = np.power(gamma_seq_from_t, powers)
        reward_seq_from_t = reward_seq[t:]
        g_t = np.sum(reward_seq_from_t * gamma_power_seq_from_t)
        return_seq[t] = g_t
        
    return return_seq

class Agent(nn.Module):
    def __init__(self, env, config):
        super().__init__()
        self.config = config
        
        d_state = env.observation_space.shape[0]
        n_action = env.action_space.n 
        
        self.encoder = nn.Sequential(
            nn.Linear(d_state, self.config.hidden_size),
            nn.ELU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ELU()
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, n_action),
            nn.Softmax(dim=-1)
        )
        
        self.batch = deque([], maxlen=self.config.batch_size)       # list of episodes
        
    def add_to_batch(self, s, a, r, s_next, done):
        if done or len(self.batch) == 0:
            episode = {
                'state': list(),
                'action': list(),
                'reward': list(),
                'state_next': list(),
                'done': list(),
            }
            self.batch.append(episode)
            
        self.batch[-1]['state'].append(s)
        self.batch[-1]['action'].append(a)
        self.batch[-1]['reward'].append(r)
        self.batch[-1]['state_next'].append(s_next)
        self.batch[-1]['done'].append(done)
        
    def set_optimizer(self):
        self.optim = torch.optim.Adam(
            self.parameters(),
            lr=self.config.lr
        )
    
    def forward(self, x):
        h_enc = self.encoder(x)
        pi = self.policy_head(h_enc)
        return pi
    
    def action(self, x):
        # used when sampling an action for a state
        with torch.no_grad():
            # embed()
            x = torch.from_numpy(x).float().reshape(1, -1)
            pi = self.forward(x)
            a = torch.distributions.Categorical(pi).sample().item()
        return a
        
    def train(self):
        num_transition = 0
        loss_actor = 0
        loss_critic = 0
        loss_exp = 0
        loss = 0
        for e in self.batch:
            num_transition += len(e['state'])
            
            state_seq_array = np.stack(e['state'], axis=0)      # (n_seq, *dim_states)
            action_seq_array = np.stack(e['action'], axis=0, dtype=np.int64)    # (n_seq)
            reward_seq_array = np.stack(e['reward'], axis=0)    # (n_seq)
            state_next_seq_array = np.stack(e['state_next'], axis=0)    # (n_seq, *dim_states)
            done_seq_array = np.stack(e['done'], axis=0)        # (n_seq)
            return_seq_array = calc_return_seq(self.config.gamma, reward_seq_array)     # (n_seq)
            
            state_seq_tensor = torch.from_numpy(state_seq_array).float()    # (n_seq, *dim_states)
            action_seq_tensor = torch.from_numpy(action_seq_array)  # (n_seq)
            reward_seq_tensor = torch.from_numpy(reward_seq_array).float()  # (n_seq)
            state_next_seq_tensor = torch.from_numpy(state_next_seq_array).float()  # (n_seq)
            return_seq_tensor = torch.from_numpy(return_seq_array).float()  # (n_seq)
            
            pi = self.forward(state_seq_tensor)     # (n_seq, n_action), (n_seq, 1)
            pi_chosen = pi.gather(dim=-1, index=action_seq_tensor.unsqueeze(-1))    # using unsqueeze(-1) function (n_seq) -> (n_seq, 1)
            pi_chosen = pi_chosen.squeeze(-1)       # (n_seq, 1) -> (n_seq)
            
            loss_actor = (
                loss_actor - torch.sum(torch.log(pi_chosen + 1e-15) * return_seq_tensor)
            )
            
        loss = loss_actor / self.config.batch_size
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.batch.clear()
        
        return loss.detach().item()
            