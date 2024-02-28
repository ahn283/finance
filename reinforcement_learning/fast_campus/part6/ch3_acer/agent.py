# agent.py

import torch
import random

import numpy as np

from torch import nn 
from collections import deque



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
        self.action_value_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ELU(),
            nn.Linear(self.config.hidden_size, n_action)
        )
        self.policy_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ELU(),
            nn.Linear(self.config.hidden_size, n_action),
            nn.Softmax(dim=-1)
        )
        
        self.replay_memory = deque([], maxlen=config.replay_capacity)
        self.on_policy_batch = deque([], maxlen=config.batch_size)
        self.trajectory = self.create_trajectory()
        
    def create_trajectory(self):
        trajectory = {
            'state': list(),
            'action': list(),
            'pi_old': list(),
            'reward': list(),
            'state_next': list(),
            'done': list()
        }
        return trajectory
    
    def add_to_batch(self, s, a, pi, r, s_next, done):
        if (
            len(self.trajectory['state']) == self.config.seq_length
        ):
            self.on_policy_batch.append(self.trajectory)
            self.trajectory = self.create_trajectory()
            
        if not done:
            length_to_append = 1
                
        else:
            # when the trajectory is done before it is full, apeend the alst data until the end
            length_to_append = self.config.seq_length - len(self.trajectory['state'])
                
        for _ in range(length_to_append):
            self.trajectory['state'].append(s)
            self.trajectory['action'].append(a)
            self.trajectory['pi_old'].append(pi)
            self.trajectory['reward'].append(r)
            self.trajectory['state_next'].append(s_next)
            self.trajectory['done'].append(done)
    
    def set_optimizer(self):
        self.optim = torch.optim.Adam(
            self.parameters(),
            lr=self.config.lr 
        )
        
    def forward(self, x):
        h_enc = self.encoder(x)
        actoin_vlalue = self.action_value_head(h_enc)
        pi = self.policy_head(h_enc)
        return pi, actoin_vlalue
    
    def action(self, x):
        # used when sampling an action for a state
        with torch.no_grad():
            x = torch.from_numpy(x).float().reshape(1, -1)
            pi, action_value = self.forward(x)
            a = torch.distributions.Categorical(pi).sample().item()
            pi = pi.squeeze(0).numpy()
            
        return a, pi
    
    def retrace(self, rho_trunc, reward_seq, done_seq, action_value_chosen_seq, pi_next, action_value_next):
        Q_ret_seq = torch.zeros(self.config.seq_length, self.config.batch_size)     # (n_seq, n_batch)
        
        value_next = torch.sum(pi_next * action_value_next, dim=-1)    # (n_seq, n_batch)
        # calculate last reward
        Q_ret = reward_seq[-1, :] + self.config.gamma * (1 - done_seq[-1, :]) * value_next[-1, :]
        Q_ret_seq[-1, :] = Q_ret
        # calculate rewards backward from the last one
        for i in range(1, self.config.seq_length):
            indice = -1 - i
            delta = Q_ret - action_value_chosen_seq[indice + 1, :]
            Q_ret = reward_seq[indice, :] + (1 - done_seq[indice, :]) * self.config.gamma * (
                rho_trunc[indice + 1, :] * delta + value_next[indice]
            )
            Q_ret_seq[indice, :] = Q_ret
        return Q_ret_seq
    
    def update(self, batch):
        state_seq_array = np.array([trajectory['state'] for trajectory in batch])   # (n_batch, n_seq, *dim_state)
        action_seq_array = np.array([trajectory['action'] for trajectory in batch], dtype=np.int64) # (n_batch, n_seq)
        pi_old_seq_array = np.array([trajectory['pi_old'] for trajectory in batch]) # (n_batch, n_seq, n_action)
        reward_seq_array = np.array([trajectory['reward'] for trajectory in batch]) # (n_batch, n_seq)
        state_next_seq_array = np.array([trajectory['state_next'] for trajectory in batch]) # (n_batch, n_seq, *dim_state)
        done_seq_array = np.array([trajectory['done'] for trajectory in batch]) # (n_batch, n_seq)
        
        state_seq_tensor = torch.from_numpy(
            state_seq_array
        ).float().transpose(0, 1)   # (n_seq, n_batch, *dim_states)
        action_seq_tensor = torch.from_numpy(action_seq_array).transpose(0, 1)  # (n_seq, n_batch)
        reward_seq_tensor = torch.from_numpy(reward_seq_array).transpose(0, 1)  # (n_seq, n_batch)
        pi_old_seq_tensor = torch.from_numpy(pi_old_seq_array).float().transpose(0, 1)  # (n_seq, n_batch)
        state_next_seq_tensor = torch.from_numpy(
            state_next_seq_array
        ).float().transpose(0, 1)   # (n_seq, n_batch, *dim_states)
        done_seq_tensor = torch.from_numpy(done_seq_array).float().transpose(0, 1)  # (n_seq, n_batch)
        
        # mask for updating policy, until the transition that its done is True
        update_mask = done_seq_tensor.roll(1, dims=0)   # (n_seq, n_batch)
        update_mask[0, :] = 0
        update_mask = 1 - update_mask   # (n_seq, n_batch)
        
        pi, action_value = self.forward(state_seq_tensor)   # (n_seq, n_batch, n_action), (n_seq, n_batch, n_action)
        pi_next, action_value_next = self.forward(state_next_seq_tensor)    # (n_seq, n_batch, 1)
        
        pi_chosen = pi.gather(dim=-1, index=action_seq_tensor.unsqueeze(-1))    # (n_seq, n_batch, 1)        
        pi_chosen = pi_chosen.squeeze(-1)   # (n_seq, n_batch)
        pi_chosen_old = pi_old_seq_tensor.gather(dim=-1, index=action_seq_tensor.unsqueeze(-1))
        pi_chosen_old = pi_chosen_old.squeeze(-1)   # (n_seq, n_batch)
        
        action_value_chosen = action_value.gather(dim=-1, index=action_seq_tensor.unsqueeze(-1))
        action_value_chosen = action_value_chosen.squeeze(-1)   # (n_seq, n_batch)
        
        rho_trunc = torch.clip(pi_chosen / pi_chosen_old, 0, self.config.c_trunc).detach()
        
        Q_ret = self.retrace(rho_trunc, reward_seq_tensor, done_seq_tensor, action_value_chosen, pi_next, action_value_next)    # (n_seq, n_batch)
        loss_critic = torch.mean(update_mask * (Q_ret.detach() - action_value_chosen) ** 2)
        
        rho = pi / pi_old_seq_tensor
        value = torch.sum(pi * action_value, dim=-1).detach()   # (n_seq, n_batch, n_action) -> (n_seq, n_batch)
        rho_clip = torch.clip((rho - self.config.c_trunc) / rho, min=0, max=None).detach()
        loss_actor = (
            rho_trunc * (Q_ret.detach() - value) * torch.log(pi_chosen + 1e-15)
            + torch.sum(
                pi.detach() * rho_clip * torch.log(pi + 1e-15) * (action_value.detach() - value.unsqueeze(-1)),
                dim=-1
            )
        )   # (n_seq, n_batch)
        
        loss_actor = -torch.mean(update_mask * loss_actor)
        loss_exp = -torch.mean(
            update_mask * torch.sum(-pi * torch.log(pi + 1e-15), dim=-1)    # (n_seq, n_batch, n_action) -> (n_seq, n_batch)
        )
        loss = self.config.c1 * loss_critic + self.config.c2 * loss_actor + self.config.c3 * loss_exp
        
        loss_critic_avg = loss_critic * self.config.seq_length * self.config.batch_size / update_mask.sum()
        entropy_avg = -loss_exp * self.config.seq_length * self.config.batch_size / update_mask.sum()
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
        return loss_critic_avg.detach().item(), entropy_avg.detach().item()
    
    def train(self):
        loss_critic_avg, entropy_avg = self.update(self.on_policy_batch)
        if len(self.replay_memory) >= self.config.replay_ratio * self.config.replay_capacity:
            for _ in range(self.config.num_replay):
                batch = random.sample(self.replay_memory, self.config.batch_size)
                self.update(batch)
        
        self.on_policy_batch.clear()
        
        return loss_critic_avg, entropy_avg
        