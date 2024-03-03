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
        
        self.value_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ELU(),
            nn.Linear(self.config.hidden_size, 1),
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
            'done': list(),
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
            # when the trajectory is done before it is full, append the last data until the end
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
        value = self.value_head(h_enc)
        pi = self.policy_head(h_enc)
        return pi, value.squeeze(-1)
    
    def action(self, x):
        # used when sampling an action for a state
        with torch.no_grad():
            x = torch.from_numpy(x).float().reshape(1, -1)
            pi, value = self.forward(x)
            a = torch.distributions.Categorical(pi).sample().item()
            pi = pi.squeeze(0).numpy()  # (n_action)
        return a, pi
    
    def vtrace(self, rho_chosen, reward_seq, done_seq, value_seq, value_next):
        vtrace_target_seq = torch.zeros(self.config.seq_length, self.config.batch_size)     # intialize vtrace
        delta_v = torch.clip(rho_chosen, min=0, max=self.config.rho_trunc) * (reward_seq + self.config.gamma * (1 - done_seq) * value_next - value_seq)     #\delta_t * V = \rho_t (r_{t+1} + \gamma V(x_{t+1}) - V(x_t))
        vtrace_target = value_seq[-1, :] + delta_v[-1, :]       # last sequence has no next value
        vtrace_target_seq[-1, :] = vtrace_target
        for i in range(1, self.config.seq_length):
            indice = -1 - i     # 역순으로 재귀적으로 계산
            vtrace_target = value_seq[indice, :] + delta_v[indice, :] + self.config.gamma * (1 - done_seq[indice, :]) * torch.clip(rho_chosen[indice, :], 0, self.config.c_trunc) * (vtrace_target - value_next[indice, :])
            vtrace_target_seq[indice, :] = vtrace_target
            
        return vtrace_target_seq
    
    def update(self, batch):
        state_seq_array = np.array([trajectory['state'] for trajectory in batch])   # (n_batch, n_seq, *dim_state)
        action_seq_array = np.array([trajectory['action'] for trajectory in batch], dtype=np.int64) # (n_batch, n_seq)
        pi_old_seq_array = np.array([trajectory['pi_old'] for trajectory in batch]) # (n_batch, n_seq)
        reward_seq_array = np.array([trajectory['reward'] for trajectory in batch]) # (n_batch, n_seq)
        state_next_seq_array = np.array([trajectory['state_next'] for trajectory in batch]) # (n_batch, n_seq, *dim_state)
        done_seq_array = np.array([trajectory['done'] for trajectory in batch])     # (n_batch, n_seq)
        
        state_seq_tensor = torch.from_numpy(state_seq_array).float().transpose(0, 1)    # (n_seq, n_batch, *dim_states)
        action_seq_tensor = torch.from_numpy(action_seq_array).transpose(0, 1)      # (n_seq, n_batch)
        reward_seq_tensor = torch.from_numpy(reward_seq_array).float().transpose(0, 1)  # (n_seq, n_batch)
        pi_old_seq_tensor = torch.from_numpy(pi_old_seq_array).transpose(0, 1)      # (n_seq, n_batch)
        state_next_seq_tensor = torch.from_numpy(state_next_seq_array).float().transpose(0, 1)  # (n_seq, n_batch, *dim_states)
        done_seq_tensor = torch.from_numpy(done_seq_array).float().transpose(0, 1)  # (n_seq, n_batch)
        
        # mask for updating policy, until the transition that its done is True
        update_mask = done_seq_tensor.roll(1, dims=0)   # (n_seq, n_batch)
        update_mask[0, :] = 0   
        update_mask = 1 - update_mask
        
        pi, value = self.forward(state_seq_tensor)  # (n_seq, n_batch, n_action), (n_seq, n_batch, 1)
        pi_next, value_next = self.forward(state_next_seq_tensor)   # (n_seq, n_batch, 1)
        
        pi_chosen = pi.gather(dim=-1, index=action_seq_tensor.unsqueeze(-1))    # (n_seq, n_batch, 1)
        pi_chosen = pi_chosen.squeeze(-1)   # (n_seq, n_batch)
        
        pi_chosen_old = pi_old_seq_tensor.gather(dim=-1, index=action_seq_tensor.unsqueeze(-1))
        pi_chosen_old = pi_chosen_old.squeeze(-1)
        rho_chosen = pi_chosen / pi_chosen_old
        
        vtrace_target_seq = self.vtrace(rho_chosen, reward_seq_tensor, done_seq_tensor, value, value_next)  # (n_seq, n_batch)
        
        criterion = nn.SmoothL1Loss(reduce='none')
        loss_critic = criterion(value, vtrace_target_seq.detach())
        loss_critic = torch.mean(update_mask * loss_critic)
        vtrace_target_next_seq = torch.cat(
            [vtrace_target_seq[1:, :]],     # vtrace_target_seq를 한칸씩 뒤로 이동
            value_next[-2:-1, :],           # 마지막 vtrace 값이 없으므로 value_next를 입력
            dim=0
        )
        
        adv = reward_seq_tensor + self.config.gamma * (1 - done_seq_tensor) * vtrace_target_next_seq - value
        rho_trunc = torch.clip(rho_chosen, min=0, max=self.config.rho_trunc)
        
        loss_actor = (
            rho_trunc.detach() * adv.detach() * torch.log(pi_chosen + 1e-10)
        )
        
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
            
        self.replay_memory.extend(self.on_policy_batch)
        self.on_policy_batch.clear()
        
        return loss_critic_avg, entropy_avg
        
        
        
        