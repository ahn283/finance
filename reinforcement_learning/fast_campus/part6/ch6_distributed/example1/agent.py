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
        self.update_step = 0

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
            nn.Linear(self.config.hidden_size, 1)
        )
        self.policy_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ELU(),
            nn.Linear(self.config.hidden_size, n_action),
            nn.Softmax(dim=-1)
        )

        self.replay_memory = deque([], maxlen=config.replay_capacity)  # list of episodes
        self.batch = list()

    def append_trajectory(self, trajectory):
        self.batch.append(trajectory)

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
        vtrace_target_seq = torch.zeros(self.config.seq_length, self.config.batch_size)
        delta_v = torch.clip(rho_chosen, min=0, max=self.config.rho_trunc) * (
                reward_seq + self.config.gamma * (1 - done_seq) * value_next - value_seq
        )
        vtrace_target = value_seq[-1, :] + delta_v[-1, :]
        vtrace_target_seq[-1, :] = vtrace_target
        for i in range(1, self.config.seq_length):
            indice = -1 - i
            vtrace_target = value_seq[indice, :] + delta_v[indice, :] + self.config.gamma * (
                (1 - done_seq[indice, :]) * torch.clip(rho_chosen[indice, :], 0, self.config.c_trunc)
                * (vtrace_target - value_next[indice, :])
            )
            vtrace_target_seq[indice, :] = vtrace_target

        return vtrace_target_seq

    def update(self, batch):
        state_seq_array = np.array([trajectory['state'] for trajectory in batch])  # (n_batch, n_seq, *dim_state)
        action_seq_array = np.array([trajectory['action'] for trajectory in batch], dtype=np.int64)  # (n_batch, n_seq)
        pi_old_seq_array = np.array([trajectory['pi_old'] for trajectory in batch])  # (n_batch, n_seq)
        reward_seq_array = np.array([trajectory['reward'] for trajectory in batch])  # (n_batch, n_seq)
        state_next_seq_array = np.array([trajectory['state_next'] for trajectory in batch])  # (n_batch, n_seq, *dim_state)
        done_seq_array = np.array([trajectory['done'] for trajectory in batch])  # (n_batch, n_seq)

        state_seq_tensor = torch.from_numpy(
            state_seq_array
        ).float().transpose(0, 1)  # (n_seq, n_batch, *dim_states)
        action_seq_tensor = torch.from_numpy(action_seq_array).transpose(0, 1)  # (n_seq, n_batch)
        reward_seq_tensor = torch.from_numpy(reward_seq_array).float().transpose(0, 1)  # (n_seq, n_batch)
        pi_old_seq_tensor = torch.from_numpy(pi_old_seq_array).transpose(0, 1)  # (n_seq, n_batch)
        state_next_seq_tensor = torch.from_numpy(
            state_next_seq_array
        ).float().transpose(0, 1)  # (n_seq, n_batch, *dim_states)
        done_seq_tensor = torch.from_numpy(done_seq_array).float().transpose(0,1)  # (n_seq, n_batch)
        
        # mask for updating policy, until the transition that its done is True
        update_mask = done_seq_tensor.roll(1, dims=0)  # (n_seq, n_batch)
        update_mask[0, :] = 0   # (n_seq, n_batch)
        update_mask = 1 - update_mask  # (n_seq, n_batch)

        pi, value = self.forward(state_seq_tensor)  # (n_seq, n_batch, n_action), (n_seq, n_batch, n_action)
        pi_next, value_next = self.forward(state_next_seq_tensor)  # (n_seq, n_batch, 1)

        pi_chosen = pi.gather(dim=-1, index=action_seq_tensor.unsqueeze(-1))  # (n_seq, n_batch, 1)
        pi_chosen = pi_chosen.squeeze(-1)  # (n_seq, n_batch)
        pi_chosen_old = pi_old_seq_tensor.gather(dim=-1, index=action_seq_tensor.unsqueeze(-1))
        pi_chosen_old = pi_chosen_old.squeeze(-1)  # (n_seq, n_batch)
        rho_chosen = pi_chosen / pi_chosen_old

        vtrace_target_seq = self.vtrace(rho_chosen, reward_seq_tensor, done_seq_tensor, value, value_next)  # (n_seq, n_batch)
        criterion = nn.SmoothL1Loss(reduction='none')
        loss_critic = criterion(value, vtrace_target_seq.detach())
        loss_critic = torch.mean(update_mask * loss_critic)
        vtrace_target_next_seq = torch.cat(
            [vtrace_target_seq[1:, :], value_next[-2:-1, :]],
            dim=0
        )

        adv = reward_seq_tensor + self.config.gamma * (1 - done_seq_tensor) * vtrace_target_next_seq - value
        rho_trunc = torch.clip(rho_chosen, min=0, max=self.config.rho_trunc)
        loss_actor = (
            rho_trunc.detach() * adv.detach() * torch.log(pi_chosen + 1e-10)
        )  # (n_seq, n_batch)

        loss_actor = -torch.mean(update_mask * loss_actor)
        loss_exp = -torch.mean(
            update_mask 
            * torch.sum(-pi * torch.log(pi + 1e-15), dim=-1)  # (n_seq, n_batch, n_action) -> (n_seq, n_batch)
        )
        loss = self.config.c1 * loss_critic + self.config.c2 * loss_actor + self.config.c3 * loss_exp

        loss_critic_avg = loss_critic * self.config.seq_length * self.config.batch_size / update_mask.sum()
        entropy_avg = -loss_exp * self.config.seq_length * self.config.batch_size / update_mask.sum()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        train_log = {
            'loss_critic': loss_critic_avg.detach().item(),
            'entropy': entropy_avg.detach().item(),
        }

        return train_log

    def train_model(self):
        train_batch = self.batch[:self.config.batch_size]
        self.batch = self.batch[self.config.batch_size:]
        train_log = self.update(train_batch)
        if len(self.replay_memory) >= self.config.replay_ratio * self.config.replay_capacity:
            for _ in range(self.config.num_replay):
                batch = random.sample(self.replay_memory, self.config.batch_size)
                self.update(batch)
        self.update_step += 1
        self.replay_memory.extend(train_batch)

        return train_log





















