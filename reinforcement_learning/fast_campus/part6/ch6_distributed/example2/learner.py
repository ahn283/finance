# learner.py
import zmq
import time
import datetime
import numpy as np

from torch.multiprocessing import Process
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from configuration import config
from utils import create_env, MPQueue
from agent import Agent


class Learner:
    def __init__(self, config):
        self.q_trajectory = MPQueue()
        self.q_env_log = MPQueue()

        env_temp = create_env(config)

        self.config = config
        self.agent = Agent(env_temp, config)
        self.agent.share_memory()
        self.agent.set_optimizer()

        self.writer = None
        self.env_logs = list()
        self.trajectories = [self.create_trajectory() for _ in range(config.num_actor)]

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

    def add_to_trajectory(self, actor_idx, transition):
        if not transition['done']:
            length_to_append = 1
        else:
            # When the trajectory is done before it is full, append the last data until the end
            length_to_append = self.config.seq_length - len(self.trajectories[actor_idx]['state'])

        for _ in range(length_to_append):
            for key, val in transition.items():
                self.trajectories[actor_idx][key].append(val)

        if len(self.trajectories[actor_idx]['state']) == self.config.seq_length:
            trajectory_to_train = self.trajectories[actor_idx]
            self.trajectories[actor_idx] = self.create_trajectory()
            self.q_trajectory.put(trajectory_to_train)

    def receive_data(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://127.0.0.1:{config.port}")

        while True:
            data_recv = socket.recv_pyobj()

            state = data_recv['state']
            action, pi_old = self.agent.action(state)

            data_send = {
                'action': action,
                'pi_old': pi_old,
            }
            socket.send_pyobj(data_send)

            transition = data_recv['transition']
            if transition is not None:
                self.add_to_trajectory(data_recv['actor_idx'], transition)

            env_log = data_recv['env_log']
            if env_log is not None:
                self.q_env_log.put(env_log)

    def write_env_log(self):
        if len(self.env_logs) < self.config.num_avg_env_log:
            return

        env_logs = self.env_logs[:self.config.num_avg_env_log]
        self.env_logs = self.env_logs[self.config.num_avg_env_log:]

        env_log_dict = dict()
        for env_log in env_logs:
            for key in env_log:
                if key not in env_log_dict:
                    env_log_dict[key] = list()  # initialize
                env_log_dict[key].append(env_log[key])

        for key in env_log_dict:
            self.writer.add_scalar(f"Env/{key}", np.mean(env_log_dict[key]), self.agent.update_step)

    def write_train_log(self, train_log):
        print(f"[{self.agent.update_step}] {train_log} num_remain_traj: {len(self.agent.batch)}")
        for key, val in train_log.items():
            self.writer.add_scalar(f"Train/{key}", val, self.agent.update_step)

    def run(self):
        print('[starting learner]')
        comm_proc = Process(target=self.receive_data, args=())
        comm_proc.start()  # Start communication thread

        dt_now = datetime.datetime.now()
        logdir = f"logdir/{dt_now.strftime('%y-%m-%d_%H-%M-%S')}"
        self.writer = SummaryWriter(logdir)

        while True:
            for _ in range(self.q_trajectory.qsize()):
                trajectory = self.q_trajectory.get()
                self.agent.append_trajectory(trajectory)

            for _ in range(self.q_env_log.qsize()):
                env_log = self.q_env_log.get()
                self.env_logs.append(env_log)
            self.write_env_log()

            if len(self.agent.batch) >= self.config.batch_size:
                train_log = self.agent.train_model()
                self.write_train_log(train_log)

            time.sleep(1e-3)  # Sleep to prevent meaningless high computational usage


if __name__ == '__main__':
    learner = Learner(config)
    learner.run()
