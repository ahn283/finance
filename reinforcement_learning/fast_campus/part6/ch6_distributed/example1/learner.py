# learner.py
import zmq
import time
import datetime
import numpy as np

from threading import Thread
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from configuration import config
from utils import create_env
from agent import Agent


class Learner:
    def __init__(self, config):
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.socket.bind(f"tcp://127.0.0.1:{config.port}")

        env_temp = create_env(config)

        self.config = config
        self.agent = Agent(env_temp, config)
        self.agent.set_optimizer()

        dt_now = datetime.datetime.now()
        self.logdir = f"logdir/{dt_now.strftime('%y-%m-%d_%H-%M-%S')}"
        self.writer = SummaryWriter(self.logdir)
        self.env_logs = deque([], maxlen=self.config.num_avg_env_log)

    def receive_data(self):
        while True:
            data_recv = self.socket.recv_pyobj()
            data_send = {
                'update_step': self.agent.update_step,
            }

            actor_update_step = data_recv['update_step']
            if actor_update_step is None or actor_update_step == self.agent.update_step:
                model_param = None  # No need to update since actor already has the latest model
            else:
                model_param = self.agent.state_dict()  # Send latest model to agent
            data_send['model_param'] = model_param
            self.socket.send_pyobj(data_send)

            trajectory = data_recv['trajectory']
            if trajectory is not None:
                self.agent.append_trajectory(trajectory)

            env_log = data_recv['env_log']
            if env_log is not None:
                self.env_logs.append(env_log)

    def write_env_log(self):
        env_log_dict = dict()
        for env_log in self.env_logs:
            for key in env_log:
                if key not in env_log_dict:
                    env_log_dict[key] = list()  # initialize
                env_log_dict[key].append(env_log[key])

        for key in env_log_dict:
            self.writer.add_scalar(f"Env/{key}", np.mean(env_log_dict[key]), self.agent.update_step)
        self.env_logs.clear()

    def write_train_log(self, train_log):
        print(f"[{self.agent.update_step}] {train_log} num_remaining_data: {len(self.agent.batch)}")
        for key, val in train_log.items():
            self.writer.add_scalar(f"Train/{key}", val, self.agent.update_step)

    def run(self):
        print('[starting learner]')
        comm_th = Thread(target=self.receive_data, args=())
        comm_th.start()  # Start communication thread

        while True:
            if len(self.agent.batch) >= self.config.batch_size:
                train_log = self.agent.train_model()
                self.write_train_log(train_log)

            if len(self.env_logs) >= self.config.num_avg_env_log:
                self.write_env_log()

            time.sleep(1e-3)  # Sleep to prevent meaningless high computational usage


if __name__ == '__main__':
    learner = Learner(config)
    learner.run()
