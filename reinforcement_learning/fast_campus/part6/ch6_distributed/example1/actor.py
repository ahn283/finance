# actor.py
import zmq
import time
import torch
torch.set_num_threads(1)

from agent import Agent
from utils import create_env
from torch.multiprocessing import Process
from configuration import config


class Actor:
    def __init__(self, config):
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(f"tcp://127.0.0.1:{config.port}")

        self.config = config
        self.env = create_env(config)
        self.agent = Agent(self.env, config)
        self.update_step = -1
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

    def add_to_trajectory(self, s, a, pi, r, s_next, done):
        if not done:
            length_to_append = 1
        else:
            # When the trajectory is done before it is full, append the last data until the end
            length_to_append = self.config.seq_length - len(self.trajectory['state'])

        for _ in range(length_to_append):
            self.trajectory['state'].append(s)
            self.trajectory['action'].append(a)
            self.trajectory['pi_old'].append(pi)
            self.trajectory['reward'].append(r)
            self.trajectory['state_next'].append(s_next)
            self.trajectory['done'].append(done)

        if len(self.trajectory['state']) == self.config.seq_length:
            trajectory_to_send = self.trajectory
            self.trajectory = self.create_trajectory()
            return trajectory_to_send

    def send_data(self, trajectory, update_step, env_log):
        data_send = {
            'trajectory': trajectory,
            'update_step': update_step,
            'env_log': env_log
        }
        #print(data_send)
        self.socket.send_pyobj(data_send)
        data_recv = self.socket.recv_pyobj()

        model_param = data_recv['model_param']
        if model_param is not None:
            self.agent.load_state_dict(model_param)
            self.update_step = data_recv['update_step']

    def run(self):
        print('[starting actor]')
        self.send_data(None, self.update_step, None)  # Request model at first
        s = self.env.reset()
        env_log = {
            'score': 0,
            'count_step': 0,
        }
        while True:
            a, pi_old = self.agent.action(s)
            s_next, r, done, _ = self.env.step(a)

            env_log['score'] += r
            env_log['count_step'] += 1

            trajectory = self.add_to_trajectory(s, a, pi_old, r, s_next, done)

            env_log_to_send = None
            if done:
                env_log_to_send = env_log
                env_log = {
                    'score': 0,
                    'count_step': 0,
                }

            update_step = None
            if trajectory is not None:
                update_step = self.update_step  # update_step is specified only when sending trajectory

            if (
                env_log_to_send is not None
                or trajectory is not None
            ):
                self.send_data(trajectory, update_step, env_log_to_send)  # send trajectory but not the env log

            s = s_next
            if done:
                s = self.env.reset()

            # time.sleep(5e-3)


def run_actor(config):
    actor = Actor(config)
    actor.run()


if __name__ == '__main__':
    procs = list()
    for _ in range(config.num_actor):
        proc = Process(target=run_actor, args=(config,))
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join()



