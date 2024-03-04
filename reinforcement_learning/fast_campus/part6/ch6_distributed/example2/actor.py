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
    def __init__(self, actor_idx, config):
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(f"tcp://127.0.0.1:{config.port}")

        self.actor_idx = actor_idx
        self.config = config
        self.env = create_env(config)

    def send_data(self, state, transition, env_log):
        data_send = {
            'actor_idx': self.actor_idx,
            'state': state,
            'transition': transition,
            'env_log': env_log
        }
        self.socket.send_pyobj(data_send)
        data_recv = self.socket.recv_pyobj()
        action = data_recv['action']
        pi_old = data_recv['pi_old']
        return action, pi_old

    def run(self):
        print('[starting actor]')
        s = self.env.reset()
        transition = None
        env_log_to_send = None
        env_log = {
            'score': 0,
            'count_step': 0,
        }
        epi_count = 0
        while True:
            a, pi_old = self.send_data(s, transition, env_log_to_send)
            s_next, r, done, _ = self.env.step(a)

            transition = {
                'state': s,
                'action': a,
                'pi_old': pi_old,
                'reward': r,
                'state_next': s_next,
                'done': done
            }

            env_log['score'] += r
            env_log['count_step'] += 1

            s = s_next
            env_log_to_send = None
            if done:
                epi_count += 1
                s = self.env.reset()
                env_log_to_send = env_log
                env_log = {
                    'score': 0,
                    'count_step': 0,
                }


def run_actor(actor_idx, config):
    actor = Actor(actor_idx, config)
    actor.run()


if __name__ == '__main__':
    procs = list()
    for actor_idx in range(config.num_actor):
        proc = Process(target=run_actor, args=(actor_idx, config))
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join()



