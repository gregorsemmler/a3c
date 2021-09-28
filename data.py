import logging
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from gym import Env
from torch.distributions import Categorical

from model import ActorCriticModel

logger = logging.getLogger(__name__)


def default_action_selector(probs):
    return Categorical(probs.detach().cpu()).sample()


class EpisodeResult(object):

    def __init__(self, env, start_state):
        self.env = env
        self.states = [start_state]
        self.actions = []
        self.rewards = []
        self.infos = []

    def append(self, action, reward, state, info=None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.infos.append(info)

    def calculate_return(self, gamma):
        total_return = 0.0
        for k in range(len(self.rewards)):
            total_return += gamma ** k * self.rewards[k]
        return total_return

    def __str__(self):
        return f"{self.actions} - {self.rewards}"

    def __len__(self):
        return len(self.states)


class EnvironmentsDataset(object):

    def __init__(self, envs: Sequence[Env], model: ActorCriticModel, n_steps, gamma, batch_size, preprocessor, device,
                 action_selector=None):
        self.envs = envs
        self.model = model
        self.num_actions = model.num_actions
        if n_steps < 1:
            raise ValueError(f"Number of steps {n_steps} needs be greater or equal to 1")
        self.n_steps = n_steps
        self.gamma = gamma
        self.batch_size = batch_size
        self.prepocessor = preprocessor
        self.device = device
        self.action_selector = default_action_selector if action_selector is None else action_selector

    def n_step_return(self, rewards, last_state_val):
        result = last_state_val
        for r in reversed(rewards):
            result = r + self.gamma * result
        return result

    def data(self):
        batch = []
        states = self.reset()
        episode_results = [EpisodeResult(e, s) for e, s in zip(self.envs, states)]

        while True:
            in_ts = torch.cat([self.prepocessor.preprocess(s) for s in states]).to(self.device)

            self.model.eval()
            with torch.no_grad():
                log_probs_out, vals_out = self.model(in_ts)
                probs_out = F.softmax(log_probs_out, dim=1)

            actions = self.action_selector(probs_out)
            stepped = self.step(actions)
            new_states, rewards, dones, infos = list(zip(*stepped))

            for er, a, (s, r, d, i) in zip(episode_results, actions, stepped):
                er.append(int(a), r, s, i)

            long_enoughs = [idx for idx in range(len(episode_results)) if len(episode_results[idx]) > self.n_steps]
            dones_ids = [idx for idx in range(len(dones)) if dones[idx]]

            if len(long_enoughs) > 0:
                last_states_vals = [float(vals_out[idx]) for idx in long_enoughs]
                reward_lists = [episode_results[idx].rewards[-self.n_steps:] for idx in long_enoughs]
                n_step_returns = [self.n_step_return(r_l, l_s_v) for r_l, l_s_v in zip(reward_lists, last_states_vals)]
                cur_states = [episode_results[idx].states[-self.n_steps] for idx in long_enoughs]

                with torch.no_grad():
                    cur_in_ts = torch.cat([self.prepocessor.preprocess(s) for s in cur_states]).to(self.device)
                    _, cur_vals_out = self.model(cur_in_ts)

                advantages = [n_r - float(c_v) for n_r, c_v in zip(n_step_returns, cur_vals_out)]

                print("")

            if len(dones_ids) > 0:
                print("")
                pass

            print("")

        # TODO
        # actions, vals = self.net(states)

        print("")
        pass

    def reset(self):
        return [e.reset() for e in self.envs]

    def step(self, actions):
        return [e.step(a) for e, a in zip(self.envs, actions)]


class Environments(Env):

    def __init__(self, envs: Sequence[Env], action_space, observation_space):
        self.envs = envs
        self.action_space = action_space
        self.observation_space = observation_space

    def step(self, actions):
        new_states = []
        rewards = []
        dones = []
        infos = []

        for env, action in zip(self.envs, actions):
            new_state, reward, done, info = env.step(action)
            new_states.append(new_state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return new_states, rewards, dones, infos

    def reset(self):
        return [e.reset() for e in self.envs]

    def render(self, mode="human"):
        # TODO display complete image instead / human render mode?
        return [e.render(mode="rgb_array") for e in self.envs]

    def close(self):
        for env in self.envs:
            env.close()
