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

    def append(self, action, reward, state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def calculate_return(self, gamma):
        total_return = 0.0
        for k in range(len(self.rewards)):
            total_return += gamma ** k * self.rewards[k]
        return total_return


class EnvironmentsDataset(object):

    def __init__(self, envs: Sequence[Env], model: ActorCriticModel, preprocessor, device, action_selector=None):
        self.envs = envs
        self.model = model
        self.prepocessor = preprocessor
        self.num_actions = model.num_actions
        self.device = device
        self.action_selector = default_action_selector if action_selector is None else action_selector

    def data(self):
        states = self.reset()
        episode_results = [EpisodeResult(e, s) for e, s in zip(self.envs, states)]

        in_ts = torch.cat([self.prepocessor.preprocess(s) for s in states]).to(self.device)

        self.model.eval()
        with torch.no_grad():
            log_probs_out, vals_out = self.model(in_ts)
            probs_out = F.softmax(log_probs_out, dim=1)

        actions = self.action_selector(probs_out)

        # TODO
        # actions, vals = self.net(states)

        print("")
        pass

    def reset(self):
        return [e.reset() for e in self.envs]


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
