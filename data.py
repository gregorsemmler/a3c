import logging
import uuid
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

    def __init__(self, env, start_state, episode_id=None, chain=True):
        self.env = env
        self.states = [start_state]
        self.actions = []
        self.rewards = []
        self.infos = []
        self.done = False
        self.chain = chain
        self.episode_id = episode_id if episode_id is not None else str(uuid.uuid4())
        self.next_episode_result = None

    def begin_new_episode(self, episode_id=None, chain=True):
        self.next_episode_result = EpisodeResult(self.env, self.env.reset(), episode_id=episode_id, chain=chain)

    def append(self, action, reward, state, done, info=None):
        if self.done:
            if not self.chain:
                raise ValueError("Can't append to done EpisodeResult.")
            else:
                self.next_episode_result.append(action, reward, state, done, info)
        self.actions.append(action)
        self.states.append(state)
        self.rewards.append(reward)
        self.done = done
        self.infos.append(info)

    def calculate_return(self, gamma):
        total_return = 0.0
        for k in range(len(self.rewards)):
            total_return += gamma ** k * self.rewards[k]
        return total_return

    def n_step_return(self, n, gamma, last_state_value):
        cur_state, action, rewards = self.n_step_stats(n)
        result = 0.0 if n >= len(self.states) and self.done else last_state_value
        for r in reversed(rewards):
            result = r + gamma * result
        return result

    def n_step_stats(self, n):
        n_step_idx = -min(n, len(self.states))
        cur_state = self.states[n_step_idx]
        rewards = self.rewards[n_step_idx:]
        action = self.actions[n_step_idx]
        return cur_state, action, rewards

    def cur_state(self, n):
        return self.states[-min(n, len(self.states))]

    def cur_action(self, n):
        return self.actions[-min(n, len(self.states))]

    @property
    def last_state(self):
        return self.states[-1]

    def __str__(self):
        return f"{self.actions} - {self.rewards}"

    def __len__(self):
        return len(self.states)


class ActorCriticSample(object):

    def __init__(self, state, action, value, advantage):
        self.state = state
        self.action = action
        self.value = value
        self.advantage = advantage


class EnvironmentsDataset(object):

    def __init__(self, envs: Sequence[Env], model: ActorCriticModel, n_steps, gamma, batch_size, preprocessor, device,
                 action_selector=None):
        self.envs = {idx: e for idx, e in enumerate(envs)}
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
        self.episode_results = {}

    def data(self):
        batch = []
        self.reset()

        while True:
            sorted_ers = sorted(self.episode_results.items())
            k_to_idx = {k: idx for idx, (k, v) in enumerate(sorted_ers)}
            idx_to_k = {v: k for k, v in k_to_idx.items()}

            in_ts = torch.cat([self.prepocessor.preprocess(er.last_state) for k, er in sorted_ers]).to(self.device)

            with torch.no_grad():
                log_probs_out, vals_out = self.model(in_ts)
                probs_out = F.softmax(log_probs_out, dim=1)

            actions = self.action_selector(probs_out)
            self.step(actions)

            to_train_ers = {k: er for k, er in self.episode_results.items() if
                            (len(er) > self.n_steps) or len(er) <= self.n_steps and er.done}
            done_ids = {k for k, er in self.episode_results.items() if er.done}

            if len(to_train_ers) > 0:
                last_states_vals = [float(vals_out[k_to_idx[k]]) for k in to_train_ers.keys()]
                batch_ers = [er for k, er in to_train_ers.items()]
                n_step_returns = [er.n_step_return(self.n_steps, self.gamma, l_v) for er, l_v in
                                  zip(batch_ers, last_states_vals)]

                with torch.no_grad():
                    cur_in_ts = torch.cat(
                        [self.prepocessor.preprocess(er.cur_state(self.n_steps)) for k, er in sorted_ers]).to(
                        self.device)
                    _, cur_vals_out = self.model(cur_in_ts)

                advantages = [n_r - float(c_v) for n_r, c_v in zip(n_step_returns, cur_vals_out)]

                for er, val, adv in zip(batch_ers, cur_vals_out, advantages):
                    batch.append(ActorCriticSample(er.cur_state(self.n_steps), er.cur_action(self.n_steps), float(val), float(adv)))

                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []

            if len (done_ids) > 0:

                for k in done_ids:
                    env = self.envs[k]
                    self.episode_results[k] = EpisodeResult(env, env.reset())

                print("")

            print("")

        pass

    def reset(self):
        self.episode_results = {k: EpisodeResult(e, e.reset()) for k, e in self.envs.items()}

    def step(self, actions):
        for (k, er), a in zip(sorted(self.episode_results.items()), actions):
            s, r, d, i = er.env.step(a)
            er.append(int(a), r, s, d, i)


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
