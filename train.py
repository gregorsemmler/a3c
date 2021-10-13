import argparse
import logging
import uuid
from collections import deque
from datetime import datetime
from os import makedirs
from os.path import join

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import envs
import numpy as np
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from atari_wrappers import make_atari, wrap_deepmind
from data import EpisodeResult, EnvironmentsDataset, default_action_selector, Policy
from model import SimpleCNNPreProcessor, AtariModel, MLPModel, NoopPreProcessor
from utils import save_checkpoint

logger = logging.getLogger(__name__)


class DummySummaryWriter(object):

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        pass


class ActorCriticTrainer(object):

    def __init__(self, config, model, model_id, trainer_id=None, optimizer=None, scheduler=None, checkpoint_path=None,
                 writer=None, batch_wise_scheduler=False):
        self.value_factor = config.value_factor
        self.policy_factor = config.policy_factor
        self.entropy_factor = config.entropy_factor
        self.max_norm = config.max_norm
        self.lr = config.lr
        self.gamma = config.gamma
        self.num_eval_episodes = config.n_eval_episodes

        if config.device_token is None:
            device_token = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_token = config.device_token
        self.device = torch.device(device_token)

        self.trainer_id = "" if trainer_id is None else f"{trainer_id}_"
        self.model = model
        self.model_id = model_id
        self.optimizer = optimizer if optimizer is not None else Adam(model.parameters(), lr=self.lr)
        self.writer = writer if writer is not None else DummySummaryWriter()

        self.scheduler = scheduler
        self.checkpoint_path = checkpoint_path
        self.curr_epoch_idx = 0
        self.curr_train_batch_idx = 0
        self.curr_val_batch_idx = 0
        self.batch_wise_scheduler = batch_wise_scheduler

    def scheduler_step(self):
        if self.scheduler is not None:
            current_lr = self.scheduler.optimizer.param_groups[0]["lr"]

            try:
                self.scheduler.step()
            except UnboundLocalError as e:  # For catching OneCycleLR errors when stepping too often
                return
            log_prefix = "batch" if self.batch_wise_scheduler else "epoch"
            log_idx = self.curr_train_batch_idx if self.batch_wise_scheduler else self.curr_epoch_idx
            self.writer.add_scalar(f"{log_prefix}/{self.trainer_id}lr", current_lr, log_idx)

    def save_checkpoint(self):
        if self.checkpoint_path is None:
            return

        filename = f"{self.model_id}_{self.curr_epoch_idx:03d}.tar"
        path = join(self.checkpoint_path, filename)
        save_checkpoint(path, self.model, self.optimizer)

    def fit(self, dataset_train, eval_env, eval_policy, num_epochs=10, training_seed=None):
        if training_seed is not None:
            np.random.seed(training_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.manual_seed(training_seed)

        self.curr_epoch_idx = 0
        self.curr_train_batch_idx = 0
        self.curr_val_batch_idx = 0

        logger.info(f"{self.trainer_id}Starting Training For {num_epochs} epochs.")
        for _ in range(num_epochs):
            logger.info(f"{self.trainer_id}Epoch {self.curr_epoch_idx}")
            logger.info(f"{self.trainer_id}Training")
            self.train(dataset_train)
            logger.info(f"{self.trainer_id}Validation")
            # TODO keep track of average returns
            self.play(eval_env, eval_policy, num_episodes=self.num_eval_episodes)

            if not self.batch_wise_scheduler:
                self.scheduler_step()
            self.curr_epoch_idx += 1
            self.save_checkpoint()

    def train(self, dataset):
        self.model.train()

        ep_l = 0.0
        ep_p_l = 0.0
        ep_v_l = 0.0
        ep_e_l = 0.0
        count_batches = 0

        for batch in dataset.data():
            b_l, p_l, v_l, e_l = self.training_step(batch, self.curr_train_batch_idx)
            self.writer.add_scalar(f"train_batch/{self.trainer_id}loss", b_l, self.curr_train_batch_idx)
            self.writer.add_scalar(f"train_batch/{self.trainer_id}policy_loss", p_l, self.curr_train_batch_idx)
            self.writer.add_scalar(f"train_batch/{self.trainer_id}value_loss", v_l, self.curr_train_batch_idx)
            self.writer.add_scalar(f"train_batch/{self.trainer_id}entropy_loss", e_l, self.curr_train_batch_idx)

            logger.info(f"{self.trainer_id}Training - Epoch: {self.curr_epoch_idx} Batch: {self.curr_train_batch_idx}: "
                        f"Loss: {b_l:.6f} Policy Loss: {p_l:.6f} Value Loss: {v_l:.6f} Entropy Loss: {e_l:.6f}")
            self.curr_train_batch_idx += 1
            count_batches += 1
            ep_l += b_l
            ep_p_l += p_l
            ep_v_l += v_l
            ep_e_l += e_l

        ep_l /= max(1.0, count_batches)
        ep_p_l /= max(1.0, count_batches)
        ep_v_l /= max(1.0, count_batches)
        ep_e_l /= max(1.0, count_batches)

        self.writer.add_scalar(f"train_epoch/{self.trainer_id}loss", ep_l, self.curr_epoch_idx)
        self.writer.add_scalar(f"train_epoch/{self.trainer_id}policy_loss", ep_p_l, self.curr_epoch_idx)
        self.writer.add_scalar(f"train_epoch/{self.trainer_id}value_loss", ep_v_l, self.curr_epoch_idx)
        self.writer.add_scalar(f"train_epoch/{self.trainer_id}entropy_loss", ep_e_l, self.curr_epoch_idx)
        logger.info(f"{self.trainer_id}Training - Epoch {self.curr_epoch_idx}: Loss: {ep_l:.6f} "
                    f"Policy Loss: {ep_p_l:.6f} Value Loss: {ep_v_l:.6f} Entropy Loss: {ep_e_l:.6f}")

    def training_step(self, batch, batch_idx):
        states_t = torch.cat(batch.states).to(self.device)
        actions = batch.actions
        values_t = torch.FloatTensor(np.array(batch.values)).to(self.device)
        advantages_t = torch.FloatTensor(np.array(batch.advantages)).to(self.device)

        log_probs_out, value_out = self.model(states_t)
        probs_out = F.softmax(log_probs_out, dim=1)

        value_loss = self.value_factor * F.mse_loss(value_out.squeeze(), values_t)
        policy_loss = advantages_t * log_probs_out[range(len(probs_out)), actions]
        policy_loss = self.policy_factor * -policy_loss.mean()
        entropy_loss = self.entropy_factor * (probs_out * log_probs_out).sum(dim=1).mean()

        loss = entropy_loss + value_loss + policy_loss

        self.optimizer.zero_grad()
        loss.backward()

        if self.max_norm is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)

        self.optimizer.step()
        if self.batch_wise_scheduler:
            self.scheduler_step()

        return loss.item(), policy_loss.item(), value_loss.item(), entropy_loss.item()

    def play(self, env, policy, num_episodes=100, render=False):
        # TODO allow multiple environments being played here?
        # TODO add writer here
        i = 0
        best_return = float("-inf")
        best_result = None
        episode_returns = []
        while i < num_episodes:
            state = env.reset()
            done = False

            episode_result = EpisodeResult(env, state)
            while not done:
                if render:
                    env.render()

                action = int(policy(state))
                new_state, reward, done, info = env.step(action)

                episode_result.append(action, reward, new_state, done, info)

                state = new_state

            episode_return = episode_result.calculate_return(self.gamma)
            if best_return < episode_return:
                best_return = episode_return
                best_result = episode_result
                logger.info("New best return: {}".format(best_return))

            episode_returns.append(episode_return)
            i += 1

            logger.info(f"Episode Length & Return: {len(episode_result.states)} {episode_return}")

        return episode_returns, best_result, best_return


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_envs", type=int, default=100)
    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--value_factor", type=float, default=0.5)
    parser.add_argument("--policy_factor", type=float, default=1.0)
    parser.add_argument("--entropy_factor", type=float, default=0.01)
    parser.add_argument("--max_norm", type=float, default=0.5)
    # parser.add_argument("--env_name", type=str, default="PongNoFrameskip-v4")
    # parser.add_argument("--is_atari", type=bool, default=True)
    parser.add_argument("--env_name", type=str, default="CartPole-v0")
    parser.add_argument("--is_atari", type=bool, default=False)
    parser.add_argument("--device_token", default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epoch_length", type=int, default=1000)
    parser.add_argument("--n_eval_episodes", type=int, default=1000)
    args = parser.parse_args()

    env_name = args.env_name
    env_count = args.n_envs
    n_steps = args.n_steps
    gamma = args.gamma
    batch_size = args.batch_size
    is_atari = args.is_atari
    epoch_length = args.epoch_length

    if args.device_token is None:
        device_token = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_token = args.device_token

    device = torch.device(device_token)

    checkpoint_path = "model_checkpoints"
    best_models_path = join(checkpoint_path, "best")

    run_id = f"testrun_{datetime.now():%d%m%Y_%H%M%S}"
    model_id = f"{run_id}"
    writer = SummaryWriter(comment=f"-{run_id}")

    makedirs(checkpoint_path, exist_ok=True)
    makedirs(best_models_path, exist_ok=True)

    env_names = sorted(envs.registry.env_specs.keys())

    if is_atari:
        env = wrap_deepmind(make_atari(env_name))
        state = env.reset()

        preprocessor = SimpleCNNPreProcessor()
        in_t = preprocessor.preprocess(state)
        n_actions = env.action_space.n
        input_shape = tuple(in_t.shape)[1:]
        model = AtariModel(input_shape, n_actions).to(device)

        environments = [wrap_deepmind(make_atari(env_name)) for _ in range(env_count)]
    else:
        env = gym.make(env_name)
        state = env.reset()
        in_states = state.shape[0]
        num_actions = env.action_space.n
        model = MLPModel(in_states, num_actions).to(device)

        preprocessor = NoopPreProcessor()
        environments = [gym.make(env_name) for _ in range(env_count)]

    dataset = EnvironmentsDataset(environments, model, n_steps, gamma, batch_size, preprocessor, device,
                                  epoch_length=epoch_length)

    trainer = ActorCriticTrainer(args, model, model_id, trainer_id=1, writer=writer)
    eval_policy = Policy(model, preprocessor, device)
    trainer.fit(dataset, env, eval_policy)

    print("")


if __name__ == "__main__":
    main()
