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
from data import EpisodeResult, EnvironmentsDataset
from model import SimplePreProcessor, AtariModel
from utils import save_checkpoint

logger = logging.getLogger(__name__)


# TODO update
def actor_critic_old(env, policy, v, num_iterations=10000, batch_size=32, gamma=0.99, alpha=0.01,
                     summary_writer: SummaryWriter = None, summary_prefix=""):
    i = 0
    total_p_losses = []
    total_v_losses = []
    episode_returns = []
    episode_lengths = []

    while i < num_iterations:
        state = env.reset()
        done = False
        episode_result = EpisodeResult(env, state)

        discount_factor = 1.0
        while not done:
            action = policy(state)
            new_state, reward, done, info = env.step(action)
            episode_result.append(action, reward, state, done)

            v_state, v_new_state = v(state), v(new_state)
            if done:
                delta = reward  # value of terminal states (v_new_state) should be zero
            else:
                delta = reward + gamma * v_new_state

            v.append_x_y_pair(state, delta)
            policy.append(state, action, discount_factor * delta)

            if len(policy.state_batches) > batch_size:
                p_losses = policy.policy_gradient_approximation(batch_size)
                v_losses = v.approximate(batch_size)
                if summary_writer is not None:
                    for l_idx, l in enumerate(p_losses):
                        summary_writer.add_scalar(f"{summary_prefix}batch_policy_loss", l, len(total_p_losses) + l_idx)
                    for l_idx, l in enumerate(v_losses):
                        summary_writer.add_scalar(f"{summary_prefix}batch_value_loss", l, len(total_v_losses) + l_idx)
                total_p_losses.extend(p_losses)
                total_v_losses.extend(v_losses)

            discount_factor *= gamma
            state = new_state

        ep_return = episode_result.calculate_return(gamma)
        ep_length = len(episode_result.states) - 1

        if summary_writer is not None:
            summary_writer.add_scalar(f"{summary_prefix}episode_length", ep_length, len(episode_lengths))
            summary_writer.add_scalar(f"{summary_prefix}episode_return", ep_return, len(episode_returns))

        episode_returns.append(ep_return)
        episode_lengths.append(ep_length)
        last_100_average = np.array(episode_returns[-100:]).mean()

        logger.info(
            f"{i}: Length: {ep_length} \t Return: {ep_return:.2f} \t Last 100 Average: {last_100_average:.2f}")

        i += 1

        if i % 100 == 0:
            print("{} iterations done".format(i))

    p_losses = policy.policy_gradient_approximation(batch_size)
    v_losses = v.approximate(batch_size)
    if summary_writer is not None:
        for l_idx, l in enumerate(p_losses):
            summary_writer.add_scalar(f"{summary_prefix}batch_policy_loss", l, len(total_p_losses) + l_idx)
        for l_idx, l in enumerate(v_losses):
            summary_writer.add_scalar(f"{summary_prefix}batch_value_loss", l, len(total_v_losses) + l_idx)
    total_p_losses.extend(p_losses)
    total_v_losses.extend(v_losses)

    return total_p_losses, total_v_losses


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_envs", type=int, default=16)
    parser.add_argument("--n_steps", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--value_factor", type=float, default=1.0)
    parser.add_argument("--policy_factor", type=float, default=1.0)
    parser.add_argument("--entropy_factor", type=float, default=0.01)
    parser.add_argument("--max_norm", type=float, default=0.5)
    parser.add_argument("--env_name", type=str, default="PongNoFrameskip-v4")
    parser.add_argument("--device_token", default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epoch_length", type=int, default=10000)
    args = parser.parse_args()

    env_name = args.env_name
    env_count = args.n_envs
    n_steps = args.n_steps
    gamma = args.gamma
    batch_size = args.batch_size

    value_factor = args.value_factor
    policy_factor = args.policy_factor
    entropy_factor = args.entropy_factor

    max_norm = args.max_norm

    lr = args.lr

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
    env = wrap_deepmind(make_atari(env_name))
    state = env.reset()

    preprocessor = SimplePreProcessor()
    in_t = preprocessor.preprocess(state)

    n_actions = env.action_space.n
    input_shape = tuple(in_t.shape)[1:]
    model = AtariModel(input_shape, n_actions).to(device)

    batch_id = 0
    train(args, model)

    print("")


class DummySummaryWriter(object):

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        pass


class ActorCriticTrainer(object):

    def __init__(self, config, model, model_id, device, optimizer=None, scheduler=None, checkpoint_path=None,
                 writer=None, batch_wise_scheduler=False):
        self.value_factor = config.value_factor
        self.policy_factor = config.policy_factor
        self.entropy_factor = config.entropy_factor
        self.max_norm = config.max_norm
        self.lr = config.lr

        self.model = model
        self.model_id = model_id
        self.optimizer = optimizer if optimizer is not None else Adam(model.parameters(), lr=self.lr)
        self.writer = writer if writer is not None else DummySummaryWriter()
        self.device = device
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
            self.writer.add_scalar(f"{log_prefix}/lr", current_lr, log_idx)

    def save_checkpoint(self):
        if self.checkpoint_path is None:
            return

        filename = f"{self.model_id}_{self.curr_epoch_idx:03d}.tar"
        path = join(self.checkpoint_path, filename)
        save_checkpoint(path, self.model, self.optimizer)

    def fit(self, dataset_train, dataset_validate, num_epochs=10, training_seed=None):
        if training_seed is not None:
            np.random.seed(training_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.manual_seed(training_seed)

        self.curr_epoch_idx = 0
        self.curr_train_batch_idx = 0
        self.curr_val_batch_idx = 0

        logger.info(f"Starting Training For {num_epochs} epochs.")
        for _ in range(num_epochs):
            logger.info(f"Epoch {self.curr_epoch_idx}")
            logger.info(f"Training")
            self.train(dataset_train)
            logger.info(f"Validation")
            self.validate(dataset_validate)

            if not self.batch_wise_scheduler:
                self.scheduler_step()
            self.curr_epoch_idx += 1
            self.save_checkpoint()

    def train(self, dataset):
        self.model.train()

        epoch_loss = 0.0
        count_batches = 0

        for batch in dataset.batches():
            batch_loss = self.training_step(batch, self.curr_train_batch_idx)
            self.writer.add_scalar("train_batch/loss", batch_loss, self.curr_train_batch_idx)
            logger.info(
                f"Training - Epoch: {self.curr_epoch_idx} Batch: {self.curr_train_batch_idx}: Loss {batch_loss}")
            self.curr_train_batch_idx += 1
            count_batches += 1
            epoch_loss += batch_loss

        epoch_loss /= max(1.0, count_batches)
        self.writer.add_scalar("train_epoch/loss", epoch_loss, self.curr_epoch_idx)
        logger.info(f"Training - Epoch {self.curr_epoch_idx}: Loss {epoch_loss}")

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

        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)

        self.optimizer.step()
        if self.batch_wise_scheduler:
            self.scheduler_step()

        return loss.item()

    def validate(self, dataset):
        self.model.eval()

        epoch_loss = 0.0
        count_batches = 0

        with torch.no_grad():
            for batch in dataset.batches():
                batch_loss = self.validation_step(batch, self.curr_val_batch_idx)
                logger.info(
                    f"Validation - Epoch: {self.curr_epoch_idx} Batch: {self.curr_val_batch_idx}: Loss {batch_loss}")
                self.writer.add_scalar("val_batch/loss", batch_loss, self.curr_val_batch_idx)

                self.curr_val_batch_idx += 1
                count_batches += 1
                epoch_loss += batch_loss

        epoch_loss /= max(1.0, count_batches)

        self.writer.add_scalar("val_epoch/loss", epoch_loss, self.curr_epoch_idx)
        logger.info(f"Validation - Epoch {self.curr_epoch_idx}: Loss {epoch_loss}")

    def validation_step(self, batch, batch_idx):
        model_in, gt = self.get_inputs_and_ground_truth(batch)

        model_out = self.model_output(model_in)
        loss = self.calculate_loss(model_out, gt)

        return loss.item()


def train(config, model, optimizer=None):
    env_name = config.env_name
    env_count = config.n_envs
    n_steps = config.n_steps
    gamma = config.gamma
    batch_size = config.batch_size

    value_factor = config.value_factor
    policy_factor = config.policy_factor
    entropy_factor = config.entropy_factor

    max_norm = config.max_norm

    lr = config.lr

    if config.device_token is None:
        device_token = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_token = config.device_token

    device = torch.device(device_token)

    preprocessor = SimplePreProcessor()

    if optimizer is None:
        optimizer = Adam(model.parameters(), lr=lr)

    environments = [wrap_deepmind(make_atari(env_name)) for _ in range(env_count)]
    dataset = EnvironmentsDataset(environments, model, n_steps, gamma, batch_size, preprocessor, device)

    for batch in dataset.data():
        states_t = torch.cat(batch.states).to(device)
        actions = batch.actions
        values_t = torch.FloatTensor(np.array(batch.values)).to(device)
        advantages_t = torch.FloatTensor(np.array(batch.advantages)).to(device)

        log_probs_out, value_out = model(states_t)
        probs_out = F.softmax(log_probs_out, dim=1)

        value_loss = value_factor * F.mse_loss(value_out.squeeze(), values_t)
        policy_loss = advantages_t * log_probs_out[range(len(probs_out)), actions]
        policy_loss = policy_factor * -policy_loss.mean()
        entropy_loss = entropy_factor * (probs_out * log_probs_out).sum(dim=1).mean()

        loss = entropy_loss + value_loss + policy_loss

        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()
    pass


if __name__ == "__main__":
    main()
