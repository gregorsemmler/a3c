import argparse
import logging
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
from data import EnvironmentsDataset, Policy
from envs import SimpleCorridorEnv
from model import SimpleCNNPreProcessor, AtariModel, NoopPreProcessor, SharedMLPModel
from play import play_environment
from utils import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


class DummySummaryWriter(object):

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        pass


class ReturnScheduler(object):

    def __init__(self, optimizer, milestones, factor=0.1):
        self.optimizer = optimizer
        self.milestones = np.sort(milestones)
        self.begin_lr = self.get_lr()
        self.factor = factor

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def set_lr(self, new_lr):
        for g in self.optimizer.param_groups:
            g["lr"] = new_lr

    def step(self, returns):
        idx = 0
        for idx in range(len(self.milestones)):
            if returns <= self.milestones[idx]:
                break

        if idx > 0:
            new_lr = self.begin_lr * self.factor ** idx
            self.set_lr(new_lr)
        else:
            self.set_lr(self.begin_lr)


class ActorCriticTrainer(object):

    def __init__(self, config, model, model_id, trainer_id=None, optimizer=None, scheduler=None, checkpoint_path=None,
                 save_optimizer=False, writer=None, batch_wise_scheduler=True, num_mean_results=100,
                 target_mean_returns=None):
        self.value_factor = config.value_factor
        self.policy_factor = config.policy_factor
        self.entropy_factor = config.entropy_factor
        self.max_norm = config.max_norm
        self.lr = config.lr
        self.gamma = config.gamma
        self.undiscounted_log = config.undiscounted_log
        self.num_eval_episodes = config.n_eval_episodes
        self.num_mean_results = num_mean_results
        self.target_mean_returns = target_mean_returns

        if config.device_token is None:
            device_token = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_token = config.device_token
        self.device = torch.device(device_token)

        self.trainer_id = "" if trainer_id is None else str(trainer_id)
        self.model = model
        self.model_id = model_id
        self.optimizer = optimizer if optimizer is not None else Adam(model.parameters(), lr=self.lr)
        self.writer = writer if writer is not None else DummySummaryWriter()

        self.scheduler = scheduler
        self.checkpoint_path = checkpoint_path
        self.save_optimizer = save_optimizer
        self.curr_epoch_idx = 0
        self.curr_train_batch_idx = 0
        self.curr_val_batch_idx = 0
        self.curr_train_episode_idx = 0
        self.count_episodes = 0
        self.batch_wise_scheduler = batch_wise_scheduler
        self.target_reached = False
        self.last_returns = deque(maxlen=self.num_mean_results)

    def scheduler_step(self, metrics=None):
        if self.scheduler is not None:
            current_lr = self.scheduler.get_lr()

            self.scheduler.step(metrics)

            log_prefix = "batch" if self.batch_wise_scheduler else "epoch"
            log_idx = self.curr_train_batch_idx if self.batch_wise_scheduler else self.curr_epoch_idx
            self.writer.add_scalar(f"{log_prefix}/{self.trainer_id}/lr", current_lr, log_idx)

    def save_checkpoint(self, filename=None, best=False):
        if self.checkpoint_path is None:
            return

        if filename is None:
            filename = f"{self.model_id}_{self.curr_epoch_idx:03d}.tar"

        if best:
            path = join(self.checkpoint_path, "best", filename)
        else:
            path = join(self.checkpoint_path, filename)

        if self.save_optimizer:
            save_checkpoint(path, self.model, self.optimizer)
        else:
            save_checkpoint(path, self.model)

    def fit(self, dataset_train, eval_env, eval_policy, num_epochs=None, training_seed=None):
        if training_seed is not None:
            np.random.seed(training_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.manual_seed(training_seed)

        self.curr_epoch_idx = 0
        self.curr_train_batch_idx = 0
        self.curr_val_batch_idx = 0
        self.curr_train_episode_idx = 0
        self.count_episodes = 0
        self.target_reached = False
        self.last_returns.clear()

        if num_epochs is None:
            logger.info(f"{self.trainer_id}# Starting training.")
        else:
            logger.info(f"{self.trainer_id}# Starting training for {num_epochs} epochs.")

        while num_epochs is None or self.curr_epoch_idx < num_epochs:
            logger.info(f"{self.trainer_id}# Epoch {self.curr_epoch_idx}")
            logger.info(f"{self.trainer_id}# Training")
            self.train(dataset_train)

            if self.target_reached:
                logger.info(f"Reached target mean returns. Ending training.")
                self.save_checkpoint(best=True)
                break

            if self.num_eval_episodes > 0:
                logger.info(f"{self.trainer_id}# Validation")
                play_environment(eval_env, eval_policy, num_episodes=self.num_eval_episodes, gamma=self.gamma)

            if not self.batch_wise_scheduler:
                self.scheduler_step()
            self.curr_epoch_idx += 1
            self.save_checkpoint()

    def get_mean_returns(self):
        return 0.0 if len(self.last_returns) == 0 else sum(self.last_returns) / len(self.last_returns)

    def train(self, dataset):
        self.model.train()

        ep_l = 0.0
        ep_p_l = 0.0
        ep_v_l = 0.0
        ep_e_l = 0.0
        ep_episode_length = 0.0
        ep_episode_returns = 0.0
        count_batches = 0
        count_epoch_episodes = 0

        for er_returns, batch in dataset.data():
            b_l, p_l, v_l, e_l = self.training_step(batch, self.curr_train_batch_idx)
            self.writer.add_scalar(f"train_batch/{self.trainer_id}/loss", b_l, self.curr_train_batch_idx)
            self.writer.add_scalar(f"train_batch/{self.trainer_id}/policy_loss", p_l, self.curr_train_batch_idx)
            self.writer.add_scalar(f"train_batch/{self.trainer_id}/value_loss", v_l, self.curr_train_batch_idx)
            self.writer.add_scalar(f"train_batch/{self.trainer_id}/entropy_loss", e_l, self.curr_train_batch_idx)

            batch_ep_len = 0.0
            batch_ep_ret = 0.0

            for length, ret, ret_u in er_returns:
                ep_ret = ret_u if self.undiscounted_log else ret
                self.writer.add_scalar(f"train_batch/{self.trainer_id}/episode_length", length,
                                       self.curr_train_episode_idx)
                self.writer.add_scalar(f"train_batch/{self.trainer_id}/episode_return", ep_ret,
                                       self.curr_train_episode_idx)
                ep_episode_length += length
                ep_episode_returns += ep_ret
                self.curr_train_episode_idx += 1
                self.count_episodes += 1
                count_epoch_episodes += 1
                batch_ep_len += length
                batch_ep_ret += ep_ret
                self.last_returns.append(ep_ret)

            mean_returns = self.get_mean_returns()
            batch_ep_len = None if len(er_returns) == 0 else batch_ep_len / len(er_returns)
            batch_ep_ret = None if len(er_returns) == 0 else batch_ep_ret / len(er_returns)

            log_msg = f"{self.trainer_id}# Epoch: {self.curr_epoch_idx} Batch: {self.curr_train_batch_idx}: " \
                      f"{self.count_episodes} Episodes, Mean{self.num_mean_results} Returns: {mean_returns:.6g}, " \
                      f"Loss: {b_l:.5g} Policy Loss: {p_l:.5g} Value Loss: {v_l:.5g} Entropy Loss: {e_l:.3g} "

            if batch_ep_len is not None:
                log_msg += f"Ep Length: {batch_ep_len:.5g} Ep Return: {batch_ep_ret:.5g}"

            logger.info(log_msg)

            self.curr_train_batch_idx += 1
            count_batches += 1
            ep_l += b_l
            ep_p_l += p_l
            ep_v_l += v_l
            ep_e_l += e_l

            if self.target_mean_returns is not None and mean_returns >= self.target_mean_returns \
                    and len(self.last_returns) >= self.num_mean_results:
                self.target_reached = True
                break

        ep_l /= max(1.0, count_batches)
        ep_p_l /= max(1.0, count_batches)
        ep_v_l /= max(1.0, count_batches)
        ep_e_l /= max(1.0, count_batches)
        ep_episode_length /= max(1.0, count_epoch_episodes)
        ep_episode_returns /= max(1.0, count_epoch_episodes)

        self.writer.add_scalar(f"train_epoch/{self.trainer_id}loss", ep_l, self.curr_epoch_idx)
        self.writer.add_scalar(f"train_epoch/{self.trainer_id}policy_loss", ep_p_l, self.curr_epoch_idx)
        self.writer.add_scalar(f"train_epoch/{self.trainer_id}value_loss", ep_v_l, self.curr_epoch_idx)
        self.writer.add_scalar(f"train_epoch/{self.trainer_id}episode_length", ep_episode_length, self.curr_epoch_idx)
        self.writer.add_scalar(f"train_epoch/{self.trainer_id}episode_return", ep_episode_returns, self.curr_epoch_idx)
        logger.info(f"{self.trainer_id}# Epoch {self.curr_epoch_idx}: Loss: {ep_l:.6g} "
                    f"Policy Loss: {ep_p_l:.6g} Value Loss: {ep_v_l:.6g} Entropy Loss: {ep_e_l:.6g} "
                    f"Episode Length: {ep_episode_length:.6g} Episode Return: {ep_episode_returns:.6g}")

    def training_step(self, batch, batch_idx):
        states_t = torch.cat(batch.states).to(self.device)
        actions = batch.actions
        values_t = torch.FloatTensor(np.array(batch.values)).to(self.device)
        advantages_t = torch.FloatTensor(np.array(batch.advantages)).to(self.device)

        policy_out, value_out = self.model(states_t)

        log_probs_out = F.log_softmax(policy_out, dim=1)
        probs_out = F.softmax(policy_out, dim=1)

        value_loss = self.value_factor * F.mse_loss(value_out.squeeze(-1), values_t)
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
            self.scheduler_step(self.get_mean_returns())

        return loss.item(), policy_loss.item(), value_loss.item(), entropy_loss.item()


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_envs", type=int, default=50)
    parser.add_argument("--n_steps", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scheduler_returns", type=lambda s: [int(e) for e in s.split(",")])
    parser.add_argument("--scheduler_factor", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--l2_regularization", type=float, default=0)
    parser.add_argument("--epoch_length", type=int, default=2000)
    parser.add_argument("--n_eval_episodes", type=int, default=0)
    parser.add_argument("--n_epochs", type=int, default=-1)
    parser.add_argument("--n_mean_results", type=int, default=100)
    parser.add_argument("--target_mean_returns", type=float)
    parser.add_argument("--undiscounted_log", type=bool, default=True)
    parser.add_argument("--value_factor", type=float, default=1.0)
    parser.add_argument("--policy_factor", type=float, default=1.0)
    parser.add_argument("--entropy_factor", type=float, default=0.01)
    parser.add_argument("--max_norm", type=float, default=0.1)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--save_optimizer", type=bool, default=False)
    parser.add_argument("--pretrained_path", default=None)
    parser.add_argument("--env_name", type=str, default="PongNoFrameskip-v4")
    parser.add_argument("--device_token", default=None)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--partial_unroll", dest="partial_unroll", action="store_true")
    parser.add_argument("--no_partial_unroll", dest="partial_unroll", action="store_false")
    parser.add_argument("--atari", dest="atari", action="store_true")
    parser.add_argument("--no_atari", dest="atari", action="store_false")
    parser.set_defaults(atari=True, partial_unroll=True)

    args = parser.parse_args()

    env_name = args.env_name
    env_count = args.n_envs
    n_steps = args.n_steps
    gamma = args.gamma
    batch_size = args.batch_size
    atari = args.atari
    epoch_length = args.epoch_length
    l2_regularization = args.l2_regularization
    target_mean_returns = args.target_mean_returns
    scheduler_milestones = args.scheduler_returns
    scheduler_factor = args.scheduler_factor
    partial_unroll = args.partial_unroll
    lr = args.lr
    eps = args.eps
    checkpoint_path = args.checkpoint_path
    num_epochs = args.n_epochs if args.n_epochs > 0 else None
    run_id = args.run_id if args.run_id is not None else f"run_{datetime.now():%d%m%Y_%H%M%S}"
    num_mean_results = args.n_mean_results
    pretrained_path = args.pretrained_path
    save_optimizer = args.save_optimizer

    if args.device_token is None:
        device_token = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_token = args.device_token

    device = torch.device(device_token)

    model_id = f"{run_id}"
    writer = SummaryWriter(comment=f"-{run_id}")

    if checkpoint_path is not None:
        best_models_path = join(checkpoint_path, "best")
        makedirs(checkpoint_path, exist_ok=True)
        makedirs(best_models_path, exist_ok=True)

    env_names = sorted(envs.registry.env_specs.keys())
    if env_name in envs.registry.env_specs:
        env_spec = envs.registry.env_specs[env_name]
        goal_return = env_spec.reward_threshold

    if env_name == "SimpleCorridor":
        env = SimpleCorridorEnv()
        state = env.reset()
        in_states = state.shape[0]
        num_actions = env.action_space.n
        model = SharedMLPModel(in_states, num_actions).to(device)

        preprocessor = NoopPreProcessor()
        environments = [SimpleCorridorEnv() for _ in range(env_count)]
    elif atari:
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
        model = SharedMLPModel(in_states, num_actions).to(device)

        preprocessor = NoopPreProcessor()
        environments = [gym.make(env_name) for _ in range(env_count)]

    dataset = EnvironmentsDataset(environments, model, n_steps, gamma, batch_size, preprocessor, device,
                                  epoch_length=epoch_length, partial_unroll=partial_unroll)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=l2_regularization, eps=eps)

    if pretrained_path is not None:
        if save_optimizer:
            load_checkpoint(pretrained_path, model, optimizer=optimizer, device=device)
        else:
            load_checkpoint(pretrained_path, model, device=device)
        logger.info(f"Loaded model from {pretrained_path}")

    if scheduler_milestones is not None:
        scheduler = ReturnScheduler(optimizer, scheduler_milestones, scheduler_factor)
    else:
        scheduler = None

    trainer = ActorCriticTrainer(args, model, model_id, trainer_id=1, writer=writer, optimizer=optimizer,
                                 num_mean_results=num_mean_results, target_mean_returns=target_mean_returns,
                                 checkpoint_path=checkpoint_path, scheduler=scheduler)
    eval_policy = Policy(model, preprocessor, device)
    trainer.fit(dataset, env, eval_policy, num_epochs=num_epochs)

    print("")


if __name__ == "__main__":
    main()
