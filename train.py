import logging
import uuid
from collections import deque
from datetime import datetime
from os import makedirs
from os.path import join

import gym
import torch
import torch.nn.functional as F
from gym import envs
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from atari_wrappers import make_atari, wrap_deepmind
from data import EpisodeResult, EnvironmentsDataset
from model import SimplePreProcessor, AtariModel

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

    env_name = "PongNoFrameskip-v4"
    env_count = 16
    n_steps = 10
    gamma = 0.99
    batch_size = 128
    entropy_beta = 0.01

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
    # in_t = preprocessor.preprocess(state)
    in_t = torch.cat([preprocessor.preprocess(state) for _ in range(16)])

    device_token = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_token)

    n_actions = env.action_space.n
    input_shape = tuple(in_t.shape)[1:]
    in_t = in_t.to(device)
    model = AtariModel(input_shape, n_actions).to(device)
    model.eval()

    with torch.no_grad():
        log_probs_out, value_out = model(in_t)
        probs_out = F.softmax(log_probs_out, dim=1)

    print("")

    environments = [wrap_deepmind(make_atari(env_name)) for _ in range(env_count)]

    dataset = EnvironmentsDataset(environments, model, n_steps, gamma, batch_size, preprocessor, device)

    batches = deque(maxlen=30)
    for batch in dataset.data():
        states_t = torch.cat(batch.states).to(device)
        actions = batch.actions
        values_t = torch.FloatTensor(np.array(batch.values)).to(device)
        advantages_t = torch.FloatTensor(np.array(batch.advantages)).to(device)

        log_probs_out, value_out = model(states_t)
        probs_out = F.softmax(log_probs_out, dim=1)

        value_loss = F.mse_loss(value_out.squeeze(), values_t)
        policy_loss = advantages_t * log_probs_out[range(len(probs_out)), actions]
        policy_loss = -policy_loss.mean()

        entropy_loss = entropy_beta * (probs_out * log_probs_out).sum(dim=1).mean()

        loss = entropy_loss + value_loss + policy_loss
        loss.backward()

        batches.append(batch)
        print("")


def main_old():
    logging.basicConfig(level=logging.INFO)

    checkpoint_path = "model_checkpoints"
    best_models_path = join(checkpoint_path, "best")
    # run_id = f"two_states_in_{datetime.now():%d%m%Y_%H%M%S}"
    # model_id = f"{run_id}"
    # writer = SummaryWriter(comment=f"-{run_id}")
    #
    # makedirs(checkpoint_path, exist_ok=True)
    # makedirs(best_models_path, exist_ok=True)
    #
    # device_token = "cuda" if torch.cuda.is_available() else "cpu"
    # device = torch.device(device_token)
    #
    # n_rows, n_cols, n_to_win = 6, 7, 4
    #
    # if pretrained:
    #     load_checkpoint(pretrained_model_path, model, device=device)
    #     logger.info(f"Loaded pretrained model from \"{pretrained_model_path}\".")
    #
    # optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=l2_regularization)
    # scheduler = MultiStepLR(optimizer, milestones=milestones)
    #
    # curr_epoch_idx = 0
    # curr_train_batch_idx = 0
    # best_model_idx = 0
    # n_total_train_games = 0
    # n_total_eval_games = 0
    #
    # graceful_exit = GracefulExit()
    #
    # while graceful_exit.run:
    #
    #     epoch_loss = 0.0
    #     epoch_policy_loss = 0.0
    #     epoch_value_loss = 0.0
    #     count_batches = 0
    #
    #     model.train()
    #     for _ in range(train_steps):
    #         valid_start_idx = 0
    #
    #         vals_t = torch.tensor([lst[-1].value for lst in batch_samples], device=device, dtype=torch.float32)
    #         probs_t = torch.tensor([lst[-1].probs for lst in batch_samples], device=device, dtype=torch.float32)
    #
    #         log_probs_out, val_out = model(states_t)
    #
    #         loss = value_loss + policy_loss
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #         # TODO scheduler step
    #
    #         batch_loss = loss.item()
    #         writer.add_scalar("batch/loss", batch_loss, curr_train_batch_idx)
    #         writer.add_scalar("batch/policy_loss", policy_loss.item(), curr_train_batch_idx)
    #         writer.add_scalar("batch/value_loss", value_loss.item(), curr_train_batch_idx)
    #
    #         best_model_log = f"Best Model Idx: {best_model_idx - 1} " if best_model_idx > 0 else ""
    #         logger.info(f"Epoch {curr_epoch_idx}: Training - "
    #                     f"{best_model_log}Batch: {curr_train_batch_idx}: Loss {batch_loss}, "
    #                     f"(Policy Loss: {policy_loss.item()}, Value Loss: {value_loss.item()})")
    #
    #         curr_train_batch_idx += 1
    #         count_batches += 1
    #
    #         epoch_loss += batch_loss
    #         epoch_policy_loss += policy_loss.item()
    #         epoch_value_loss += value_loss.item()
    #
    #     epoch_loss /= max(1.0, count_batches)
    #     epoch_policy_loss /= max(1.0, count_batches)
    #     epoch_value_loss /= max(1.0, count_batches)
    #
    #
    #     writer.add_scalar("epoch/loss", epoch_loss, curr_epoch_idx)
    #     writer.add_scalar("epoch/policy_loss", epoch_policy_loss, curr_epoch_idx)
    #     writer.add_scalar("epoch/value_loss", epoch_value_loss, curr_epoch_idx)
    #     logger.info(f"Epoch {curr_epoch_idx}: Loss: {epoch_loss}, "
    #                 f"Policy Loss: {epoch_policy_loss}, Value Loss: {epoch_value_loss}")
    #
    #     curr_epoch_idx += 1
    pass


if __name__ == "__main__":
    main()
