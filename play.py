import argparse
import logging
import os
from datetime import datetime
from os.path import join

import gym
from gym.wrappers import Monitor
import torch

from atari_wrappers import wrap_deepmind, make_atari
from data import Policy, EpisodeResult
from envs import SimpleCorridorEnv
from model import SharedMLPModel, NoopPreProcessor, SimpleCNNPreProcessor, CNNModel
from utils import load_checkpoint


logger = logging.getLogger(__name__)


def play_environment(env, policy, num_episodes=100, render=False, gamma=1.0, video_save_path=None, run_id=None):
    i = 0
    best_return = float("-inf")
    best_result = None
    episode_returns = []

    if video_save_path is not None:
        sub_folder = f"{datetime.now():%d%m%Y_%H%M%S}" if run_id is None else str(run_id)
        sub_path = join(video_save_path, sub_folder)
        os.makedirs(sub_path, exist_ok=True)
        env = Monitor(env, sub_path, video_callable=lambda x: True, force=True)

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

        episode_return = episode_result.calculate_return(gamma)
        if best_return < episode_return:
            best_return = episode_return
            best_result = episode_result
            logger.info("New best return: {}".format(best_return))

        episode_returns.append(episode_return)
        logger.info(f"Episode {i} Length & Return: {len(episode_result.states)} {episode_return:.3f}")

        i += 1

    return episode_returns, best_result, best_return


def evaluate_model():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--video_path")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--env_name", type=str, default="PongNoFrameskip-v4")
    parser.add_argument("--render", type=bool, default=True)
    parser.add_argument("--device_token", default=None)
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--partial_unroll", dest="partial_unroll", action="store_true")
    parser.add_argument("--no_partial_unroll", dest="partial_unroll", action="store_false")
    parser.add_argument("--atari", dest="atari", action="store_true")
    parser.add_argument("--no_atari", dest="atari", action="store_false")
    parser.set_defaults(atari=True, partial_unroll=True)

    args = parser.parse_args()

    env_name = args.env_name
    atari = args.atari
    model_path = args.model_path
    gamma = args.gamma
    render = args.render
    video_path = args.video_path
    num_episodes = args.n_episodes
    run_id = args.run_id

    if args.device_token is None:
        device_token = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_token = args.device_token

    device = torch.device(device_token)

    if env_name == "SimpleCorridor":
        env = SimpleCorridorEnv()
        state = env.reset()
        in_states = state.shape[0]
        num_actions = env.action_space.n
        model = SharedMLPModel(in_states, num_actions).to(device)

        preprocessor = NoopPreProcessor()
        env = SimpleCorridorEnv()
    elif atari:
        env = wrap_deepmind(make_atari(env_name))
        state = env.reset()

        preprocessor = SimpleCNNPreProcessor()
        in_t = preprocessor.preprocess(state)
        n_actions = env.action_space.n
        input_shape = tuple(in_t.shape)[1:]
        model = CNNModel(input_shape, n_actions).to(device)

        env = wrap_deepmind(make_atari(env_name))
    else:
        env = gym.make(env_name)
        state = env.reset()
        in_states = state.shape[0]
        num_actions = env.action_space.n
        model = SharedMLPModel(in_states, num_actions).to(device)

        preprocessor = NoopPreProcessor()
        env = gym.make(env_name)

    load_checkpoint(model_path, model, device=device)

    policy = Policy(model, preprocessor, device)

    play_environment(env, policy, num_episodes=num_episodes, render=render, gamma=gamma, video_save_path=video_path,
                     run_id=run_id)


if __name__ == "__main__":
    evaluate_model()
