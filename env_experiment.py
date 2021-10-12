from collections import deque

import torch
import torch.nn.functional as F
from gym import envs

from atari_wrappers import wrap_deepmind, make_atari
from data import EnvironmentsDataset
from model import SimplePreProcessor, AtariModel


def main():
    env_name = "PongNoFrameskip-v4"
    env_count = 1
    # env_count = 16
    n_steps = 10
    gamma = 0.99
    batch_size = 10
    # batch_size = 128

    env_names = sorted(envs.registry.env_specs.keys())
    env = wrap_deepmind(make_atari(env_name))
    state = env.reset()

    preprocessor = SimplePreProcessor()
    in_t = preprocessor.preprocess(state)

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
    for xxx in dataset.data():
        batches.append(xxx)
        print("")


def main2():
    yield_test = TestYield()

    while True:
        print("New Round")

        for x in yield_test.yield_test():
            print(x)


class TestYield(object):

    def __init__(self):
        self.counter = 0

    def yield_test(self):
        n = 0
        while True:
            yield self.counter
            self.counter += 1
            n += 1
            if n >= 10:
                return


if __name__ == "__main__":
    main2()
