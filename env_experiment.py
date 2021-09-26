import gym
from gym import envs
import cv2 as cv

from atari_wrappers import wrap_deepmind, make_atari


def main():
    env_name = "PongNoFrameskip-v4"
    env_names = sorted(envs.registry.env_specs.keys())

    env1 = gym.make(env_name)
    state1 = env1.reset()

    env2 = wrap_deepmind(gym.make(env_name))
    state2 = env2.reset()

    env3 = wrap_deepmind(make_atari(env_name))
    state3 = env3.reset()

    env = env3
    state = state3

    while True:
        cv.imshow("state", state.squeeze())
        cv.waitKey()
        cv.destroyAllWindows()

        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        print(f"Action {action}, Reward {reward}")

        if done:
            print("Done")
            state = env.reset()
        pass

    pass


if __name__ == "__main__":
    main()
