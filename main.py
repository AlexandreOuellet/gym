import argparse
import sys

import gym
from gym import wrappers, logger
import gym_snake



import agents
from agents.random import RandomAgent

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='snake-v0', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)

    env.seed(0)
    agent = RandomAgent(env.action_space)

    episode_count = 100
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        while True:
            # print(ob) # Shows ascii representation
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)

            if done:
                break
            env.render()
