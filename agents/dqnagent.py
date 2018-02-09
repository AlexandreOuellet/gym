import argparse
import sys

import gym
from gym import logger
import gym_snake

from agents.agent import Agent

from keras.models import Sequential
from keras.layers import Dense, Input, Embedding, Conv2D, Flatten, Activation, MaxPooling2D, Dropout
from keras.optimizers import Adam

import numpy as np


class DQNAgent(Agent):
    def __init__(self, action_space, input_shape):
        super().__init__(action_space)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.model = self._build_model(input_shape)

    def _build_model(self, input_shape):

        asdf = Input(shape=(input_shape[0], input_shape[1], input_shape[2]), name='input')

        model = Conv2D(32, (8, 8), name='conv1', activation='relu')(asdf)
        # guylaine = MaxPooling2D(pool_size=(2, 2), name='guylaine_maxpool1')(guylaine)

        model = Conv2D(64, (3, 3), name='conv2', activation='relu')(model)
        # guylaine = MaxPooling2D(pool_size=(2, 2), name='guylaine_maxpool2')(guylaine)

        # model = Conv2D(64, (2, 2), name='conv3', activation='relu')(model)

        model = Flatten()(model)

        model = Dense(512, name='dense1', activation='relu')(model)

        output = Dense(self.action_space.n, name='output', activation='relu')(model)
        return output

    def act(self, observation, reward, done):
        return self.action_space.sample()
        act_values = self.model.predict(observation)
        return np.argmax(act_values[0])  # returns action
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='snake-v0', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)

    env.seed(0)

    # samples = 2
    channel = 1
    rows = columns = 10
    agent = DQNAgent(env.action_space, [rows, columns, channel])

    episode_count = 1000
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
            # env.render()
