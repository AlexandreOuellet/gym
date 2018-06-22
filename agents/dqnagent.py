import argparse
import sys

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from collections import deque
import random

import gym
from gym import logger
from gym.spaces import Discrete, Box

from agents.agent import Agent

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np

import tensorflow as tf

# Hyperparamters
seed = 1
activation = 'tanh'
min_episode = 100
epsilon = 1
nb_hidden_layer = 2
layer_width = 16
memory_len = 200
batch_size = 64

np.random.seed(seed)
tf.set_random_seed(seed)

class DQNAgent(Agent):
    def __init__(self, observation_space, action_space, seed=0, min_episode_before_acting=100):
        super().__init__(action_space)

        self._random = random.Random(seed)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self._memory = deque(maxlen=200)

        self._observation_space = observation_space
        self._action_space = action_space

        self._model = self._build_model()

        self._min_episode_before_acting = min_episode_before_acting

    def _build_model(self):

        model = Sequential()

        if type(self._observation_space) is Box:
            input_shape = self._observation_space.shape

        for i in range(nb_hidden_layer):
            if i == 0:
                model.add(Dense(layer_width, name='dense'+i, activation=activation, input_shape=input_shape))
            else:
                model.add(Dense(16, name='dense2', activation=activation))

        if type(self._action_space) is Discrete:
            output_shape = self._action_space.n 
        elif type(self._action_space) is Box:
            output_shape = self._action_space.high.shape

        model.add(Dense(output_shape, name='output', activation='linear'))

        model.compile(loss='mse',
              optimizer=Adam(),
              metrics=['accuracy'])

        return model

    def act(self, observation):
        if self._random.random() <= self.epsilon or self._min_episode_before_acting > 0:
            return self._action_space.sample()
        else:
            state = self._reshapeState(observation)
            act_values = self._model.predict(state)
            return np.argmax(act_values)  # returns action

    def train(self, batch_size=None):
        x_batch, y_batch = [], []
        if batch_size == None:
            batch_size = len(self._memory)
        # minibatch = self._random.sample(
        #     self._memory, min(len(self._memory), batch_size))
        for state, action, reward, next_state, done in self._memory:
            y_target = self._model.predict(state)
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self._model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])
        
        self._model.fit(np.array(x_batch), np.array(y_batch), batch_size=batch_size, verbose=0)

        # don't act before you've received enough information
        if self._min_episode_before_acting > 0:
            self._min_episode_before_acting -= 1
        else:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def remember(self, state, action, reward, next_state, done):
        state = self._reshapeState(state)
        next_state = self._reshapeState(next_state)
        self._memory.append([state, action, reward, next_state, done])
    
    def _reshapeState(self, state):
        return np.reshape(state, (1, self._observation_space.shape[0]))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v0', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)

    env.seed(seed)
    agent = DQNAgent(env.observation_space, env.action_space, seed)

    episode_count = 1000
    reward = 0
    done = False

    scores = deque(maxlen=episode_count)

    solved_score = 200
    first_solved = None
    mean_score_at_250 = None
    mean_score_at_500 = None
    mean_score_at_750 = None

    for i in range(episode_count):
        ob = env.reset()
        score = 0
        while True:
            action = agent.act(ob)
            ob1 = ob
            ob, reward, done, _ = env.step(action)
            ob2 = ob

            agent.remember(ob1, action, reward, ob2, done)
            score += reward

            if done:
                scores.append(score)

                if score >= solved_score and first_solved == None:
                    first_solved = i

                if score == 250:
                    mean_score_at_250 = np.mean(scores)
                elif score == 500:
                    mean_score_at_500 = np.mean(scores)
                if score == 750:
                    mean_score_at_750 = np.mean(scores)

                print('Episode: {}\t Epsilon: {}\t Score: {}\t Mean Score:{}\t First Soled:{}'.format(i, agent.epsilon, score, np.mean(scores), first_200))
                agent.train(batch_size=batch_size)
                break
            env.render()

    print("HyperParameters:")
    print("seed = {}".format(seed))
    print("activation = {}".format(activation))
    print("min_episode = {}".format(min_episode))
    print("epsilon = {}".format(epsilon))
    print("nb_hidden_layer = {}".format(nb_hidden_layer))
    print("layer_width = {}".format(layer_width))
    print("memory_len = {}".format(memory_len))
    print("batch_size = {}".format(batch_size))
    print('Mean Score:{}\t First Solved:{}'.format(np.mean(scores), first_solved))