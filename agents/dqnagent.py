import argparse
import sys

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

from collections import deque
import random

import gym
from gym import logger
from gym.spaces import Discrete, Box

from agents.agent import Agent

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import plot_model


from keras import backend as K


from gym.wrappers import Monitor

import numpy as np

import tensorflow as tf

import uuid

import random as rn
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# tf.set_random_seed(1234)
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)



seed=0
min_episode_before_acting=0
activation="tanh"
epsilon=1
layer_width=16
nb_hidden_layer=4
memory_length=256*10
experiment_id="{}".format(uuid.uuid4())
batch_size=32

class DQNAgent(Agent):
    def __init__(self, observation_space, action_space,
        seed=0,
        min_episode_before_acting=0,
        activation="tanh",
        epsilon=1,
        layer_width=16,
        nb_hidden_layer=4,
        memory_length=200):

        np.random.seed(seed)
        tf.set_random_seed(seed)

        super().__init__(action_space)

        self._random = random.Random(seed)

        self.gamma = 0.95    # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self._memory = deque(maxlen=memory_length)

        self._observation_space = observation_space
        self._action_space = action_space

        self._model = self._build_model(nb_hidden_layer, layer_width, activation)
        # self._gpu_model = self._build_model(nb_hidden_layer, layer_width, activation, True)
        self._gpu_model = self._model

        # plot_model(self._model, to_file='cpuModel.png')
        # plot_model(self._gpu_model, to_file='gpuModel.png')


        self._min_episode_before_acting = min_episode_before_acting

    def _build_model(self, nb_hidden_layer, layer_width, activation, use_cpu=True):
        device = '/gpu:0'
        if use_cpu:
            device = '/cpu:0'
        
        with tf.device(device):
            model = Sequential()

            if type(self._observation_space) is Box:
                input_shape = self._observation_space.shape

            for i in range(nb_hidden_layer):
                if i == 0:
                    model.add(Dense(layer_width, name='dense{}'.format(i), activation=activation, input_shape=input_shape))
                else:
                    model.add(Dense(layer_width, name='dense{}'.format(i), activation=activation))

            if type(self._action_space) is Discrete:
                output_shape = self._action_space.n 
            elif type(self._action_space) is Box:
                output_shape = self._action_space.high.shape

            model.add(Dense(output_shape, name='output', activation='linear'))

            model.compile(loss='mse',
                optimizer=Adam(),
                metrics=['accuracy'])

        return model

    def act(self, observation, force_predict=False):
        if (self._random.random() <= self.epsilon or self._min_episode_before_acting > 0) and not force_predict:
            return self._action_space.sample()
        else:
            state = self._reshapeState(observation)
            act_values = self._model.predict(state)

            return np.argmax(act_values)  # returns action

    def train(self, batch_size=8, epochs=5, verbose=0, provided_replay=None, lock=None):
        x_batch, y_batch = [], []
        replay = self._memory
        if provided_replay is not None:
            replay = provided_replay

        if batch_size == None:
            batch_size = len(replay)
        # minibatch = self._random.sample(
        #     self._memory, min(len(self._memory), batch_size))

        for state, action, reward, next_state, done in replay:
            y_target = self._model.predict(state)
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self._gpu_model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])
        
        # with tf.device('/gpu:0'):
        self._gpu_model.fit(np.array(x_batch), np.array(y_batch), batch_size=batch_size, verbose=verbose, epochs=epochs)

        if lock is not None:
            lock.acquire(True)
        self._gpu_model.save_weights("./weights", True)
        if lock is not None:
            lock.release()

        # don't act before you've received enough information
        if self._min_episode_before_acting > 0:
            self._min_episode_before_acting -= 1
        else:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def remember(self, state, action, reward, next_state, done):
        state = self._reshapeState(state)
        next_state = self._reshapeState(next_state)
        state = [state, action, reward, next_state, done]
        self._memory.append(state)
        return state
    
    def _reshapeState(self, state):
        return np.reshape(state, (1, self._observation_space.shape[0]))

    def getUpdatedWeights(self):
        self._model.load_weights("./weights")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v0', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)
    # env = Monitor(env, "./monitoring/{}/".format(experiment_id), video_callable=False, force=True, resume=False,
    #              write_upon_reset=False, uid=None, mode=None)

    env.seed(seed)
    agent = DQNAgent(env.observation_space, env.action_space, seed,min_episode_before_acting,activation,epsilon,layer_width,nb_hidden_layer,memory_length)

    scores = deque()
    sw_scores = deque(maxlen=100)

    solved_score = 200
    first_solved = None
    mean_score_at_250 = None
    mean_score_at_500 = None
    mean_score_at_750 = None

    episode = 0
    while len(sw_scores) == 0 or np.mean(sw_scores) < 195:
        episode += 1

        state = env.reset()
        reward = 0
        done = False
        episode_score = 0
        

        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)

            state = next_state

            episode_score += reward

            if done:
                scores.append(episode_score)
                sw_scores.append(episode_score)

                print('Episode: {}\t Epsilon: {}\t Score: {}\t Mean Score:{}\t Sliding Score:{}\t'.format(episode, agent.epsilon, episode_score, np.mean(scores), np.mean(sw_scores)))
                if episode > 100:
                    agent.train(batch_size=batch_size)
                    agent._memory.clear()
                    episode = 0

                break
            # env.render()

    # print("HyperParameters:")
    # print("seed = {}".format(seed))
    # print("activation = {}".format(activation))
    # print("min_episode = {}".format(min_episode))
    # print("epsilon = {}".format(epsilon))
    # print("nb_hidden_layer = {}".format(nb_hidden_layer))
    # print("layer_width = {}".format(layer_width))
    # print("memory_len = {}".format(memory_len))
    # print("batch_size = {}".format(batch_size))
    # print('Mean Score:{}\tat 250:{}\tat 500:{}\tat 750:{}\tFirst Solved:{}'.format(np.mean(scores), mean_score_at_250, mean_score_at_500, mean_score_at_750, first_solved))