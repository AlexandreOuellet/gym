import argparse
import sys

import gym
from gym import wrappers, logger

from collections import deque

import agents
from agents.random import RandomAgent
from agents.dqnagent import DQNAgent

from multiprocessing import Process, Queue, Lock

experience_length = 10000
experience = deque(maxlen=experience_length)
lock = Lock()

def train(experience, lock, observation_space, action_space):
    print("Creating Learner Agent!")
    learner = DQNAgent(observation_space, action_space, memory_length=200, epsilon=0.5, nb_hidden_layer=16, layer_width=12, min_episode_before_acting=0)
    print("learner agent created!")
    
    while True:
        print("acquiring lock")
        with lock:
            try:                
                print("training!")
                learner.train(8, 10, provided_replay=experience, verbose=1)
                print("training completed!")
            except Exception as e:
                print(e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v0', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)

    env.seed(0)

    agent = DQNAgent(env.observation_space, env.action_space, memory_length=200, epsilon=0.5, nb_hidden_layer=16, layer_width=12, min_episode_before_acting=0)

    p = Process(target=train, args=(experience, lock, env.observation_space, env.action_space))
    # p.daemon = True
    # p.start()
    # p.join()

    current_episode = 0
    episode_count = 1000

    while True:
        current_episode += 1
        reward = 0
        done = False
        episode_score = 0

        state = env.reset()

        while not done:
            action = agent.act(state, True)
            next_state, reward, done, _ = env.step(action)
            experience_datapoint = agent.remember(state, action, reward, next_state, done)
            # experience.append(experience_datapoint)

            state = next_state

            episode_score += reward

            if done:
                print('Episode: {}\t Epsilon: {}\t Score: {}\t'.format(current_episode, agent.epsilon, episode_score))
                experience.extend(agent._memory)

                # for s in agent._memory:
                #     experience.append(s)
                if len(experience) >= experience_length - 200:
                    print('Training!')
                    agent.train(batch_size=1024, epochs=10, verbose=1, provided_replay=experience)
                    agent.getUpdatedWeights()

                    experience = deque(maxlen=experience_length)