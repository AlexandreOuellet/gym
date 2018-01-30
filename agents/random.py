import gym
from gym import logger
from agents.agent import Agent

class RandomAgent(Agent):
    def __init__(self, action_space):
        super().__init__(action_space)

    def act(self, observation, reward, done):
        return self.action_space.sample()