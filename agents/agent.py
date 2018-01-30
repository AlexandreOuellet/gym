from abc import ABC, abstractmethod


class Agent(ABC):

    def __init__(self, action_space):
        self.action_space = action_space

    @abstractmethod
    def act(self, observation, reward, done):
        pass
