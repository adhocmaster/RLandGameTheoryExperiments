from abc import abstractmethod
from gym.wrappers import LazyFrames

class Agent:

    def __init__(self, state_shape, action_shape, device = None) -> None:
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        pass

    @abstractmethod
    def initNet(self):
        raise Exception("Not implemented initNet")

    @abstractmethod
    def getAction(self, state:LazyFrames) -> int:
        raise Exception("Not implemented getAction")


    @abstractmethod
    def learn(self):
        raise Exception("Not implemented learn")