from abc import abstractmethod
from game.RewardManager import RewardManager

class Game(RewardManager):
    
    @abstractmethod
    def play(self, steps):
        pass

    @abstractmethod
    def nextStep(self, playerId, stepNo):
        pass