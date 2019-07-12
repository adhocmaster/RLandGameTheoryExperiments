from abc import abstractmethod

class Game():
    
    @abstractmethod
    def play(self, steps):
        pass

    @abstractmethod
    def nextStep(self, playerId, stepNo):
        pass
    
    @abstractmethod
    def reward(self, outcome, actionTaken, reward):
        pass