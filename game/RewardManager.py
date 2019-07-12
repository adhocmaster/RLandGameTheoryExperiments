from abc import ABC, abstractmethod
class RewardManager(ABC):
    
    @abstractmethod
    def getReward(self, playerId, actionTaken):
        pass