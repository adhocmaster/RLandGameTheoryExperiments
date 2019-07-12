'''
In a two agent unit resource game, each player plays simultaneously for a single resource. They don't know each other's action profile. A penalty is given for a collision.
'''
from agent.MemoryAgent import MemoryAgent
from agent.AgentType import AgentType
class UnitResourceGame:

    def __init__(self, numberOfPlayers=2, playerType=AgentType.MEMORY):
        self.numberOfPlayers = numberOfPlayers
        self.playerType = playerType
        self.players = []
        # if playerType == AgentType.MEMORY:
        #     self.players = self.makeMemoryPlayers()
    
        # pass


    def makeMemoryPlayers(self, memorySize, actionSize):
        for i in range(1, self.numberOfPlayers + 1):
            self.players[i] = MemoryAgent(i, memorySize, actionSize, self)

