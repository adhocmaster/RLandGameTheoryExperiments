import numpy as np
from library.initKeras import *

class MemoryAgent:

    def __init__(self, agentId, memorySize, actionSize, rewardManager, gamma=0.5, alpha = 0.9, epsilon = 0.2, explorationStrategy = None):
        self.id = agentId
        self.memory = np.zeros(memorySize, dtype=np.float)
        self.actions = np.arange(actionSize).astype(float)
        self.rewardManager = rewardManager
        self.epsilon = epsilon
        self.explorationStrategy = explorationStrategy
        self.gamma = gamma
        self.alpha = alpha
        self.v = np.zeros(memorySize)
        self.q = self.initQModel()
        self.policy = self.initPolicy()
    
    def takeAction(self, actionNo):
        self.memory[:-1] = self.memory[1:]
        self.memory[-1] = actionNo
    
    def takeNoAction(self, noActionNo = 0):
        self.memory[:-1] = self.memory[1:]
        self.memory[-1] = noActionNo

    def initQModel(self):
        # a simple ffn

        model_input = layers.Input(shape = (len(self.memory) + 1, )) # 1 for action. We are assuming actions are single valued integers.
        x = layers.Dense(16)(model_input)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Dense(32)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Dense(16)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.Dropout(0.2)(x)

        output = layers.Dense(1)(x)

        model = models.Model(model_input, output, name = f'QA_{self.id}')
        model.compile(optimizer=optimizers.Adam(lr=0.001),
             loss = losses.MSE,
             metrics = [metrics.MSE, metrics.MAE])
        
        return model
    
    def initPolicy(self):

        model_input = layers.Input(shape = (len(self.memory), )) # 1 for action. We are assuming actions are single valued integers.
        x = layers.Dense(16)(model_input)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Dense(32)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Dense(16)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.Dropout(0.2)(x)

        output = layers.Dense(len(self.actions), actvation=activations.softmax)

        model = models.Model(model_input, output, name = f'QA_{self.id}')
        model.compile(optimizer=optimizers.Adam(lr=0.001),
             loss = losses.categorical_crossentropy,
             metrics = [metrics.categorical_crossentropy, metrics.categorical_accuracy])
        
        return model

    def getAction(self, state):

        # consider epsilon.
        if np.random.choice(2, 1, p=[self.epsilon, 1 - self.epsilon])[0] == 0:
            return np.random.choice(self.actions, 1)[0]
        
        # else get the best action
        else:
            _,a = self.getBestQA(state)
            return a


    def getBestQA(self, state):

        maxV = float('-inf')
        maxA = -1
        for action in self.actions:
            q_s_a = self.getQVal(state, action) 
            if q_s_a > maxV:
                maxV = q_s_a
                maxA = action
        
        return maxV, maxA

    def getQVal(self, state, action):
        #TODO improve this method with all action predictions

        return self.q.predict([np.append(state, action)])[0]


    def updateQ(self, oldState, newState, actionTaken):
        
        # Update q values Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Q[new_state, :]) — Q[state, action]

        # 1 update q(oldState,actionTaken)

        q_oldS_a = self.getQVal(oldState, actionTaken)
        newVmax, newAmax = self.getBestQA(newState)

        newQval = q_oldS_a + self.alpha * (self.rewardManager.getReward(self.id, actionTaken) + self.gamma * newVmax - q_oldS_a )
        



        # 2 update policy for newState?


