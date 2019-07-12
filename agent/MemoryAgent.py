import numpy as np
from collections import deque 
from library.initKeras import *

class MemoryAgent:

    def __init__(self, agentId, memorySize, actionSize, experienceSize=0, experienceReplayRatio=0., gamma=0.5, alpha = 0.9, epsilon = 0.2, explorationStrategy = None):

        self.floatx = 'float16'
        k.set_floatx(self.floatx)
        k.set_epsilon(1e-4)


        self.id = agentId
        self.memorySize = memorySize # number of items in the state.
        self.memory = deque(np.zeros(memorySize, dtype=np.float16), maxlen=memorySize)

        self.actions = np.arange(actionSize).astype(self.floatx)

        self.epsilon = epsilon # for epsilon explorationStrategy
        self.explorationStrategy = explorationStrategy

        self.gamma = gamma # discount
        self.alpha = alpha # learning rate
        self.v = np.zeros(memorySize)
        self.q = self.initQModel()
        self.policy = self.initPolicy()
        # self.memory = np.zeros(memorySize, dtype=self.floatx)
        # if memorySize > experienceSize:
        #     self.memory = np.zeros(memorySize, dtype=self.floatx)
        # else:
        #     self.memory = np.zeros(experienceSize, dtype=self.floatx)

        self.experienceReplayRatio = experienceReplayRatio
    
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

        model = models.Model(model_input, output, name = f'A_{self.id}')
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

    def getCurrentState(self):
        return np.asarray(self.memory)

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


    def updateQ(self, oldState, newState, actionTaken, reward):
        
        # Update q values Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Q[new_state, :]) — Q[state, action]

        # 1 update q(oldState,actionTaken)

        q_oldS_a = self.getQVal(oldState, actionTaken)
        newVmax, newAmax = self.getBestQA(newState)

        newQval = q_oldS_a + self.alpha * (reward + self.gamma * newVmax - q_oldS_a)
        
        self.trainQModel(oldState, actionTaken, newQval)

        pass


    def trainQModel(self, state, action, newQval):

        X = np.append(state, [action]).reshape(-1, 1)
        self.q.fit(X, newQval, epochs=1, batch_size=1)

    
    def move(self, stepNo):
        # return action
        return self.getAction(self.getCurrentState)
    
    def reward(self, outcome, actionTaken, reward):
        # this player got this reward for this action

        oldState = self.getCurrentState()
        self.memory.append(outcome) # this is the current state now
        newState = self.getCurrentState()
        self.updateQ(oldState, newState, actionTaken, reward)
