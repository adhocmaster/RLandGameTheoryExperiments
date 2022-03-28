from matplotlib.pyplot import axis
from .Agent import Agent
import numpy as np
from gym.wrappers import LazyFrames
import torch
from .CNN50x50 import CNN50x50
from collections import deque
import random

class ForgetfulAgent(Agent):
    
    def __init__(self, 
                    state_shape, 
                    action_shape, 
                    device = None,
                    ) -> None:
        super().__init__(
            state_shape=state_shape,
            action_shape=action_shape,
            device=device
        )

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99
        self.exploration_rate_min = 0.1
        self.current_step = 0

        self.memory = deque(maxlen=10_000)
        self.batchSize = 32


    #region high-level
    def initNet(self):
        self.net = CNN50x50(self.state_shape, self.action_shape)
        if self.device:
            self.net = self.net.to(self.device)

    def getAction(self, state:LazyFrames) -> int:
        if self._explore():
            action = np.random.randint(self.action_shape)
        else:
            action = self._exploit(state)

        

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate, self.exploration_rate_min)

        self.current_step += 1

        return action


    def learn(self):
        raise Exception("Not implemented learn")

    #endregion

    #region low-level    
    def _explore(self):
        return np.random.rand() < self.exploration_rate

    def _exploit(self, state:LazyFrames) -> int:

        stateArr = state.__array__() # lazy frames to ndarray
        stateTs = torch.tensor(stateArr, device=self.device)

        # add batch dimension
        stateInput = stateTs.unsqueeze(0)
        actionVals = self.net(state, model="online")
        bestAction = torch.argmax(actionVals, axis=1).item()

        return bestAction


    def cache(self, state, next_state, action, reward, done):
        state = state.__array__()
        next_state = next_state.__array__()
        state = torch.tensor(state, device=self.device)
        next_state = torch.tensor(next_state, device=self.device)
        action = torch.tensor([action], device=self.device)
        reward = torch.tensor([reward], device=self.device)
        done = torch.tensor([done], device=self.device)
        self.memory.append((state, next_state, action, reward, done,))


    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch)) # zip(*batch) -> seperate lists of states, next_states, ...
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze() # what is the point of unsqueezing in caching then



    #endregion