from .Agent import Agent
import numpy as np
from gym.wrappers import LazyFrames
import torch

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

    #region high-level
    def initNet(self):
        raise Exception("Not implemented initNet")

    def getAction(self, state:LazyFrames) -> int:
        if self._explore():
            return np.random.randint(self.action_shape)
        
        return self._exploit(state)


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
        



    #endregion