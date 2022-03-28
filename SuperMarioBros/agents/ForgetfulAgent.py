from matplotlib.pyplot import axis
from .DQNAgent import DQNAgent
import numpy as np
import torch
from .CNN50x50 import CNN50x50
from collections import deque
import random

class ForgetfulAgent(DQNAgent):
    
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

        # self.exploration_rate = 1
        # self.exploration_rate_decay = 0.99
        # self.exploration_rate_min = 0.1
        # self.current_step = 0

        # self.memory = deque(maxlen=10_000)
        self.gamma

        self._optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self._loss_func = torch.nn.SmoothL1Loss()

        # learning parameters
        self.onlinePeriod = 5
        self.targetPeriod = 1e4
        self.burnInPeriod = 1000

    @property
    def net(self):
        return self._net
    
    @property
    def lr(self):
        return 0.0001

    @property
    def gamma(self):
        return 0.9


    @property
    def lossFunc(self):
        return self._loss_func

    @property
    def optimizer(self):
        return self._optim

    def initNet(self):
        self._net = CNN50x50(self.state_shape, self.action_shape)
        if self.device:
            self._net = self._net.to(self.device)



    def learn(self):
        """
        Plan: 
        onlinePeriod - learn every online_period experiences
        targetPeriod - transfer weights from online to target every target_period experiences
        burnInPeriod - gather burnIn_period experiences before starting to learn.

        returns average batch qOnline and average batch loss

        """

        if self.current_step < self.burnInPeriod:
            return None, None
        
        if self.current_step % self.onlinePeriod != 0:
            return None, None

        
        if self.current_step % self.targetPeriod == 0:
            self.net.updateTarget()
        
        return self.replayExperiences()

    #endregion