import torch
from torch import nn
from abc import abstractmethod
from  torch.cuda.amp import autocast

class DoubleNet(nn.Module):

    @property
    @abstractmethod
    def name(self):
        raise Exception("Not implemented name")

    @property
    @abstractmethod
    def online(self):
        raise Exception("Not implemented online")

    @property
    @abstractmethod
    def target(self):
        raise Exception("Not implemented target")

    def updateTarget(self):
        self.target.load_state_dict(
            self.online.state_dict()
        )
        
    def forward(self, input, model):
        with autocast():
            if model == "online":
                return self.online(input)
            else:
                return self.target(input)