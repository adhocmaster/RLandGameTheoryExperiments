"""
Adapted from https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
"""
from turtle import forward
import torch
from torch import nn
import copy
from .DoubleNet import DoubleNet

class CNN84x84(DoubleNet):

    def __init__(self, input_shape, output_shape):
        super().__init__()

        self.name="CNN84x84"

        inputChannels, h, w = input_shape
        self.activation = nn.ReLU()
        self.batch_flattener = nn.Flatten(start_dim=1) # make flat from dim 1 (So, batch dim is kept)

        if h != 84:
            raise ValueError(f"Height must be 84, got {h}")
        if w != 84:
            raise ValueError(f"Width must be 84, got {w}")

        self._online = nn.Sequential(
            nn.Conv2d(inputChannels, 32, 8, 4), #50x50 -> (49-2) / 2 = 23 => 2, 4, ... 46 => output = (23x23)x32, params = 5x5x5x32 + 32 = 
            self.activation, 
            nn.Conv2d(32, 64, 4, 2), # (23x23)x32 -> 21x21x64
            self.activation,
            nn.Conv2d(64, 64, 3, 1), # (23x23)x32 -> 21x21x64
            self.activation,
            self.batch_flattener,
            nn.Linear(3136, 512),
            self.activation,
            nn.Linear(512, output_shape)
        )

        self._target = copy.deepcopy(self._online)

        #freeze target params
        for param in self._target.parameters():
            param.requires_grad = False

            
    @property
    def online(self):
        return self._online

    @property
    def target(self):
        return self._target
        
    



