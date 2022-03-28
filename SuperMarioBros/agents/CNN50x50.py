from turtle import forward
import torch
from torch import nn
import copy

class CNN50x50(nn.Module):

    def __init__(self, input_shape, output_shape):
        super().__init__()

        inputChannels, h, w = input_shape
        self.activation = nn.LeakyReLU(0.01)
        self.batch_flattener = nn.Flatten(start_dim=1) # make flat from dim 1 (So, batch dim is kept)

        if h != 50:
            raise ValueError(f"Height must be 50, got {h}")
        if w != 50:
            raise ValueError(f"Width must be 50, got {w}")

        self.online = nn.Sequential(
            nn.Conv2d(inputChannels, 32, 5, 2), #50x50 -> (49-2) / 2 = 23 => 2, 4, ... 46 => (23x23)x32
            self.activation, 
            # nn.Conv2d(32, 64, 2, 2), # 11x11x32 -> 
            # self.activation,
            nn.Conv2d(32, 64, 3, 1), # (23x23)x32 -> 22x22x64
            self.activation,
            nn.Flatten(), # we are not doing batch?
            nn.Linear(22*22*64, 512),
            self.activation,
            nn.Linear(512, output_shape)
        )

        self.target = copy.deepcopy(self.online)

        #freeze target params
        for param in self.target.parameters():
            param.requires_grad = False
        
    
    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        else:
            return self.target(input)



