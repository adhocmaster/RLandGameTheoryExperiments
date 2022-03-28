import gym
import numpy as np
import torch
from torchvision import transforms as T
from .image_utils import ImageUtils

class EnvWrapperGrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, isFinalTransformer=False):
        super().__init__(env)
        #observation space is in PIL image format
        grayShape = self.observation_space.shape[:2] # first two dimensions, the color dim will be converted to gray
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=grayShape, dtype=np.uint8)
        print("shape after grayscaler:", self.observation_space.shape)
        self.isFinalTransformer = isFinalTransformer

    
    def observation(self, obs):
        # print("gray")
        # print(obs.shape)
        # print(type(obs))
        imgTensor = ImageUtils.permuteHWCtoCHWTensor(obs)
        imgGray = ImageUtils.toTorchGray(imgTensor)
        if self.isFinalTransformer:
            return imgGray.squeeze(0)
        return imgGray

    

class EnvWrapperObservationResizer(gym.ObservationWrapper):
    """Resizes first two dimensions. Assumes observation is in PIL format (HWC)

    Args:
        gym (_type_): _description_
    """

    def __init__(self, env: gym.Env, shape, isFinalTransformer=True) -> None:
        super().__init__(env)
        self.shape = shape
        obsShape = self.shape + self.observation_space.shape[2:]
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obsShape, dtype=np.uint8)
        print("shape after resizer:", self.observation_space.shape)
        self.isFinalTransformer = isFinalTransformer

    
    def observation(self, observation):
        # print("resize")
        # print(observation.shape)
        # print(type(observation))
        
        transforms = T.Compose(
            [
                T.Resize(self.shape),
                T.Normalize(0, 255) # a bit of hack as (x - mean) / std can be used as min-max normalizer here 
            ]
        )

        if self.isFinalTransformer:
            return transforms(observation).squeeze(0)
        return transforms(observation)


class EnvWrapperFactory():

    @staticmethod
    def convert(env, shape, gray=True):
        print("shape before any transformations:", env.observation_space.shape)
        # if gray:
        #     env = EnvWrapperGrayScaleObservation(env)
        env = EnvWrapperGrayScaleObservation(env)
        env = EnvWrapperObservationResizer(env, shape)
        env = gym.wrappers.FrameStack(env, 5)
        print("shape after all transformations:", env.observation_space.shape)
        return env
    






