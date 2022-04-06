from matplotlib.pyplot import axis
import torch
from .Agent import Agent
from abc import abstractmethod
import numpy as np
from collections import deque
from .DoubleNet import DoubleNet
from gym.wrappers import LazyFrames
import random
import logging
import os
from datetime import date


class DQNAgent(Agent):

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
        self.exploration_rate_decay = 0.9999995
        self.exploration_rate_min = 0.1
        self.current_step = 0

        self.memory = deque(maxlen=10_000)
        self.batchSize = 32


    @property
    @abstractmethod
    def net(self) -> DoubleNet:
        return self._net

    @property
    @abstractmethod
    def lr(self):
        raise Exception("Not implemented lr")

    @property
    @abstractmethod
    def gamma(self):
        raise Exception("Not implemented gamma")

    @property
    @abstractmethod
    def lossFunc(self):
        raise Exception("Not implemented lossFunc")

    @property
    @abstractmethod
    def optimizer(self):
        raise Exception("Not implemented optimizer")

    @abstractmethod
    def learn(self):
        raise Exception("Not implemented learn")

    """
    Q_target = Q_target(next_state, a_best) 

    Q_online(state) = Q_online(state) + lr * (r + gamma * Q_target - Q_online(state))
                      = (1 - lr) * Q_online(state) + lr * (r + gamma * Q_target)
    
    loss = Q_online - Q_target
    Q_online = Q_online + lr * gradient(loss)

    We update the target network periodically

     
    """
    
    def getOnlineQs(self, states, actions):
        """
        Assumes batch
        """
        # raise Exception("Not implemented qOnline")
        currentQ = self.net(states, model="online") [
            np.arange(0, self.batchSize), actions
        ]

        return currentQ

    def getTargetQs(self, rewards, next_states, actions, dones) -> float:
        """
        Assumes batch
        """
        # raise Exception("Not implemented qTarget")
        nextStateQs = self.net(next_states, model="online")
        bestActions = torch.argmax(nextStateQs, axis=1) # axis or dim parameter is the one to reduce. this is different from tensorflow. axis = 0 means batch dim will be reduced. It does not mean arg max will be applied on each row.
        targetQs = self.net(next_states, model="target")[ np.arange(0, self.batchSize), bestActions ]

        return (
            rewards
            + (1 - dones) * self.gamma * targetQs # when done, current state is a terminal state, no future rewards.
        ).float()

    
    def updateQonlineAndGetLoss(self, qOnline, qTarget):

        loss = self.lossFunc(qOnline, qTarget)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def updateQtarget(self):
        logging.debug(f"updating target network")
        self.net.updateTarget()


    def replayExperiences(self):
        
        # step 1: sample self.batchSize experiences
        states, next_states, actions, rewards, dones = self.sampleExperienceBatch()


        # step 2: compute q online and q target

        onlineQs = self.getOnlineQs(states, actions)
        targetQs = self.getTargetQs(
                                    rewards=rewards,
                                    next_states=next_states,
                                    actions=actions,
                                    dones=dones
                                    )

        # step 3: update online net
        loss = self.updateQonlineAndGetLoss(onlineQs, targetQs)

        return onlineQs.mean().item(), loss
    

    #region high-level

    def getAction(self, state:LazyFrames) -> int:
        if self._explore():
            action = np.random.randint(self.action_shape)
        else:
            action = self._exploit(state)

        

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate, self.exploration_rate_min)

        self.current_step += 1

        return action

    
    def getBestAction(self, state:LazyFrames) -> int:
        return self._exploitTarget(state)


    def cache(self, state, next_state, action, reward, done):
        """
        We are converting everything to tensor for GPU computing. Having 
        """
        state = state.__array__()
        next_state = next_state.__array__()
        state = torch.tensor(state, device=self.device)
        next_state = torch.tensor(next_state, device=self.device)
        # action = torch.tensor([action], device=self.device)
        # reward = torch.tensor([reward], device=self.device)
        # done = torch.tensor([done], device=self.device, dtype=torch.float) # no bool
        action = torch.tensor(action, device=self.device)
        reward = torch.tensor(reward, device=self.device)
        done = torch.tensor(done, device=self.device, dtype=torch.float) # no bool
        self.memory.append((state, next_state, action, reward, done,))


    def sampleExperienceBatch(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batchSize)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch)) # zip(*batch) -> seperate lists of states, next_states, ...
        return state, next_state, action, reward, done # what is the point of unsqueezing in caching then

    #endregion

    #region saving

    def save(self, dir, epoch):
        day = date.today().strftime("%b-%d-%Y")
        path = os.path.join(dir, f"{self.name}-checkpoint-{day}-{epoch}.pytorch")

        print(f"saving model to {path}")
        
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "exploration_rate": self.exploration_rate
        }, path)

    def load(self, pathStr, eval=True):
        checkpoint = torch.load(pathStr)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint['epoch']
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.exploration_rate = checkpoint["exploration_rate"]

        if eval:
            self.net.eval()
        else:
            self.net.train()

        return epoch

    #end region



    #region low-level    
    def _explore(self):
        return np.random.rand() < self.exploration_rate

    def _exploit(self, state:LazyFrames) -> int:

        stateArr = state.__array__() # lazy frames to ndarray
        stateTs = torch.tensor(stateArr, device=self.device)

        # add batch dimension
        stateInput = stateTs.unsqueeze(0)
        actionVals = self._net(stateInput, model="online")
        bestAction = torch.argmax(actionVals, axis=1).item()

        return bestAction

    def _exploitTarget(self, state:LazyFrames) -> int:

        stateArr = state.__array__() # lazy frames to ndarray
        stateTs = torch.tensor(stateArr, device=self.device)

        # add batch dimension
        stateInput = stateTs.unsqueeze(0)
        actionVals = self._net(stateInput, model="target")
        bestAction = torch.argmax(actionVals, axis=1).item()

        return bestAction
    #endregion

    