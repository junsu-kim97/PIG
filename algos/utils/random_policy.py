import numpy as np
import torch
from torch import nn


class RandomPolicy(nn.Module):
    def __init__(self, action_space, is_binary=False):
        nn.Module.__init__(self)
        self.action_space = action_space
        self.is_binary = is_binary
        self.discrete = "n" in vars(self.action_space)

    def random(self):
        if self.discrete:
            return np.random.randint(self.action_space.n)
        else:
            low = np.array(self.action_space.low)
            high = np.array(self.action_space.high)
            if self.is_binary:
                return np.random.randint(3, size=self.action_space.shape) - 1
            return np.random.random(size=self.action_space.shape) * (high - low) + low

    def forward(self, obs, *args):
        if isinstance(obs, dict):  # goal conditioned environment
            obs = obs["observation"]
        act = torch.Tensor(np.stack([self.random() for i in range(len(obs))], axis=0))
        if self.discrete:
            act = act.long()
        return act

    def reset(self, i):
        pass
