import torch
import torch.nn as nn
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=64):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )

    def forward(self, obs):
        logits = self.net(obs)
        dist = Categorical(logits=logits)
        return dist

