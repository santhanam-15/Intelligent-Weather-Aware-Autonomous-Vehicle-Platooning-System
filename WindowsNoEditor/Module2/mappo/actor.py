import torch
import torch.nn as nn
from torch.distributions import Categorical

class SharedActor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(SharedActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )

    def forward(self, obs):
        """
        obs: Tensor of shape (batch_size, obs_dim) or (num_vehicles, obs_dim)
        returns logits of shape (batch_size, action_dim)
        """
        return self.net(obs)

    def sample_action(self, obs):
        """Samples actions from the policy distribution."""
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs

    def evaluate_actions(self, obs, actions):
        """Evaluates probability and entropy of given actions."""
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy

