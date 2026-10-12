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

    @staticmethod
    def _masked_logits(logits, action_mask):
        if action_mask is None:
            return logits
        if action_mask.shape != logits.shape:
            raise ValueError("action_mask must have the same shape as policy logits")
        if not action_mask.any(dim=-1).all():
            raise ValueError("each agent must have at least one valid action")
        return logits.masked_fill(~action_mask, -1e9)

    def sample_action(self, obs, action_mask=None, deterministic=False, return_details=False):
        """Sample or greedily select actions from the shared policy using per-agent validity masks."""
        logits = self._masked_logits(self.forward(obs), action_mask)
        dist = Categorical(logits=logits)
        if deterministic:
            actions = torch.argmax(logits, dim=-1)
        else:
            actions = dist.sample()
        log_probs = dist.log_prob(actions)
        if return_details:
            return actions, log_probs, logits, dist.probs
        return actions, log_probs


    def evaluate_actions(self, obs, actions, action_mask=None):
        """Evaluate actions under the same validity masks used during sampling."""
        logits = self._masked_logits(self.forward(obs), action_mask)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy

