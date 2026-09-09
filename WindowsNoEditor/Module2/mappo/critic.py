import torch
import torch.nn as nn

class CentralizedCritic(nn.Module):
    def __init__(self, global_state_dim):
        super(CentralizedCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, global_state):
        """
        global_state: Tensor of shape (batch_size, global_state_dim)
        returns value estimates of shape (batch_size, 1)
        """
        return self.net(global_state)

