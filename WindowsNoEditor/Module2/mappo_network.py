import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedActor(nn.Module):
    """
    Shared Actor Policy for all platoon vehicles.
    Takes the local observation of a vehicle and outputs the tactical action logits.
    """
    def __init__(self, obs_dim, action_dim, hidden_size=128):
        super(SharedActor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


class CentralizedCritic(nn.Module):
    """
    Centralized Critic used only during training.
    Takes the global state (concatenation of all local observations + global info)
    and outputs the state value.
    """
    def __init__(self, global_state_dim, hidden_size=256):
        super(CentralizedCritic, self).__init__()
        self.fc1 = nn.Linear(global_state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

