import torch
import numpy as np

class RolloutBuffer:
    def __init__(self):
        self.observations = []
        self.global_states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
    def add(self, obs, global_state, action, log_prob, reward, value, done):
        self.observations.append(obs)
        self.global_states.append(global_state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.observations.clear()
        self.global_states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        
    def get_data(self, device):
        # returns tensors
        obs_t = torch.FloatTensor(np.array(self.observations)).to(device)
        gs_t = torch.FloatTensor(np.array(self.global_states)).to(device)
        actions_t = torch.LongTensor(np.array(self.actions)).to(device)
        log_probs_t = torch.FloatTensor(np.array(self.log_probs)).to(device)
        rewards_t = torch.FloatTensor(np.array(self.rewards)).to(device)
        values_t = torch.FloatTensor(np.array(self.values)).to(device)
        dones_t = torch.FloatTensor(np.array(self.dones)).to(device)
        
        return obs_t, gs_t, actions_t, log_probs_t, rewards_t, values_t, dones_t

