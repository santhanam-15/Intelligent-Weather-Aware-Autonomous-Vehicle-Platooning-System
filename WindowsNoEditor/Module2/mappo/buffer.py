import torch

class RolloutBuffer:
    def __init__(self):
        self.local_obs = []
        self.global_states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []

    def clear(self):
        self.local_obs.clear()
        self.global_states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.is_terminals.clear()
        self.values.clear()

    def store_transition(self, obs, global_state, action, log_prob, reward, value, done):
        self.local_obs.append(obs)
        self.global_states.append(global_state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.is_terminals.append(done)

    def is_full(self, rollout_length):
        return len(self.local_obs) >= rollout_length

