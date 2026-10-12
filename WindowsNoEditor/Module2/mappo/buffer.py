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
        self.next_global_states = []

    def clear(self):
        self.local_obs.clear()
        self.global_states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.is_terminals.clear()
        self.values.clear()
        self.next_global_states.clear()

    def store_transition(
        self,
        obs,
        global_state,
        action,
        log_prob,
        reward,
        value,
        done,
        next_global_state=None,
    ):
        self.local_obs.append(obs)
        self.global_states.append(global_state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.is_terminals.append(done)
        self.next_global_states.append(
            global_state if next_global_state is None else next_global_state
        )

    def is_full(self, rollout_length):
        return len(self.local_obs) >= rollout_length

