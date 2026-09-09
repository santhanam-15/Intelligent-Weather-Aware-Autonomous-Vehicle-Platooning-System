import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np

from .mappo_network import SharedActor, CentralizedCritic

class MAPPOAgent:
    def __init__(self, obs_dim, action_dim, global_state_dim, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, K_epochs=4, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.actor = SharedActor(obs_dim, action_dim)
        self.critic = CentralizedCritic(global_state_dim)

        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr_actor},
            {'params': self.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = SharedActor(obs_dim, action_dim)
        self.policy_old.load_state_dict(self.actor.state_dict())
        
        self.loss_fn = nn.MSELoss()

        self.buffer = RolloutBuffer()

    def select_action(self, local_obs):
        """
        Select actions for all vehicles in the platoon based on their local observations.
        local_obs: Tensor of shape (num_vehicles, obs_dim)
        """
        with torch.no_grad():
            logits = self.policy_old(local_obs)
            dist = Categorical(logits=logits)
            actions = dist.sample()
            action_logprobs = dist.log_prob(actions)
        
        return actions.numpy(), action_logprobs.numpy()

    def update(self):
        if len(self.buffer.rewards) == 0:
            return

        # Calculate returns and advantages
        rewards = []
        discounted_reward = 0
        # For simplicity in this demo, we assume collective rewards or independent returns
        # self.buffer.rewards is a list of tensors of shape (num_vehicles,)
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Shape: (T, num_vehicles)
        rewards = torch.stack(rewards)
        # Normalize rewards
        rewards = (rewards - rewards.mean(dim=0)) / (rewards.std(dim=0) + 1e-7)
        
        old_states = torch.squeeze(torch.stack(self.buffer.global_states, dim=0)).detach() # (T, global_state_dim)
        old_obs = torch.squeeze(torch.stack(self.buffer.local_obs, dim=0)).detach() # (T, num_vehicles, obs_dim)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach() # (T, num_vehicles)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach() # (T, num_vehicles)

        T, num_vehicles, _ = old_obs.shape

        for _ in range(self.K_epochs):
            # Flatten to evaluate all actors
            flat_obs = old_obs.view(-1, old_obs.shape[-1])
            logits = self.actor(flat_obs)
            dist = Categorical(logits=logits)
            
            flat_actions = old_actions.view(-1)
            logprobs = dist.log_prob(flat_actions)
            dist_entropy = dist.entropy()
            
            logprobs = logprobs.view(T, num_vehicles)
            dist_entropy = dist_entropy.view(T, num_vehicles)
            
            # Evaluate Critic
            # Critic predicts global value. Here we just use mean of individual returns for simplicity,
            # or we could make critic output (num_vehicles) values.
            # Following standard MAPPO, critic outputs 1 value representing the global state value.
            state_values = self.critic(old_states).squeeze() # shape: (T)
            
            # Calculate global returns (mean of all vehicles' returns)
            global_returns = rewards.mean(dim=1) # shape: (T)
            
            # Global advantage
            advantages = global_returns - state_values.detach()
            
            # PPO Actor loss (using global advantage for all agents)
            ratios = torch.exp(logprobs - old_logprobs)
            # Expand advantages to match ratios (T, num_vehicles)
            adv_expanded = advantages.unsqueeze(1).expand_as(ratios)
            
            surr1 = ratios * adv_expanded
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * adv_expanded
            
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * dist_entropy.mean()
            
            # Critic loss
            critic_loss = self.loss_fn(state_values, global_returns)
            
            loss = actor_loss + 0.5 * critic_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.actor.state_dict())
        self.buffer.clear()
        
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
        
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.actor.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


class RolloutBuffer:
    def __init__(self):
        self.global_states = []
        self.local_obs = []
        self.actions = []
        self.logprobs = []
        self.rewards = [] # List of arrays of shape (num_vehicles,)
        self.is_terminals = []

    def clear(self):
        del self.global_states[:]
        del self.local_obs[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

