import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from .actor import Actor
from .critic import Critic
from .rollout_buffer import RolloutBuffer
from .training_config import TrainingConfig

class WeatherAwareMAPPO:
    def __init__(self, obs_dim, global_state_dim, action_dim, config=None, device="cpu"):
        self.config = config or TrainingConfig()
        self.device = device
        
        self.actor = Actor(obs_dim, action_dim, self.config.hidden_size).to(device)
        self.critic = Critic(global_state_dim, self.config.hidden_size).to(device)
        
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.config.learning_rate)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.config.learning_rate)
        
        self.buffer = RolloutBuffer()
        
    def select_actions(self, observations):
        """
        observations: dict {vehicle_id: obs_numpy_array}
        Returns:
            actions: dict {vehicle_id: action_int}
            log_probs: dict {vehicle_id: log_prob_float}
        """
        actions = {}
        log_probs = {}
        with torch.no_grad():
            for vid, obs in observations.items():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                dist = self.actor(obs_t)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
                actions[vid] = action.item()
                log_probs[vid] = log_prob.item()
        return actions, log_probs

    def get_value(self, global_state):
        with torch.no_grad():
            gs_t = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
            value = self.critic(gs_t)
        return value.item()

    def train_step(self):
        if len(self.buffer.observations) == 0:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}
            
        obs_t, gs_t, actions_t, old_log_probs_t, rewards_t, values_t, dones_t = self.buffer.get_data(self.device)
        
        # Calculate advantages using GAE
        advantages = torch.zeros_like(rewards_t).to(self.device)
        lastgaelam = 0
        
        # We need the next value for GAE, assuming value of terminal state is 0
        with torch.no_grad():
            next_value = self.critic(gs_t[-1].unsqueeze(0)).squeeze() # simplified
        
        # Better GAE calculation:
        for t in reversed(range(len(rewards_t))):
            if t == len(rewards_t) - 1:
                nextnonterminal = 1.0 - dones_t[t]
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones_t[t]
                nextvalues = values_t[t+1]
            delta = rewards_t[t] + self.config.gamma * nextvalues * nextnonterminal - values_t[t]
            advantages[t] = lastgaelam = delta + self.config.gamma * self.config.gae_lambda * nextnonterminal * lastgaelam
        
        returns = advantages + values_t
        
        actor_losses = []
        critic_losses = []
        entropies = []
        
        dataset_size = len(obs_t)
        batch_size = self.config.minibatch_size
        
        for _ in range(self.config.epochs):
            for indices in BatchSampler(SubsetRandomSampler(range(dataset_size)), batch_size, drop_last=False):
                mb_obs = obs_t[indices]
                mb_gs = gs_t[indices]
                mb_actions = actions_t[indices]
                mb_old_log_probs = old_log_probs_t[indices]
                mb_advantages = advantages[indices]
                mb_returns = returns[indices]
                
                # Normalize advantages
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Actor loss
                dist = self.actor(mb_obs)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.config.ppo_clip, 1.0 + self.config.ppo_clip) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.config.entropy_coefficient * entropy
                
                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()
                
                # Critic loss
                new_values = self.critic(mb_gs).squeeze(-1)
                critic_loss = nn.MSELoss()(new_values, mb_returns)
                
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()
                
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.item())
                
        self.buffer.clear()
        return {
            "actor_loss": sum(actor_losses) / len(actor_losses) if actor_losses else 0.0,
            "critic_loss": sum(critic_losses) / len(critic_losses) if critic_losses else 0.0,
            "entropy": sum(entropies) / len(entropies) if entropies else 0.0
        }

    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer_actor': self.optimizer_actor.state_dict(),
            'optimizer_critic': self.optimizer_critic.state_dict()
        }, path)

    def load_checkpoint(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor'])
            self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic'])

