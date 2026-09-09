import torch
import torch.nn as nn
import torch.optim as optim
import os
from .actor import SharedActor
from .critic import CentralizedCritic
from .buffer import RolloutBuffer
from .. import config

class MAPPOTrainer:
    def __init__(self, obs_dim, action_dim, global_state_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = SharedActor(obs_dim, action_dim).to(self.device)
        self.critic = CentralizedCritic(global_state_dim).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.LEARNING_RATE_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.LEARNING_RATE_CRITIC)
        
        self.buffer = RolloutBuffer()
        
        self.gamma = config.GAMMA
        self.gae_lambda = config.GAE_LAMBDA
        self.clip_range = config.PPO_CLIP_RANGE
        self.k_epochs = config.K_EPOCHS
        
        self.checkpoint_dir = config.CHECKPOINT_DIR
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.update_step = 0

    def select_action(self, local_obs):
        """Returns actions, log_probs, and value estimates."""
        obs_tensor = torch.FloatTensor(local_obs).to(self.device)
        with torch.no_grad():
            actions, log_probs = self.actor.sample_action(obs_tensor)
        return actions.cpu().numpy(), log_probs.cpu().numpy()

    def get_value(self, global_state):
        state_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = self.critic(state_tensor)
        return value.item()

    def train(self):
        if len(self.buffer.rewards) == 0:
            return

        # Convert lists to tensors
        old_obs = torch.FloatTensor(self.buffer.local_obs).to(self.device) # (T, num_vehicles, obs_dim)
        old_states = torch.FloatTensor(self.buffer.global_states).to(self.device) # (T, global_state_dim)
        old_actions = torch.LongTensor(self.buffer.actions).to(self.device) # (T, num_vehicles)
        old_logprobs = torch.FloatTensor(self.buffer.log_probs).to(self.device) # (T, num_vehicles)
        
        rewards = self.buffer.rewards
        values = self.buffer.values
        dones = self.buffer.is_terminals

        T = len(rewards)
        num_vehicles = old_obs.shape[1]

        # Calculate GAE
        returns = []
        advantages = []
        gae = 0
        
        for t in reversed(range(T)):
            next_value = values[t + 1] if t + 1 < T else 0.0
            mask = 1.0 - float(dones[t])
            
            # Use mean reward across vehicles for global critic update
            mean_reward = sum(rewards[t]) / max(1, len(rewards[t]))
            
            delta = mean_reward + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            
            returns.insert(0, gae + values[t])
            advantages.insert(0, gae)

        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Optimize for K epochs
        for _ in range(self.k_epochs):
            # Flatten for actor
            flat_obs = old_obs.view(-1, old_obs.shape[-1])
            flat_actions = old_actions.view(-1)
            
            logprobs, dist_entropy = self.actor.evaluate_actions(flat_obs, flat_actions)
            logprobs = logprobs.view(T, num_vehicles)
            dist_entropy = dist_entropy.view(T, num_vehicles)

            # State values for critic
            state_values = self.critic(old_states).squeeze()

            # Ratios
            ratios = torch.exp(logprobs - old_logprobs)
            
            # Expanded advantages for actor (each vehicle uses global advantage)
            adv_expanded = advantages.unsqueeze(1).expand_as(ratios)

            surr1 = ratios * adv_expanded
            surr2 = torch.clamp(ratios, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv_expanded
            
            actor_loss = -torch.min(surr1, surr2).mean() - config.ENTROPY_COEF * dist_entropy.mean()
            critic_loss = nn.MSELoss()(state_values, returns)
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), config.MAX_GRAD_NORM)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), config.MAX_GRAD_NORM)
            self.critic_optimizer.step()

        self.buffer.clear()
        self.update_step += 1
        
        if self.update_step % config.CHECKPOINT_INTERVAL_UPDATES == 0:
            self.save_checkpoint()

    def save_checkpoint(self):
        path = os.path.join(self.checkpoint_dir, f"mappo_ckpt_{self.update_step}.pt")
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_opt': self.actor_optimizer.state_dict(),
            'critic_opt': self.critic_optimizer.state_dict(),
            'update_step': self.update_step
        }, path)

