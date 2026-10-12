import datetime
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .actor import SharedActor
from .buffer import RolloutBuffer
from .critic import CentralizedCritic
from .. import config

logger = logging.getLogger("MAPPOTrainer")


class MAPPOTrainer:
    """Shared-policy MAPPO trainer with a centralized value critic."""

    CKPT_PATH = "mappo_checkpoint.pt"

    def __init__(self, obs_dim, action_dim, global_state_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = SharedActor(obs_dim, action_dim).to(self.device)
        self.critic = CentralizedCritic(global_state_dim).to(self.device)

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=config.LEARNING_RATE_ACTOR
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=config.LEARNING_RATE_CRITIC
        )
        self.buffer = RolloutBuffer()

        self.gamma = config.GAMMA
        self.gae_lambda = config.GAE_LAMBDA
        self.clip_range = config.PPO_CLIP_RANGE
        self.k_epochs = config.K_EPOCHS
        self.mini_batch_size = config.MINI_BATCH_SIZE
        self.checkpoint_dir = config.CHECKPOINT_DIR
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.update_step = 0
        self.last_metrics = None

    def select_action(self, local_obs, deterministic=False):
        """Run the one shared actor once for every local observation row."""
        obs_array = np.asarray(local_obs, dtype=np.float32)
        if obs_array.ndim != 2:
            raise ValueError("local_obs must have shape (num_vehicles, obs_dim)")
        if not np.isfinite(obs_array).all():
            raise ValueError("local_obs contains NaN or Inf")

        obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=self.device)
        action_mask = self._action_masks(obs_array.shape[0])
        with torch.no_grad():
            actions, log_probs = self.actor.sample_action(
                obs_tensor, action_mask, deterministic=deterministic
            )
        return actions.cpu().numpy(), log_probs.cpu().numpy()

    def get_action_log_probs(self, local_obs, actions):
        """Evaluate the effective actions that were actually executed."""
        obs_array = np.asarray(local_obs, dtype=np.float32)
        action_array = np.asarray(actions, dtype=np.int64).reshape(-1)
        if obs_array.ndim != 2 or obs_array.shape[0] != action_array.shape[0]:
            raise ValueError("local observations and actions must have matching agent dimensions")
        if not np.isfinite(obs_array).all():
            raise ValueError("local_obs contains NaN or Inf")

        obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=self.device)
        action_tensor = torch.as_tensor(action_array, dtype=torch.long, device=self.device)
        with torch.no_grad():
            log_probs, _ = self.actor.evaluate_actions(
                obs_tensor, action_tensor, self._action_masks(obs_array.shape[0])
            )
        return log_probs.cpu().numpy()

    def _action_masks(self, num_vehicles):
        """Return valid actions for each ordered platoon slot.

        The actor is shared, but lane changes are leader-only.  Masking invalid
        follower actions keeps the sampled action, executed action, and PPO
        log-probability identical.
        """
        masks = np.ones((num_vehicles, 5), dtype=np.bool_)
        if num_vehicles > 1:
            masks[1:, 3:] = False
        return torch.as_tensor(masks, dtype=torch.bool, device=self.device)

    def get_value(self, global_state):
        state_array = np.asarray(global_state, dtype=np.float32).reshape(1, -1)
        if not np.isfinite(state_array).all():
            raise ValueError("global_state contains NaN or Inf")
        state_tensor = torch.as_tensor(state_array, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            value = self.critic(state_tensor)
        return float(value.squeeze().item())

    def _validate_rollout(self, old_obs, old_states, old_actions, old_logprobs):
        if old_obs.ndim != 3:
            raise ValueError("rollout observations must have shape (T, N, obs_dim)")
        if old_states.ndim != 2:
            raise ValueError("rollout global states must have shape (T, global_state_dim)")
        if old_actions.shape[:2] != old_obs.shape[:2]:
            raise ValueError("action and observation agent dimensions do not match")
        if old_logprobs.shape != old_actions.shape:
            raise ValueError("log probability and action dimensions do not match")
        for tensor in (old_obs, old_states, old_actions, old_logprobs):
            if not torch.isfinite(tensor.float()).all():
                raise ValueError("rollout contains NaN or Inf")

    def train(self):
        """Perform a clipped PPO/MAPPO update over the collected rollout."""
        if len(self.buffer.rewards) == 0:
            return None

        old_obs = torch.as_tensor(
            np.asarray(self.buffer.local_obs), dtype=torch.float32, device=self.device
        )
        old_states = torch.as_tensor(
            np.asarray(self.buffer.global_states), dtype=torch.float32, device=self.device
        )
        next_states = torch.as_tensor(
            np.asarray(self.buffer.next_global_states),
            dtype=torch.float32,
            device=self.device,
        )
        old_actions = torch.as_tensor(
            np.asarray(self.buffer.actions), dtype=torch.long, device=self.device
        )
        old_logprobs = torch.as_tensor(
            np.asarray(self.buffer.log_probs), dtype=torch.float32, device=self.device
        )
        self._validate_rollout(old_obs, old_states, old_actions, old_logprobs)

        rewards = np.asarray(self.buffer.rewards, dtype=np.float32)
        values = np.asarray(self.buffer.values, dtype=np.float32).reshape(-1)
        dones = np.asarray(self.buffer.is_terminals, dtype=np.bool_).reshape(-1)
        if rewards.ndim != 2 or rewards.shape[0] != old_obs.shape[0]:
            raise ValueError("rewards must have shape (T, num_vehicles)")
        if values.shape[0] != old_obs.shape[0] or dones.shape[0] != old_obs.shape[0]:
            raise ValueError("value and done lengths do not match rollout length")
        if not np.isfinite(rewards).all() or not np.isfinite(values).all():
            raise ValueError("rewards or values contain NaN or Inf")

        T, num_vehicles = old_obs.shape[:2]
        with torch.no_grad():
            next_values = self.critic(next_states).squeeze(-1).cpu().numpy()

        # Cooperative MAPPO uses one global value and one team advantage.  The
        # reward is the mean of the correctly indexed per-vehicle rewards.
        mean_rewards = rewards.mean(axis=1)
        advantages_np = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            mask = 0.0 if dones[t] else 1.0
            delta = mean_rewards[t] + self.gamma * next_values[t] * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages_np[t] = gae

        returns_np = advantages_np + values
        advantages = torch.as_tensor(advantages_np, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(returns_np, dtype=torch.float32, device=self.device)
        advantage_std = advantages.std(unbiased=False)
        advantages = (advantages - advantages.mean()) / (advantage_std + 1e-8)
        if not torch.isfinite(advantages).all() or not torch.isfinite(returns).all():
            raise ValueError("advantages or returns contain NaN or Inf")

        flat_obs = old_obs.reshape(T * num_vehicles, -1)
        flat_actions = old_actions.reshape(T * num_vehicles)
        flat_old_logprobs = old_logprobs.reshape(T * num_vehicles)
        flat_advantages = advantages.unsqueeze(1).expand(T, num_vehicles).reshape(-1)
        flat_action_masks = self._action_masks(num_vehicles).repeat(T, 1)

        actor_loss = torch.zeros((), device=self.device)
        entropy_mean = torch.zeros((), device=self.device)
        value_loss = torch.zeros((), device=self.device)

        for _ in range(self.k_epochs):
            actor_order = torch.randperm(T * num_vehicles, device=self.device)
            for start in range(0, T * num_vehicles, self.mini_batch_size):
                indices = actor_order[start : start + self.mini_batch_size]
                logprobs, entropy = self.actor.evaluate_actions(
                    flat_obs[indices], flat_actions[indices], flat_action_masks[indices]
                )
                ratios = torch.exp(logprobs - flat_old_logprobs[indices])
                if not torch.isfinite(ratios).all():
                    raise ValueError("PPO probability ratio contains NaN or Inf")
                batch_advantages = flat_advantages[indices]
                surrogate_1 = ratios * batch_advantages
                surrogate_2 = torch.clamp(
                    ratios, 1.0 - self.clip_range, 1.0 + self.clip_range
                ) * batch_advantages
                actor_loss = -torch.minimum(surrogate_1, surrogate_2).mean()
                entropy_mean = entropy.mean()
                actor_objective = actor_loss - config.ENTROPY_COEF * entropy_mean
                self.actor_optimizer.zero_grad()
                actor_objective.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), config.MAX_GRAD_NORM)
                self.actor_optimizer.step()

            critic_order = torch.randperm(T, device=self.device)
            for start in range(0, T, self.mini_batch_size):
                indices = critic_order[start : start + self.mini_batch_size]
                predicted_values = self.critic(old_states[indices]).squeeze(-1)
                value_loss = nn.functional.mse_loss(predicted_values, returns[indices])
                critic_objective = config.VALUE_LOSS_COEF * value_loss
                self.critic_optimizer.zero_grad()
                critic_objective.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), config.MAX_GRAD_NORM)
                self.critic_optimizer.step()

        self.buffer.clear()
        self.update_step += 1
        self.last_metrics = {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(value_loss.item()),
            "entropy": float(entropy_mean.item()),
            "advantage_mean": float(advantages.mean().item()),
            "advantage_std": float(advantages.std(unbiased=False).item()),
        }
        self._write_training_log(self.last_metrics)
        self.save_checkpoint()
        return dict(self.last_metrics)

    def _write_training_log(self, metrics):
        try:
            actor_grad_norm = sum(
                p.grad.data.norm(2).item() ** 2
                for p in self.actor.parameters()
                if p.grad is not None
            ) ** 0.5
            critic_grad_norm = sum(
                p.grad.data.norm(2).item() ** 2
                for p in self.critic.parameters()
                if p.grad is not None
            ) ** 0.5
            actor_layer_norms = {
                name: param.data.norm(2).item()
                for name, param in self.actor.named_parameters()
                if param.requires_grad
            }
            critic_layer_norms = {
                name: param.data.norm(2).item()
                for name, param in self.critic.named_parameters()
                if param.requires_grad
            }
            now = datetime.datetime.now().strftime("%H:%M:%S")
            with open("mappo_training.log", "w") as handle:
                sep = "=" * 52
                handle.write(sep + "\n")
                handle.write("  MAPPO DRL NEURAL UPDATE  [%s]\n" % now)
                handle.write(sep + "\n")
                handle.write("  Update Step  : %d\n" % self.update_step)
                handle.write("  Actor Loss   : %+0.4f\n" % metrics["actor_loss"])
                handle.write("  Critic Loss  : %0.4f\n" % metrics["critic_loss"])
                handle.write("  Entropy      : %0.4f\n" % metrics["entropy"])
                handle.write(
                    "  Adv mean/std : %+0.4f / %0.4f\n"
                    % (metrics["advantage_mean"], metrics["advantage_std"])
                )
                handle.write("  Actor GradN  : %0.4f\n" % actor_grad_norm)
                handle.write("  Critic GradN : %0.4f\n" % critic_grad_norm)
                handle.write(sep + "\n")
                handle.write("  ACTOR layer weight L2 norms:\n")
                for name, norm in actor_layer_norms.items():
                    handle.write("    %-22s %6.3f\n" % (name, norm))
                handle.write(sep + "\n")
                handle.write("  CRITIC layer weight L2 norms:\n")
                for name, norm in critic_layer_norms.items():
                    handle.write("    %-22s %6.3f\n" % (name, norm))
        except Exception:
            pass

    def get_checkpoint_path(self, path=None):
        if path is None:
            return os.path.join(self.checkpoint_dir, self.CKPT_PATH)
        return path

    def save_checkpoint(self, path=None):
        target_path = self.get_checkpoint_path(path)
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_opt": self.actor_optimizer.state_dict(),
                "critic_opt": self.critic_optimizer.state_dict(),
                "update_step": self.update_step,
                "obs_dim": self.actor.net[0].in_features,
                "action_dim": self.actor.net[-1].out_features,
                "global_state_dim": self.critic.net[0].in_features,
            },
            target_path,
        )
        logger.info(
            "[MAPPO] Saved checkpoint to '%s' (update_step=%d).",
            os.path.abspath(target_path),
            self.update_step,
        )

    def load_checkpoint(self, path=None):
        target_path = self.get_checkpoint_path(path)
        abs_path = os.path.abspath(target_path)
        if not os.path.exists(target_path):
            logger.warning("[MAPPO] Checkpoint file does not exist: %s", abs_path)
            return False
        try:
            checkpoint = torch.load(target_path, map_location=self.device)
            if not isinstance(checkpoint, dict):
                raise ValueError(
                    "Checkpoint at '%s' is not a valid dictionary (got %s)"
                    % (abs_path, type(checkpoint).__name__)
                )

            # Explicitly validate actor observation dimension
            actor_state = checkpoint.get("actor", {})
            if "net.0.weight" in actor_state:
                ckpt_obs_dim = actor_state["net.0.weight"].shape[1]
                expected_obs_dim = self.actor.net[0].in_features
                if ckpt_obs_dim != expected_obs_dim:
                    raise ValueError(
                        "Incompatible actor observation dimension in '%s': "
                        "checkpoint has %d dims (e.g. historical schema), "
                        "but current model requires %d dims."
                        % (abs_path, ckpt_obs_dim, expected_obs_dim)
                    )

            # Explicitly validate critic global state dimension
            critic_state = checkpoint.get("critic", {})
            if "net.0.weight" in critic_state:
                ckpt_critic_dim = critic_state["net.0.weight"].shape[1]
                expected_critic_dim = self.critic.net[0].in_features
                if ckpt_critic_dim != expected_critic_dim:
                    raise ValueError(
                        "Incompatible critic global state dimension in '%s': "
                        "checkpoint has %d dims, but current critic requires %d dims."
                        % (abs_path, ckpt_critic_dim, expected_critic_dim)
                    )

            self.actor.load_state_dict(checkpoint["actor"])
            self.critic.load_state_dict(checkpoint["critic"])
            if "actor_opt" in checkpoint and self.actor_optimizer is not None:
                self.actor_optimizer.load_state_dict(checkpoint["actor_opt"])
            if "critic_opt" in checkpoint and self.critic_optimizer is not None:
                self.critic_optimizer.load_state_dict(checkpoint["critic_opt"])
            self.update_step = checkpoint.get("update_step", 0)
            logger.info(
                "[MAPPO] Successfully loaded checkpoint from '%s' (update_step=%d).",
                abs_path,
                self.update_step,
            )
            return True
        except Exception as exc:
            logger.error(
                "[MAPPO] Failed to load checkpoint from '%s': %s",
                abs_path,
                exc,
                exc_info=True,
            )
            return False
