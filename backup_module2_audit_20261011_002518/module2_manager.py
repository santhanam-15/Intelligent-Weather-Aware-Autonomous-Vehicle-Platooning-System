import logging
import math
import os

import numpy as np

from . import config
from .action.action_manager import ActionManager
from .integration.module1_adapter import Module1Adapter
from .mappo.trainer import MAPPOTrainer
from .observation.observation_builder import ObservationBuilder
from .reward.reward_manager import RewardManager

logger = logging.getLogger("Module2Manager")


class Module2Manager:
    """Coordinate 15 Hz MAPPO decisions with 30 Hz CARLA control ticks."""

    def __init__(self, uncertainty_manager, num_vehicles, mode="train"):
        if not 1 <= num_vehicles <= config.MAX_PLATOON_SIZE:
            raise ValueError(
                "num_vehicles must be between 1 and MAX_PLATOON_SIZE (%d)"
                % config.MAX_PLATOON_SIZE
            )

        self.num_vehicles = num_vehicles
        self.mode = str(mode).lower()
        if self.mode not in ("train", "evaluate"):
            raise ValueError("mode must be 'train' or 'evaluate'")
        import torch
        torch.set_num_threads(1)
        self.uncertainty_manager = uncertainty_manager
        self.obs_builder = ObservationBuilder(uncertainty_manager)
        self.trainer = MAPPOTrainer(
            obs_dim=self.obs_builder.obs_dim,
            action_dim=5,
            global_state_dim=self.obs_builder.global_state_dim,
        )
        self.action_manager = ActionManager(num_vehicles)
        self.reward_manager = RewardManager(uncertainty_manager)
        self.adapter = Module1Adapter()

        self.last_debug_time = 0.0
        self.last_decision_time = -math.inf
        self.last_tactical_actions = ["CRUISE"] * num_vehicles
        self.last_actions_dict = self.action_manager._to_dict(self.last_tactical_actions)
        self.action_updated = False
        self.pending_transition = None
        self.episode_done = False

        loaded = self.trainer.load_checkpoint()
        ckpt_path = os.path.abspath(self.trainer.get_checkpoint_path())
        if loaded:
            logger.info(
                "[MODULE 2] Loaded persistent checkpoint from '%s' (update_step=%d).",
                ckpt_path,
                self.trainer.update_step,
            )
        else:
            if self.mode == "evaluate":
                raise RuntimeError(
                    "Evaluation mode requires a valid, compatible trained MAPPO checkpoint at '%s'. "
                    "Loading failed or file does not exist; silent fallback to random weights is prohibited."
                    % ckpt_path
                )
            logger.warning(
                "[MODULE 2] No compatible checkpoint loaded from '%s'; starting fresh training with random weights.",
                ckpt_path,
            )

        if self.mode == "evaluate":
            self.trainer.actor.eval()
            self.trainer.critic.eval()
        else:
            self.trainer.actor.train()
            self.trainer.critic.train()

        self.current_local_obs = None
        self.current_global_state = None
        self.current_raw_actions = None
        self.current_log_probs = None
        self.current_value = None

    def set_control_context(self, controller):
        """Expose route/safety state to observations without adding control paths."""
        self.obs_builder.set_control_context(controller)
        self.reward_manager.set_control_context(controller)

    def _collect_state(self, platoon):
        nearby_actors = None
        for v in platoon:
            if v and getattr(v, "is_alive", True):
                try:
                    world = v.get_world()
                    nearby_actors = list(world.get_actors().filter("vehicle.*"))
                except Exception:
                    nearby_actors = None
                break
        local_obs_list = [
            self.obs_builder.get_local_observation(vehicle, index, platoon, nearby_actors=nearby_actors)
            for index, vehicle in enumerate(platoon)
        ]
        global_state = self.obs_builder.get_global_state(local_obs_list)
        return local_obs_list, global_state

    @staticmethod
    def _is_terminal(platoon):
        """Detect actor loss or an actual adjacent-platoon collision."""
        for vehicle in platoon:
            if not vehicle or not vehicle.is_alive:
                return True
        for index in range(1, len(platoon)):
            ahead = platoon[index - 1]
            follower = platoon[index]
            if not ahead or not follower:
                return True
            try:
                if follower.get_location().distance(ahead.get_location()) <= 4.7:
                    return True
            except Exception:
                return True
        return False

    def _store_pending_transition(self, platoon, next_global_state=None):
        if self.pending_transition is None:
            return False

        if next_global_state is None:
            _, next_global_state = self._collect_state(platoon)
        rewards = self.reward_manager.compute_rewards(
            platoon, self.pending_transition["raw_actions"]
        )
        done = self._is_terminal(platoon)
        if len(rewards) != self.num_vehicles:
            raise ValueError("reward count does not match platoon size")

        self.trainer.buffer.store_transition(
            self.pending_transition["local_obs"],
            self.pending_transition["global_state"],
            self.pending_transition["raw_actions"],
            self.pending_transition["log_probs"],
            rewards,
            self.pending_transition["value"],
            done,
            next_global_state=next_global_state,
        )
        self.pending_transition = None
        return done

    def update(self, platoon, current_time):
        """Return cached actions between 15 Hz decisions and collect transitions.

        A transition is closed at the next decision boundary, after CARLA has
        advanced from the previous action.  Thus stored actions, rewards and
        next observations refer to the same environment transition.
        """
        if len(platoon) != self.num_vehicles:
            raise ValueError("platoon size changed during Module 2 execution")

        self.action_updated = False

        if self.episode_done:
            return self.last_tactical_actions, self.last_actions_dict

        if current_time - self.last_decision_time < config.STATE_EXTRACTION_INTERVAL:
            return self.last_tactical_actions, self.last_actions_dict

        local_obs_list, global_state = self._collect_state(platoon)

        if self.pending_transition is not None and self.mode == "train":
            if self._store_pending_transition(platoon, next_global_state=global_state):
                self.episode_done = True
                return self.last_tactical_actions, self.last_actions_dict
        elif self.mode == "evaluate":
            self.pending_transition = None
        raw_actions, log_probs = self.trainer.select_action(
            local_obs_list, deterministic=(self.mode == "evaluate")
        )
        tactical_actions, actions_dict = self.action_manager.process_actions(
            raw_actions, current_time
        )
        effective_raw_actions = np.asarray(
            self.action_manager.last_effective_raw_actions, dtype=np.int64
        )
        if not np.array_equal(effective_raw_actions, np.asarray(raw_actions, dtype=np.int64)):
            # PPO must use the log probability of the action that reached the
            # controller, not the discarded candidate action.
            log_probs = self.trainer.get_action_log_probs(
                local_obs_list, effective_raw_actions
            )
        value = self.trainer.get_value(global_state)

        self.last_decision_time = current_time
        self.last_tactical_actions = list(tactical_actions)
        self.last_actions_dict = actions_dict
        self.current_local_obs = local_obs_list
        self.current_global_state = global_state
        self.current_raw_actions = effective_raw_actions
        self.current_log_probs = np.asarray(log_probs, dtype=np.float32)
        self.current_value = float(value)
        if self.mode == "train":
            self.pending_transition = {
                "local_obs": np.asarray(local_obs_list, dtype=np.float32),
                "global_state": np.asarray(global_state, dtype=np.float32),
                "raw_actions": effective_raw_actions,
                "log_probs": np.asarray(log_probs, dtype=np.float32),
                "value": float(value),
            }
        self.action_updated = True

        if config.DEBUG_MODE and current_time - self.last_debug_time >= config.DEBUG_INTERVAL:
            self.last_debug_time = current_time
            self._print_debug(actions_dict)

        return self.last_tactical_actions, self.last_actions_dict

    def apply_actions(self, tactical_actions, leader_controller, platoon_controller, platoon):
        """Apply the cached individual tactical action to each vehicle."""
        self.adapter.apply_actions(
            tactical_actions, leader_controller, platoon_controller, platoon
        )

    def record_transition(self, platoon):
        """Explicitly flush one pending transition, if a caller needs it."""
        return self._store_pending_transition(platoon)

    def train_if_ready(self):
        if self.mode == "evaluate":
            return None
        if self.trainer.buffer.is_full(config.ROLLOUT_LENGTH):
            logger.info("[MODULE 2] Rollout full. Training MAPPO...")
            return self.trainer.train()
        return None

    def _print_debug(self, actions_dict):
        try:
            with open("mappo_telemetry.log", "w") as handle:
                handle.write("=" * 48 + "\n")
                handle.write("MODULE 2 (MAPPO LIVE DECISIONS)\n")
                handle.write("=" * 48 + "\n")
                for key, action in actions_dict.items():
                    handle.write("Vehicle: %s\n" % key)
                    handle.write("MAPPO Action: %s\n" % action)
                    handle.write("-" * 20 + "\n")
                handle.write("=" * 48 + "\n")
        except Exception:
            pass
