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

    def __init__(self, uncertainty_manager, num_vehicles, mode="train", checkpoint_path=None):
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

        self.checkpoint_path = checkpoint_path
        self._controller = None
        self.current_decision_map = {}
        loaded = self.trainer.load_checkpoint(checkpoint_path)
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
        self._controller = controller
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

    def _is_terminal(self, platoon):
        """Detect actor loss or an actual adjacent-platoon collision among active vehicles."""
        disabled_set = getattr(self._controller, "disabled_vehicles", set()) if self._controller else set()
        active_live = 0
        for vehicle in platoon:
            if not vehicle or not vehicle.is_alive or vehicle.id in disabled_set:
                continue
            active_live += 1
        if active_live == 0:
            return True

        pred_map = getattr(self._controller, "predecessor_map", None) if self._controller else None
        for index in range(len(platoon)):
            follower = platoon[index]
            if not follower or not follower.is_alive or follower.id in disabled_set:
                continue
            pred_idx = pred_map.get(index) if pred_map is not None else (index - 1 if index > 0 else None)
            if pred_idx is not None and 0 <= pred_idx < len(platoon):
                ahead = platoon[pred_idx]
                if not ahead or not ahead.is_alive or ahead.id in disabled_set:
                    continue
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
        raw_actions, log_probs, raw_logits, action_probs = self.trainer.select_action(
            local_obs_list, deterministic=(self.mode == "evaluate"), return_details=True
        )

        active_lead_idx = getattr(self._controller, "active_leader_index", 0) if self._controller else 0
        disabled_set = getattr(self._controller, "disabled_vehicles", set()) if self._controller else set()
        disabled_indices = {i for i, v in enumerate(platoon) if v and v.id in disabled_set}

        tactical_actions, actions_dict = self.action_manager.process_actions(
            raw_actions,
            current_time,
            active_leader_index=active_lead_idx,
            disabled_indices=disabled_indices,
        )

        frame = 0
        try:
            for v in platoon:
                if v and getattr(v, "is_alive", False):
                    frame = v.get_world().get_snapshot().frame
                    break
        except Exception:
            pass

        decision_map = {}
        for i, veh in enumerate(platoon):
            vid = veh.id if (veh and getattr(veh, "is_alive", False)) else i
            dec_id = "DEC-%d-%d-%d" % (frame, vid, int(current_time * 1000))
            if vid in disabled_set:
                role_str = "Disabled Leader"
            elif i == active_lead_idx:
                role_str = "Active Leader" if i > 0 else "Leader"
            else:
                role_str = f"Follower {i}"
            decision_map[vid] = {
                "decision_id": dec_id,
                "frame": frame,
                "time": current_time,
                "veh_id": vid,
                "role": role_str,
                "checkpoint_path": self.trainer.active_checkpoint_path or self.trainer.get_checkpoint_path(),
                "checkpoint_update": self.trainer.update_step,
                "obs_shape": tuple(local_obs_list[i].shape),
                "obs_finite": bool(np.isfinite(local_obs_list[i]).all()),
                "inference_mode": self.mode,
                "raw_logits": [round(float(x), 4) for x in raw_logits[i]],
                "action_probs": [round(float(x), 4) for x in action_probs[i]],
                "sampled_action": int(raw_actions[i]),
                "tactical_action": tactical_actions[i],
            }
        self.current_decision_map = decision_map
        if self._controller is not None and hasattr(self._controller, "set_decision_telemetry"):
            self._controller.set_decision_telemetry(decision_map)
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
