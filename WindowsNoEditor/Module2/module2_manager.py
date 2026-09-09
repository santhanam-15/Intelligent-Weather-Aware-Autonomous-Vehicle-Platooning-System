import time
import logging
from .observation.observation_builder import ObservationBuilder
from .mappo.trainer import MAPPOTrainer
from .action.action_manager import ActionManager
from .reward.reward_manager import RewardManager
from .integration.module1_adapter import Module1Adapter
from . import config

logger = logging.getLogger("Module2Manager")

class Module2Manager:
    def __init__(self, uncertainty_manager, num_vehicles):
        self.num_vehicles = num_vehicles
        self.uncertainty_manager = uncertainty_manager
        
        # Initialize components
        self.obs_builder = ObservationBuilder(uncertainty_manager)
        
        self.trainer = MAPPOTrainer(
            obs_dim=self.obs_builder.obs_dim,
            action_dim=5, # 5 discrete actions
            global_state_dim=self.obs_builder.global_state_dim
        )
        
        self.action_manager = ActionManager(num_vehicles)
        self.reward_manager = RewardManager(uncertainty_manager)
        self.adapter = Module1Adapter()
        
        self.last_debug_time = 0.0
        
        # For rollout collection
        self.current_local_obs = None
        self.current_global_state = None
        self.current_raw_actions = None
        self.current_log_probs = None
        self.current_value = None

    def update(self, platoon, current_time):
        """
        Gathers observations and decides actions.
        Returns actions dict for debugging/Module 3.
        """
        # 1. Build observations
        local_obs_list = []
        for i, vehicle in enumerate(platoon):
            obs = self.obs_builder.get_local_observation(vehicle, i, platoon)
            local_obs_list.append(obs)
            
        global_state = self.obs_builder.get_global_state(local_obs_list)
        
        self.current_local_obs = local_obs_list
        self.current_global_state = global_state
        
        # 2. MAPPO action selection
        raw_actions, log_probs = self.trainer.select_action(local_obs_list)
        self.current_raw_actions = raw_actions
        self.current_log_probs = log_probs
        self.current_value = self.trainer.get_value(global_state)
        
        # 3. Process actions (Persistence / State Machine)
        tactical_actions, actions_dict = self.action_manager.process_actions(raw_actions, current_time)
        
        # Debug Output
        if config.DEBUG_MODE and current_time - self.last_debug_time >= config.DEBUG_INTERVAL:
            self.last_debug_time = current_time
            self._print_debug(actions_dict)
            
        return tactical_actions, actions_dict

    def apply_actions(self, tactical_actions, leader_controller, platoon_controller):
        """Passes the tactical strings to Module 1."""
        self.adapter.apply_actions(tactical_actions, leader_controller, platoon_controller)
        
    def record_transition(self, platoon):
        """Calculates rewards and stores transition in the buffer."""
        rewards = self.reward_manager.compute_rewards(platoon, self.current_raw_actions)
        
        # Simple done logic (e.g. timeout or collision, assuming False for this scope)
        done = False 
        
        self.trainer.buffer.store_transition(
            self.current_local_obs,
            self.current_global_state,
            self.current_raw_actions,
            self.current_log_probs,
            rewards,
            self.current_value,
            done
        )

    def train_if_ready(self):
        """Calls MAPPO train if rollout length is reached."""
        if self.trainer.buffer.is_full(config.ROLLOUT_LENGTH):
            logger.info("[MODULE 2] Rollout full. Training MAPPO...")
            self.trainer.train()

    def _print_debug(self, actions_dict):
        try:
            with open("mappo_telemetry.log", "w") as f:
                f.write("="*48 + "\n")
                f.write("MODULE 2 (MAPPO LIVE DECISIONS)\n")
                f.write("="*48 + "\n")
                for key, act in actions_dict.items():
                    f.write(f"Vehicle: {key}\n")
                    f.write(f"MAPPO Action: {act}\n")
                    f.write("-" * 20 + "\n")
                f.write("="*48 + "\n")
        except Exception:
            pass

