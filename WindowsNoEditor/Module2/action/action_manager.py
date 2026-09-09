from .. import config
from .lane_change_manager import LaneChangeManager

ACTION_MAP = {
    0: "CRUISE",
    1: "ACCELERATE",
    2: "DECELERATE",
    3: "CHANGE_LEFT",
    4: "CHANGE_RIGHT"
}

class ActionManager:
    def __init__(self, num_vehicles):
        self.num_vehicles = num_vehicles
        self.last_actions = ["CRUISE" for _ in range(num_vehicles)]
        self.last_decision_time = 0.0
        self.lane_change_manager = LaneChangeManager(num_vehicles)

    def process_actions(self, raw_actions, current_time):
        """
        raw_actions: list of integers from the policy
        current_time: float sim time
        Returns a list of string tactical actions and a dict for Module 3 format.
        """
        # Anti-Wobble / Decision Interval Check
        if current_time - self.last_decision_time < config.TACTICAL_DECISION_INTERVAL:
            # Return persisted actions
            return self.last_actions, self._to_dict(self.last_actions)
        
        self.last_decision_time = current_time
        processed_actions = []
        
        for i in range(self.num_vehicles):
            raw_action = raw_actions[i]
            action_str = ACTION_MAP.get(raw_action, "CRUISE")
            
            # Update lane change state machine
            self.lane_change_manager.update_state(i, current_time)
            lc_state = self.lane_change_manager.get_state(i)
            
            # If currently in lane change, override action
            if lc_state != "NORMAL_DRIVING":
                action_str = self.last_actions[i] # Maintain previous intent or ignore new lane changes
                if raw_action in [3, 4]:
                     action_str = "CRUISE" # Ignore new LC request while one is active
            else:
                # Handle new lane change requests
                if raw_action in [3, 4]:
                    direction = "LEFT" if raw_action == 3 else "RIGHT"
                    accepted = self.lane_change_manager.request_lane_change(i, direction, current_time)
                    if not accepted:
                        action_str = "CRUISE"

            processed_actions.append(action_str)
            
        self.last_actions = processed_actions
        return processed_actions, self._to_dict(processed_actions)

    def _to_dict(self, actions):
        output = {}
        for i, act in enumerate(actions):
            key = "Leader" if i == 0 else f"F{i}"
            output[key] = act
        return output

