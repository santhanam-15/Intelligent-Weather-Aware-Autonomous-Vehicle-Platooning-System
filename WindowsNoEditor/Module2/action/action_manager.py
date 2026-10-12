from .. import config

# Full action map (5 actions for the neural net)
ACTION_MAP = {
    0: "CRUISE",
    1: "ACCELERATE",
    2: "DECELERATE",
    3: "CHANGE_LEFT",   # Only valid for leader
    4: "CHANGE_RIGHT"   # Only valid for leader
}

# Followers ONLY get 3 safe actions – lane changes are forcibly remapped to CRUISE
FOLLOWER_ACTION_MAP = {
    0: "CRUISE",
    1: "ACCELERATE",
    2: "DECELERATE",
    3: "CRUISE",   # Remap → CRUISE (followers must never change lanes)
    4: "CRUISE"    # Remap → CRUISE
}

class ActionManager:
    def __init__(self, num_vehicles):
        self.num_vehicles = num_vehicles
        self.last_actions = ["CRUISE" for _ in range(num_vehicles)]
        self.last_effective_raw_actions = [0 for _ in range(num_vehicles)]
        self._candidate_raw_actions = [None for _ in range(num_vehicles)]
        self._candidate_counts = [0 for _ in range(num_vehicles)]
        self._initialized = False
        self.last_decision_time = None

    @staticmethod
    def _normalize_raw_action(index, raw_action, active_leader_index=0):
        raw_action = int(raw_action)
        if index == active_leader_index:
            return raw_action if raw_action in ACTION_MAP else 0
        return raw_action if raw_action in FOLLOWER_ACTION_MAP else 0

    def process_actions(self, raw_actions, current_time, active_leader_index=0, disabled_indices=None):
        """
        raw_actions: list of integers from the policy
        current_time: float sim time
        active_leader_index: index of authoritative active leader
        Returns a list of string tactical actions and a dict for Module 3 format.
        """
        self.last_decision_time = current_time
        processed_actions = []
        effective_raw_actions = []
        
        for i in range(self.num_vehicles):
            proposed = raw_actions[i] if i < len(raw_actions) else 0
            proposed = self._normalize_raw_action(i, proposed, active_leader_index=active_leader_index)

            # A tactical command must persist long enough for the low-level
            # controller to react.  Require two consecutive decisions before
            # committing a changed command, preventing ACCELERATE/BRAKE or
            # LEFT/RIGHT chatter from tiny observation changes.
            if not self._initialized:
                effective = proposed
            elif proposed == self.last_effective_raw_actions[i]:
                effective = proposed
                self._candidate_raw_actions[i] = None
                self._candidate_counts[i] = 0
            else:
                if proposed != self._candidate_raw_actions[i]:
                    self._candidate_raw_actions[i] = proposed
                    self._candidate_counts[i] = 1
                else:
                    self._candidate_counts[i] += 1

                if self._candidate_counts[i] >= config.ACTION_HYSTERESIS_STEPS:
                    effective = proposed
                    self._candidate_raw_actions[i] = None
                    self._candidate_counts[i] = 0
                else:
                    effective = self.last_effective_raw_actions[i]

            effective_raw_actions.append(effective)
            
            if i == active_leader_index:
                # Active Leader: full 5-action space
                action_str = ACTION_MAP.get(effective, "CRUISE")
            else:
                # Followers: lane changes are NEVER allowed – remap to CRUISE
                action_str = FOLLOWER_ACTION_MAP.get(effective, "CRUISE")

            processed_actions.append(action_str)

        self._initialized = True
        self.last_effective_raw_actions = effective_raw_actions
        self.last_actions = processed_actions
        return processed_actions, self._to_dict(
            processed_actions,
            active_leader_index=active_leader_index,
            disabled_indices=disabled_indices,
        )

    def _to_dict(self, actions, active_leader_index=0, disabled_indices=None):
        output = {}
        for i, act in enumerate(actions):
            if i == active_leader_index:
                key = "Leader"
            elif disabled_indices and i in disabled_indices:
                key = f"Disabled_F{i}"
            else:
                key = f"F{i}"
            output[key] = act
        return output

