import time
from .. import config

class LaneChangeState:
    NORMAL_DRIVING = "NORMAL_DRIVING"
    LANE_CHANGE_REQUESTED = "LANE_CHANGE_REQUESTED"
    LANE_CHANGE_IN_PROGRESS = "LANE_CHANGE_IN_PROGRESS"
    LANE_CHANGE_COMPLETE = "LANE_CHANGE_COMPLETE"
    RETURN_TO_NORMAL = "RETURN_TO_NORMAL"

class LaneChangeManager:
    def __init__(self, num_vehicles):
        self.num_vehicles = num_vehicles
        self.states = [LaneChangeState.NORMAL_DRIVING for _ in range(num_vehicles)]
        self.request_times = [0.0 for _ in range(num_vehicles)]
        
    def request_lane_change(self, vehicle_idx, direction, current_time):
        """direction: 'LEFT' or 'RIGHT'. Returns True if accepted."""
        state = self.states[vehicle_idx]
        if state == LaneChangeState.NORMAL_DRIVING:
            if current_time - self.request_times[vehicle_idx] > config.LANE_CHANGE_COOLDOWN:
                self.states[vehicle_idx] = LaneChangeState.LANE_CHANGE_REQUESTED
                self.request_times[vehicle_idx] = current_time
                return True
        return False
        
    def update_state(self, vehicle_idx, current_time):
        """Updates internal state machines for lane changes based on time elapsed."""
        state = self.states[vehicle_idx]
        elapsed = current_time - self.request_times[vehicle_idx]
        
        if state == LaneChangeState.LANE_CHANGE_REQUESTED:
            self.states[vehicle_idx] = LaneChangeState.LANE_CHANGE_IN_PROGRESS
            
        elif state == LaneChangeState.LANE_CHANGE_IN_PROGRESS:
            if elapsed > config.LANE_CHANGE_DURATION:
                self.states[vehicle_idx] = LaneChangeState.LANE_CHANGE_COMPLETE
                
        elif state == LaneChangeState.LANE_CHANGE_COMPLETE:
            self.states[vehicle_idx] = LaneChangeState.RETURN_TO_NORMAL
            
        elif state == LaneChangeState.RETURN_TO_NORMAL:
            self.states[vehicle_idx] = LaneChangeState.NORMAL_DRIVING

    def get_state(self, vehicle_idx):
        return self.states[vehicle_idx]

