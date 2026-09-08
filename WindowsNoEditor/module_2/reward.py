import math
from .observation_builder import actor_speed_kmh, actor_location

class RewardCalculator:
    def __init__(self, desired_clearance=7.5):
        self.desired_clearance = desired_clearance
        
    def calculate_reward(self, platoon, actions, weather_info, collision_monitor=None, cache=None):
        """
        Returns:
            dict of rewards for each vehicle id
        """
        rewards = {}
        if not platoon:
            return rewards
            
        # Extract friction
        preset = weather_info.get("preset_name", "CLEAR") if weather_info else "CLEAR"
        mapping = {
            "CLEAR": 1.0, "CLOUDY": 0.95, "LIGHT_RAIN": 0.8,
            "MODERATE_RAIN": 0.7, "HEAVY_RAIN": 0.55,
            "FOG": 0.9, "HEAVY_FOG": 0.8, "RAIN_FOG": 0.6
        }
        mu = mapping.get(preset, 1.0)
        
        # Calculate platoon collective states
        speeds = []
        headways = []
        for i, v in enumerate(platoon):
            if v and v.is_alive:
                c_data = cache.get(v.id) if cache else None
                speed = c_data["speed_kmh"] if c_data else actor_speed_kmh(v)
                speeds.append(speed)
                
                if i > 0 and platoon[i-1] and platoon[i-1].is_alive:
                    p_data = cache.get(platoon[i-1].id) if cache else None
                    loc = c_data["location"] if c_data else actor_location(v)
                    prev_loc = p_data["location"] if p_data else actor_location(platoon[i-1])
                    if loc and prev_loc:
                        headways.append(loc.distance(prev_loc))
                        
        avg_speed = sum(speeds)/max(1, len(speeds))
        avg_headway = sum(headways)/max(1, len(headways))
        
        # Collective Reward Components
        r_speed_coord = -abs(avg_speed - 28.0) * 0.05  # target 28 km/h
        r_headway_coord = -abs(avg_headway - self.desired_clearance - 5.0) * 0.1
        
        r_collective = r_speed_coord + r_headway_coord
        
        for i, vehicle in enumerate(platoon):
            if not vehicle or not vehicle.is_alive:
                continue
                
            vid = vehicle.id
            action = actions.get(vid, 0) # default CRUISE
            
            # Individual Reward Components
            r_collision = 0.0
            if collision_monitor and collision_monitor.collision_count > 0:
                # Basic check, proper implementation would check specific vehicle involvement
                r_collision = -10.0
                
            r_lane = 0.0 # simplified
            
            c_data = cache.get(vid) if cache else None
            speed = c_data["speed_kmh"] if c_data else actor_speed_kmh(vehicle)
            r_speed = -abs(speed - 28.0) * 0.01
            
            r_acceleration = 0.0
            if action in [1, 2]: # ACCELERATE, DECELERATE
                r_acceleration = -0.05
                
            # Weather Safety Reward
            r_weather_safety = 0.0
            
            # Action space: 0=CRUISE, 1=ACCELERATE, 2=DECELERATE, 3=LEFT, 4=RIGHT
            if mu < 0.7:
                # Penalize aggressive actions in low friction
                if action == 1: # ACCELERATE
                    r_weather_safety -= 2.0
                elif action in [3, 4]: # Lane changes
                    r_weather_safety -= 1.0
                elif action == 2: # DECELERATE
                    r_weather_safety += 0.5 # Encourage caution
                    
            r_individual = r_collision + r_lane + r_speed + r_acceleration
            
            r_total = r_individual + r_collective + r_weather_safety
            rewards[vid] = r_total
            
        return rewards

