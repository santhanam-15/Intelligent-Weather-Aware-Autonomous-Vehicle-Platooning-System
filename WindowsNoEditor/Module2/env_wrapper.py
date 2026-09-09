import carla
import math
import numpy as np
import torch

from uncertainty.weather_uncertainty import WeatherPreset

class MAPPOEnvWrapper:
    def __init__(self, world, platoon, uncertainty_manager, map_instance):
        self.world = world
        self.platoon = platoon
        self.uncertainty_manager = uncertainty_manager
        self.map = map_instance
        
        self.action_dim = 5
        self.obs_dim = 15 # Will compute later
        
        # Mapping actions to high-level decisions
        self.action_mapping = {
            0: "CRUISE",
            1: "ACCELERATE",
            2: "DECELERATE",
            3: "CHANGE_LEFT",
            4: "CHANGE_RIGHT"
        }
        self.last_actions = np.zeros(len(platoon), dtype=int)

    def _get_friction_from_weather(self, weather_preset):
        if weather_preset == WeatherPreset.CLEAR:
            return 1.0
        elif weather_preset == WeatherPreset.CLOUDY:
            return 0.9
        elif weather_preset == WeatherPreset.LIGHT_RAIN:
            return 0.6
        elif weather_preset == WeatherPreset.HEAVY_RAIN:
            return 0.4
        elif weather_preset == WeatherPreset.HEAVY_FOG:
            return 0.7
        return 1.0

    def _calculate_vehicle_obs(self, vehicle, index):
        """
        Extract observation for a single vehicle.
        Return shape: (obs_dim,)
        """
        if not vehicle or not vehicle.is_alive:
            return np.zeros(self.obs_dim, dtype=np.float32)

        # 1. Kinematics
        velocity = vehicle.get_velocity()
        speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        transform = vehicle.get_transform()
        
        # 2. Platoon state
        is_leader = 1.0 if index == 0 else 0.0
        clearance = 0.0
        rel_speed = 0.0
        if index > 0:
            ahead = self.platoon[index - 1]
            if ahead and ahead.is_alive:
                ahead_vel = ahead.get_velocity()
                ahead_speed = 3.6 * math.sqrt(ahead_vel.x**2 + ahead_vel.y**2 + ahead_vel.z**2)
                rel_speed = ahead_speed - speed
                
                # Center distance minus vehicle length approx 4.7m
                dist = transform.location.distance(ahead.get_transform().location)
                clearance = max(0.0, dist - 4.7)

        # 3. Environment (Weather & Friction)
        current_weather = getattr(self.uncertainty_manager.weather, 'current_weather', WeatherPreset.CLEAR)
        friction = self._get_friction_from_weather(current_weather)

        # 4. Traffic Light State
        tl_state = vehicle.get_traffic_light_state()
        tl_val = 0.0 # Green
        if tl_state == carla.TrafficLightState.Red:
            tl_val = 1.0
        elif tl_state == carla.TrafficLightState.Yellow:
            tl_val = 0.5
            
        is_at_tl = 1.0 if vehicle.is_at_traffic_light() else 0.0

        # Construct vector
        obs = [
            speed / 50.0, # normalized speed
            velocity.x / 20.0,
            velocity.y / 20.0,
            is_leader,
            clearance / 50.0, # normalized clearance
            rel_speed / 20.0,
            friction,
            tl_val,
            is_at_tl,
            # Add placeholders to reach 15 if needed
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]
        
        self.obs_dim = len(obs)
        return np.array(obs, dtype=np.float32)

    def get_state(self):
        """
        Returns local observations and global state.
        local_obs: (num_vehicles, obs_dim)
        global_state: (global_state_dim,)
        """
        local_obs = []
        for i, vehicle in enumerate(self.platoon):
            obs = self._calculate_vehicle_obs(vehicle, i)
            local_obs.append(obs)
            
        local_obs_np = np.array(local_obs, dtype=np.float32)
        global_state_np = local_obs_np.flatten() # Simple concatenation for global state
        
        return local_obs_np, global_state_np

    def get_rewards(self):
        """
        Returns individual rewards for each vehicle.
        shape: (num_vehicles,)
        """
        rewards = []
        for i, vehicle in enumerate(self.platoon):
            if not vehicle or not vehicle.is_alive:
                rewards.append(-10.0)
                continue
                
            r_i = 0.0
            
            # 1. Traffic Light Compliance
            if vehicle.is_at_traffic_light() and vehicle.get_traffic_light_state() == carla.TrafficLightState.Red:
                speed = math.sqrt(vehicle.get_velocity().x**2 + vehicle.get_velocity().y**2)
                if speed > 0.5:
                    r_i -= 5.0 # Large penalty for moving at red light
                    
            # 2. Safe headway (for followers)
            if i > 0:
                ahead = self.platoon[i - 1]
                if ahead and ahead.is_alive:
                    dist = vehicle.get_location().distance(ahead.get_location())
                    clearance = max(0.0, dist - 4.7)
                    
                    # Weather aware safe distance
                    current_weather = getattr(self.uncertainty_manager.weather, 'current_weather', WeatherPreset.CLEAR)
                    friction = self._get_friction_from_weather(current_weather)
                    
                    # Desired clearance increases when friction is low
                    desired_clearance = 7.5 + (1.0 - friction) * 10.0
                    
                    if clearance < desired_clearance * 0.5:
                        r_i -= 2.0 # unsafe headway
                    elif clearance > desired_clearance * 2.0:
                        r_i -= 0.5 # too far
                    else:
                        r_i += 0.5 # good headway
                        
            # 3. Efficiency
            speed = 3.6 * math.sqrt(vehicle.get_velocity().x**2 + vehicle.get_velocity().y**2)
            if speed > 10.0:
                r_i += 0.1
                
            rewards.append(r_i)
            
        return np.array(rewards, dtype=np.float32)

    def apply_actions(self, actions, leader_controller, platoon_controller):
        """
        Translate MAPPO tactical actions to low-level controller commands.
        actions: list of integers mapping to tactical actions.
        """
        for i, vehicle in enumerate(self.platoon):
            action_idx = actions[i]
            tactical = self.action_mapping[action_idx]
            
            # Map to vehicle controllers
            if i == 0:
                # Leader Controller
                if tactical == "ACCELERATE":
                    leader_controller.target_speed_kmh = min(40.0, leader_controller.target_speed_kmh + 2.0)
                elif tactical == "DECELERATE":
                    leader_controller.target_speed_kmh = max(0.0, leader_controller.target_speed_kmh - 5.0)
                # CRUISE maintains speed. CHANGE_LEFT/RIGHT can be implemented if leader controller supports it.
            else:
                # Platoon Follower Controller
                # Platoon controller uses desired_clearance. We can adjust it or temporarily override brake/throttle.
                if tactical == "DECELERATE":
                    platoon_controller.desired_clearance += 2.0
                elif tactical == "ACCELERATE":
                    platoon_controller.desired_clearance = max(5.0, platoon_controller.desired_clearance - 1.0)
                
        # To prevent oscillation, clamp values
        platoon_controller.desired_clearance = max(5.0, min(25.0, platoon_controller.desired_clearance))
