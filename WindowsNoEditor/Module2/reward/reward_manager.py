import carla
import math
from .. import config
import numpy as np
from uncertainty.weather_uncertainty import WeatherPreset

class RewardManager:
    def __init__(self, uncertainty_manager):
        self.uncertainty_manager = uncertainty_manager

    def _get_friction(self):
        try:
            weather_str = str(getattr(self.uncertainty_manager.weather, 'current_weather', 'CLEAR'))
            if 'CLEAR' in weather_str: return 1.0
            if 'CLOUDY' in weather_str: return 0.9
            if 'LIGHT_RAIN' in weather_str: return 0.6
            if 'HEAVY_RAIN' in weather_str: return 0.4
            if 'FOG' in weather_str: return 0.7
            return 1.0
        except:
            return 1.0

    def calculate_desired_headway(self, speed, rel_speed, friction):
        """Weather-aware headway calculation"""
        # Base headway + speed dependency + weather/friction adjustment
        base = config.DEFAULT_DESIRED_HEADWAY
        speed_factor = max(0.0, speed) * 0.5
        friction_factor = (1.0 - friction) * 10.0
        return base + speed_factor + friction_factor

    def compute_rewards(self, platoon, actions):
        """Computes individual rewards for each vehicle.
        Returns: list of floats
        """
        rewards = []
        friction = self._get_friction()
        
        for i, vehicle in enumerate(platoon):
            if not vehicle or not vehicle.is_alive:
                rewards.append(-10.0)
                continue
            
            r_i = 0.0
            vel = vehicle.get_velocity()
            speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            
            # 1. Traffic Light Compliance
            tl_state = vehicle.get_traffic_light_state()
            if vehicle.is_at_traffic_light() and tl_state == carla.TrafficLightState.Red:
                if speed > 0.5:
                    r_i += config.RED_LIGHT_PENALTY
            
            # 2. Headway Quality (All vehicles check their immediate front)
            fwd = vehicle.get_transform().get_forward_vector()
            world = vehicle.get_world()
            all_vehicles = world.get_actors().filter('vehicle.*')
            
            nearest_v = None
            min_dist = 50.0
            
            for v in all_vehicles:
                if v.id == vehicle.id:
                    continue
                v_loc = v.get_location()
                dist = vehicle.get_location().distance(v_loc)
                if dist < min_dist:
                    dir_vec = carla.Vector3D(v_loc.x - vehicle.get_location().x, v_loc.y - vehicle.get_location().y, 0)
                    dir_vec_mag = math.sqrt(dir_vec.x**2 + dir_vec.y**2)
                    if dir_vec_mag > 0.0:
                        dot = (dir_vec.x * fwd.x + dir_vec.y * fwd.y) / dir_vec_mag
                        if dot > 0.85:
                            min_dist = dist
                            nearest_v = v
                            
            if nearest_v is not None:
                clearance = max(0.0, min_dist - 4.7)
                
                ahead_vel = nearest_v.get_velocity()
                ahead_speed = math.sqrt(ahead_vel.x**2 + ahead_vel.y**2 + ahead_vel.z**2)
                rel_speed = ahead_speed - speed
                
                desired_hw = self.calculate_desired_headway(speed, rel_speed, friction)
                
                if clearance < desired_hw * 0.5:
                    r_i += config.UNSAFE_HEADWAY_PENALTY
                elif abs(clearance - desired_hw) <= config.HEADWAY_DEADBAND:
                    r_i += config.GOOD_HEADWAY_REWARD
                elif clearance > desired_hw * 1.5 and i > 0:
                    r_i -= 0.5 # penalty for being too far from vehicle ahead (mostly applies to followers needing to keep up)

            # 3. Efficiency / Speed tracking
            if speed > 5.0 and tl_state != carla.TrafficLightState.Red:
                r_i += config.EFFICIENCY_REWARD
                
            rewards.append(r_i)

        # Collective Reward
        collective_reward = sum(rewards) / max(1, len(rewards))
        
        # Final combined reward
        final_rewards = [r + config.COLLECTIVE_REWARD_WEIGHT * collective_reward for r in rewards]
        
        return final_rewards

