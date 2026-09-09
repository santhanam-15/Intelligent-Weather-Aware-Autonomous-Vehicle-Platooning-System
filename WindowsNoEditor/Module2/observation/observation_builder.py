import math
import numpy as np
import carla
from ..config import MAX_PLATOON_SIZE

class ObservationBuilder:
    def __init__(self, uncertainty_manager):
        self.uncertainty_manager = uncertainty_manager
        # Define dimensions
        # own_speed(1), vel_x(1), vel_y(1), is_leader(1),
        # dist_ahead(1), rel_speed_ahead(1), dist_leader(1), rel_speed_leader(1)
        # tl_val(1), is_at_tl(1), tl_dist(1)
        # weather_friction(1),
        # wp_dist(1), wp_dir(1)
        self.obs_dim = 14
        self.global_state_dim = self.obs_dim * MAX_PLATOON_SIZE

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

    def get_local_observation(self, vehicle, index, platoon):
        """Returns a normalized numpy array of the local observation for a vehicle."""
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        if not vehicle or not vehicle.is_alive:
            return obs

        try:
            vel = vehicle.get_velocity()
            speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            loc = vehicle.get_location()

            is_leader = 1.0 if index == 0 else 0.0

            # Relation to vehicle ahead
            dist_ahead = 50.0 # Default max sight
            rel_speed_ahead = 0.0
            
            # Find nearest vehicle ahead (obstacle or platoon)
            fwd = vehicle.get_transform().get_forward_vector()
            world = vehicle.get_world()
            all_vehicles = world.get_actors().filter('vehicle.*')
            
            nearest_v = None
            min_dist = 50.0
            
            for v in all_vehicles:
                if v.id == vehicle.id:
                    continue
                v_loc = v.get_location()
                dist = loc.distance(v_loc)
                if dist < min_dist:
                    dir_vec = carla.Vector3D(v_loc.x - loc.x, v_loc.y - loc.y, 0)
                    dir_vec_mag = math.sqrt(dir_vec.x**2 + dir_vec.y**2)
                    if dir_vec_mag > 0.0:
                        dot = (dir_vec.x * fwd.x + dir_vec.y * fwd.y) / dir_vec_mag
                        if dot > 0.85: # roughly within a forward cone
                            min_dist = dist
                            nearest_v = v
                            
            if nearest_v is not None:
                ahead_vel = nearest_v.get_velocity()
                ahead_speed = math.sqrt(ahead_vel.x**2 + ahead_vel.y**2 + ahead_vel.z**2)
                dist_ahead = max(0.0, min_dist - 4.7)
                rel_speed_ahead = ahead_speed - speed

            # Relation to Leader
            dist_leader = 0.0
            rel_speed_leader = 0.0
            if index > 0 and platoon[0] and platoon[0].is_alive:
                leader_v = platoon[0]
                leader_vel = leader_v.get_velocity()
                leader_speed = math.sqrt(leader_vel.x**2 + leader_vel.y**2 + leader_vel.z**2)
                dist_leader = max(0.0, loc.distance(leader_v.get_location()) - 4.7)
                rel_speed_leader = leader_speed - speed

            # Traffic light
            tl_state = vehicle.get_traffic_light_state()
            tl_val = 0.0 # Green / None
            if tl_state == carla.TrafficLightState.Red:
                tl_val = 1.0
            elif tl_state == carla.TrafficLightState.Yellow:
                tl_val = 0.5
            
            is_at_tl = 1.0 if vehicle.is_at_traffic_light() else 0.0
            
            tl_actor = vehicle.get_traffic_light()
            tl_dist = 50.0 # Default far away
            if tl_actor is not None:
                tl_dist = loc.distance(tl_actor.get_location())

            friction = self._get_friction()

            # Normalize and populate
            obs[0] = speed / 50.0
            obs[1] = vel.x / 30.0
            obs[2] = vel.y / 30.0
            obs[3] = is_leader
            obs[4] = dist_ahead / 100.0
            obs[5] = rel_speed_ahead / 30.0
            obs[6] = dist_leader / 100.0
            obs[7] = rel_speed_leader / 30.0
            obs[8] = tl_val
            obs[9] = is_at_tl
            obs[10] = tl_dist / 100.0
            obs[11] = friction
            obs[12] = 0.0 # Placeholder for wp dist
            obs[13] = 0.0 # Placeholder for wp dir
        except Exception as e:
            # Fallback to zeros on error
            pass
            
        return obs

    def get_global_state(self, local_obs_list):
        """Flattens local observations into a global state vector, padding if necessary."""
        global_state = np.zeros(self.global_state_dim, dtype=np.float32)
        idx = 0
        for obs in local_obs_list:
            if idx + self.obs_dim <= self.global_state_dim:
                global_state[idx : idx + self.obs_dim] = obs
                idx += self.obs_dim
        return global_state

