import math
import numpy as np
from .action_space import TacticalAction

def actor_speed_kmh(actor) -> float:
    try:
        velocity = actor.get_velocity()
        return 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    except Exception:
        return 0.0

def actor_location(actor):
    try:
        if actor and actor.is_alive:
            return actor.get_location()
    except Exception:
        pass
    return None

class ObservationBuilder:
    def __init__(self, desired_clearance=7.5):
        self.desired_clearance = desired_clearance
        # Local Obs: 
        # 1. Rel long pos
        # 2. Rel lat pos
        # 3. Rel long vel
        # 4. Rel lat vel
        # 5. Headway distance
        # 6. Platoon membership indicator
        # 7. Current vehicle speed
        # 8. Current lane
        # 9. Weather/friction coefficient mu
        self.obs_dim = 9
        
    def get_friction(self, weather_info):
        if weather_info is None:
            return 1.0 # default clear
        preset = weather_info.get("preset_name", "CLEAR")
        # Example mapping based on common sense
        mapping = {
            "CLEAR": 1.0,
            "CLOUDY": 0.95,
            "LIGHT_RAIN": 0.8,
            "MODERATE_RAIN": 0.7,
            "HEAVY_RAIN": 0.55,
            "FOG": 0.9,
            "HEAVY_FOG": 0.8,
            "RAIN_FOG": 0.6
        }
        return mapping.get(preset, 1.0)
        
    def build_local_observations(self, platoon, weather_info, cache=None):
        """
        platoon: List of CARLA vehicle objects (Leader is index 0)
        weather_info: dict from uncertainty_manager.weather.get_weather_info()
        Returns dict of observations keyed by vehicle id.
        """
        observations = {}
        mu = self.get_friction(weather_info)
        
        for i, vehicle in enumerate(platoon):
            if not vehicle or not vehicle.is_alive:
                continue
                
            vid = vehicle.id
            is_leader = 1.0 if i == 0 else 0.0
            
            c_data = cache.get(vid) if cache else None
            
            speed = (c_data["speed_kmh"] if c_data else actor_speed_kmh(vehicle)) / 100.0  # normalize roughly
            lane = 0.0
            try:
                # Use a cached map instead of calling get_map() every frame which downloads the entire map
                if not hasattr(self, '_cached_map') or self._cached_map is None:
                    self._cached_map = vehicle.get_world().get_map()
                
                loc = c_data["location"] if c_data else vehicle.get_location()
                waypoint = self._cached_map.get_waypoint(loc)
                lane = float(waypoint.lane_id)
            except Exception as e:
                pass
                
            # relative info
            rel_long_pos = 0.0
            rel_lat_pos = 0.0
            rel_long_vel = 0.0
            rel_lat_vel = 0.0
            headway = 0.0
            
            if i > 0 and platoon[i-1] and platoon[i-1].is_alive:
                prev = platoon[i-1]
                p_data = cache.get(prev.id) if cache else None
                
                loc = c_data["location"] if c_data else actor_location(vehicle)
                prev_loc = p_data["location"] if p_data else actor_location(prev)
                if loc and prev_loc:
                    # simplified relative positioning
                    dx = prev_loc.x - loc.x
                    dy = prev_loc.y - loc.y
                    
                    yaw = math.radians(c_data["rotation"].yaw if c_data else vehicle.get_transform().rotation.yaw)
                    rel_long_pos = dx * math.cos(yaw) + dy * math.sin(yaw)
                    rel_lat_pos = -dx * math.sin(yaw) + dy * math.cos(yaw)
                    
                    headway = (math.sqrt(dx**2 + dy**2) - 5.0) / 20.0 # roughly normalized
                    
                    my_vel = c_data["velocity"] if c_data else vehicle.get_velocity()
                    prev_vel = p_data["velocity"] if p_data else prev.get_velocity()
                    dvx = prev_vel.x - my_vel.x
                    dvy = prev_vel.y - my_vel.y
                    rel_long_vel = dvx * math.cos(yaw) + dvy * math.sin(yaw)
                    rel_lat_vel = -dvx * math.sin(yaw) + dvy * math.cos(yaw)
            
            obs = np.array([
                rel_long_pos / 50.0,
                rel_lat_pos / 10.0,
                rel_long_vel / 20.0,
                rel_lat_vel / 5.0,
                headway,
                is_leader,
                speed,
                lane / 5.0,
                mu
            ], dtype=np.float32)
            
            observations[vid] = obs
            
        return observations

    def build_global_state(self, platoon, weather_info, cache=None):
        """
        Global State = 
        [
            vehicle_1_state,
            vehicle_2_state,
            ...,
            vehicle_N_state,
            weather_state,
            friction_coefficient,
            platoon_average_speed,
            headway_info
        ]
        """
        mu = self.get_friction(weather_info)
        global_state = []
        
        speeds = []
        headways = []
        
        # Max 5 vehicles in global state to keep dimension fixed for critic
        MAX_VEHICLES = 5
        for i in range(MAX_VEHICLES):
            if i < len(platoon) and platoon[i] and platoon[i].is_alive:
                v = platoon[i]
                c_data = cache.get(v.id) if cache else None
                speed = c_data["speed_kmh"] if c_data else actor_speed_kmh(v)
                speeds.append(speed)
                
                loc = c_data["location"] if c_data else actor_location(v)
                global_state.extend([loc.x/1000.0, loc.y/1000.0, speed/100.0])
                
                if i > 0 and platoon[i-1] and platoon[i-1].is_alive:
                    p_data = cache.get(platoon[i-1].id) if cache else None
                    prev_loc = p_data["location"] if p_data else actor_location(platoon[i-1])
                    if prev_loc:
                        headways.append(loc.distance(prev_loc))
            else:
                global_state.extend([0.0, 0.0, 0.0])
                
        avg_speed = sum(speeds)/max(1, len(speeds)) / 100.0
        avg_headway = sum(headways)/max(1, len(headways)) / 50.0
        
        weather_val = 1.0 if mu > 0.8 else (0.5 if mu > 0.6 else 0.0)
        
        global_state.extend([
            weather_val,
            mu,
            avg_speed,
            avg_headway
        ])
        
        return np.array(global_state, dtype=np.float32)

