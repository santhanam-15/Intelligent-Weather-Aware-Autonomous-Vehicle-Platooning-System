import logging
import math

import carla
import numpy as np

from ..config import MAX_PLATOON_SIZE

logger = logging.getLogger("ObservationBuilder")


class ObservationBuilder:
    """Build ordered local observations and the centralized state."""

    def __init__(self, uncertainty_manager, control_context=None):
        self.uncertainty_manager = uncertainty_manager
        # All continuous values are normalized/clipped.  Route and safety
        # fields come from the sole low-level-controller owner, not from a
        # second waypoint/controller implementation.
        self.control_context = control_context
        self.obs_dim = 32
        self.global_state_dim = self.obs_dim * MAX_PLATOON_SIZE
        self._carla_map = None

    @staticmethod
    def _weather_name(weather):
        current = getattr(weather, "current_preset", None)
        if current is None:
            current = getattr(weather, "current_weather", "CLEAR")
        return getattr(current, "name", str(current)).upper().split(".")[-1]

    def _get_friction(self, vehicle=None):
        """Return the active friction multiplier used by Module 2."""
        try:
            friction_manager = getattr(self.uncertainty_manager, "friction", None)
            if friction_manager is not None and vehicle is not None:
                managed_mu = friction_manager.get_friction(vehicle)
                if managed_mu is not None:
                    return float(managed_mu)
        except Exception:
            pass

        try:
            weather = getattr(self.uncertainty_manager, "weather", None)
            name = self._weather_name(weather)
            return {
                "CLEAR": 1.0,
                "CLOUDY": 0.9,
                "LIGHT_RAIN": 0.6,
                "MODERATE_RAIN": 0.6,
                "HEAVY_RAIN": 0.4,
                "FOG": 0.7,
                "HEAVY_FOG": 0.7,
                "RAIN_FOG": 0.4,
            }.get(name, 1.0)
        except Exception:
            return 1.0

    def set_control_context(self, control_context):
        self.control_context = control_context

    def _context(self, vehicle, index):
        try:
            if self.control_context is not None:
                return self.control_context.observation_context(vehicle, index)
        except Exception:
            pass
        return {}

    def _nearest_vehicle(self, vehicle, platoon_ids, nearby_actors=None):
        """Compact nearest-traffic feature in ego coordinates."""
        best = None
        best_distance = 80.0
        try:
            actors = nearby_actors
            if actors is None:
                actors = vehicle.get_world().get_actors().filter("vehicle.*")
            for other in actors:
                if other.id == vehicle.id or not other.is_alive:
                    continue
                distance = vehicle.get_location().distance(other.get_location())
                if distance < best_distance:
                    best, best_distance = other, distance
        except Exception:
            pass
        if best is None:
            return 0.0, 0.0, 0.0, 0.0
        rel_x, rel_y, rel_vx, _ = self._relative_components(vehicle, best)
        return rel_x, rel_y, rel_vx, 1.0 if best.id in platoon_ids else 0.0

    @staticmethod
    def _relative_components(vehicle, ahead):
        """Return ahead-minus-vehicle pose and velocity in ego axes."""
        loc = vehicle.get_location()
        ahead_loc = ahead.get_location()
        velocity = vehicle.get_velocity()
        ahead_velocity = ahead.get_velocity()
        forward = vehicle.get_transform().get_forward_vector()

        dx = ahead_loc.x - loc.x
        dy = ahead_loc.y - loc.y
        dvx = ahead_velocity.x - velocity.x
        dvy = ahead_velocity.y - velocity.y
        return (
            dx * forward.x + dy * forward.y,
            -dx * forward.y + dy * forward.x,
            dvx * forward.x + dvy * forward.y,
            -dvx * forward.y + dvy * forward.x,
        )

    def get_local_observation(self, vehicle, index, platoon, nearby_actors=None):
        """Return one finite, normalized observation for one platoon vehicle."""
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        if not vehicle or not vehicle.is_alive:
            return obs

        try:
            platoon_size = len(platoon)
            if not 0 <= index < platoon_size:
                raise IndexError("vehicle index is outside the platoon")

            velocity = vehicle.get_velocity()
            speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
            is_leader = 1.0 if index == 0 else 0.0
            agent_slot = index / max(1.0, float(MAX_PLATOON_SIZE - 1))
            size_norm = platoon_size / float(MAX_PLATOON_SIZE)

            rel_x = rel_y = rel_vx = rel_vy = 0.0
            clearance = 50.0
            rel_speed_ahead = 0.0

            # Use only the ordered platoon predecessor for platoon headway.
            if index > 0:
                ahead = platoon[index - 1]
                if ahead and ahead.is_alive:
                    rel_x, rel_y, rel_vx, rel_vy = self._relative_components(vehicle, ahead)
                    center_distance = vehicle.get_location().distance(ahead.get_location())
                    clearance = max(0.0, center_distance - 4.7)
                    ahead_velocity = ahead.get_velocity()
                    rel_speed_ahead = math.sqrt(
                        ahead_velocity.x ** 2
                        + ahead_velocity.y ** 2
                        + ahead_velocity.z ** 2
                    ) - speed

            dist_leader = 0.0
            rel_speed_leader = 0.0
            if index > 0 and platoon[0] and platoon[0].is_alive:
                leader = platoon[0]
                leader_velocity = leader.get_velocity()
                leader_speed = math.sqrt(
                    leader_velocity.x ** 2
                    + leader_velocity.y ** 2
                    + leader_velocity.z ** 2
                )
                dist_leader = max(0.0, vehicle.get_location().distance(leader.get_location()) - 4.7)
                rel_speed_leader = leader_speed - speed

            context = self._context(vehicle, index)
            tl_state = vehicle.get_traffic_light_state()
            tl_val = 1.0 if tl_state == carla.TrafficLightState.Red else 0.0
            if tl_state == carla.TrafficLightState.Yellow:
                tl_val = 0.5
            is_at_tl = 1.0 if vehicle.is_at_traffic_light() else 0.0
            tl_actor = vehicle.get_traffic_light()
            tl_dist = vehicle.get_location().distance(tl_actor.get_location()) if tl_actor else 50.0
            friction = self._get_friction(vehicle)
            platoon_ids = {v.id for v in platoon if v and v.is_alive}
            nearby_x, nearby_y, nearby_vx, nearby_is_platoon = self._nearest_vehicle(
                vehicle, platoon_ids, nearby_actors=nearby_actors
            )
            if self._carla_map is None:
                try:
                    self._carla_map = vehicle.get_world().get_map()
                except Exception:
                    self._carla_map = None
            waypoint = None
            if self._carla_map is not None:
                try:
                    waypoint = self._carla_map.get_waypoint(
                        vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving
                    )
                except Exception:
                    waypoint = None
            lane_id = float(waypoint.lane_id) / 10.0 if waypoint else 0.0
            road_id = float(waypoint.road_id) / 100.0 if waypoint else 0.0
            left_ok = right_ok = 0.0
            if waypoint:
                left = waypoint.get_left_lane()
                right = waypoint.get_right_lane()
                left_ok = 1.0 if left and left.lane_type == carla.LaneType.Driving else 0.0
                right_ok = 1.0 if right and right.lane_type == carla.LaneType.Driving else 0.0
            platoon_state = str(context.get("platoon_state", "NORMAL_PLATOON"))
            state_value = {
                "NORMAL_PLATOON": 0.0, "INTRUSION_DETECTED": 0.25,
                "PLATOON_SPLIT": 0.5, "SAFE_FOLLOWING": 0.65,
                "REJOIN_ELIGIBLE": 0.8, "REJOINING": 1.0,
            }.get(platoon_state, 0.0)

            obs[:] = np.asarray(
                [
                    speed / 50.0,
                    velocity.x / 30.0,
                    velocity.y / 30.0,
                    is_leader,
                    agent_slot,
                    size_norm,
                    rel_x / 100.0,
                    rel_y / 100.0,
                    rel_vx / 30.0,
                    rel_vy / 30.0,
                    clearance / 100.0,
                    rel_speed_ahead / 30.0,
                    dist_leader / 100.0,
                    rel_speed_leader / 30.0,
                    tl_val,
                    is_at_tl,
                    tl_dist / 100.0,
                    friction,
                    _clamp(context.get("route_progress", 0.0), 0.0, 1.0),
                    _clamp(context.get("distance_to_waypoint", 50.0) / 100.0, 0.0, 1.0),
                    _clamp(context.get("lateral_error", 0.0) / 6.0, -1.0, 1.0),
                    _clamp(context.get("heading_error", 0.0) / math.pi, -1.0, 1.0),
                    lane_id,
                    road_id,
                    left_ok,
                    right_ok,
                    nearby_x / 80.0,
                    nearby_y / 12.0,
                    nearby_vx / 30.0,
                    nearby_is_platoon,
                    float(context.get("intrusion", 0.0)),
                    state_value,
                ],
                dtype=np.float32,
            )
        except Exception as exc:
            logger.error(
                "Exception in observation building for vehicle %s (index %s): %s",
                getattr(vehicle, "id", "?"),
                index,
                exc,
                exc_info=True,
            )


        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    def get_global_state(self, local_obs_list):
        """Concatenate ordered local observations and pad to MAX_PLATOON_SIZE."""
        if len(local_obs_list) > MAX_PLATOON_SIZE:
            raise ValueError(
                "platoon has %d vehicles but MAX_PLATOON_SIZE is %d"
                % (len(local_obs_list), MAX_PLATOON_SIZE)
            )

        global_state = np.zeros(self.global_state_dim, dtype=np.float32)
        for index, obs in enumerate(local_obs_list):
            array = np.asarray(obs, dtype=np.float32).reshape(-1)
            if array.size != self.obs_dim:
                raise ValueError("local observation has wrong dimension")
            start = index * self.obs_dim
            global_state[start : start + self.obs_dim] = array

        return np.nan_to_num(global_state, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)


def _clamp(value, low, high):
    return max(low, min(high, float(value)))
