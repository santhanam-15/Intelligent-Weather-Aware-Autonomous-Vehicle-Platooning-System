"""Route-owned, safety-constrained CARLA control for Module 2.

This module deliberately owns *all* ``apply_control`` calls for platoon
actors.  Traffic Manager remains free to control NPC/ghost actors, but never
the platoon.  The policy supplies a low-rate tactical intent; this controller
turns that intent into a bounded, route-following vehicle command every CARLA
tick.
"""

from __future__ import annotations

import json
import logging
import math
import os

import carla

from Module2.config import (
    DECISION_LOG_FILE,
    DESIRED_HEADWAY_TIME_SEC,
    MIN_STANDSTILL_GAP_M,
    CATCHUP_GAP_THRESHOLD_RATIO,
    CATCHUP_MAX_SPEED_INCREMENT_MPS,
    CATCHUP_KP_GAIN,
    LEADER_FAILURE_TIMEOUT_SEC,
    LEADER_COLLISION_TIMEOUT_SEC,
    LEADER_SUSPECT_TIMEOUT_SEC,
    LEADER_RECOVERY_ATTEMPT_TIMEOUT_SEC,
    LEADER_STOPPED_SPEED_THRESHOLD_MPS,
    LEADER_RECOVERY_SPEED_THRESHOLD_MPS,
)
from Module2.overtake_state_machine import OvertakeStateMachine, OvertakeState
from Module2.logger import MAPPODecisionLogger


class LeaderHealthState(object):
    """Explicit leader lifecycle state machine."""
    ACTIVE = "ACTIVE"
    SUSPECTED_FAILURE = "SUSPECTED_FAILURE"
    RECOVERY_ATTEMPT = "RECOVERY_ATTEMPT"
    FAILED = "FAILED"
    PROMOTION_PENDING = "PROMOTION_PENDING"
    NEW_LEADER_ACTIVE = "NEW_LEADER_ACTIVE"
    NO_ELIGIBLE_LEADER = "NO_ELIGIBLE_LEADER"


logger = logging.getLogger("Module2MissionController")


def _clamp(value, low, high):
    return max(low, min(high, value))


def _speed_mps(vehicle):
    velocity = vehicle.get_velocity()
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)


def _angle_error_deg(current, target):
    return (target - current + 180.0) % 360.0 - 180.0


class RoutePlanner(object):
    """Create one deterministic road route and give every CAV its own cursor."""

    def __init__(self, carla_map, origin, distance_m=150.0, spacing_m=2.0):
        self.map = carla_map
        self.spacing_m = float(spacing_m)
        self.route = self._build_route(origin, float(distance_m))
        if len(self.route) < 2:
            raise RuntimeError("Could not build a valid mission route from platoon origin")
        self.destination = self.route[-1].transform.location
        self.cursors = {}

    def _build_route(self, origin, distance_m):
        current = self.map.get_waypoint(
            origin, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if current is None:
            return []
        route = [current]
        travelled = 0.0
        # Junctions in CARLA OpenDRIVE can have gap distances across entry waypoints.
        # Bounding continuation jump to max(12.0, spacing*6.0) allows legitimate junction
        # continuations while preventing jumps across opposite highway lanes (>20m).
        max_jump = max(12.0, self.spacing_m * 6.0)

        while travelled < distance_m:
            next_points = current.next(self.spacing_m)
            if not next_points:
                break
            try:
                current = min(
                    next_points,
                    key=lambda wp: (
                        wp.road_id != current.road_id,
                        abs(wp.lane_id - current.lane_id),
                        abs(_angle_error_deg(current.transform.rotation.yaw, wp.transform.rotation.yaw)),
                    ),
                )
            except Exception:
                current = next_points[0]

            if route[-1].transform.location.distance(current.transform.location) > max_jump:
                break
            route.append(current)
            travelled += self.spacing_m
        return route

    def extend_route(self, distance_m=150.0):
        """Append continuous road waypoints along the road network if available."""
        if not self.route:
            return 0
        current = self.route[-1]
        travelled = 0.0
        added = 0
        max_jump = max(12.0, self.spacing_m * 6.0)

        while travelled < distance_m:
            try:
                next_points = current.next(self.spacing_m)
            except Exception:
                break
            if not next_points:
                break
            try:
                current = min(
                    next_points,
                    key=lambda wp: (
                        wp.road_id != current.road_id,
                        abs(wp.lane_id - current.lane_id),
                        abs(_angle_error_deg(current.transform.rotation.yaw, wp.transform.rotation.yaw)),
                    ),
                )
            except Exception:
                current = next_points[0]

            if self.route[-1].transform.location.distance(current.transform.location) > max_jump:
                break

            self.route.append(current)
            self.destination = current.transform.location
            travelled += self.spacing_m
            added += 1

        return added

    def progress_for(self, vehicle):
        """Return a vehicle-local cursor; it only advances by physically passing points.

        Determines sequential progress along the ordered route using forward
        projection along the segment vector, path corridor constraints, and
        bounded forward advancement. This prevents nearby parallel roads, loops,
        or spawn offsets from causing cursor jumps to distant route sections.
        """
        vehicle_id = vehicle.id
        last = self.cursors.get(vehicle_id, 0)
        loc = vehicle.get_location()
        cursor = last

        # Bounded advancement per call prevents runaway cursor jumps in a single tick.
        max_advance = 3
        advanced = 0

        # Check if near junction for corridor sizing
        is_junc = False
        if cursor < len(self.route):
            is_junc = getattr(self.route[cursor], "is_junction", False)
        if not is_junc and cursor + 1 < len(self.route):
            is_junc = getattr(self.route[cursor + 1], "is_junction", False)

        # Lateral corridor: vehicle must be within reasonable lateral distance of segment.
        # In junctions/turns, allow wider corridor (up to 8.5m) to prevent false deadlock.
        max_lateral = max(8.5, self.spacing_m * 4.0) if is_junc else max(5.0, self.spacing_m * 2.5)
        # Reach corridor: vehicle must be in the spatial vicinity of the active segment.
        max_reach = max(12.0, self.spacing_m * 5.0)

        while cursor + 1 < len(self.route) and advanced < max_advance:
            curr_loc = self.route[cursor].transform.location
            next_loc = self.route[cursor + 1].transform.location

            dx = next_loc.x - curr_loc.x
            dy = next_loc.y - curr_loc.y
            seg_len = math.hypot(dx, dy)

            # Skip degenerate / zero-length segments safely.
            if seg_len < 1e-3:
                cursor += 1
                advanced += 1
                continue

            tx = dx / seg_len
            ty = dy / seg_len

            # Vector from current waypoint to vehicle in horizontal plane.
            vx = loc.x - curr_loc.x
            vy = loc.y - curr_loc.y

            # Longitudinal projection along the route segment.
            s_proj = vx * tx + vy * ty
            # Lateral perpendicular offset from the segment centerline.
            lat_offset = abs(-vx * ty + vy * tx)

            dist_curr = math.hypot(vx, vy)
            dist_next = math.hypot(loc.x - next_loc.x, loc.y - next_loc.y)

            # Sequential passing condition:
            # The vehicle has physically progressed through curr_wp toward next_wp
            # if it has covered at least half the segment along the forward tangent
            # and is closer to next_wp, or has passed next_wp longitudinally.
            passed_midpoint = (s_proj >= seg_len * 0.5) and (dist_next < dist_curr)
            passed_segment = s_proj >= seg_len

            # Proximity corridor check:
            # Reject if vehicle is far from this segment unless it is physically
            # closer to next_wp than curr_wp and within reach.
            if dist_curr > max_reach and dist_next > max_reach:
                break
            if lat_offset > max_lateral and not (passed_midpoint and dist_next < max_reach):
                break

            if passed_midpoint or passed_segment:
                cursor += 1
                advanced += 1
            else:
                break

        # Bounded forward recovery search if cursor stalled due to lateral excursion:
        if advanced == 0 and cursor + 1 < len(self.route):
            curr_dist = math.hypot(loc.x - self.route[cursor].transform.location.x, loc.y - self.route[cursor].transform.location.y)
            search_end = min(len(self.route), cursor + 8)
            best_idx = cursor
            best_dist = curr_dist
            for k in range(cursor + 1, search_end):
                d = math.hypot(loc.x - self.route[k].transform.location.x, loc.y - self.route[k].transform.location.y)
                if d < best_dist and d < max_reach:
                    best_dist = d
                    best_idx = k
            if best_idx > cursor:
                cursor = best_idx

        self.cursors[vehicle_id] = cursor
        return cursor

    def target_for(self, vehicle, lookahead_m):
        cursor = self.progress_for(vehicle)
        steps = max(1, int(lookahead_m / self.spacing_m))
        target_idx = min(len(self.route) - 1, cursor + steps)
        return self.route[target_idx], cursor, target_idx

    def route_progress(self, vehicle):
        cursor = self.progress_for(vehicle)
        return cursor / float(max(1, len(self.route) - 1))


class PlatoonState(object):
    NORMAL = "NORMAL_PLATOON"
    INTRUSION = "INTRUSION_DETECTED"
    SPLIT = "PLATOON_SPLIT"
    SAFE_FOLLOWING = "SAFE_FOLLOWING"
    REJOIN_ELIGIBLE = "REJOIN_ELIGIBLE"
    REJOINING = "REJOINING"


class MissionPlatoonController(object):
    """The sole platoon control owner: route -> safety -> low-level control."""

    CONTROL_OWNER = "MAPPO->SAFETY->LOW_LEVEL"

    def __init__(self, world, platoon, cruise_speed_kmh=28.0, friction_provider=None):
        self.world = world
        self.map = world.get_map()
        self.platoon = list(platoon)
        origin = self.platoon[0].get_location()
        if origin.x == 0.0 and origin.y == 0.0 and origin.z == 0.0:
            try:
                if hasattr(self.world, "tick"):
                    self.world.tick()
                    origin = self.platoon[0].get_location()
            except Exception:
                pass
        self.route = RoutePlanner(self.map, origin, distance_m=300.0)
        self.cruise_speed_mps = float(cruise_speed_kmh) / 3.6
        self.friction_provider = friction_provider
        self.tactical_actions = ["CRUISE"] * len(self.platoon)
        self.emergency_brakes = [False] * len(self.platoon)
        self.last_steer = {}
        self.last_control = {}
        self.state = PlatoonState.NORMAL
        self.state_since = 0.0
        self.intruder_id = None
        self.intruded_between = None
        self.rejoin_ready_since = None
        self.lane_states = ["NORMAL" for _ in self.platoon]
        self._last_override = ["" for _ in self.platoon]
        self._last_context = [{} for _ in self.platoon]
        self._stuck_since = {}
        self._lateral_offsets = {}
        # Leader failover & promotion tracking
        self.active_leader_index = 0
        self.disabled_vehicles = set()
        self._leader_stopped_since = None
        self._leader_failed = False
        self.leader_health_state = LeaderHealthState.ACTIVE
        self.leader_state_since = 0.0
        self.leader_state_history = []
        self.collision_events = {}  # {vehicle_id: (timestamp, intensity)}
        self.promotion_events_count = 0
        self.predecessor_map = {i: (i - 1 if i > 0 else None) for i in range(len(self.platoon))}
        self.desired_headway_time_sec = DESIRED_HEADWAY_TIME_SEC
        self.min_standstill_gap_m = MIN_STANDSTILL_GAP_M
        self.catchup_gap_threshold_ratio = CATCHUP_GAP_THRESHOLD_RATIO
        self.catchup_max_speed_increment_mps = CATCHUP_MAX_SPEED_INCREMENT_MPS
        self.catchup_kp_gain = CATCHUP_KP_GAIN
        self.leader_failure_timeout_sec = LEADER_FAILURE_TIMEOUT_SEC
        self.leader_collision_timeout_sec = LEADER_COLLISION_TIMEOUT_SEC
        self.leader_suspect_timeout_sec = LEADER_SUSPECT_TIMEOUT_SEC
        self.leader_recovery_attempt_timeout_sec = LEADER_RECOVERY_ATTEMPT_TIMEOUT_SEC
        self.leader_stopped_speed_threshold_mps = LEADER_STOPPED_SPEED_THRESHOLD_MPS
        self.leader_recovery_speed_threshold_mps = LEADER_RECOVERY_SPEED_THRESHOLD_MPS
        self.follower_speed_telemetry = {}
        # Centralized stop tracking and diagnostics
        self._prev_target_speed = {}
        self._stop_reasons = {}
        self._stop_sources = {}
        # Intelligent Overtaking state machine & parameter tuning
        self.overtake_sm = OvertakeStateMachine(
            min_speed_diff_mps=3.0,
            eval_distance_m=28.0,
            min_safe_time_gap_sec=2.0,
            cooldown_sec=8.0,
            lateral_offset_m=3.5,
        )
        self.overtake_lateral_offsets = {}
        self._decision_log_file = DECISION_LOG_FILE
        self._decision_log_buffer = []
        self._last_flush_time = 0.0
        logger.info(
            "Mission route created: %d waypoints, destination=(%.1f, %.1f)",
            len(self.route.route), self.route.destination.x, self.route.destination.y,
        )

    def set_tactical_actions(self, actions):
        if len(actions) != len(self.platoon):
            raise ValueError("tactical actions must have one entry per platoon vehicle")
        self.tactical_actions = list(actions)

    def set_emergency_brakes(self, emergency_flags):
        """Set auxiliary emergency braking signals (e.g. from Module 3 Safety Shield)."""
        if len(emergency_flags) != len(self.platoon):
            raise ValueError("emergency flags must have one entry per platoon vehicle")
        self.emergency_brakes = [bool(f) for f in emergency_flags]

    def set_lateral_offsets(self, offsets_dict):
        """Set auxiliary lateral evasion offsets from Module 3 (e.g. +/- 3.5m)."""
        self._lateral_offsets = dict(offsets_dict) if offsets_dict else {}

    def _friction(self, vehicle):
        try:
            value = self.friction_provider(vehicle) if self.friction_provider else 1.0
            return _clamp(float(value), 0.2, 1.2)
        except Exception:
            return 1.0

    @staticmethod
    def _local_components(vehicle, other):
        tf = vehicle.get_transform()
        loc = tf.location
        other_loc = other.get_location()
        forward = tf.get_forward_vector()
        dx, dy = other_loc.x - loc.x, other_loc.y - loc.y
        return dx * forward.x + dy * forward.y, -dx * forward.y + dy * forward.x

    def _vehicle_clearance(self, ahead, behind):
        return max(0.0, ahead.get_location().distance(behind.get_location()) - 4.7)

    def _desired_headway(self, vehicle, predecessor=None):
        speed = _speed_mps(vehicle)
        friction = self._friction(vehicle)
        closing = 0.0
        if predecessor and predecessor.is_alive:
            closing = max(0.0, speed - _speed_mps(predecessor))
        time_hw = getattr(self, "desired_headway_time_sec", 1.2)
        min_gap = getattr(self, "min_standstill_gap_m", 4.5)
        # Bounded weather-aware headway
        return _clamp(min_gap + time_hw * speed + 0.3 * closing + (1.0 - friction) * 6.0, 4.5, 30.0)

    def _closest_forward_obstacle(self, vehicle):
        best = None
        best_x = 80.0
        platoon_ids = {v.id for v in self.platoon if v and v.is_alive and v.id not in getattr(self, "disabled_vehicles", set())}
        try:
            actors = list(self.world.get_actors().filter("vehicle.*"))
            walkers = list(self.world.get_actors().filter("walker.pedestrian.*"))
            cursor = self.route.cursors.get(vehicle.id, 0) if hasattr(self, "route") and self.route else 0
            forward_wps = []
            if hasattr(self, "route") and self.route and hasattr(self.route, "route"):
                forward_wps = self.route.route[cursor : min(cursor + 15, len(self.route.route))]

            for other in actors + walkers:
                if other.id == vehicle.id or other.id in platoon_ids or not other.is_alive:
                    continue

                other_loc = other.get_location()
                x, y = self._local_components(vehicle, other)

                # 1. Strict forward check: discard any actor behind or level with ego vehicle.
                # A rear-approaching vehicle (e.g. ghost driver, follower) must NEVER be treated as a forward obstacle!
                if x <= 0.2 or x >= best_x:
                    continue

                # 2. Lateral corridor boundaries:
                # Vehicle half-width is ~1.0m; lane width is 3.0-3.5m.
                # Objects with |y| > 2.5m are in adjacent lanes or outside the roadway.
                if abs(y) > 2.5:
                    continue

                corridor_half = 1.65 if x < 20.0 else 1.85

                # 3. Drivable road check: if CARLA map is available, verify if other actor is on a driving lane
                if hasattr(self, "map") and self.map:
                    try:
                        other_wp = self.map.get_waypoint(other_loc, project_to_road=False)
                        if other_wp is None or getattr(other_wp, "lane_type", None) != carla.LaneType.Driving:
                            # Actor is on the sidewalk, shoulder, or off-road.
                            # Only treat as obstacle if it physically intrudes into the ego vehicle's width envelope (abs(y) < 1.35m)
                            if abs(y) > 1.35:
                                continue
                    except Exception:
                        pass

                # 4. Route corridor verification:
                is_in_path = False
                if forward_wps and x >= 4.0:
                    closest_wp = min(forward_wps, key=lambda wp: other_loc.distance(wp.transform.location))
                    wp_loc = closest_wp.transform.location
                    wp_fwd = closest_wp.transform.get_forward_vector()
                    dx_w = other_loc.x - wp_loc.x
                    dy_w = other_loc.y - wp_loc.y
                    wp_lat = abs(-dx_w * wp_fwd.y + dy_w * wp_fwd.x)
                    lane_w = getattr(closest_wp, "lane_width", 3.5)
                    half_w = min(1.75, lane_w * 0.5)
                    if wp_lat < half_w and abs(y) < corridor_half:
                        is_in_path = True
                else:
                    if abs(y) < corridor_half:
                        is_in_path = True

                if is_in_path and x < best_x:
                    best = other
                    best_x = x
        except Exception:
            pass
        return best, max(0.0, best_x - 4.7) if best else 80.0

    def _detect_intrusion(self, current_time):
        platoon_ids = {v.id for v in self.platoon if v and v.is_alive}
        detected = None
        for index in range(1, len(self.platoon)):
            ahead, behind = self.platoon[index - 1], self.platoon[index]
            if not ahead or not behind or not ahead.is_alive or not behind.is_alive:
                continue
            bx, by = self._local_components(behind, ahead)
            if bx <= 4.7 or bx > 28.0:
                continue
            for other in self.world.get_actors().filter("vehicle.*"):
                if other.id in platoon_ids or not other.is_alive:
                    continue
                ox, oy = self._local_components(behind, other)
                if 1.5 < ox < bx - 1.5 and abs(oy) < 3.2:
                    detected = (other, index - 1, index)
                    break
            if detected:
                break

        if detected:
            intruder, front_idx, rear_idx = detected
            if self.intruder_id != intruder.id:
                logger.warning(
                    "INTRUSION ghost=%d between platoon[%d]=%d and platoon[%d]=%d",
                    intruder.id, front_idx, self.platoon[front_idx].id,
                    rear_idx, self.platoon[rear_idx].id,
                )
                self.state = PlatoonState.INTRUSION
                self.state_since = current_time
            self.intruder_id = intruder.id
            self.intruded_between = (front_idx, rear_idx)
            if current_time - self.state_since > 0.5:
                self.state = PlatoonState.SPLIT
            self.rejoin_ready_since = None
            return

        if self.intruder_id is not None:
            # A removal from the immediate gap is necessary but not sufficient
            # for rejoin.  Hold safe-following until a gap remains available.
            self.intruder_id = None
            self.state = PlatoonState.SAFE_FOLLOWING
            self.state_since = current_time

        if self.state in (PlatoonState.SPLIT, PlatoonState.SAFE_FOLLOWING):
            all_rejoined = True
            for i in range(1, len(self.platoon)):
                a, b = self.platoon[i - 1], self.platoon[i]
                if a and b and a.is_alive and b.is_alive:
                    g = self._vehicle_clearance(a, b)
                    des = self._desired_headway(b, a)
                    if g > des * 1.4:
                        all_rejoined = False
                        break
            if all_rejoined:
                self.state = PlatoonState.NORMAL
                self.intruded_between = None
                MAPPODecisionLogger.get_instance().log_platoon_event(
                    current_time, 0, "Platoon", "REJOIN_COMPLETE", "All followers restored formation headway"
                )

        if self.state == PlatoonState.SAFE_FOLLOWING and self.intruded_between:
            rear = self.platoon[self.intruded_between[1]]
            front = self.platoon[self.intruded_between[0]]
            if rear and front and rear.is_alive and front.is_alive:
                gap = self._vehicle_clearance(front, rear)
                needed = self._desired_headway(rear, front) * 1.25
                if gap >= needed and not self._near_junction(rear):
                    if self.rejoin_ready_since is None:
                        self.rejoin_ready_since = current_time
                        self.state = PlatoonState.REJOIN_ELIGIBLE
                    elif current_time - self.rejoin_ready_since >= 2.0:
                        self.state = PlatoonState.REJOINING
                        MAPPODecisionLogger.get_instance().log_platoon_event(
                            current_time, rear.id, "Follower", "REJOIN_PENDING", f"Merging gap={gap:.1f}m needed={needed:.1f}m"
                        )
                else:
                    self.rejoin_ready_since = None
            if self.state == PlatoonState.REJOINING and current_time - self.state_since > 8.0:
                self.state = PlatoonState.NORMAL
                self.intruded_between = None
                MAPPODecisionLogger.get_instance().log_platoon_event(
                    current_time, 0, "Platoon", "REJOIN_COMPLETE", "Formation restored after rejoining"
                )

    def _near_junction(self, vehicle):
        try:
            waypoint = self.map.get_waypoint(vehicle.get_location(), project_to_road=True)
            if waypoint is None:
                return True
            return waypoint.is_junction or any(wp.is_junction for wp in waypoint.next(15.0))
        except Exception:
            return True

    def _traffic_light_constraint(self, vehicle):
        """Return (hard_stop, distance, label) only for this vehicle's relevant signal."""
        try:
            light = vehicle.get_traffic_light()
            state = vehicle.get_traffic_light_state()
            if not light:
                return False, 80.0, "NONE"

            loc = vehicle.get_location()
            tf = vehicle.get_transform()
            fwd = tf.get_forward_vector()

            # Identify stop line trigger volume center if available
            ext_x = 2.0
            if hasattr(light, "trigger_volume") and light.trigger_volume:
                try:
                    light_tf = light.get_transform()
                    light_loc = light_tf.transform(light.trigger_volume.location)
                    ext_x = getattr(light.trigger_volume.extent, "x", 2.0)
                except Exception:
                    light_loc = light.get_location()
            else:
                light_loc = light.get_location()

            # Longitudinal distance along vehicle forward heading
            dx = light_loc.x - loc.x
            dy = light_loc.y - loc.y
            longitudinal = dx * fwd.x + dy * fwd.y
            lateral = abs(-dx * fwd.y + dy * fwd.x)

            # 1. Ignore if signal pole/trigger is BEHIND vehicle (passed under or through)
            if longitudinal < -2.0:
                return False, 80.0, "CLEARED_LIGHT"

            # 2. Ignore if vehicle is already inside an intersection junction
            waypoint = self.map.get_waypoint(loc, project_to_road=True)
            if waypoint and waypoint.is_junction:
                return False, 80.0, "INSIDE_JUNCTION"

            # 3. Ignore if light is far to the side (adjacent junction/road)
            if lateral > 14.0:
                return False, 80.0, "ADJACENT_SIGNAL"

            label = str(state).split(".")[-1].upper()

            # CARLA native trigger check: True if vehicle is at the stop line box
            is_at_stop_line = False
            try:
                is_at_stop_line = vehicle.is_at_traffic_light()
            except Exception:
                pass

            hard_stop = False
            stop_dist = max(0.0, longitudinal - ext_x)

            if state == carla.TrafficLightState.Red:
                if is_at_stop_line or (0.0 < longitudinal < 35.0 and lateral < 10.0):
                    hard_stop = True
            elif state == carla.TrafficLightState.Yellow:
                # Dynamic stopping calculation based on speed, braking capability, and road friction
                speed = _speed_mps(vehicle)
                friction = self._friction(vehicle)
                a_brake = _clamp(3.5 * friction, 1.5, 4.5)
                d_needed = (speed ** 2) / (2.0 * a_brake) + speed * 0.35
                if is_at_stop_line:
                    hard_stop = True
                elif stop_dist < d_needed * 0.65 and speed > 2.0:
                    # Dilemma zone: cannot stop safely without harsh skidding; proceed
                    hard_stop = False
                    label = "AMBER_PROCEED"
                elif 0.0 < longitudinal < 28.0 and lateral < 10.0:
                    hard_stop = True

            effective_dist = stop_dist if hard_stop else 80.0
            return hard_stop, effective_dist, label
        except Exception:
            return False, 80.0, "UNKNOWN"

    def _route_geometry(self, vehicle):
        speed = _speed_mps(vehicle)
        waypoint, cursor, target_idx = self.route.target_for(vehicle, _clamp(5.0 + speed * 0.7, 6.0, 16.0))
        tf = vehicle.get_transform()
        target = waypoint.transform.location
        dx, dy = target.x - tf.location.x, target.y - tf.location.y
        yaw = math.radians(tf.rotation.yaw)
        local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
        local_y = -dx * math.sin(yaw) + dy * math.cos(yaw)

        # In CARLA UE4 left-handed coordinates, +Y is to the right.
        # lat_offset: negative = shift left, positive = shift right.
        ev_offset = getattr(self, "_lateral_offsets", {}).get(vehicle.id, 0.0)
        ot_offset = getattr(self, "overtake_lateral_offsets", {}).get(vehicle.id, 0.0)
        tracker = getattr(self, "overtake_sm", None).trackers.get(vehicle.id) if hasattr(self, "overtake_sm") else None
        if tracker and abs(tracker.current_lateral_offset) > 0.05:
            ot_offset = tracker.current_lateral_offset
        lat_offset = ev_offset if abs(ev_offset) > 0.1 else ot_offset
        effective_local_y = local_y - lat_offset

        alpha = math.atan2(effective_local_y, max(1.0, local_x))
        lookahead = max(4.0, math.hypot(local_x, effective_local_y))
        raw_steer = math.atan2(2.7 * math.sin(alpha), lookahead) * 1.8
        heading_error = math.radians(_angle_error_deg(tf.rotation.yaw, waypoint.transform.rotation.yaw))
        return waypoint, cursor, target_idx, _clamp(raw_steer, -0.65, 0.65), effective_local_y, heading_error

    def _safe_lane_change(self, vehicle, action):
        if action not in ("CHANGE_LEFT", "CHANGE_RIGHT") or self._near_junction(vehicle):
            return False
        try:
            waypoint = self.map.get_waypoint(vehicle.get_location(), project_to_road=True)
            candidate = waypoint.get_left_lane() if action == "CHANGE_LEFT" else waypoint.get_right_lane()
            if candidate is None or candidate.lane_type != carla.LaneType.Driving:
                return False
            for other in self.world.get_actors().filter("vehicle.*"):
                if other.id == vehicle.id or not other.is_alive:
                    continue
                other_wp = self.map.get_waypoint(other.get_location(), project_to_road=True)
                if other_wp and other_wp.road_id == candidate.road_id and other_wp.lane_id == candidate.lane_id:
                    if other.get_location().distance(candidate.transform.location) < 15.0:
                        return False
            return True
        except Exception:
            return False

    def _target_speed(self, vehicle, index, action, predecessor, obstacle, clearance):
        action_delta = {"ACCELERATE": 1.1, "DECELERATE": -1.8, "BRAKE": -8.0}.get(action, 0.0)
        friction = self._friction(vehicle) if vehicle else 1.0
        reason = "NORMAL"

        is_active_lead = (index == getattr(self, "active_leader_index", 0) or predecessor is None)

        if is_active_lead:
            # Leader: directly modulates baseline cruise speed
            target = (self.cruise_speed_mps + action_delta) * (0.65 + 0.35 * friction)
            reason = "LEADER_CRUISE"
        else:
            # Followers: configurable gap-aware longitudinal control
            pred_speed = _speed_mps(predecessor) if (predecessor and hasattr(predecessor, "get_velocity")) else self.cruise_speed_mps
            desired = self._desired_headway(vehicle, predecessor) if vehicle else 8.0
            gap = self._vehicle_clearance(predecessor, vehicle) if (vehicle and predecessor) else desired
            catchup_ratio = getattr(self, "catchup_gap_threshold_ratio", 1.15)
            max_catchup_inc = getattr(self, "catchup_max_speed_increment_mps", 3.5)
            max_allowed_speed = self.cruise_speed_mps + 4.0

            # 4 Gap Regimes:
            if pred_speed < 0.5 and gap < desired * 1.5:
                # Predecessor stopped: smoothly bring follower to safe stop
                stop_dist = max(0.0, gap - 3.8)
                target = math.sqrt(max(0.0, 2.0 * 2.5 * friction * stop_dist))
                reason = f"PREDECESSOR_STOPPED_GAP_{gap:.1f}M"
            elif gap < 3.8:
                # Critical proximity: must stop immediately
                target = 0.0
                reason = f"CRITICAL_PROXIMITY_GAP_{gap:.1f}M"
            elif gap < desired * 0.85:
                # Close proximity: restrict target speed below predecessor
                speed_deficit = min(3.5, (desired - gap) * 0.40)
                target = max(0.0, pred_speed - speed_deficit)
                reason = f"GAP_CLOSING_DECEL_GAP_{gap:.1f}M_DESIRED_{desired:.1f}M"
            elif gap > desired * catchup_ratio:
                # CATCH-UP REGIME: gap exceeds desired gap threshold, actively accelerate to catch up!
                gap_excess = gap - desired
                catch_up_delta = min(max_catchup_inc, max(0.8, gap_excess * 0.30))
                target = min(max_allowed_speed, max(pred_speed + catch_up_delta, self.cruise_speed_mps + 1.2))
                reason = f"CATCH_UP_ACTIVE_GAP_{gap:.1f}M_DESIRED_{desired:.1f}M_DELTA_{catch_up_delta:.1f}MPS"
            else:
                # Nominal headway tracking: smooth speed matching with fine bias
                gap_error = gap - desired
                headway_bias = _clamp(gap_error * 0.25, -1.0, 1.0)
                target = max(0.0, pred_speed + action_delta + headway_bias)
                reason = f"NOMINAL_TRACKING_GAP_{gap:.1f}M_DESIRED_{desired:.1f}M"

            # Record telemetry for logging
            if vehicle and hasattr(self, "follower_speed_telemetry"):
                self.follower_speed_telemetry[vehicle.id] = {
                    "gap": gap,
                    "desired_gap": desired,
                    "target_speed": target,
                    "current_speed": _speed_mps(vehicle) if vehicle else 0.0,
                    "reason": reason,
                }

        if obstacle is not None:
            safe_speed = math.sqrt(max(0.0, 2.0 * 3.0 * friction * max(0.0, clearance - 2.0)))
            if safe_speed < target:
                target = safe_speed
                reason = f"FORWARD_OBSTACLE_CLEARANCE_{clearance:.1f}M"

        if self.state == PlatoonState.INTRUSION and self.intruded_between:
            if index >= self.intruded_between[1]:
                target = min(target, self.cruise_speed_mps * 0.75)

        return _clamp(target, 0.0, self.cruise_speed_mps + 4.5)

    def _compute_longitudinal_control(
        self,
        target_speed,
        current_speed,
        hard_stop=False,
        light_distance=80.0,
        obstacle=None,
        obstacle_clearance=80.0,
        predecessor=None,
        predecessor_clearance=80.0,
        override="NONE",
        emergency_brake=False,
    ):
        """Compute feedforward-proportional throttle and brake with strict safety gates.

        Preserves all existing traffic-light, obstacle, headway, destination-stop,
        and emergency-braking behavior while addressing proportional steady-state speed droop.
        """
        # 0. Emergency brake override (e.g. from Module 3 Safety Shield)
        if emergency_brake:
            return 0.0, 1.0

        # 1. Commanded stop condition (destination, red light, obstacle, headway)
        if target_speed <= 0.05:
            throttle = 0.0
            is_critical = (
                (obstacle is not None and obstacle_clearance < 2.8)
                or (predecessor is not None and predecessor_clearance < 3.8)
                or (hard_stop and light_distance < 4.0)
            )
            brake = 1.0 if (current_speed > 0.1 or is_critical) else 0.5  # Hold brake while stationary
            return throttle, brake

        error = target_speed - current_speed
        overspeed_deadband = 0.35  # m/s (~1.25 km/h) to avoid hunting / chatter around target speed

        # 2. Driving throttle & brake calculation
        if error >= 0.0:
            # Under target speed: apply feedforward + proportional tracking
            k_ff = _clamp(0.038 * target_speed, 0.0, 0.38)
            k_p = 0.24
            throttle = _clamp(k_ff + k_p * error, 0.0, 0.65)
            brake = 0.0
        elif -overspeed_deadband <= error < 0.0:
            # Minor overspeed within deadband: coast naturally (zero throttle, zero brake)
            throttle = 0.0
            brake = 0.0
        else:
            # Genuine overspeed beyond deadband: proportional braking, feedforward inhibited
            throttle = 0.0
            brake = _clamp(-0.42 * (error + overspeed_deadband), 0.0, 1.0)

        # 3. Standstill breakout guard: MUST NEVER override any stop or safety constraint
        is_breakout_safe = (
            current_speed < 0.4
            and target_speed >= 2.5
            and error > 1.0
            and not hard_stop
            and light_distance >= 12.0
            and (obstacle is None or obstacle_clearance >= 6.0)
            and (predecessor is None or predecessor_clearance >= 6.0)
            and (override in ("", "NONE", "NORMAL") or "CATCH_UP" in override or "LANE_CHANGE" in override)
        )
        if is_breakout_safe:
            throttle = max(throttle, 0.38)
            brake = 0.0

        # 4. Existing safety constraint brake enforcements
        if hard_stop and light_distance < 8.0:
            if current_speed > 0.3:
                brake = max(brake, 0.40)
            elif current_speed < 0.4:
                brake = max(brake, 0.50)
        if predecessor and getattr(predecessor, "is_alive", False):
            if predecessor_clearance < 3.8:
                brake = 1.0
            elif predecessor_clearance < 5.0:
                if current_speed > 0.2:
                    brake = max(brake, 0.65)
                elif current_speed < 0.4:
                    brake = max(brake, 0.50)
        if obstacle and getattr(obstacle, "is_alive", False):
            if obstacle_clearance < 3.0:
                brake = 1.0
            elif obstacle_clearance < 5.0 and current_speed < 0.4:
                brake = max(brake, 0.50)

        # 5. Strict mutual exclusion: throttle and brake NEVER overlap
        if brake > 0.01:
            throttle = 0.0

        return throttle, brake

    def set_decision_telemetry(self, decision_map):
        """Receive detailed MAPPO decision records for tracing."""
        self._decision_map = dict(decision_map) if decision_map else {}

    def record_collision(self, vehicle_id: int, intensity: float = 1.0, current_time: float = 0.0):
        """Record collision event for vehicle health monitoring."""
        self.collision_events[int(vehicle_id)] = (float(current_time), float(intensity))

    def _set_leader_health_state(self, new_state: str, current_time: float, reason: str):
        if self.leader_health_state == new_state:
            return
        prev = self.leader_health_state
        self.leader_health_state = new_state
        self.leader_state_since = current_time
        record = {
            "time": round(float(current_time), 3),
            "from_state": prev,
            "to_state": new_state,
            "reason": str(reason),
            "active_leader_index": self.active_leader_index,
        }
        self.leader_state_history.append(record)
        logger.info(
            "[LEADER HEALTH] t=%.2fs | Transition: %s -> %s | Reason: %s",
            current_time, prev, new_state, reason,
        )

    def get_active_leader(self):
        """Return the authoritative active leader actor if valid."""
        if 0 <= self.active_leader_index < len(self.platoon):
            leader = self.platoon[self.active_leader_index]
            if leader and getattr(leader, "is_alive", False):
                return leader
        return None

    def _attempt_leader_recovery(self, leader, current_time):
        """Attempt safe non-destructive breakout recovery for stalled leader."""
        if not leader or not getattr(leader, "is_alive", False):
            return
        obstacle, clearance = self._closest_forward_obstacle(leader)
        hard_stop, _, _ = self._traffic_light_constraint(leader)
        if not hard_stop and (obstacle is None or clearance >= 4.0):
            try:
                leader.apply_control(carla.VehicleControl(throttle=0.35, brake=0.0, hand_brake=False))
            except Exception:
                pass

    def _check_leader_status(self, current_time):
        """Monitor active leader mobility and trigger failover state transitions."""
        if getattr(self, "active_leader_index", 0) >= len(self.platoon):
            return

        leader = self.platoon[self.active_leader_index]
        if not leader or not getattr(leader, "is_alive", False):
            self._set_leader_health_state(LeaderHealthState.FAILED, current_time, "LEADER_DESTROYED")
            self._handle_leader_promotion(current_time, "LEADER_DESTROYED")
            return

        col_info = self.collision_events.get(leader.id)
        has_collided = (
            col_info is not None
            and (current_time - col_info[0]) <= getattr(self, "leader_collision_timeout_sec", 5.0) * 2.0
        )

        leader_speed = _speed_mps(leader)
        stop_thresh = getattr(self, "leader_stopped_speed_threshold_mps", 0.2)
        recov_thresh = getattr(self, "leader_recovery_speed_threshold_mps", 0.35)

        if leader_speed < stop_thresh:
            if self._leader_stopped_since is None:
                self._leader_stopped_since = current_time

            duration = current_time - self._leader_stopped_since

            # Distinguish legitimate stops:
            # 1. Red traffic light
            hard_stop, tl_dist, tl_state = self._traffic_light_constraint(leader)
            if hard_stop:
                self._leader_stopped_since = current_time  # Reset while legitimately waiting at red light
                if self.leader_health_state != LeaderHealthState.ACTIVE:
                    self._set_leader_health_state(LeaderHealthState.ACTIVE, current_time, "WAITING_AT_RED_LIGHT")
                return

            # 2. Destination reached
            cursor = self.route.cursors.get(leader.id, 0)
            remaining_wps = len(self.route.route) - cursor
            dist_dest = leader.get_location().distance(self.route.destination)
            if remaining_wps <= 3 or dist_dest < 6.0:
                self._leader_stopped_since = current_time
                if self.leader_health_state != LeaderHealthState.ACTIVE:
                    self._set_leader_health_state(LeaderHealthState.ACTIVE, current_time, "DESTINATION_REACHED")
                return

            # 3. Temporary forward obstacle (pedestrian / NPC crossing / traffic jam)
            obstacle, clearance = self._closest_forward_obstacle(leader)
            if obstacle is not None and clearance < 8.0 and not has_collided:
                self._leader_stopped_since = current_time
                if self.leader_health_state != LeaderHealthState.ACTIVE:
                    self._set_leader_health_state(LeaderHealthState.ACTIVE, current_time, f"WAITING_FOR_OBSTACLE_{obstacle.id}")
                return

            # 4. Safety shield emergency brake intervention
            if (
                self.active_leader_index < len(self.emergency_brakes)
                and self.emergency_brakes[self.active_leader_index]
                and not has_collided
            ):
                self._leader_stopped_since = current_time
                return

            # If stopped following a collision beyond the collision timeout:
            col_timeout = getattr(self, "leader_collision_timeout_sec", 5.0)
            if has_collided and duration >= col_timeout:
                self._set_leader_health_state(LeaderHealthState.FAILED, current_time, f"COLLISION_IMMOBILIZED_{duration:.1f}S")
                self._handle_leader_promotion(current_time, f"COLLISION_IMMOBILIZED_{duration:.1f}S")
                return

            # Non-collision immobilization state machine
            suspect_timeout = getattr(self, "leader_suspect_timeout_sec", 3.0)
            fail_timeout = getattr(self, "leader_failure_timeout_sec", 8.0)

            if duration >= fail_timeout:
                self._set_leader_health_state(LeaderHealthState.FAILED, current_time, f"LEADER_IMMOBILIZED_{duration:.1f}S")
                self._handle_leader_promotion(current_time, f"LEADER_IMMOBILIZED_{duration:.1f}S")
            elif duration >= suspect_timeout:
                if self.leader_health_state == LeaderHealthState.ACTIVE:
                    self._set_leader_health_state(LeaderHealthState.SUSPECTED_FAILURE, current_time, f"IMMOBILIZED_{duration:.1f}S")
                if self.leader_health_state in (LeaderHealthState.SUSPECTED_FAILURE, LeaderHealthState.RECOVERY_ATTEMPT):
                    self._set_leader_health_state(LeaderHealthState.RECOVERY_ATTEMPT, current_time, "ATTEMPTING_BREAKOUT_RECOVERY")
                    self._attempt_leader_recovery(leader, current_time)
        else:
            # Moving: reset stopped timer
            self._leader_stopped_since = None
            if leader_speed >= recov_thresh and self.leader_health_state in (
                LeaderHealthState.SUSPECTED_FAILURE,
                LeaderHealthState.RECOVERY_ATTEMPT,
            ):
                self._set_leader_health_state(LeaderHealthState.ACTIVE, current_time, "LEADER_RESUMED_MOTION")

    def _handle_leader_promotion(self, current_time, reason):
        """Atomically promote next eligible follower to active leader and transfer route continuity."""
        if self.leader_health_state == LeaderHealthState.NEW_LEADER_ACTIVE and self._leader_failed:
            return

        self._set_leader_health_state(LeaderHealthState.PROMOTION_PENDING, current_time, reason)
        old_idx = self.active_leader_index
        old_leader = self.platoon[old_idx] if old_idx < len(self.platoon) else None
        if old_leader:
            self.disabled_vehicles.add(old_leader.id)
        self._leader_failed = True

        new_idx = None
        for i in range(old_idx + 1, len(self.platoon)):
            cand = self.platoon[i]
            if not cand or not getattr(cand, "is_alive", False) or cand.id in self.disabled_vehicles:
                continue
            # Validate candidate is controllable and has queryable location
            try:
                loc = cand.get_location()
                if loc is not None and math.isfinite(loc.x) and math.isfinite(loc.y):
                    new_idx = i
                    break
            except Exception:
                continue

        if new_idx is None:
            self._set_leader_health_state(LeaderHealthState.NO_ELIGIBLE_LEADER, current_time, "NO_ELIGIBLE_FOLLOWERS")
            logger.error("[FAILOVER] No eligible live followers available to promote. Platoon holding safe state.")
            return

        self.active_leader_index = new_idx
        new_leader = self.platoon[new_idx]
        self.promotion_events_count += 1

        # Re-target predecessors
        self.predecessor_map[new_idx] = None
        last_v = new_idx
        for i in range(new_idx + 1, len(self.platoon)):
            if self.platoon[i] and getattr(self.platoon[i], "is_alive", False) and self.platoon[i].id not in self.disabled_vehicles:
                self.predecessor_map[i] = last_v
                last_v = i

        # Transfer route progress to promoted leader
        cursor = self.route.progress_for(new_leader)
        self.route.cursors[new_leader.id] = cursor
        try:
            self.route.target_for(new_leader, lookahead_m=6.0)
        except Exception:
            pass

        self._set_leader_health_state(LeaderHealthState.NEW_LEADER_ACTIVE, current_time, f"PROMOTED_VEHICLE_{new_leader.id}")

        # Stream failover event to MAPPO logger
        try:
            from Module2.logger import MAPPODecisionLogger
            MAPPODecisionLogger.get_instance().log_role_failover(
                sim_time_sec=current_time,
                old_leader_id=old_leader.id if old_leader else 0,
                promoted_leader_id=new_leader.id,
                reason=reason,
            )
        except Exception:
            pass

        # Stream failover event to Module 3 logger
        try:
            from Module3.logger import Module3DecisionLogger
            Module3DecisionLogger.get_instance().log_state_transition(
                sim_time_sec=current_time,
                actor_id=new_leader.id,
                prev_state=f"Follower {new_idx}",
                new_state="Active Leader",
                reason=f"LEADER_FAILOVER: {reason}",
            )
        except Exception:
            pass

        logger.warning(
            "\n" + "=" * 76 + "\n"
            f"  [AUTOMATIC LEADER FAILOVER] Reason: {reason} | SimTime: {current_time:.2f}s\n"
            f"  Old Leader: Index {old_idx} (ID {old_leader.id if old_leader else 'None'}) -> DISABLED\n"
            f"  Promoted Leader: Index {new_idx} (ID {new_leader.id}) -> ACTIVE LEADER\n"
            f"  New Leader Route Cursor: {cursor} / {len(self.route.route)}\n"
            f"  Predecessor Hierarchy: {self.predecessor_map}\n"
            + "=" * 76 + "\n"
        )

    def _trigger_leader_failover(self, current_time, reason):
        """Backward-compatible entry point for leader failover."""
        self._handle_leader_promotion(current_time, reason)

    def tick(self, current_time):
        """Apply exactly one bounded control per live platoon vehicle."""
        self._detect_intrusion(current_time)
        self._check_leader_status(current_time)
        frame = 0
        try:
            frame = self.world.get_snapshot().frame
        except Exception:
            pass

        for index, vehicle in enumerate(self.platoon):
            if not vehicle or not vehicle.is_alive:
                continue

            # Command full hold brake for disabled vehicles
            if vehicle.id in getattr(self, "disabled_vehicles", set()):
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
                continue

            action = self.tactical_actions[index]
            pred_idx = self.predecessor_map.get(index)
            predecessor = self.platoon[pred_idx] if (pred_idx is not None and pred_idx < len(self.platoon)) else None

            waypoint, cursor, target_idx, raw_steer, lateral, heading_error = self._route_geometry(vehicle)
            previous_steer = self.last_steer.get(vehicle.id, 0.0)
            steer = _clamp(previous_steer * 0.78 + raw_steer * 0.22, previous_steer - 0.08, previous_steer + 0.08)
            self.last_steer[vehicle.id] = steer

            # Active leader dynamically extends route horizon if approaching current end
            if index == getattr(self, "active_leader_index", 0):
                remaining_wps = len(self.route.route) - cursor
                if remaining_wps < 45:
                    self.route.extend_route(distance_m=150.0)

            obstacle, clearance = self._closest_forward_obstacle(vehicle)
            hard_stop, light_distance, light_state = self._traffic_light_constraint(vehicle)
            self._last_traffic_light_stop = hard_stop
            override = ""

            # ── INTELLIGENT OVERTAKING EVALUATION ──
            # Evaluate overtake state machine for active leader and followers
            veh_role = "Active Leader" if index == getattr(self, "active_leader_index", 0) else f"Follower {index}"
            ot_act, ot_state, ot_lat_offset = self.overtake_sm.step(
                vehicle_idx=index,
                ego=vehicle,
                role=veh_role,
                obstacle=obstacle,
                obstacle_clearance=clearance,
                controller=self,
                current_time=current_time,
            )
            self.overtake_lateral_offsets[vehicle.id] = ot_lat_offset

            # Tactical action arbitration: prioritize evasion if active, then overtake, then policy action
            ev_offset = getattr(self, "_lateral_offsets", {}).get(vehicle.id, 0.0)
            is_evading = abs(ev_offset) > 1.0
            is_overtaking = abs(ot_lat_offset) > 1.0 or ot_state in (
                OvertakeState.LANE_CHANGE_OUT,
                OvertakeState.LANE_CHANGE_IN_PROGRESS,
                OvertakeState.PASSING,
                OvertakeState.RETURN_GAP_VALIDATION,
                OvertakeState.LANE_CHANGE_BACK,
            )

            if is_evading:
                self.lane_states[index] = "EVASION_LANE_CHANGE"
                override = "EVASION_LANE_CHANGE"
            elif is_overtaking and ot_act in ("CHANGE_LEFT", "CHANGE_RIGHT", "ACCELERATE"):
                action = ot_act
                self.lane_states[index] = f"OVERTAKE_{ot_state.value}"
                override = f"OVERTAKE_{action}"
            elif action in ("CHANGE_LEFT", "CHANGE_RIGHT") and not is_overtaking and not is_evading:
                dir_req = "LEFT" if action == "CHANGE_LEFT" else "RIGHT"
                ok, reason, req_offset = self.overtake_sm.request_lane_change(
                    index, vehicle, veh_role, dir_req, self, current_time
                )
                if ok:
                    self.overtake_lateral_offsets[vehicle.id] = req_offset
                    self.lane_states[index] = f"LANE_CHANGE_{action}"
                    override = f"LANE_CHANGE_{action}"
                else:
                    self.lane_states[index] = f"REQUEST_REJECTED_{reason}"
                    override = f"LANE_CHANGE_REJECTED_{reason}"
            elif is_overtaking:
                self.lane_states[index] = f"OVERTAKE_{ot_state.value}"
                override = f"OVERTAKE_{ot_state.value}"
            else:
                self.lane_states[index] = "NORMAL"

            target_speed = self._target_speed(vehicle, index, action, predecessor, obstacle, clearance)
            speed = _speed_mps(vehicle)

            stop_reason = "NONE"
            stop_source = "NONE"

            # Observability: Rate-limited follower wait state diagnostic
            if index > 0 and (target_speed <= 0.05 or speed < 0.5):
                if not hasattr(self, "_last_wait_log"):
                    self._last_wait_log = {}
                last_w = self._last_wait_log.get(vehicle.id, -999.0)
                if current_time - last_w >= 3.0:
                    self._last_wait_log[vehicle.id] = current_time
                    w_rsn = "RED_LIGHT_STOP" if hard_stop else ("LOW_TARGET_SPEED" if target_speed <= 0.05 else "HEADWAY_WAIT")
                    MAPPODecisionLogger.get_instance().log_platoon_event(
                        current_time, vehicle.id, veh_role, "FOLLOWER_WAIT_REASON", w_rsn
                    )

            # Check if destination complete
            remaining_wps = len(self.route.route) - cursor
            dist_to_dest = vehicle.get_location().distance(self.route.destination)
            is_dest_reached = (remaining_wps <= 3) and (dist_to_dest < 6.0)

            if is_dest_reached:
                target_speed = 0.0
                override = "destination_complete"
                stop_reason = "DESTINATION_REACHED"
                stop_source = "RoutePlanner.destination_check"
            elif hard_stop:
                stopping = max(0.0, light_distance - 2.0)
                light_speed = math.sqrt(max(0.0, 2.0 * 2.8 * self._friction(vehicle) * stopping))
                if stopping < 1.0:
                    target_speed = 0.0
                else:
                    target_speed = min(target_speed, light_speed)
                override = "RED_LIGHT_STOP" if light_state == "RED" else "AMBER_STOP"
                if target_speed <= 0.05:
                    stop_reason = override
                    stop_source = "_traffic_light_constraint"
            elif obstacle is not None and clearance < 3.0:
                target_speed = 0.0
                override = "COLLISION_RISK"
                stop_reason = "COLLISION_RISK"
                stop_source = "_closest_forward_obstacle"
            elif predecessor and predecessor.is_alive and self._vehicle_clearance(predecessor, vehicle) < 3.8:
                target_speed = 0.0
                override = "INSUFFICIENT_HEADWAY"
                stop_reason = "INSUFFICIENT_HEADWAY"
                stop_source = "_desired_headway"
            elif target_speed <= 0.05:
                stop_reason = "SAFETY_OVERRIDE" if override else "CONTROLLER_STOP"
                stop_source = "_target_speed"

            # Longitudinal throttle and brake control
            pred_gap = self._vehicle_clearance(predecessor, vehicle) if (predecessor and predecessor.is_alive) else 80.0
            is_emerg_brake = self.emergency_brakes[index] if index < len(self.emergency_brakes) else False
            if is_emerg_brake:
                override = override or "MODULE3_EMERGENCY_BRAKE"
                stop_reason = "MODULE3_EMERGENCY_BRAKE"
                stop_source = "Module3.SafetyShield"
            throttle, brake = self._compute_longitudinal_control(
                target_speed=target_speed,
                current_speed=speed,
                hard_stop=hard_stop,
                light_distance=light_distance,
                obstacle=obstacle,
                obstacle_clearance=clearance,
                predecessor=predecessor,
                predecessor_clearance=pred_gap,
                override=override or "NONE",
                emergency_brake=is_emerg_brake,
            )

            # Centralized Stop Transition and Recovery Diagnostic
            prev_target = self._prev_target_speed.get(vehicle.id, self.cruise_speed_mps)
            was_stopping_before = (prev_target <= 0.05)
            is_stopping_now = (target_speed <= 0.05)
            if vehicle.id in getattr(self, "disabled_vehicles", set()):
                role = "Disabled Leader"
            elif index == getattr(self, "active_leader_index", 0):
                role = "Active Leader" if index > 0 else "Leader"
            else:
                role = "Follower %d" % index

            # Transition: moving -> stopped
            if is_stopping_now and not was_stopping_before:
                logger.warning(
                    "\n" + "=" * 68 + "\n"
                    "  [STOP EVENT] Frame: %d | Time: %.2fs | veh_id=%d (%s)\n"
                    "  EXACT STOP REASON:   %s\n"
                    "  SOURCE FUNCTION:     %s\n"
                    "  Target Speed:        %.1f km/h -> %.1f km/h\n"
                    "  Actual Speed:        %.1f km/h | thr=%.2f brk=%.2f steer=%.3f\n"
                    "  Tactical Action:     %s\n"
                    "  Route Cursor:        %d / %d (%.1f%%) | Dist to Next WP: %.1fm | Dist to Dest: %.1fm\n"
                    "  Traffic Light:       %s (dist=%.1fm)\n"
                    "  Forward Obstacle:    %s (dist=%.1fm)\n"
                    "  Predecessor:         %s (gap=%.1fm)\n"
                    "  Safety Override:     %s\n"
                    + "=" * 68,
                    frame, current_time, vehicle.id, role,
                    stop_reason, stop_source,
                    prev_target * 3.6, target_speed * 3.6,
                    speed * 3.6, throttle, brake, steer,
                    action,
                    cursor, len(self.route.route), (cursor / max(1, len(self.route.route) - 1)) * 100.0,
                    vehicle.get_location().distance(waypoint.transform.location),
                    dist_to_dest,
                    light_state, light_distance,
                    obstacle.id if obstacle else "None", clearance,
                    predecessor.id if predecessor else "None",
                    self._vehicle_clearance(predecessor, vehicle) if predecessor else 80.0,
                    override or "NONE"
                )
                self._stop_reasons[vehicle.id] = stop_reason
                self._stop_sources[vehicle.id] = stop_source

            # Transition: stopped -> resumed driving
            elif not is_stopping_now and was_stopping_before:
                cleared_reason = self._stop_reasons.get(vehicle.id, "NONE")
                logger.info(
                    "[STOP RECOVERY] Frame: %d | Time: %.2fs | veh_id=%d (%s) RESUMING DRIVE: "
                    "prev_target=%.1f km/h -> new_target=%.1f km/h | actual_speed=%.1f km/h | tactical=%s | cleared_condition=%s",
                    frame, current_time, vehicle.id, role,
                    prev_target * 3.6, target_speed * 3.6,
                    speed * 3.6, action, cleared_reason
                )
                self._stop_reasons[vehicle.id] = "NONE"
                self._stop_sources[vehicle.id] = "NONE"

            self._prev_target_speed[vehicle.id] = target_speed

            control = carla.VehicleControl(
                throttle=float(throttle), brake=float(brake), steer=float(_clamp(steer, -0.65, 0.65)),
                hand_brake=False, reverse=False,
            )
            vehicle.apply_control(control)
            self.last_control[vehicle.id] = control
            self._last_override[index] = override

            # Match with unique decision record if available
            dec_rec = getattr(self, "_decision_map", {}).get(vehicle.id, {})
            dec_id = dec_rec.get("decision_id", f"DEC-{frame}-{vehicle.id}")

            self._last_context[index] = {
                "decision_id": dec_id,
                "route_progress": cursor / float(max(1, len(self.route.route) - 1)),
                "distance_to_waypoint": vehicle.get_location().distance(waypoint.transform.location),
                "lateral_error": lateral,
                "heading_error": heading_error,
                "target_speed": target_speed,
                "traffic_light": light_state,
                "traffic_light_distance": light_distance,
                "predecessor_id": predecessor.id if predecessor else -1,
                "predecessor_gap": self._vehicle_clearance(predecessor, vehicle) if predecessor and predecessor.is_alive else 80.0,
                "nearby_clearance": clearance,
                "intrusion": 1.0 if self.intruder_id else 0.0,
                "lane_state": self.lane_states[index],
                "platoon_state": self.state,
                "safety_override": override or "NONE",
                "stop_reason": stop_reason if is_stopping_now else "NONE",
                "stop_source": stop_source if is_stopping_now else "NONE",
                "is_stopped": is_stopping_now,
                "dist_to_destination": dist_to_dest,
                "controller_owner": self.CONTROL_OWNER,
                "waypoint_index": target_idx,
                "applied_throttle": throttle,
                "applied_brake": brake,
                "applied_steer": steer,
                "role": role,
                "active_leader_index": self.active_leader_index,
                "leader_health_state": self.leader_health_state,
            }

            # Record structured decision entry connecting MAPPO inference to applied control
            rec = {
                "decision_id": dec_id,
                "frame": frame,
                "sim_time": round(current_time, 3),
                "vehicle_id": vehicle.id,
                "role": role,
                "checkpoint_path": dec_rec.get("checkpoint_path", "unknown"),
                "update_step": dec_rec.get("update_step", 0),
                "obs_shape": dec_rec.get("obs_shape", [21]),
                "obs_valid": dec_rec.get("obs_valid", True),
                "inference_mode": dec_rec.get("mode", "evaluate"),
                "logits": dec_rec.get("logits"),
                "probs": dec_rec.get("probs"),
                "sampled_action": dec_rec.get("raw_action"),
                "tactical_action": action,
                "requested_target_speed_mps": round(target_speed, 2),
                "safety_override": override or "NONE",
                "override_reason": stop_reason if is_stopping_now and stop_reason != "NONE" else (override or "NONE"),
                "final_throttle": round(float(throttle), 3),
                "final_brake": round(float(brake), 3),
                "final_steer": round(float(steer), 3),
                "speed_before_mps": round(speed, 2),
            }
            self._decision_log_buffer.append(rec)

        # Periodic non-blocking flush to mappo_decisions.log
        if len(self._decision_log_buffer) >= 20 or (current_time - self._last_flush_time >= 1.0 and self._decision_log_buffer):
            try:
                os.makedirs(os.path.dirname(self._decision_log_file), exist_ok=True)
                with open(self._decision_log_file, "a") as f:
                    for r in self._decision_log_buffer:
                        f.write(json.dumps(r) + "\n")
                self._decision_log_buffer = []
                self._last_flush_time = current_time
            except Exception:
                pass

    def observation_context(self, vehicle, index):
        if 0 <= index < len(self._last_context):
            return dict(self._last_context[index])
        return {}

    def diagnostics(self, vehicle, index):
        return self.observation_context(vehicle, index)
