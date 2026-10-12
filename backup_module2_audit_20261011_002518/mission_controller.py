"""Route-owned, safety-constrained CARLA control for Module 2.

This module deliberately owns *all* ``apply_control`` calls for platoon
actors.  Traffic Manager remains free to control NPC/ghost actors, but never
the platoon.  The policy supplies a low-rate tactical intent; this controller
turns that intent into a bounded, route-following vehicle command every CARLA
tick.
"""

from __future__ import annotations

import logging
import math

import carla


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

        # Lateral corridor: vehicle must be within reasonable lateral distance of segment.
        max_lateral = max(4.5, self.spacing_m * 2.0)
        # Reach corridor: vehicle must be in the spatial vicinity of the active segment.
        max_reach = max(10.0, self.spacing_m * 4.0)

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

            # Proximity corridor check: reject if vehicle is far from this segment.
            if lat_offset > max_lateral:
                break
            if dist_curr > max_reach and dist_next > max_reach:
                break

            # Sequential passing condition:
            # The vehicle has physically progressed through curr_wp toward next_wp
            # if it has covered at least half the segment along the forward tangent
            # and is closer to next_wp, or has passed next_wp longitudinally.
            passed_midpoint = (s_proj >= seg_len * 0.5) and (dist_next < dist_curr)
            passed_segment = s_proj >= seg_len

            if passed_midpoint or passed_segment:
                cursor += 1
                advanced += 1
            else:
                break

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
        # Centralized stop tracking and diagnostics
        self._prev_target_speed = {}
        self._stop_reasons = {}
        self._stop_sources = {}
        logger.info(
            "Mission route created: %d waypoints, destination=(%.1f, %.1f)",
            len(self.route.route), self.route.destination.x, self.route.destination.y,
        )

    def set_tactical_actions(self, actions):
        if len(actions) != len(self.platoon):
            raise ValueError("tactical actions must have one entry per platoon vehicle")
        self.tactical_actions = list(actions)

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
        # Reaction distance plus friction-sensitive braking distance.
        return _clamp(5.0 + 0.9 * speed + 0.25 * closing + (1.0 - friction) * 10.0, 6.0, 30.0)

    def _closest_forward_obstacle(self, vehicle):
        best = None
        best_x = 80.0
        platoon_ids = {v.id for v in self.platoon if v and v.is_alive}
        try:
            actors = list(self.world.get_actors().filter("vehicle.*"))
            walkers = list(self.world.get_actors().filter("walker.pedestrian.*"))
            for other in actors + walkers:
                if other.id == vehicle.id or other.id in platoon_ids or not other.is_alive:
                    continue
                x, y = self._local_components(vehicle, other)
                # In-lane forward corridor:
                # Car half-width is ~0.9m; lane width is 3.0-3.5m.
                # A narrow corridor (1.65m near, 1.95m far) excludes vehicles in
                # adjacent lanes (y > 2.5m) while reliably detecting in-lane hazards.
                corridor_half = 1.65 if x < 20.0 else 1.95
                if 0.0 < x < best_x and abs(y) < corridor_half:
                    best, best_x = other, x
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
            if bx <= 4.7:
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
                else:
                    self.rejoin_ready_since = None
            if self.state == PlatoonState.REJOINING and current_time - self.state_since > 8.0:
                self.state = PlatoonState.NORMAL
                self.intruded_between = None

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
            light_loc = light.get_location()

            # Longitudinal distance along vehicle forward heading
            dx = light_loc.x - loc.x
            dy = light_loc.y - loc.y
            longitudinal = dx * fwd.x + dy * fwd.y
            lateral = abs(-dx * fwd.y + dy * fwd.x)

            # 1. Ignore if signal pole is BEHIND vehicle (passed under or through)
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
            if state == carla.TrafficLightState.Red:
                if is_at_stop_line:
                    hard_stop = True
                elif 0.0 < longitudinal < 28.0 and lateral < 10.0:
                    hard_stop = True
            elif state == carla.TrafficLightState.Yellow:
                if is_at_stop_line or (8.0 < longitudinal < 20.0 and lateral < 10.0):
                    hard_stop = True

            effective_dist = max(0.0, longitudinal) if hard_stop else 80.0
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
        alpha = math.atan2(local_y, max(1.0, local_x))
        lookahead = max(4.0, math.hypot(local_x, local_y))
        # In CARLA UE4 left-handed coordinates, +Y is to the right.
        # Negative steer in VehicleControl is a left turn, and positive steer is a right turn.
        # When local_y > 0 (waypoint to the right), raw_steer > 0 (steer right).
        # When local_y < 0 (waypoint to the left), raw_steer < 0 (steer left).
        raw_steer = math.atan2(2.7 * math.sin(alpha), lookahead) * 1.8
        heading_error = math.radians(_angle_error_deg(tf.rotation.yaw, waypoint.transform.rotation.yaw))
        return waypoint, cursor, target_idx, _clamp(raw_steer, -0.65, 0.65), local_y, heading_error

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
        target = self.cruise_speed_mps
        target += {"ACCELERATE": 1.1, "DECELERATE": -1.8, "BRAKE": -8.0}.get(action, 0.0)
        target *= 0.65 + 0.35 * self._friction(vehicle)
        desired = self._desired_headway(vehicle, predecessor)
        if predecessor and predecessor.is_alive:
            pred_speed = _speed_mps(predecessor)
            gap = self._vehicle_clearance(predecessor, vehicle)
            if pred_speed < 1.0 and gap < desired * 1.5:
                stop_dist = max(0.0, gap - 3.8)
                target = min(target, math.sqrt(max(0.0, 2.0 * 2.5 * stop_dist)))
            elif gap < desired:
                target = min(target, pred_speed - min(2.0, (desired - gap) * 0.30))
            elif gap < desired * 1.2:
                target = min(target, pred_speed + 0.8)
            else:
                # Gap recovery: actively close distance when platoon predecessor is ahead
                catch_up_bonus = min(2.5, (gap - desired) * 0.12)
                target = min(self.cruise_speed_mps + 2.0, pred_speed + catch_up_bonus)
        if obstacle is not None:
            safe_speed = math.sqrt(max(0.0, 2.0 * 3.0 * self._friction(vehicle) * max(0.0, clearance - 2.0)))
            target = min(target, safe_speed)
        if self.state in (PlatoonState.INTRUSION, PlatoonState.SPLIT, PlatoonState.SAFE_FOLLOWING) and self.intruded_between:
            if index >= self.intruded_between[1]:
                target = min(target, self.cruise_speed_mps * 0.65)
        return _clamp(target, 0.0, self.cruise_speed_mps + 2.0)

    def tick(self, current_time):
        """Apply exactly one bounded control per live platoon vehicle."""
        self._detect_intrusion(current_time)
        for index, vehicle in enumerate(self.platoon):
            if not vehicle or not vehicle.is_alive:
                continue
            action = self.tactical_actions[index]
            predecessor = self.platoon[index - 1] if index else None
            waypoint, cursor, target_idx, raw_steer, lateral, heading_error = self._route_geometry(vehicle)
            previous_steer = self.last_steer.get(vehicle.id, 0.0)
            steer = _clamp(previous_steer * 0.78 + raw_steer * 0.22, previous_steer - 0.08, previous_steer + 0.08)
            self.last_steer[vehicle.id] = steer
            # Leader dynamically extends route horizon if approaching current end
            if index == 0:
                remaining_wps = len(self.route.route) - cursor
                if remaining_wps < 45:
                    self.route.extend_route(distance_m=150.0)

            obstacle, clearance = self._closest_forward_obstacle(vehicle)
            hard_stop, light_distance, light_state = self._traffic_light_constraint(vehicle)
            override = ""
            if action in ("CHANGE_LEFT", "CHANGE_RIGHT"):
                if self._safe_lane_change(vehicle, action):
                    # Current mission lane remains route-compatible.  The
                    # action is retained as a stateful request for telemetry,
                    # but no transform/teleport is ever used.
                    self.lane_states[index] = "REQUEST_REJECTED_ROUTE_LOCK"
                    override = "route_lane_lock"
                else:
                    self.lane_states[index] = "REQUEST_REJECTED_UNSAFE"
                    override = "unsafe_lane_change"
            else:
                self.lane_states[index] = "NORMAL"

            target_speed = self._target_speed(vehicle, index, action, predecessor, obstacle, clearance)
            speed = _speed_mps(vehicle)

            stop_reason = "NONE"
            stop_source = "NONE"

            # Check if route terminus has actually been reached (only if road cannot be extended
            # AND vehicle has physically arrived at the destination location).
            remaining_wps = len(self.route.route) - cursor
            dist_to_dest = vehicle.get_location().distance(self.route.destination)
            is_dest_reached = (remaining_wps <= 3) and (dist_to_dest < 6.0)

            if is_dest_reached:
                target_speed = 0.0
                override = "destination_complete"
                stop_reason = "DESTINATION_REACHED"
                stop_source = "RoutePlanner.destination_check"
            elif hard_stop:
                stopping = max(0.0, light_distance - 3.5)
                light_speed = math.sqrt(max(0.0, 2.0 * 2.5 * self._friction(vehicle) * stopping))
                if stopping < 1.5:
                    target_speed = 0.0
                else:
                    target_speed = min(target_speed, light_speed)
                override = "traffic_light_%s" % light_state.lower()
                if target_speed <= 0.05:
                    stop_reason = "RED_LIGHT_STOP"
                    stop_source = "_traffic_light_constraint"
            elif obstacle is not None and clearance < 3.0:
                target_speed = 0.0
                override = "obstacle_emergency"
                stop_reason = "COLLISION_RISK"
                stop_source = "_closest_forward_obstacle"
            elif predecessor and predecessor.is_alive and self._vehicle_clearance(predecessor, vehicle) < 3.6:
                target_speed = 0.0
                override = "headway_emergency"
                stop_reason = "UNSAFE_HEADWAY"
                stop_source = "_desired_headway"
            elif target_speed <= 0.05:
                stop_reason = "SAFETY_OVERRIDE" if override else "CONTROLLER_STOP"
                stop_source = "_target_speed"

            # Longitudinal throttle and brake control
            error = target_speed - speed
            if target_speed <= 0.05:
                throttle = 0.0
                brake = 1.0 if speed > 0.1 else 0.5  # hold brake while stationary
            else:
                throttle = _clamp(0.23 * error, 0.0, 0.65)
                # Overcome vehicle static inertia when transitioning from standstill to drive
                if speed < 0.5 and error > 1.0:
                    throttle = max(throttle, 0.38)
                brake = _clamp(-0.42 * error, 0.0, 1.0)
                if hard_stop and light_distance < 8.0 and speed > 0.3:
                    brake = max(brake, 0.35)
                if predecessor and predecessor.is_alive:
                    pred_gap = self._vehicle_clearance(predecessor, vehicle)
                    if pred_gap < 5.0 and speed > 0.2:
                        brake = max(brake, 0.6)
                    if pred_gap < 3.8:
                        brake = 1.0
                if brake > 0.02:
                    throttle = 0.0

            # Centralized Stop Transition and Recovery Diagnostic
            prev_target = self._prev_target_speed.get(vehicle.id, self.cruise_speed_mps)
            is_stopping_now = (target_speed <= 0.05)
            was_stopping_before = (prev_target <= 0.05)
            role = "Leader" if index == 0 else "Follower %d" % index

            frame = 0
            try:
                frame = self.world.get_snapshot().frame
            except Exception:
                pass

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
            self._last_context[index] = {
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
                "safety_override": override,
                "stop_reason": stop_reason if is_stopping_now else "NONE",
                "stop_source": stop_source if is_stopping_now else "NONE",
                "is_stopped": is_stopping_now,
                "dist_to_destination": dist_to_dest,
                "controller_owner": self.CONTROL_OWNER,
                "waypoint_index": target_idx,
            }

    def observation_context(self, vehicle, index):
        if 0 <= index < len(self._last_context):
            return dict(self._last_context[index])
        return {}

    def diagnostics(self, vehicle, index):
        return self.observation_context(vehicle, index)
