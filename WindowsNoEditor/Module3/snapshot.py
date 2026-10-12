# WindowsNoEditor/Module3/snapshot.py
"""Immutable Safety Snapshot extracted from the authoritative CARLA world (World A).

Extracts read-only vehicle states, obstacles, traffic lights, and environmental conditions.
Safely handles destroyed actors, missing fields, non-finite values, and sensor dropouts.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (ValueError, TypeError):
        return default


class Vector2D:
    """Read-only 2D vector for safety calculations in the road plane."""
    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float):
        self.x = _safe_float(x)
        self.y = _safe_float(y)

    def distance_to(self, other: "Vector2D") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def __repr__(self) -> str:
        return f"Vector2D({self.x:.2f}, {self.y:.2f})"


class VehicleSnapshot:
    """Immutable representation of a single vehicle's state."""
    __slots__ = (
        "id", "role", "is_alive", "position", "speed", "heading_rad",
        "length", "width", "friction", "predecessor_id", "predecessor_gap"
    )

    def __init__(
        self,
        vehicle_id: int,
        role: str,
        is_alive: bool,
        position: Vector2D,
        speed: float,
        heading_rad: float,
        length: float = 4.7,
        width: float = 2.0,
        friction: float = 1.0,
        predecessor_id: Optional[int] = None,
        predecessor_gap: float = 80.0,
    ):
        self.id = int(vehicle_id)
        self.role = str(role)
        self.is_alive = bool(is_alive)
        self.position = position
        self.speed = max(0.0, _safe_float(speed))
        self.heading_rad = _safe_float(heading_rad)
        self.length = max(1.0, _safe_float(length, 4.7))
        self.width = max(0.5, _safe_float(width, 2.0))
        self.friction = max(0.1, min(1.5, _safe_float(friction, 1.0)))
        self.predecessor_id = predecessor_id
        self.predecessor_gap = max(0.0, _safe_float(predecessor_gap, 80.0))

    def __repr__(self) -> str:
        return f"VehicleSnapshot(id={self.id}, role={self.role}, speed={self.speed:.1f}m/s, gap={self.predecessor_gap:.1f}m)"


class ObstacleSnapshot:
    """Immutable representation of an external obstacle (NPC vehicle, pedestrian)."""
    __slots__ = ("id", "position", "speed", "heading_rad", "length", "width")

    def __init__(
        self,
        obstacle_id: int,
        position: Vector2D,
        speed: float,
        heading_rad: float,
        length: float = 4.5,
        width: float = 2.0,
    ):
        self.id = int(obstacle_id)
        self.position = position
        self.speed = max(0.0, _safe_float(speed))
        self.heading_rad = _safe_float(heading_rad)
        self.length = max(0.5, _safe_float(length, 4.5))
        self.width = max(0.5, _safe_float(width, 2.0))


class TrafficLightSnapshot:
    """Immutable traffic light state along the vehicle's path."""
    __slots__ = ("state", "distance", "is_hard_stop")

    def __init__(self, state: str, distance: float, is_hard_stop: bool):
        self.state = str(state).upper()
        self.distance = max(0.0, _safe_float(distance, 80.0))
        self.is_hard_stop = bool(is_hard_stop)


class WorldSnapshot:
    """Top-level immutable snapshot of World A at a specific simulation timestamp."""
    __slots__ = (
        "timestamp", "frame", "platoon", "obstacles", "traffic_light",
        "weather_preset", "visibility_m", "nominal_friction", "is_valid", "error_message"
    )

    def __init__(
        self,
        timestamp: float,
        frame: int,
        platoon: List[VehicleSnapshot],
        obstacles: List[ObstacleSnapshot],
        traffic_light: TrafficLightSnapshot,
        weather_preset: str = "CLEAR",
        visibility_m: float = 120.0,
        nominal_friction: float = 1.0,
        is_valid: bool = True,
        error_message: str = "",
    ):
        self.timestamp = _safe_float(timestamp)
        self.frame = int(frame)
        self.platoon = tuple(platoon)
        self.obstacles = tuple(obstacles)
        self.traffic_light = traffic_light
        self.weather_preset = str(weather_preset)
        self.visibility_m = max(5.0, _safe_float(visibility_m, 120.0))
        self.nominal_friction = max(0.1, min(1.5, _safe_float(nominal_friction, 1.0)))
        self.is_valid = bool(is_valid)
        self.error_message = str(error_message)


def build_world_snapshot(
    world: Any,
    platoon: List[Any],
    current_time: float,
    friction_provider: Optional[Any] = None,
    uncertainty_manager: Optional[Any] = None,
    active_leader_index: int = 0,
    disabled_vehicles: Optional[set] = None,
    predecessor_map: Optional[dict] = None,
    controller: Optional[Any] = None,
) -> WorldSnapshot:
    """Safely extracts a WorldSnapshot from live CARLA world objects with full defensive handling."""
    if controller is not None:
        if active_leader_index == 0 and hasattr(controller, "active_leader_index"):
            active_leader_index = controller.active_leader_index
        if disabled_vehicles is None and hasattr(controller, "disabled_vehicles"):
            disabled_vehicles = set(controller.disabled_vehicles)
        if predecessor_map is None and hasattr(controller, "predecessor_map"):
            predecessor_map = dict(controller.predecessor_map)

    if disabled_vehicles is None:
        disabled_vehicles = set()

    frame = 0
    try:
        if world and hasattr(world, "get_snapshot"):
            snap = world.get_snapshot()
            frame = snap.frame
    except Exception:
        pass

    if not platoon:
        return WorldSnapshot(
            timestamp=current_time,
            frame=frame,
            platoon=[],
            obstacles=[],
            traffic_light=TrafficLightSnapshot("NONE", 80.0, False),
            is_valid=False,
            error_message="Platoon vehicle list is empty or None",
        )

    # 1. Extract platoon vehicles
    platoon_snaps = []
    for idx, v in enumerate(platoon):
        is_disabled = (v.id in disabled_vehicles) if (v and hasattr(v, "id")) else False
        if is_disabled:
            v_role = "Disabled Leader"
        elif idx == active_leader_index:
            v_role = "Leader"
        else:
            v_role = f"Follower {idx}"

        if not v or not getattr(v, "is_alive", False):
            platoon_snaps.append(
                VehicleSnapshot(
                    vehicle_id=getattr(v, "id", idx),
                    role=v_role,
                    is_alive=False,
                    position=Vector2D(0.0, 0.0),
                    speed=0.0,
                    heading_rad=0.0,
                )
            )
            continue

        try:
            tf = v.get_transform()
            loc = tf.location
            yaw_deg = tf.rotation.yaw
            heading_rad = math.radians(yaw_deg)
            pos = Vector2D(loc.x, loc.y)
            vel = v.get_velocity()
            speed = math.hypot(vel.x, vel.y)
            
            fric = 1.0
            if friction_provider:
                try:
                    fric = float(friction_provider(v))
                except Exception:
                    fric = 1.0

            pred_id = None
            pred_gap = 80.0
            pred_idx = predecessor_map.get(idx) if predecessor_map is not None else (idx - 1 if idx > 0 else None)
            if pred_idx is not None and 0 <= pred_idx < len(platoon) and platoon[pred_idx] and getattr(platoon[pred_idx], "is_alive", False):
                prev_v = platoon[pred_idx]
                pred_id = prev_v.id
                raw_d = prev_v.get_location().distance(v.get_location())
                pred_gap = max(0.0, raw_d - 4.7)

            bb = getattr(v, "bounding_box", None)
            l = bb.extent.x * 2.0 if bb else 4.7
            w = bb.extent.y * 2.0 if bb else 2.0

            platoon_snaps.append(
                VehicleSnapshot(
                    vehicle_id=v.id,
                    role=v_role,
                    is_alive=True,
                    position=pos,
                    speed=speed,
                    heading_rad=heading_rad,
                    length=l,
                    width=w,
                    friction=fric,
                    predecessor_id=pred_id,
                    predecessor_gap=pred_gap,
                )
            )
        except Exception as e:
            platoon_snaps.append(
                VehicleSnapshot(
                    vehicle_id=getattr(v, "id", idx),
                    role=v_role,
                    is_alive=False,
                    position=Vector2D(0.0, 0.0),
                    speed=0.0,
                    heading_rad=0.0,
                )
            )

    # 2. Extract nearby obstacles (NPC vehicles, pedestrians, and disabled vehicles)
    obstacle_snaps = []
    # Platoon IDs that are active and NOT disabled
    active_platoon_ids = {
        v.id for v in platoon
        if v and getattr(v, "is_alive", False) and v.id not in disabled_vehicles
    }

    try:
        if world and hasattr(world, "get_actors"):
            actors = list(world.get_actors().filter("vehicle.*")) + list(world.get_actors().filter("walker.pedestrian.*"))
            for a in actors:
                if not a or not getattr(a, "is_alive", False) or a.id in active_platoon_ids:
                    continue
                tf = a.get_transform()
                pos = Vector2D(tf.location.x, tf.location.y)
                # Include obstacles within an 85m radius of any live active platoon vehicle
                is_near_platoon = any(pos.distance_to(p.position) <= 85.0 for p in platoon_snaps if p.is_alive)
                if is_near_platoon:
                    vel = a.get_velocity()
                    spd = math.hypot(vel.x, vel.y)
                    hd = math.radians(tf.rotation.yaw)
                    bb = getattr(a, "bounding_box", None)
                    l = bb.extent.x * 2.0 if bb else 4.5
                    w = bb.extent.y * 2.0 if bb else 2.0
                    obstacle_snaps.append(
                        ObstacleSnapshot(
                            obstacle_id=a.id,
                            position=pos,
                            speed=spd,
                            heading_rad=hd,
                            length=l,
                            width=w,
                        )
                    )
    except Exception:
        pass

    # 3. Extract traffic light constraints from authoritative active leader
    tl_snap = TrafficLightSnapshot("NONE", 80.0, False)
    active_leader = platoon[active_leader_index] if (0 <= active_leader_index < len(platoon)) else (platoon[0] if platoon else None)
    if active_leader and getattr(active_leader, "is_alive", False):
        try:
            if hasattr(active_leader, "get_traffic_light"):
                tl = active_leader.get_traffic_light()
                if tl is not None:
                    state = tl.get_state()
                    # CARLA TrafficLightState enum: Red=0, Yellow=1, Green=2, Off=3, Unknown=4
                    import carla
                    state_str = "GREEN"
                    if state == carla.TrafficLightState.Red:
                        state_str = "RED"
                    elif state == carla.TrafficLightState.Yellow:
                        state_str = "AMBER"
                    
                    dist = 80.0
                    if hasattr(active_leader, "get_traffic_light_waypoints"):
                        wps = active_leader.get_traffic_light_waypoints()
                        if wps:
                            dist = active_leader.get_location().distance(wps[0].transform.location)
                    
                    is_stop = (state_str in ("RED", "AMBER") and dist < 30.0)
                    tl_snap = TrafficLightSnapshot(state_str, dist, is_stop)
        except Exception:
            pass

    # 4. Extract environmental weather and visibility
    weather_preset = "CLEAR"
    visibility_m = 120.0
    nominal_friction = 1.0

    if uncertainty_manager and hasattr(uncertainty_manager, "weather"):
        try:
            curr = uncertainty_manager.weather.current_preset
            weather_preset = curr.name if hasattr(curr, "name") else str(curr)
            if "FOG" in weather_preset:
                visibility_m = 25.0
            elif "RAIN" in weather_preset:
                visibility_m = 60.0
        except Exception:
            pass

    return WorldSnapshot(
        timestamp=current_time,
        frame=frame,
        platoon=platoon_snaps,
        obstacles=obstacle_snaps,
        traffic_light=tl_snap,
        weather_preset=weather_preset,
        visibility_m=visibility_m,
        nominal_friction=nominal_friction,
        is_valid=True,
    )

