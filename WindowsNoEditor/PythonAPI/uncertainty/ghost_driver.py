#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
GHOST DRIVER: bounded one-vehicle adversarial uncertainty for CARLA 0.9.9
================================================================================
The ghost never spawns objects, teleports actors, or takes over a group. It only
possesses one non-platoon vehicle already inside the 50 m ego/platoon bubble,
nudges it with Traffic Manager controls where possible, and then restores normal
Traffic Manager behavior after a 5-10 s possession window.
================================================================================
"""

from __future__ import annotations

import enum
import logging
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import carla

logger = logging.getLogger("GhostDriver")


def _set_tm_speed_diff(tm: Any, vehicle: carla.Vehicle, diff: float) -> None:
    """Set CARLA Traffic Manager speed difference across 0.9.9/newer APIs."""
    try:
        if hasattr(tm, "vehicle_percentage_speed_difference"):
            tm.vehicle_percentage_speed_difference(vehicle, float(diff))
        elif hasattr(tm, "set_percentage_speed_difference"):
            tm.set_percentage_speed_difference(vehicle, float(diff))
    except Exception:
        pass


def _safe_tm_call(fn, *args) -> None:
    try:
        fn(*args)
    except Exception:
        pass


class GhostState(enum.Enum):
    LURKING = "LURKING"
    POSSESSING = "POSSESSING"


class DisruptionType(enum.Enum):
    LEADER_CUT_IN = "Leader Cut-In"
    LEAD_EMERGENCY_BRAKE = "Lead Emergency Braking"
    GAP_INTRUSION = "Gap Intrusion"
    LANE_OBSTACLE = "Platoon Lane Obstacle"
    JUNCTION_RULE_DISOBEDIENCE = "Junction Rule Disobedience"


class GhostDriver:
    """One-at-a-time adversarial driver bounded to a 50 m platoon bubble."""

    BRAKE_DURATION = 2.8

    def __init__(
        self,
        world: carla.World,
        traffic_manager: carla.TrafficManager,
        ego_vehicle: carla.Vehicle,
        npc_vehicles: List[carla.Vehicle],
        platoon_vehicles: Optional[List[carla.Vehicle]] = None,
        walkers: Optional[List[carla.Actor]] = None,
        walker_controllers: Optional[List[carla.Actor]] = None,
        min_cooldown: float = 18.0,
        max_cooldown: float = 32.0,
        min_possession: float = 5.0,
        max_possession: float = 10.0,
        proximity_radius: float = 50.0,
        seed: Optional[int] = None,
    ):
        self.world = world
        self.tm = traffic_manager
        self.ego = ego_vehicle
        self.npcs = npc_vehicles
        self.platoon = platoon_vehicles or [ego_vehicle]

        # Accepted for backwards-compatible construction; vehicle-only mode keeps
        # walkers untouched and supports NUM_PEDESTRIANS = 0 cleanly.
        self.walkers = walkers or []
        self.controllers = walker_controllers or []

        self.min_cooldown = float(min_cooldown)
        self.max_cooldown = float(max_cooldown)
        self.min_possession = float(min_possession)
        self.max_possession = float(max_possession)
        self.proximity_radius = float(proximity_radius)
        self._rng = random.Random(seed)

        self.state = GhostState.LURKING
        self.current_target: Optional[carla.Vehicle] = None
        self.current_disruption: Optional[DisruptionType] = None
        self.possession_start_time = 0.0
        self.possession_duration = 0.0
        self.possession_end_time = 0.0
        self.next_possession_time = self._rng.uniform(self.min_cooldown, self.max_cooldown)

        self._target_lane_change_right = True
        self._manual_brake_until = 0.0
        self._manual_released = True
        self._last_tm_refresh_time = -999.0
        self._last_distance_to_platoon = 0.0

        logger.info(
            "GhostDriver ready: radius=%.1fm, possession=%.1f-%.1fs, cooldown=%.1f-%.1fs, platoon=%d",
            self.proximity_radius,
            self.min_possession,
            self.max_possession,
            self.min_cooldown,
            self.max_cooldown,
            len(self.platoon),
        )

    @property
    def is_possessing(self) -> bool:
        return (
            self.state == GhostState.POSSESSING
            and self.current_target is not None
            and self.current_target.is_alive
        )

    def get_status(self, current_time: float) -> Dict[str, Any]:
        if self.is_possessing:
            dist = self._distance_to_platoon(self.current_target)
            elapsed = max(0.0, current_time - self.possession_start_time)
            return {
                "active": True,
                "state": self.state.value,
                "target_id": self.current_target.id,
                "disruption_name": self.current_disruption.value if self.current_disruption else "Unknown",
                "elapsed_seconds": elapsed,
                "remaining_seconds": max(0.0, self.possession_end_time - current_time),
                "total_duration": self.possession_duration,
                "distance_to_ego": self._distance_to_ego(self.current_target),
                "distance_to_platoon": dist,
                "bubble_radius": self.proximity_radius,
            }

        return {
            "active": False,
            "state": self.state.value,
            "target_id": None,
            "disruption_name": "None",
            "elapsed_seconds": 0.0,
            "remaining_seconds": 0.0,
            "total_duration": 0.0,
            "cooldown_remaining": max(0.0, self.next_possession_time - current_time),
            "distance_to_ego": 0.0,
            "distance_to_platoon": 0.0,
            "bubble_radius": self.proximity_radius,
        }

    def tick(self, delta_seconds: float, current_time: float):
        """Advance the ghost state machine. Returns alert tuple on possession."""
        if self.state == GhostState.POSSESSING:
            if not self.is_possessing:
                self._exorcise(current_time)
                return None

            # Hard boundary: a car outside the 50 m platoon bubble is released.
            self._last_distance_to_platoon = self._distance_to_platoon(self.current_target)
            if self._last_distance_to_platoon > self.proximity_radius:
                logger.info(
                    "[GHOST] Vehicle #%d left %.1fm bubble at %.1fm; released early.",
                    self.current_target.id,
                    self.proximity_radius,
                    self._last_distance_to_platoon,
                )
                self._exorcise(current_time)
                return None

            self._apply_maneuver_tick(current_time, delta_seconds)
            if current_time >= self.possession_end_time:
                self._exorcise(current_time)
            return None

        if current_time >= self.next_possession_time:
            alert = self._trigger_next_disruption(current_time)
            if alert is not None:
                return alert
            self.next_possession_time = current_time + self._rng.uniform(4.0, 7.0)
        return None

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _trigger_next_disruption(self, current_time: float):
        if not self.ego or not self.ego.is_alive:
            return None

        choices = [
            DisruptionType.LEADER_CUT_IN,
            DisruptionType.LEAD_EMERGENCY_BRAKE,
            DisruptionType.GAP_INTRUSION,
            DisruptionType.LANE_OBSTACLE,
            DisruptionType.JUNCTION_RULE_DISOBEDIENCE,
        ]
        self._rng.shuffle(choices)

        for maneuver in choices:
            target_info = self._select_target(maneuver)
            if target_info is not None:
                target, local_x, local_y, dist = target_info
                return self._start_vehicle_disruption(
                    target=target,
                    maneuver=maneuver,
                    current_time=current_time,
                    local_y=local_y,
                    distance=dist,
                )
        return None

    def _select_target(
        self,
        maneuver: DisruptionType,
    ) -> Optional[Tuple[carla.Vehicle, float, float, float]]:
        candidates = self._find_vehicle_candidates()
        if not candidates:
            return None

        if maneuver == DisruptionType.LEADER_CUT_IN:
            flank = [
                item for item in candidates
                if -8.0 <= item[1] <= 18.0 and 2.2 <= abs(item[2]) <= 8.0 and item[3] <= 35.0
            ]
            return min(flank, key=lambda item: item[3]) if flank else None

        if maneuver == DisruptionType.LEAD_EMERGENCY_BRAKE:
            ahead = [
                item for item in candidates
                if 8.0 <= item[1] <= 42.0 and abs(item[2]) <= 4.5
            ]
            return min(ahead, key=lambda item: item[3]) if ahead else None

        if maneuver == DisruptionType.GAP_INTRUSION:
            gap_target = self._get_platoon_gap_location()
            if gap_target is None:
                return None
            scored = []
            for npc, local_x, local_y, dist in candidates:
                try:
                    gap_dist = npc.get_location().distance(gap_target)
                    if gap_dist <= 35.0 and 1.8 <= abs(local_y) <= 9.0:
                        scored.append((gap_dist, npc, local_x, local_y, dist))
                except Exception:
                    continue
            if scored:
                _, npc, local_x, local_y, dist = min(scored, key=lambda item: item[0])
                return npc, local_x, local_y, dist
            return None

        if maneuver == DisruptionType.LANE_OBSTACLE:
            ahead = [
                item for item in candidates
                if 12.0 <= item[1] <= 45.0 and abs(item[2]) <= 5.0
            ]
            return min(ahead, key=lambda item: item[3]) if ahead else None

        if maneuver == DisruptionType.JUNCTION_RULE_DISOBEDIENCE:
            junctions = [item for item in candidates if self._is_approaching_junction(item[0])]
            return min(junctions, key=lambda item: item[3]) if junctions else None

        return None

    def _find_vehicle_candidates(self) -> List[Tuple[carla.Vehicle, float, float, float]]:
        """Return (vehicle, ego-local-x, ego-local-y, nearest-platoon-distance)."""
        platoon_ids = {v.id for v in self.platoon if v and v.is_alive}
        ego_tf = self.ego.get_transform()
        candidates: List[Tuple[carla.Vehicle, float, float, float]] = []

        for npc in self.npcs:
            if not npc or not npc.is_alive or npc.id in platoon_ids:
                continue
            try:
                dist = self._distance_to_platoon(npc)
                if 4.0 < dist <= self.proximity_radius:
                    local_x, local_y = self._project_to_local(ego_tf, npc.get_location())
                    candidates.append((npc, local_x, local_y, dist))
            except Exception:
                continue

        candidates.sort(key=lambda item: item[3])
        return candidates

    # ------------------------------------------------------------------
    # Possession setup and per-frame actions
    # ------------------------------------------------------------------

    def _start_vehicle_disruption(
        self,
        target: carla.Vehicle,
        maneuver: DisruptionType,
        current_time: float,
        local_y: float,
        distance: float,
    ):
        if self.state == GhostState.POSSESSING:
            return None

        self.state = GhostState.POSSESSING
        self.current_target = target
        self.current_disruption = maneuver
        self.possession_duration = round(self._rng.uniform(self.min_possession, self.max_possession), 1)
        self.possession_start_time = current_time
        self.possession_end_time = current_time + self.possession_duration
        self._target_lane_change_right = local_y < 0.0
        self._manual_brake_until = 0.0
        self._manual_released = True
        self._last_tm_refresh_time = -999.0
        self._last_distance_to_platoon = distance

        self._configure_maneuver(current_time, initial=True)

        logger.info(
            "[GHOST] Possessing vehicle #%d for %.1fs: %s (dist %.1fm)",
            target.id,
            self.possession_duration,
            maneuver.value,
            distance,
        )

        return (
            f"👻 GHOST DRIVER: {maneuver.value.upper()}",
            f"Vehicle #{target.id} possessed in 50m bubble! Dist: {distance:.1f}m",
            self.possession_duration,
        )

    def _apply_maneuver_tick(self, current_time: float, delta_seconds: float) -> None:
        if not self.current_target or not self.current_target.is_alive or not self.current_disruption:
            return

        if self.current_disruption == DisruptionType.LEAD_EMERGENCY_BRAKE:
            if current_time < self._manual_brake_until:
                try:
                    self.current_target.apply_control(
                        carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0, hand_brake=False)
                    )
                except Exception:
                    pass
                return
            if not self._manual_released:
                self._release_manual_brake()

        if current_time - self._last_tm_refresh_time >= 0.8:
            self._configure_maneuver(current_time, initial=False)

    def _configure_maneuver(self, current_time: float, initial: bool) -> None:
        target = self.current_target
        maneuver = self.current_disruption
        if not target or not target.is_alive or not maneuver:
            return

        if maneuver == DisruptionType.LEAD_EMERGENCY_BRAKE and initial:
            try:
                target.set_autopilot(False)
                target.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0, hand_brake=False))
            except Exception:
                pass
            self._manual_brake_until = current_time + self.BRAKE_DURATION
            self._manual_released = False
            self._last_tm_refresh_time = current_time
            return

        _safe_tm_call(target.set_autopilot, True, self.tm.get_port())
        _safe_tm_call(self.tm.ignore_vehicles_percentage, target, 0.0)
        _safe_tm_call(self.tm.ignore_walkers_percentage, target, 0.0)

        if maneuver == DisruptionType.LEADER_CUT_IN:
            _safe_tm_call(self.tm.auto_lane_change, target, True)
            _safe_tm_call(self.tm.distance_to_leading_vehicle, target, 1.6)
            _set_tm_speed_diff(self.tm, target, -25.0)
            _safe_tm_call(self.tm.force_lane_change, target, self._target_lane_change_right)

        elif maneuver == DisruptionType.GAP_INTRUSION:
            _safe_tm_call(self.tm.auto_lane_change, target, True)
            _safe_tm_call(self.tm.distance_to_leading_vehicle, target, 1.2)
            _set_tm_speed_diff(self.tm, target, -15.0)
            _safe_tm_call(self.tm.force_lane_change, target, self._target_lane_change_right)

        elif maneuver == DisruptionType.LANE_OBSTACLE:
            _safe_tm_call(self.tm.auto_lane_change, target, False)
            _safe_tm_call(self.tm.distance_to_leading_vehicle, target, 8.0)
            _set_tm_speed_diff(self.tm, target, 60.0)

        elif maneuver == DisruptionType.JUNCTION_RULE_DISOBEDIENCE:
            _safe_tm_call(self.tm.auto_lane_change, target, True)
            _safe_tm_call(self.tm.distance_to_leading_vehicle, target, 2.0)
            _safe_tm_call(self.tm.ignore_lights_percentage, target, 100.0)
            _safe_tm_call(self.tm.ignore_signs_percentage, target, 100.0)
            _set_tm_speed_diff(self.tm, target, -10.0)

        elif maneuver == DisruptionType.LEAD_EMERGENCY_BRAKE:
            _safe_tm_call(self.tm.auto_lane_change, target, True)
            _safe_tm_call(self.tm.distance_to_leading_vehicle, target, 3.5)
            _set_tm_speed_diff(self.tm, target, 0.0)

        self._last_tm_refresh_time = current_time

    def _release_manual_brake(self) -> None:
        if not self.current_target or not self.current_target.is_alive:
            return
        try:
            self.current_target.apply_control(carla.VehicleControl(throttle=0.25, brake=0.0, steer=0.0))
            self.current_target.set_autopilot(True, self.tm.get_port())
        except Exception:
            pass
        _safe_tm_call(self.tm.auto_lane_change, self.current_target, True)
        _safe_tm_call(self.tm.distance_to_leading_vehicle, self.current_target, 3.5)
        _set_tm_speed_diff(self.tm, self.current_target, 0.0)
        self._manual_released = True
        logger.info("[GHOST] Vehicle #%d emergency brake released after %.1fs.", self.current_target.id, self.BRAKE_DURATION)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _distance_to_ego(self, actor: carla.Actor) -> float:
        try:
            return self.ego.get_location().distance(actor.get_location())
        except Exception:
            return 0.0

    def _distance_to_platoon(self, actor: carla.Actor) -> float:
        try:
            loc = actor.get_location()
            distances = [
                v.get_location().distance(loc)
                for v in self.platoon
                if v and v.is_alive
            ]
            return min(distances) if distances else self._distance_to_ego(actor)
        except Exception:
            return float("inf")

    def _get_platoon_gap_location(self) -> Optional[carla.Location]:
        live = [v for v in self.platoon if v and v.is_alive]
        if len(live) < 2:
            return None
        p1 = live[0].get_location()
        p2 = live[1].get_location()
        return carla.Location(x=(p1.x + p2.x) * 0.5, y=(p1.y + p2.y) * 0.5, z=(p1.z + p2.z) * 0.5)

    def _project_to_local(self, reference_tf: carla.Transform, target_loc: carla.Location) -> Tuple[float, float]:
        ref_loc = reference_tf.location
        dx = target_loc.x - ref_loc.x
        dy = target_loc.y - ref_loc.y
        yaw = math.radians(reference_tf.rotation.yaw)
        local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
        local_y = -dx * math.sin(yaw) + dy * math.cos(yaw)
        return local_x, local_y

    def _is_approaching_junction(self, vehicle: carla.Vehicle) -> bool:
        try:
            carla_map = self.world.get_map()
            wp = carla_map.get_waypoint(vehicle.get_location())
            if wp.is_junction:
                return True
            for lookahead in (8.0, 16.0, 24.0):
                next_wps = wp.next(lookahead)
                if any(next_wp.is_junction for next_wp in next_wps):
                    return True
        except Exception:
            return False
        return False

    # ------------------------------------------------------------------
    # Exorcism and cleanup
    # ------------------------------------------------------------------

    def _exorcise(self, current_time: float) -> None:
        target = self.current_target
        if target and target.is_alive:
            try:
                target.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0, hand_brake=False))
                target.set_autopilot(True, self.tm.get_port())
            except Exception:
                pass

            _safe_tm_call(self.tm.auto_lane_change, target, True)
            _safe_tm_call(self.tm.distance_to_leading_vehicle, target, 3.5)
            _safe_tm_call(self.tm.ignore_lights_percentage, target, 0.0)
            _safe_tm_call(self.tm.ignore_signs_percentage, target, 0.0)
            _safe_tm_call(self.tm.ignore_vehicles_percentage, target, 0.0)
            _safe_tm_call(self.tm.ignore_walkers_percentage, target, 0.0)
            _set_tm_speed_diff(self.tm, target, 0.0)
            logger.info("[GHOST] Exorcised vehicle #%d. Traffic Manager restored.", target.id)

        self.state = GhostState.LURKING
        self.current_target = None
        self.current_disruption = None
        self.possession_start_time = 0.0
        self.possession_duration = 0.0
        self.possession_end_time = 0.0
        self._manual_brake_until = 0.0
        self._manual_released = True
        self._last_tm_refresh_time = -999.0

        cooldown = self._rng.uniform(self.min_cooldown, self.max_cooldown)
        self.next_possession_time = current_time + cooldown
        logger.info("[GHOST] Lurking for %.1fs. Next possession at t=%.1fs.", cooldown, self.next_possession_time)

    def cleanup(self) -> None:
        if self.current_target and self.current_target.is_alive:
            target = self.current_target
            try:
                target.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0, hand_brake=False))
                target.set_autopilot(True, self.tm.get_port())
            except Exception:
                pass
            _safe_tm_call(self.tm.auto_lane_change, target, True)
            _safe_tm_call(self.tm.distance_to_leading_vehicle, target, 3.5)
            _safe_tm_call(self.tm.ignore_lights_percentage, target, 0.0)
            _safe_tm_call(self.tm.ignore_signs_percentage, target, 0.0)
            _safe_tm_call(self.tm.ignore_vehicles_percentage, target, 0.0)
            _set_tm_speed_diff(self.tm, target, 0.0)

        self.current_target = None
        self.current_disruption = None
        self.state = GhostState.LURKING
