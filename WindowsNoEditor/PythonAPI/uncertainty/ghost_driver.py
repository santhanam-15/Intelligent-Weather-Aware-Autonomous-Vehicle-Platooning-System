#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
GHOST DRIVER: bounded one-vehicle adversarial uncertainty for CARLA 0.9.9
================================================================================
The ghost driver possesses non-platoon vehicles within a 50m bubble around the
platoon to introduce deterministic, realistic traffic disruptions (gap intrusions,
leader/follower cut-ins, approach from behind, adjacent lane blocking, clearing,
and persistent intrusions).
================================================================================
"""

from __future__ import annotations

import enum
import json
import logging
import math
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import carla
import numpy as np

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


def _clamp(val: float, low: float, high: float) -> float:
    return max(low, min(val, high))


def _angle_diff_deg(yaw1: float, yaw2: float) -> float:
    diff = (yaw2 - yaw1 + 180.0) % 360.0 - 180.0
    return diff


def _speed_mps(vehicle: carla.Vehicle) -> float:
    try:
        v = vehicle.get_velocity()
        return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    except Exception:
        return 0.0


class GhostState(enum.Enum):
    LURKING = "LURKING"
    POSSESSING = "POSSESSING"
    CLEARING = "CLEARING"
    ABORTED = "ABORTED"


class DisruptionType(enum.Enum):
    NORMAL_BACKGROUND = "Normal Background Traffic"
    APPROACH_FROM_BEHIND = "Approach From Behind"
    GAP_INTRUSION = "Gap Intrusion"
    LEADER_CUT_IN = "Leader Cut-In"
    FOLLOWER_CUT_IN = "Follower Cut-In"
    ADJACENT_LANE_BLOCK = "Adjacent Lane Block"
    INTRUSION_CLEARING = "Intrusion Clearing"
    INTRUSION_PERSISTENT = "Intrusion Persistent"
    # Backwards-compatible aliases
    LEAD_EMERGENCY_BRAKE = "Lead Emergency Braking"
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

        self.walkers = walkers or []
        self.controllers = walker_controllers or []

        self.min_cooldown = float(min_cooldown)
        self.max_cooldown = float(max_cooldown)
        self.min_possession = float(min_possession)
        self.max_possession = float(max_possession)
        self.proximity_radius = float(proximity_radius)
        self._seed = seed if seed is not None else 42
        self._rng = random.Random(self._seed)

        self.state = GhostState.LURKING
        self.current_target: Optional[carla.Vehicle] = None
        self.current_disruption: Optional[DisruptionType] = None
        self.possession_start_time = 0.0
        self.possession_duration = 0.0
        self.possession_end_time = 0.0
        self.next_possession_time = self._rng.uniform(self.min_cooldown, self.max_cooldown)

        # Scenario tracking and deterministic script queue
        self.current_scenario_id: Optional[str] = None
        self.current_seed: int = self._seed
        self.scripted_queue: List[Dict[str, Any]] = []
        self.maneuver_progress: float = 0.0
        self.maneuver_outcome: str = "IDLE"
        self.mission_controller: Any = None
        self._current_transitions: List[Tuple[float, str]] = []
        self._last_platoon_state: Optional[str] = None

        # Closed-loop maneuver internal states
        self._target_lane_change_right = True
        self._manual_brake_until = 0.0
        self._manual_released = True
        self._last_tm_refresh_time = -999.0
        self._last_distance_to_platoon = 0.0
        self._subphase = "APPROACHING"  # APPROACHING, INTRUDING, HOLDING, CLEARING, COMPLETED
        self._subphase_start_time = 0.0
        self._initial_target_offset = 0.0
        self._init_record: Dict[str, Any] = {}

        # Log file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self._log_dir = os.path.normpath(os.path.join(script_dir, "..", "logs"))
        if not os.path.exists(self._log_dir):
            try:
                os.makedirs(self._log_dir, exist_ok=True)
            except Exception:
                pass
        self._log_file = os.path.join(self._log_dir, "ghost_driver_scenarios.log")

        logger.info(
            "GhostDriver initialized: radius=%.1fm, possession=%.1f-%.1fs, cooldown=%.1f-%.1fs, platoon=%d, seed=%d",
            self.proximity_radius,
            self.min_possession,
            self.max_possession,
            self.min_cooldown,
            self.max_cooldown,
            len(self.platoon),
            self._seed,
        )

    def set_mission_controller(self, controller: Any) -> None:
        """Bind mission controller to monitor platoon state transitions."""
        self.mission_controller = controller

    def set_scripted_scenario(
        self,
        scenario_name: str,
        seed: Optional[int] = 42,
        target_vehicle: Optional[carla.Vehicle] = None,
        duration: Optional[float] = None,
    ) -> None:
        """Configure a deterministic scripted scenario to trigger on next tick."""
        name_map = {
            "normal": DisruptionType.NORMAL_BACKGROUND,
            "normal_background": DisruptionType.NORMAL_BACKGROUND,
            "approach_behind": DisruptionType.APPROACH_FROM_BEHIND,
            "approach_from_behind": DisruptionType.APPROACH_FROM_BEHIND,
            "gap_intrusion": DisruptionType.GAP_INTRUSION,
            "leader_cut_in": DisruptionType.LEADER_CUT_IN,
            "follower_cut_in": DisruptionType.FOLLOWER_CUT_IN,
            "adjacent_block": DisruptionType.ADJACENT_LANE_BLOCK,
            "adjacent_lane_block": DisruptionType.ADJACENT_LANE_BLOCK,
            "clearing": DisruptionType.INTRUSION_CLEARING,
            "intrusion_clearing": DisruptionType.INTRUSION_CLEARING,
            "persistent": DisruptionType.INTRUSION_PERSISTENT,
            "intrusion_persistent": DisruptionType.INTRUSION_PERSISTENT,
            "lead_brake": DisruptionType.LEAD_EMERGENCY_BRAKE,
        }
        enum_type = name_map.get(str(scenario_name).lower(), DisruptionType.GAP_INTRUSION)
        used_seed = seed if seed is not None else 42
        self._seed = used_seed
        self._rng = random.Random(used_seed)
        self.scripted_queue.append({
            "disruption": enum_type,
            "seed": used_seed,
            "target": target_vehicle,
            "duration": duration,
        })
        self.next_possession_time = 0.0  # Trigger immediately on next tick
        logger.info(
            "[GHOST] Scripted scenario queued: %s (seed=%d, queue_len=%d)",
            enum_type.name, used_seed, len(self.scripted_queue)
        )

    @property
    def is_possessing(self) -> bool:
        return (
            self.state in (GhostState.POSSESSING, GhostState.CLEARING)
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
                "scenario_id": self.current_scenario_id or "NONE",
                "seed": self.current_seed,
                "target_id": self.current_target.id,
                "disruption_name": self.current_disruption.value if self.current_disruption else "Unknown",
                "elapsed_seconds": elapsed,
                "remaining_seconds": max(0.0, self.possession_end_time - current_time),
                "total_duration": self.possession_duration,
                "distance_to_ego": self._distance_to_ego(self.current_target),
                "distance_to_platoon": dist,
                "bubble_radius": self.proximity_radius,
                "progress": round(self.maneuver_progress, 2),
                "subphase": self._subphase,
                "outcome": self.maneuver_outcome,
            }

        return {
            "active": False,
            "state": self.state.value,
            "scenario_id": self.current_scenario_id or "NONE",
            "seed": self.current_seed,
            "target_id": None,
            "disruption_name": "None",
            "elapsed_seconds": 0.0,
            "remaining_seconds": 0.0,
            "total_duration": 0.0,
            "cooldown_remaining": max(0.0, self.next_possession_time - current_time),
            "distance_to_ego": 0.0,
            "distance_to_platoon": 0.0,
            "bubble_radius": self.proximity_radius,
            "progress": 0.0,
            "subphase": "IDLE",
            "outcome": self.maneuver_outcome,
        }

    def tick(self, delta_seconds: float, current_time: float):
        """Advance ghost driver state machine. Returns alert tuple on possession."""
        # Monitor platoon state transitions for active scenario logging
        if self.mission_controller and hasattr(self.mission_controller, "state"):
            curr_p_state = str(self.mission_controller.state).split(".")[-1]
            if curr_p_state != self._last_platoon_state:
                self._current_transitions.append((round(current_time, 2), curr_p_state))
                self._last_platoon_state = curr_p_state

        if self.is_possessing:
            # Check bubble distance boundary
            self._last_distance_to_platoon = self._distance_to_platoon(self.current_target)
            if self._last_distance_to_platoon > self.proximity_radius:
                logger.info(
                    "[GHOST] Vehicle #%d left %.1fm bubble at %.1fm; releasing.",
                    self.current_target.id,
                    self.proximity_radius,
                    self._last_distance_to_platoon,
                )
                self.maneuver_outcome = "BUBBLE_EXIT_RELEASE"
                self._exorcise(current_time)
                return None

            self._apply_maneuver_tick(current_time, delta_seconds)

            if current_time >= self.possession_end_time and self.state != GhostState.CLEARING:
                if self.current_disruption == DisruptionType.INTRUSION_CLEARING and self._subphase != "COMPLETED":
                    # Transition to active clearing
                    self.state = GhostState.CLEARING
                    self._subphase = "CLEARING"
                    self._subphase_start_time = current_time
                    self.possession_end_time = current_time + 4.0
                else:
                    self._exorcise(current_time)
            elif self.state == GhostState.CLEARING and (current_time >= self.possession_end_time or self._subphase == "COMPLETED"):
                self.maneuver_outcome = "CLEARED"
                self._exorcise(current_time)

            return None

        # Check possession trigger
        if current_time >= self.next_possession_time:
            alert = self._trigger_next_disruption(current_time)
            if alert is not None:
                return alert
            self.next_possession_time = current_time + self._rng.uniform(4.0, 7.0)
        return None

    # ------------------------------------------------------------------
    # Target Selection and Disruption Initialization
    # ------------------------------------------------------------------

    def _trigger_next_disruption(self, current_time: float):
        if not self.ego or not self.ego.is_alive:
            return None

        # Check scripted queue first
        if self.scripted_queue:
            item = self.scripted_queue.pop(0)
            maneuver = item["disruption"]
            self.current_seed = item["seed"]
            self._rng = random.Random(self.current_seed)
            target = item.get("target")
            dur = item.get("duration")

            if target and target.is_alive:
                ego_tf = self.ego.get_transform()
                lx, ly = self._project_to_local(ego_tf, target.get_location())
                dist = self._distance_to_platoon(target)
                return self._start_vehicle_disruption(target, maneuver, current_time, ly, dist, override_dur=dur)
            else:
                target_info = self._select_target(maneuver)
                if target_info is not None:
                    target, lx, ly, dist = target_info
                    return self._start_vehicle_disruption(target, maneuver, current_time, ly, dist, override_dur=dur)
                else:
                    logger.warning("[GHOST] Scripted scenario %s could not find eligible candidate vehicle. Retrying next tick.", maneuver.name)
                    return None

        # Autonomous mode: select from pool
        choices = [
            DisruptionType.GAP_INTRUSION,
            DisruptionType.LEADER_CUT_IN,
            DisruptionType.FOLLOWER_CUT_IN,
            DisruptionType.APPROACH_FROM_BEHIND,
            DisruptionType.ADJACENT_LANE_BLOCK,
            DisruptionType.INTRUSION_CLEARING,
            DisruptionType.INTRUSION_PERSISTENT,
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

        if maneuver == DisruptionType.NORMAL_BACKGROUND:
            return candidates[0]

        if maneuver in (DisruptionType.LEADER_CUT_IN, DisruptionType.LEAD_EMERGENCY_BRAKE):
            flank = [
                item for item in candidates
                if -6.0 <= item[1] <= 22.0 and 1.8 <= abs(item[2]) <= 8.5 and item[3] <= 38.0
            ]
            return min(flank, key=lambda item: item[3]) if flank else None

        if maneuver in (DisruptionType.GAP_INTRUSION, DisruptionType.INTRUSION_CLEARING, DisruptionType.INTRUSION_PERSISTENT):
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

        if maneuver == DisruptionType.FOLLOWER_CUT_IN:
            if len(self.platoon) > 1:
                f_tf = self.platoon[1].get_transform()
                scored = []
                for npc, local_x, local_y, dist in candidates:
                    try:
                        fl_x, fl_y = self._project_to_local(f_tf, npc.get_location())
                        if -5.0 <= fl_x <= 18.0 and 1.8 <= abs(fl_y) <= 8.5:
                            scored.append((dist, npc, local_x, local_y, dist))
                    except Exception:
                        continue
                if scored:
                    _, npc, local_x, local_y, dist = min(scored, key=lambda item: item[0])
                    return npc, local_x, local_y, dist
            return None

        if maneuver == DisruptionType.APPROACH_FROM_BEHIND:
            last_veh = self.platoon[-1]
            last_tf = last_veh.get_transform()
            scored = []
            for npc, local_x, local_y, dist in candidates:
                try:
                    lx, ly = self._project_to_local(last_tf, npc.get_location())
                    if lx < -2.0 and abs(ly) <= 3.8 and dist <= 40.0:
                        scored.append((dist, npc, local_x, local_y, dist))
                except Exception:
                    continue
            if scored:
                _, npc, local_x, local_y, dist = min(scored, key=lambda item: item[0])
                return npc, local_x, local_y, dist
            return None

        if maneuver == DisruptionType.ADJACENT_LANE_BLOCK:
            alongside = [
                item for item in candidates
                if -12.0 <= item[1] <= 15.0 and 2.5 <= abs(item[2]) <= 5.5 and item[3] <= 25.0
            ]
            return min(alongside, key=lambda item: item[3]) if alongside else None

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
                if 2.5 < dist <= self.proximity_radius:
                    local_x, local_y = self._project_to_local(ego_tf, npc.get_location())
                    candidates.append((npc, local_x, local_y, dist))
            except Exception:
                continue

        candidates.sort(key=lambda item: item[3])
        return candidates

    def _start_vehicle_disruption(
        self,
        target: carla.Vehicle,
        maneuver: DisruptionType,
        current_time: float,
        local_y: float,
        distance: float,
        override_dur: Optional[float] = None,
    ):
        if self.is_possessing:
            return None

        self.state = GhostState.POSSESSING
        self.current_target = target
        self.current_disruption = maneuver

        if override_dur is not None:
            self.possession_duration = float(override_dur)
        elif maneuver == DisruptionType.INTRUSION_PERSISTENT:
            self.possession_duration = round(self._rng.uniform(14.0, 18.0), 1)
        elif maneuver == DisruptionType.INTRUSION_CLEARING:
            self.possession_duration = round(self._rng.uniform(6.0, 9.0), 1)
        else:
            self.possession_duration = round(self._rng.uniform(self.min_possession, self.max_possession), 1)

        self.possession_start_time = current_time
        self.possession_end_time = current_time + self.possession_duration
        self._target_lane_change_right = (local_y < 0.0)
        self._manual_brake_until = 0.0
        self._manual_released = True
        self._last_tm_refresh_time = -999.0
        self._last_distance_to_platoon = distance
        self._subphase = "APPROACHING"
        self._subphase_start_time = current_time
        self.maneuver_progress = 0.0
        self.maneuver_outcome = "IN_PROGRESS"
        self._current_transitions = [(round(current_time, 2), "POSSESSING")]

        # Determine target descriptor
        if maneuver in (DisruptionType.GAP_INTRUSION, DisruptionType.INTRUSION_CLEARING, DisruptionType.INTRUSION_PERSISTENT):
            target_desc = "Platoon Gap 0-1"
        elif maneuver == DisruptionType.LEADER_CUT_IN:
            target_desc = "Leader (Front Cut-In)"
        elif maneuver == DisruptionType.FOLLOWER_CUT_IN:
            target_desc = "Follower 1 (Front Cut-In)"
        elif maneuver == DisruptionType.APPROACH_FROM_BEHIND:
            target_desc = "Rear Platoon Tailgate"
        elif maneuver == DisruptionType.ADJACENT_LANE_BLOCK:
            target_desc = "Adjacent Platoon Corridor"
        else:
            target_desc = "General Traffic Corridor"

        # Unique scenario ID
        self.current_scenario_id = f"SCEN-{int(current_time * 10):04d}-{maneuver.name}"

        # Record initial conditions
        tf = target.get_transform()
        init_speed_mps = _speed_mps(target)
        init_lane = "unknown"
        try:
            wp = self.world.get_map().get_waypoint(tf.location, project_to_road=True)
            if wp:
                init_lane = wp.lane_id
        except Exception:
            pass

        self._initial_target_offset = abs(local_y)
        self._init_record = {
            "scenario_id": self.current_scenario_id,
            "seed": self.current_seed,
            "ghost_vehicle_id": target.id,
            "initial_position": (round(tf.location.x, 2), round(tf.location.y, 2), round(tf.location.z, 2)),
            "initial_lane": init_lane,
            "initial_speed_kmh": round(init_speed_mps * 3.6, 2),
            "target": target_desc,
            "intended_maneuver": maneuver.name,
            "detection_time": round(current_time, 2),
            "classification": maneuver.name,
            "safety_response": "OBSERVED",
        }

        # Take manual closed-loop control of possessed vehicle
        if maneuver != DisruptionType.NORMAL_BACKGROUND:
            try:
                target.set_autopilot(False)
            except Exception:
                pass

        logger.info(
            "\n%s\n  [GHOST SCENARIO TRIGGERED]\n%s\n"
            "  Scenario ID:        %s\n"
            "  Seed:               %d\n"
            "  Ghost Vehicle ID:   %d\n"
            "  Initial Position:   (%.1f, %.1f, %.1f) | Lane: %s | Spd: %.1f km/h\n"
            "  Target Corridor:    %s\n"
            "  Intended Maneuver:  %s\n"
            "  Duration:           %.1fs (until t=%.1fs)\n%s",
            "=" * 64, "=" * 64,
            self.current_scenario_id, self.current_seed, target.id,
            tf.location.x, tf.location.y, tf.location.z, init_lane, init_speed_mps * 3.6,
            target_desc, maneuver.name, self.possession_duration, self.possession_end_time,
            "=" * 64,
        )
        self._log_scenario_record("TRIGGER", {"progress": 0.0, "status": "TRIGGERED"})

        return (
            f"👻 GHOST DRIVER: {maneuver.value.upper()}",
            f"Vehicle #{target.id} possessed in 50m bubble! Dist: {distance:.1f}m",
            self.possession_duration,
        )

    # ------------------------------------------------------------------
    # Closed-Loop Maneuver Execution
    # ------------------------------------------------------------------

    def _apply_maneuver_tick(self, current_time: float, delta_seconds: float) -> None:
        target = self.current_target
        maneuver = self.current_disruption
        if not target or not target.is_alive or not maneuver:
            return

        if maneuver == DisruptionType.NORMAL_BACKGROUND:
            self.maneuver_progress = 1.0
            self.maneuver_outcome = "NORMAL_BACKGROUND_FLOW"
            return

        if maneuver == DisruptionType.LEAD_EMERGENCY_BRAKE:
            try:
                target.set_autopilot(False)
                target.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0, hand_brake=False))
            except Exception:
                pass
            self.maneuver_progress = 1.0
            self.maneuver_outcome = "EMERGENCY_BRAKE_ACTIVE"
            return

        # Closed-loop target calculation
        tf_t = target.get_transform()
        v_curr = _speed_mps(target)

        dest_pos = None
        target_speed_mps = 8.0

        if maneuver in (DisruptionType.GAP_INTRUSION, DisruptionType.INTRUSION_CLEARING, DisruptionType.INTRUSION_PERSISTENT):
            p0 = self.platoon[0]
            p1 = self.platoon[1] if len(self.platoon) > 1 else p0
            p0_loc, p1_loc = p0.get_location(), p1.get_location()
            dest_pos = carla.Location(
                x=(p0_loc.x + p1_loc.x) * 0.5,
                y=(p0_loc.y + p1_loc.y) * 0.5,
                z=p0_loc.z,
            )
            target_speed_mps = max(5.0, _speed_mps(p0))

        elif maneuver == DisruptionType.LEADER_CUT_IN:
            p0 = self.platoon[0]
            p0_tf = p0.get_transform()
            yaw_r = math.radians(p0_tf.rotation.yaw)
            dest_pos = carla.Location(
                x=p0_tf.location.x + 10.0 * math.cos(yaw_r),
                y=p0_tf.location.y + 10.0 * math.sin(yaw_r),
                z=p0_tf.location.z,
            )
            target_speed_mps = max(6.0, _speed_mps(p0) + 0.8)

        elif maneuver == DisruptionType.FOLLOWER_CUT_IN:
            pf = self.platoon[1] if len(self.platoon) > 1 else self.platoon[0]
            pf_tf = pf.get_transform()
            yaw_r = math.radians(pf_tf.rotation.yaw)
            dest_pos = carla.Location(
                x=pf_tf.location.x + 8.0 * math.cos(yaw_r),
                y=pf_tf.location.y + 8.0 * math.sin(yaw_r),
                z=pf_tf.location.z,
            )
            target_speed_mps = max(5.0, _speed_mps(pf))

        elif maneuver == DisruptionType.APPROACH_FROM_BEHIND:
            plast = self.platoon[-1]
            plast_tf = plast.get_transform()
            yaw_r = math.radians(plast_tf.rotation.yaw)
            dest_pos = carla.Location(
                x=plast_tf.location.x - 4.5 * math.cos(yaw_r),
                y=plast_tf.location.y - 4.5 * math.sin(yaw_r),
                z=plast_tf.location.z,
            )
            last_spd = _speed_mps(plast)
            dist_to_rear = tf_t.location.distance(plast_tf.location)
            target_speed_mps = last_spd + (2.0 if dist_to_rear > 8.0 else 0.0)

        elif maneuver == DisruptionType.ADJACENT_LANE_BLOCK:
            p0 = self.platoon[0]
            p0_tf = p0.get_transform()
            yaw_r = math.radians(p0_tf.rotation.yaw)
            sign = 1.0 if self._target_lane_change_right else -1.0
            dest_pos = carla.Location(
                x=p0_tf.location.x - sign * 3.5 * math.sin(yaw_r),
                y=p0_tf.location.y + sign * 3.5 * math.cos(yaw_r),
                z=p0_tf.location.z,
            )
            target_speed_mps = max(5.0, _speed_mps(p0))

        if dest_pos is None:
            self._abort_maneuver(current_time, "Destination position unavailable")
            return

        # Vector towards dest_pos
        dx = dest_pos.x - tf_t.location.x
        dy = dest_pos.y - tf_t.location.y
        dist_to_dest = math.hypot(dx, dy)

        # Lateral steering control
        desired_yaw = math.degrees(math.atan2(dy, dx))
        angle_err = _angle_diff_deg(tf_t.rotation.yaw, desired_yaw)
        steer = _clamp(angle_err / 35.0, -0.65, 0.65)

        # Longitudinal throttle/brake control
        spd_err = target_speed_mps - v_curr
        if spd_err > 0.4:
            thr = _clamp(0.3 + 0.25 * spd_err, 0.0, 0.75)
            brk = 0.0
        elif spd_err < -0.4:
            thr = 0.0
            brk = _clamp(-0.35 * spd_err, 0.0, 0.85)
        else:
            thr = 0.28
            brk = 0.0

        # State and progress progression
        if self._subphase == "APPROACHING":
            self.maneuver_progress = _clamp(1.0 - (dist_to_dest / max(1.0, self.proximity_radius)), 0.0, 0.95)
            if dist_to_dest < 3.2:
                self._subphase = "INTRUDING"
                self._subphase_start_time = current_time
                self.maneuver_progress = 0.85
                logger.info("[GHOST] Vehicle #%d achieved corridor intrusion (dist=%.1fm)", target.id, dist_to_dest)

        elif self._subphase == "INTRUDING":
            self.maneuver_progress = 0.95
            if current_time - self._subphase_start_time > 1.0:
                self._subphase = "HOLDING"
                self._subphase_start_time = current_time
                self.maneuver_progress = 1.0

        elif self._subphase == "HOLDING":
            self.maneuver_progress = 1.0
            if maneuver == DisruptionType.INTRUSION_CLEARING and current_time - self._subphase_start_time > 3.0:
                self._subphase = "CLEARING"
                self._subphase_start_time = current_time
                self.state = GhostState.CLEARING
                logger.info("[GHOST] Vehicle #%d initiating clearance maneuver away from platoon.", target.id)

        elif self._subphase == "CLEARING":
            # Steer outward to adjacent lane
            steer = -0.35 if self._target_lane_change_right else 0.35
            thr = 0.55
            brk = 0.0
            self.maneuver_progress = 1.0
            if current_time - self._subphase_start_time > 2.5:
                self._subphase = "COMPLETED"
                self.maneuver_outcome = "CLEARED"

        # Apply vehicle control
        try:
            control = carla.VehicleControl(
                throttle=float(thr),
                brake=float(brk),
                steer=float(steer),
                hand_brake=False,
                reverse=False,
            )
            target.apply_control(control)
        except Exception as exc:
            self._abort_maneuver(current_time, f"Apply control failed: {exc}")

    # ------------------------------------------------------------------
    # Exorcism, Abort, and Structured Logging
    # ------------------------------------------------------------------

    def _abort_maneuver(self, current_time: float, reason: str) -> None:
        """Safely abort maneuver if path cannot be followed or road geometry is unsuitable."""
        logger.warning(
            "[GHOST SCENARIO ABORT] Scenario %s aborted at t=%.2fs. Reason: %s",
            self.current_scenario_id, current_time, reason
        )
        self.state = GhostState.ABORTED
        self.maneuver_outcome = f"ABORTED: {reason}"
        self._exorcise(current_time)

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

        if self.maneuver_outcome == "IN_PROGRESS":
            self.maneuver_outcome = "COMPLETED"

        self._log_scenario_record("FINISH", {
            "progress": self.maneuver_progress,
            "outcome": self.maneuver_outcome,
            "elapsed": round(current_time - self.possession_start_time, 2),
        })

        self.state = GhostState.LURKING
        self.current_target = None
        self.current_disruption = None
        self.possession_start_time = 0.0
        self.possession_duration = 0.0
        self.possession_end_time = 0.0
        self._manual_brake_until = 0.0
        self._manual_released = True
        self._last_tm_refresh_time = -999.0
        self._subphase = "IDLE"

        cooldown = self._rng.uniform(self.min_cooldown, self.max_cooldown)
        self.next_possession_time = current_time + cooldown
        logger.info("[GHOST] Lurking for %.1fs. Next possession at t=%.1fs.", cooldown, self.next_possession_time)

    def _log_scenario_record(self, event_type: str, details: Dict[str, Any]) -> None:
        rec = dict(self._init_record)
        rec.update(details)
        rec["event"] = event_type
        rec["transitions"] = list(self._current_transitions)
        line = (
            f"[GHOST SCENARIO] Event={event_type} | ID={rec.get('scenario_id')} | Seed={rec.get('seed')} | "
            f"GhostID={rec.get('ghost_vehicle_id')} | InitPos={rec.get('initial_position')} | "
            f"InitLane={rec.get('initial_lane')} | InitSpd={rec.get('initial_speed_kmh')} km/h | "
            f"Target={rec.get('target')} | Maneuver={rec.get('intended_maneuver')} | "
            f"Progress={rec.get('progress') * 100:.1f}% | Outcome={rec.get('outcome', 'RUNNING')} | "
            f"Transitions={rec.get('transitions')}"
        )
        logger.info(line)
        try:
            with open(self._log_file, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass

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
