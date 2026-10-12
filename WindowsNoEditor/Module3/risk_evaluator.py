# WindowsNoEditor/Module3/risk_evaluator.py
"""Mathematical Risk Evaluator for Module 3.

Evaluates forward collision risks, TTC, headway, weather-dependent stopping distances,
traffic-light compliance, and lateral clearances based on World B trajectory predictions.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
from Module3.config import (
    CRITICAL_TTC_THRESHOLD_SEC,
    WARNING_TTC_THRESHOLD_SEC,
    MIN_COLLISION_SEPARATION_M,
    MAX_EMERGENCY_DECEL_MPS2,
    DRIVER_REACTION_TIME_SEC,
    MIN_BUFFER_DISTANCE_M,
    LANE_CHANGE_TARGET_CLEAR_M,
)
from Module3.snapshot import WorldSnapshot, VehicleSnapshot, Vector2D
from Module3.twin_world import PredictedState


def compute_ttc(distance_m: float, closing_speed_mps: float, epsilon: float = 0.05) -> float:
    """Exact TTC mathematical definition:
    TTC = 0.0 if distance <= 0
    TTC = distance / closing_speed if distance > 0 and closing_speed > epsilon
    TTC = infinity if closing_speed <= epsilon
    """
    d = float(distance_m)
    v_c = float(closing_speed_mps)

    if not math.isfinite(d) or not math.isfinite(v_c):
        return 0.0  # Conservative critical fallback on invalid numbers

    if d <= 0.0:
        return 0.0
    if v_c > epsilon:
        return d / v_c
    return float("inf")


def compute_stopping_distance(
    follower_speed_mps: float,
    lead_speed_mps: float,
    friction: float,
    reaction_time_sec: float = DRIVER_REACTION_TIME_SEC,
    buffer_distance_m: float = MIN_BUFFER_DISTANCE_M,
) -> float:
    """Computes weather-dependent stopping distance accounting for reaction delay and effective deceleration.
    
    d_stop = v_f * t_r + max(0, (v_f^2 - v_l^2) / (2 * a_eff)) + d_buffer
    where a_eff = MAX_EMERGENCY_DECEL * friction.
    """
    v_f = max(0.0, float(follower_speed_mps))
    v_l = max(0.0, float(lead_speed_mps))
    fric = max(0.2, min(1.2, float(friction)))

    a_eff = max(1.0, MAX_EMERGENCY_DECEL_MPS2 * fric)
    reaction_dist = v_f * reaction_time_sec

    speed_diff_sq = max(0.0, (v_f ** 2) - (v_l ** 2))
    braking_dist = speed_diff_sq / (2.0 * a_eff)

    return reaction_dist + braking_dist + buffer_distance_m


class RiskAssessment:
    """Structured risk metrics for a single vehicle under a candidate action."""
    __slots__ = (
        "vehicle_id", "action", "is_safe", "is_critical", "risk_reasons",
        "min_predicted_distance", "min_ttc", "stopping_distance_required",
        "headway_margin", "traffic_light_risk", "lane_change_risk"
    )

    def __init__(
        self,
        vehicle_id: int,
        action: str,
        is_safe: bool = True,
        is_critical: bool = False,
        risk_reasons: Optional[List[str]] = None,
        min_predicted_distance: float = 80.0,
        min_ttc: float = float("inf"),
        stopping_distance_required: float = 10.0,
        headway_margin: float = 10.0,
        traffic_light_risk: bool = False,
        lane_change_risk: bool = False,
    ):
        self.vehicle_id = int(vehicle_id)
        self.action = str(action)
        self.is_safe = bool(is_safe)
        self.is_critical = bool(is_critical)
        self.risk_reasons = list(risk_reasons or [])
        self.min_predicted_distance = float(min_predicted_distance)
        self.min_ttc = float(min_ttc)
        self.stopping_distance_required = float(stopping_distance_required)
        self.headway_margin = float(headway_margin)
        self.traffic_light_risk = bool(traffic_light_risk)
        self.lane_change_risk = bool(lane_change_risk)

    def __repr__(self) -> str:
        status = "CRITICAL" if self.is_critical else ("SAFE" if self.is_safe else "UNSAFE")
        return f"Risk({self.vehicle_id}, act={self.action}, {status}, min_d={self.min_predicted_distance:.1f}m, min_ttc={self.min_ttc:.1f}s)"


class RiskEvaluator:
    """Evaluates multi-vehicle safety risks across predicted trajectory futures."""

    def evaluate_vehicle_risk(
        self,
        vehicle_idx: int,
        action: str,
        snapshot: WorldSnapshot,
        platoon_trajs: Dict[int, List[PredictedState]],
        obstacle_trajs: Dict[int, List[PredictedState]],
    ) -> RiskAssessment:
        if not snapshot.is_valid or vehicle_idx >= len(snapshot.platoon):
            return RiskAssessment(
                vehicle_id=vehicle_idx,
                action=action,
                is_safe=False,
                is_critical=True,
                risk_reasons=["INVALID_OR_MISSING_SNAPSHOT"],
                min_predicted_distance=0.0,
                min_ttc=0.0,
            )

        veh = snapshot.platoon[vehicle_idx]
        if not veh.is_alive:
            return RiskAssessment(veh.id, action, is_safe=True)

        my_traj = platoon_trajs.get(veh.id, [])
        reasons: List[str] = []
        is_critical = False
        min_dist = 80.0
        min_ttc = float("inf")

        # 1. Current State Invariant Checks (Zero/Overlapping distance)
        if veh.predecessor_id is not None and veh.predecessor_gap <= MIN_COLLISION_SEPARATION_M:
            reasons.append("IMMEDIATE_PREDECESSOR_COLLISION_RISK")
            is_critical = True
            min_dist = veh.predecessor_gap
            min_ttc = 0.0

        # 2. Predecessor Trajectory Check
        if veh.predecessor_id is not None:
            pred_veh = snapshot.platoon[vehicle_idx - 1]
            pred_traj = platoon_trajs.get(veh.predecessor_id, [])
            
            d_stop_req = compute_stopping_distance(
                follower_speed_mps=veh.speed,
                lead_speed_mps=pred_veh.speed,
                friction=veh.friction,
            )

            # Compare step-by-step future positions
            steps = min(len(my_traj), len(pred_traj))
            for s in range(steps):
                my_st = my_traj[s]
                pred_st = pred_traj[s]
                dist = my_st.position.distance_to(pred_st.position) - veh.length
                dist = max(0.0, dist)

                if dist < min_dist:
                    min_dist = dist

                closing = my_st.speed - pred_st.speed
                ttc = compute_ttc(dist, closing)
                if ttc < min_ttc:
                    min_ttc = ttc

                if dist < MIN_COLLISION_SEPARATION_M:
                    is_critical = True
                    reasons.append("PREDICTED_PREDECESSOR_COLLISION")
                    break

            if min_ttc < CRITICAL_TTC_THRESHOLD_SEC:
                is_critical = True
                reasons.append(f"CRITICAL_TTC_PREDECESSOR_{min_ttc:.1f}s")
            elif min_ttc < WARNING_TTC_THRESHOLD_SEC:
                reasons.append(f"LOW_TTC_PREDECESSOR_{min_ttc:.1f}s")

            # Stopping distance margin check
            headway_margin = veh.predecessor_gap - d_stop_req
            if headway_margin < 0.0 and action == "ACCELERATE":
                reasons.append("INSUFFICIENT_HEADWAY_FOR_ACCELERATION")
        else:
            d_stop_req = compute_stopping_distance(veh.speed, 0.0, veh.friction)
            headway_margin = 80.0

        # 3. Obstacle Trajectory Checks (External NPCs / pedestrians)
        min_target_lane_dist = 80.0
        for obs_id, obs_traj in obstacle_trajs.items():
            obs_init = next((o for o in snapshot.obstacles if o.id == obs_id), None)
            if obs_init is not None:
                dx0 = obs_init.position.x - veh.position.x
                dy0 = obs_init.position.y - veh.position.y
                lon0 = dx0 * math.cos(veh.heading_rad) + dy0 * math.sin(veh.heading_rad)
                lat0 = -dx0 * math.sin(veh.heading_rad) + dy0 * math.cos(veh.heading_rad)
                # If obstacle starts behind ego in ego's current lane (rear traffic),
                # ignore it during forward collision evaluation; it is handled by rear threat logic.
                if lon0 < -1.0 and abs(lat0) < 1.9:
                    continue

            steps = min(len(my_traj), len(obs_traj))
            for s in range(steps):
                my_st = my_traj[s]
                obs_st = obs_traj[s]
                dist = max(0.0, my_st.position.distance_to(obs_st.position) - (veh.length + 2.0) * 0.5)

                if dist < min_dist:
                    min_dist = dist

                closing = my_st.speed - obs_st.speed
                ttc = compute_ttc(dist, closing)
                if ttc < min_ttc:
                    min_ttc = ttc

                if dist < MIN_COLLISION_SEPARATION_M:
                    is_critical = True
                    reasons.append(f"PREDICTED_OBSTACLE_COLLISION_ID_{obs_id}")
                    break

                # For lane-change clearance, verify obstacles in the target lane corridor
                # (Ignore vehicles in the original lane being overtaken or rear traffic)
                dx = obs_st.position.x - veh.position.x
                dy = obs_st.position.y - veh.position.y
                lon = dx * math.cos(veh.heading_rad) + dy * math.sin(veh.heading_rad)
                lat = -dx * math.sin(veh.heading_rad) + dy * math.cos(veh.heading_rad)

                is_adjacent_corridor = (abs(lat) >= 0.7)
                if is_adjacent_corridor and lon > -6.0 and dist < min_target_lane_dist:
                    min_target_lane_dist = dist

            if min_ttc < CRITICAL_TTC_THRESHOLD_SEC:
                is_critical = True
                reasons.append(f"CRITICAL_TTC_OBSTACLE_{obs_id}")

        # 4. Traffic Light Stopping Risk (Active leader check: no predecessor)
        tl_risk = False
        if veh.predecessor_id is None and snapshot.traffic_light.is_hard_stop:
            tl_dist = snapshot.traffic_light.distance
            if tl_dist < d_stop_req + 2.0:
                tl_risk = True
                if action == "ACCELERATE":
                    reasons.append("RED_LIGHT_APPROACH_ACCEL_REJECTED")
                if tl_dist < 6.0 and veh.speed > 0.5:
                    is_critical = True
                    reasons.append("RED_LIGHT_OVERSHOOT_CRITICAL")

        # 5. Lane-Change Safety Validation
        lc_risk = False
        if action in ("CHANGE_LEFT", "CHANGE_RIGHT"):
            # Lane changes require unobstructed clearance in the target lane
            if min_target_lane_dist < LANE_CHANGE_TARGET_CLEAR_M:
                lc_risk = True
                reasons.append(f"LANE_CHANGE_INSUFFICIENT_CLEARANCE_{min_target_lane_dist:.1f}m")
            if min_ttc < WARNING_TTC_THRESHOLD_SEC and not is_critical:
                lc_risk = True
                reasons.append(f"LANE_CHANGE_LOW_TTC_{min_ttc:.1f}s")

        # 6. Environmental Visibility Check
        if snapshot.visibility_m < 35.0:  # Heavy fog
            # Safe speed under fog is capped to stopping distance <= visibility
            if d_stop_req > snapshot.visibility_m and action == "ACCELERATE":
                reasons.append("FOG_VISIBILITY_EXCEEDED_FOR_ACCELERATION")

        is_safe = (len(reasons) == 0 and not is_critical)
        return RiskAssessment(
            vehicle_id=veh.id,
            action=action,
            is_safe=is_safe,
            is_critical=is_critical,
            risk_reasons=reasons,
            min_predicted_distance=min_dist,
            min_ttc=min_ttc,
            stopping_distance_required=d_stop_req,
            headway_margin=headway_margin,
            traffic_light_risk=tl_risk,
            lane_change_risk=lc_risk,
        )

