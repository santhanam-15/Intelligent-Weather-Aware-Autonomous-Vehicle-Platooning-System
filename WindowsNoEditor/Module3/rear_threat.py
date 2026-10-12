# WindowsNoEditor/Module3/rear_threat.py
"""Rear-vehicle detection, kinematic trajectory projection, and rear-impact risk prediction.

Continuously monitors rear-approaching traffic for every platoon member in both
stationary (e.g. stopped at red signal) and moving operational regimes.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from Module3.config import (
    ENABLE_REAR_EVASION,
    REAR_PREDICTION_HORIZON_SEC,
    REAR_TTC_CRITICAL_SEC,
    REAR_TTC_HIGH_RISK_SEC,
    REAR_TTC_CAUTION_SEC,
    REAR_DISTANCE_THRESHOLD_M,
    NPC_CONSERVATIVE_BRAKE_MPS2,
    NPC_REACTION_TIME_SEC,
    REAR_HYSTERESIS_FRAMES,
    REAR_SAFE_CLEARANCE_FRAMES,
)
from Module3.snapshot import WorldSnapshot, VehicleSnapshot, ObstacleSnapshot, Vector2D


class RearThreatAssessment:
    """Immutable assessment of rear-impact risk for a single platoon vehicle."""
    __slots__ = (
        "ego_id", "ego_role", "threat_id", "risk_level", "rear_distance_m",
        "closing_speed_mps", "rear_ttc_s", "npc_speed_mps", "npc_stopping_dist_m",
        "ego_speed_mps", "is_in_lane", "is_converging", "threat_passed", "reasons"
    )

    def __init__(
        self,
        ego_id: int,
        ego_role: str,
        threat_id: Optional[int],
        risk_level: str,
        rear_distance_m: float,
        closing_speed_mps: float,
        rear_ttc_s: float,
        npc_speed_mps: float,
        npc_stopping_dist_m: float,
        ego_speed_mps: float,
        is_in_lane: bool,
        is_converging: bool,
        threat_passed: bool = False,
        reasons: Optional[List[str]] = None,
    ):
        self.ego_id = int(ego_id)
        self.ego_role = str(ego_role)
        self.threat_id = int(threat_id) if threat_id is not None else None
        self.risk_level = str(risk_level).upper()  # "NORMAL", "CAUTION", "HIGH_RISK", "CRITICAL"
        self.rear_distance_m = max(0.0, float(rear_distance_m))
        self.closing_speed_mps = float(closing_speed_mps)
        self.rear_ttc_s = float(rear_ttc_s)
        self.npc_speed_mps = max(0.0, float(npc_speed_mps))
        self.npc_stopping_dist_m = max(0.0, float(npc_stopping_dist_m))
        self.ego_speed_mps = max(0.0, float(ego_speed_mps))
        self.is_in_lane = bool(is_in_lane)
        self.is_converging = bool(is_converging)
        self.threat_passed = bool(threat_passed)
        self.reasons = list(reasons or [])

    @property
    def is_critical(self) -> bool:
        return self.risk_level == "CRITICAL"

    @property
    def is_actionable(self) -> bool:
        return self.risk_level in ("HIGH_RISK", "CRITICAL")

    def __repr__(self) -> str:
        ttc_str = f"{self.rear_ttc_s:.2f}s" if math.isfinite(self.rear_ttc_s) else "inf"
        tid = self.threat_id if self.threat_id is not None else "None"
        return (
            f"RearThreat(ego={self.ego_id}, threat={tid}, risk={self.risk_level}, "
            f"dist={self.rear_distance_m:.1f}m, closing={self.closing_speed_mps:.1f}m/s, "
            f"TTC={ttc_str})"
        )


class RearThreatPredictor:
    """Evaluates rear-impact trajectories, closing kinematics, and risk levels with hysteresis."""

    def __init__(
        self,
        critical_ttc: float = REAR_TTC_CRITICAL_SEC,
        high_risk_ttc: float = REAR_TTC_HIGH_RISK_SEC,
        caution_ttc: float = REAR_TTC_CAUTION_SEC,
        max_range: float = REAR_DISTANCE_THRESHOLD_M,
        npc_brake_accel: float = NPC_CONSERVATIVE_BRAKE_MPS2,
        reaction_time: float = NPC_REACTION_TIME_SEC,
        hysteresis_frames: int = REAR_HYSTERESIS_FRAMES,
        clearance_frames: int = REAR_SAFE_CLEARANCE_FRAMES,
    ):
        self.critical_ttc = float(critical_ttc)
        self.high_risk_ttc = float(high_risk_ttc)
        self.caution_ttc = float(caution_ttc)
        self.max_range = float(max_range)
        self.npc_brake_accel = float(npc_brake_accel)
        self.reaction_time = float(reaction_time)
        self.hysteresis_frames = int(hysteresis_frames)
        self.clearance_frames = int(clearance_frames)

        # Per-vehicle persistence filters: vehicle_id -> list of raw risk assessments
        self._history: Dict[int, List[str]] = {}
        self._last_active_threat: Dict[int, Optional[int]] = {}

    def reset(self) -> None:
        self._history.clear()
        self._last_active_threat.clear()

    @staticmethod
    def _project_relative(ego: VehicleSnapshot, other_pos: Vector2D) -> Tuple[float, float]:
        """Project other position into ego vehicle coordinates (longitudinal along heading, lateral perpendicular)."""
        dx = other_pos.x - ego.position.x
        dy = other_pos.y - ego.position.y
        cos_h = math.cos(ego.heading_rad)
        sin_h = math.sin(ego.heading_rad)
        lon = dx * cos_h + dy * sin_h
        lat = -dx * sin_h + dy * cos_h
        return lon, lat

    def compute_npc_stopping_distance(self, speed_mps: float, friction: float = 1.0) -> float:
        """Conservative estimated stopping distance for approaching NPC including reaction delay."""
        eff_friction = max(0.2, min(1.2, friction))
        a_brake = max(1.5, self.npc_brake_accel * eff_friction)
        d_reaction = speed_mps * self.reaction_time
        d_brake = (speed_mps ** 2) / (2.0 * a_brake)
        return d_reaction + d_brake

    def evaluate_threat_for_vehicle(
        self,
        ego_idx: int,
        snapshot: WorldSnapshot,
    ) -> RearThreatAssessment:
        """Evaluate rear-impact threats targeting the specified platoon vehicle."""
        if not snapshot.is_valid or ego_idx >= len(snapshot.platoon):
            return RearThreatAssessment(
                ego_id=ego_idx,
                ego_role=f"Vehicle {ego_idx}",
                threat_id=None,
                risk_level="NORMAL",
                rear_distance_m=80.0,
                closing_speed_mps=0.0,
                rear_ttc_s=float("inf"),
                npc_speed_mps=0.0,
                npc_stopping_dist_m=0.0,
                ego_speed_mps=0.0,
                is_in_lane=False,
                is_converging=False,
            )

        ego = snapshot.platoon[ego_idx]
        if not ego.is_alive:
            return RearThreatAssessment(
                ego_id=ego.id,
                ego_role=ego.role,
                threat_id=None,
                risk_level="NORMAL",
                rear_distance_m=80.0,
                closing_speed_mps=0.0,
                rear_ttc_s=float("inf"),
                npc_speed_mps=0.0,
                npc_stopping_dist_m=0.0,
                ego_speed_mps=0.0,
                is_in_lane=False,
                is_converging=False,
            )

        # Ignore other platoon members as rear threats; only external obstacles
        platoon_ids = {p.id for p in snapshot.platoon if p.is_alive}
        friction = snapshot.nominal_friction

        candidates: List[Tuple[float, float, float, float, ObstacleSnapshot, List[str]]] = []

        for obs in snapshot.obstacles:
            if obs.id in platoon_ids:
                continue

            lon, lat = self._project_relative(ego, obs.position)

            # Must be behind ego (lon < 0) within maximum tracking range
            if lon >= -0.5 or abs(lon) > (self.max_range + ego.length):
                continue

            # Bumper-to-bumper rear gap
            rear_gap = max(0.0, abs(lon) - (ego.length + obs.length) * 0.5)

            # Lateral lane corridor:
            # Standard CARLA lane is ~3.5m wide. Half-width corridor is 1.75m.
            # Allow up to 1.9m for tire overhang/lane edge, but reject > 2.2m (adjacent lane).
            is_in_lane = (abs(lat) < 1.90)
            if not is_in_lane:
                continue

            # Heading alignment:
            # NPC must be traveling in roughly the same direction along the road (not oncoming)
            angle_diff = abs(obs.heading_rad - ego.heading_rad)
            while angle_diff > math.pi:
                angle_diff = abs(angle_diff - 2.0 * math.pi)

            if angle_diff > (math.pi / 3.0):  # > 60 degrees misaligned
                continue

            # Longitudinal velocity along ego heading
            npc_fwd_spd = obs.speed * math.cos(angle_diff)
            closing_spd = npc_fwd_spd - ego.speed

            # Stopping distance
            npc_stop_dist = self.compute_npc_stopping_distance(obs.speed, friction)

            # TTC calculation: only finite if actively closing
            if closing_spd > 0.35 and rear_gap > 0.05:
                rear_ttc = rear_gap / closing_spd
            else:
                rear_ttc = float("inf")

            reasons: List[str] = []
            if closing_spd > 0.5:
                reasons.append(f"CLOSING_AT_{closing_spd:.1f}MPS")

            candidates.append((rear_gap, closing_spd, rear_ttc, npc_stop_dist, obs, reasons))

        if not candidates:
            # Check if previous threat has passed
            prev_threat = self._last_active_threat.get(ego.id, None)
            threat_passed = (prev_threat is not None)
            if threat_passed:
                self._last_active_threat[ego.id] = None

            raw_assessment = RearThreatAssessment(
                ego_id=ego.id,
                ego_role=ego.role,
                threat_id=None,
                risk_level="NORMAL",
                rear_distance_m=80.0,
                closing_speed_mps=0.0,
                rear_ttc_s=float("inf"),
                npc_speed_mps=0.0,
                npc_stopping_dist_m=0.0,
                ego_speed_mps=ego.speed,
                is_in_lane=False,
                is_converging=False,
                threat_passed=threat_passed,
            )
            return self._apply_hysteresis(ego.id, raw_assessment)

        # Select the most critical candidate (smallest TTC, or smallest gap if equal)
        candidates.sort(key=lambda c: (c[2] if math.isfinite(c[2]) else 999.0, c[0]))
        best_gap, best_closing, best_ttc, best_stop_dist, best_obs, reasons = candidates[0]

        # Determine raw risk level
        raw_risk = "NORMAL"

        # Threat severity logic:
        # A. Stationary ego (e.g. stopped at red light or in queue):
        # An approaching NPC is ONLY a collision threat if it is physically unable to stop
        # in the available bumper-to-bumper gap (d_gap < d_stop_npc + margin).
        if ego.speed < 0.4:
            npc_can_stop_comfortably = (best_gap >= (best_stop_dist + 2.5))
            if npc_can_stop_comfortably:
                # NPC has sufficient distance to stop safely; this is normal queue approach
                if best_gap < 10.0 and best_ttc <= self.caution_ttc:
                    raw_risk = "CAUTION"
                    reasons.append(f"STATIONARY_APPROACH_ABLE_TO_STOP_GAP_{best_gap:.1f}M")
                else:
                    raw_risk = "NORMAL"
            else:
                # Physical stopping deficit: NPC cannot stop without collision or excessive braking
                reasons.append(f"STATIONARY_GAP_{best_gap:.1f}M_LESS_THAN_STOPPING_{best_stop_dist:.1f}M")
                if best_obs.speed > 3.0:
                    if best_ttc <= self.critical_ttc or best_gap < 12.0:
                        raw_risk = "CRITICAL"
                    elif best_ttc <= self.high_risk_ttc or best_gap < 20.0:
                        raw_risk = "HIGH_RISK"
                    else:
                        raw_risk = "CAUTION"
                else:
                    # Low-speed creeping in traffic
                    raw_risk = "NORMAL"
        else:
            # B. Moving ego:
            # Evaluates relative stopping capability and closing time
            ego_stop_dist = (ego.speed ** 2) / (2.0 * max(1.5, 6.0 * friction))
            rel_stop_deficit = max(0.0, best_stop_dist - ego_stop_dist)

            if best_closing < 1.0:
                raw_risk = "NORMAL"
            elif best_ttc <= self.critical_ttc and best_closing > 2.5 and best_gap < (rel_stop_deficit + 8.0):
                raw_risk = "CRITICAL"
                reasons.append(f"MOVING_CRITICAL_TTC_{best_ttc:.2f}S_CLOSING_{best_closing:.1f}MPS")
            elif best_ttc <= self.high_risk_ttc and best_closing > 1.8 and best_gap < (rel_stop_deficit + 5.0):
                raw_risk = "HIGH_RISK"
                reasons.append(f"MOVING_HIGH_RISK_TTC_{best_ttc:.2f}S")
            elif best_ttc <= self.caution_ttc:
                raw_risk = "CAUTION"
                reasons.append(f"MOVING_CAUTION_TTC_{best_ttc:.2f}S")

        raw_assessment = RearThreatAssessment(
            ego_id=ego.id,
            ego_role=ego.role,
            threat_id=best_obs.id,
            risk_level=raw_risk,
            rear_distance_m=best_gap,
            closing_speed_mps=best_closing,
            rear_ttc_s=best_ttc,
            npc_speed_mps=best_obs.speed,
            npc_stopping_dist_m=best_stop_dist,
            ego_speed_mps=ego.speed,
            is_in_lane=True,
            is_converging=(best_closing > 0.5),
            threat_passed=False,
            reasons=reasons,
        )

        return self._apply_hysteresis(ego.id, raw_assessment)

    def _apply_hysteresis(self, ego_id: int, assessment: RearThreatAssessment) -> RearThreatAssessment:
        """Filter raw assessments through multi-frame hysteresis to prevent rapid mode oscillation."""
        if ego_id not in self._history:
            self._history[ego_id] = []

        hist = self._history[ego_id]
        hist.append(assessment.risk_level)
        if len(hist) > 10:
            hist.pop(0)

        # Escalation requires at least hysteresis_frames of high/critical level
        escalated_levels = [r for r in hist[-self.hysteresis_frames:] if r in ("HIGH_RISK", "CRITICAL")]
        deescalated_levels = [r for r in hist[-self.clearance_frames:] if r == "NORMAL"]

        filtered_risk = assessment.risk_level

        if assessment.risk_level in ("HIGH_RISK", "CRITICAL"):
            if len(escalated_levels) < self.hysteresis_frames:
                # Still validating persistence; hold at CAUTION if elevating
                filtered_risk = "CAUTION"
            else:
                self._last_active_threat[ego_id] = assessment.threat_id
        elif assessment.risk_level == "NORMAL":
            if len(deescalated_levels) < self.clearance_frames and len(hist) >= self.clearance_frames:
                # Hold previous caution while clearing
                if any(r in ("HIGH_RISK", "CRITICAL", "CAUTION") for r in hist[-self.clearance_frames:]):
                    filtered_risk = "CAUTION"

        return RearThreatAssessment(
            ego_id=assessment.ego_id,
            ego_role=assessment.ego_role,
            threat_id=assessment.threat_id,
            risk_level=filtered_risk,
            rear_distance_m=assessment.rear_distance_m,
            closing_speed_mps=assessment.closing_speed_mps,
            rear_ttc_s=assessment.rear_ttc_s,
            npc_speed_mps=assessment.npc_speed_mps,
            npc_stopping_dist_m=assessment.npc_stopping_dist_m,
            ego_speed_mps=assessment.ego_speed_mps,
            is_in_lane=assessment.is_in_lane,
            is_converging=assessment.is_converging,
            threat_passed=assessment.threat_passed,
            reasons=assessment.reasons,
        )

    def detect_all_rear_threats(
        self,
        snapshot: WorldSnapshot,
    ) -> Dict[int, RearThreatAssessment]:
        """Assess rear threat for every platoon member in parallel."""
        results: Dict[int, RearThreatAssessment] = {}
        for idx in range(len(snapshot.platoon)):
            veh = snapshot.platoon[idx]
            results[veh.id] = self.evaluate_threat_for_vehicle(idx, snapshot)
        return results

