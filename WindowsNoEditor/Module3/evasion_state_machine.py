# WindowsNoEditor/Module3/evasion_state_machine.py
"""Emergency Evasion State Machine and Multi-Candidate Lane Validation Engine.

Manages the lifecycle of rear-threat evasion:
NORMAL -> REAR_RISK_DETECTED -> EVASION_EVALUATION -> EVASION_PREPARATION ->
EVADING_LEFT / EVADING_RIGHT -> SAFE_HOLD -> WAITING_FOR_SAFE_REJOIN ->
REJOINING -> RECOVERY -> NORMAL (or EMERGENCY_NO_SAFE_ESCAPE when blocked).
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from Module3.config import (
    EVASION_LANE_CHANGE_CLEARANCE_M,
    EVASION_LATERAL_OFFSET_M,
    REJOIN_SAFE_GAP_M,
)
from Module3.snapshot import WorldSnapshot, VehicleSnapshot, Vector2D
from Module3.rear_threat import RearThreatAssessment


class EvasionState(str, Enum):
    NORMAL = "NORMAL"
    REAR_RISK_DETECTED = "REAR_RISK_DETECTED"
    EVASION_EVALUATION = "EVASION_EVALUATION"
    EVASION_PREPARATION = "EVASION_PREPARATION"
    EVADING_LEFT = "EVADING_LEFT"
    EVADING_RIGHT = "EVADING_RIGHT"
    SAFE_HOLD = "SAFE_HOLD"
    WAITING_FOR_SAFE_REJOIN = "WAITING_FOR_SAFE_REJOIN"
    REJOINING = "REJOINING"
    RECOVERY = "RECOVERY"
    EMERGENCY_NO_SAFE_ESCAPE = "EMERGENCY_NO_SAFE_ESCAPE"


class EvasionCandidate:
    """Evaluation result for a prospective candidate evasion maneuver."""
    __slots__ = (
        "action", "direction", "target_lane_id", "is_valid", "predicted_collision",
        "min_predicted_clearance_m", "rejection_reason"
    )

    def __init__(
        self,
        action: str,
        direction: str,
        target_lane_id: Optional[int] = None,
        is_valid: bool = False,
        predicted_collision: bool = False,
        min_predicted_clearance_m: float = 0.0,
        rejection_reason: str = "",
    ):
        self.action = str(action)
        self.direction = str(direction)
        self.target_lane_id = target_lane_id
        self.is_valid = bool(is_valid)
        self.predicted_collision = bool(predicted_collision)
        self.min_predicted_clearance_m = max(0.0, float(min_predicted_clearance_m))
        self.rejection_reason = str(rejection_reason)

    def __repr__(self) -> str:
        status = "SAFE" if self.is_valid else "REJECTED"
        return f"Candidate({self.action}, {status}, clearance={self.min_predicted_clearance_m:.1f}m, reason='{self.rejection_reason}')"


class VehicleEvasionTracker:
    """Maintains state machine and trajectory state for a single platoon vehicle."""

    def __init__(self, vehicle_id: int, role: str):
        self.vehicle_id = int(vehicle_id)
        self.role = str(role)
        self.state = EvasionState.NORMAL
        self.state_since_sec = 0.0
        self.evasion_direction: Optional[str] = None  # "LEFT" or "RIGHT"
        self.target_lateral_offset_m: float = 0.0
        self.active_threat_id: Optional[int] = None
        self.last_candidates: List[EvasionCandidate] = []
        self.last_transition_reason: str = ""
        self.threat_passed_timer_sec: float = 0.0

    def transition_to(self, new_state: EvasionState, current_time: float, reason: str) -> None:
        self.state = new_state
        self.state_since_sec = current_time
        self.last_transition_reason = reason


class EvasionStateMachine:
    """Evaluates multi-vehicle evasion candidates, enforces traffic/safety rules, and drives transitions."""

    def __init__(self):
        self.trackers: Dict[int, VehicleEvasionTracker] = {}

    def get_or_create_tracker(self, veh_id: int, role: str) -> VehicleEvasionTracker:
        if veh_id not in self.trackers:
            self.trackers[veh_id] = VehicleEvasionTracker(veh_id, role)
        return self.trackers[veh_id]

    @staticmethod
    def _check_target_lane_clearance(
        ego: VehicleSnapshot,
        target_lat_offset: float,
        snapshot: WorldSnapshot,
    ) -> Tuple[bool, float, str]:
        """Check if adjacent target lane has adequate longitudinal and lateral clearance."""
        cos_h = math.cos(ego.heading_rad)
        sin_h = math.sin(ego.heading_rad)

        # Expected center position of ego in target lane
        target_x = ego.position.x - target_lat_offset * sin_h
        target_y = ego.position.y + target_lat_offset * cos_h
        target_pos = Vector2D(target_x, target_y)

        min_clearance = 80.0
        platoon_ids = {p.id for p in snapshot.platoon if p.is_alive}

        # Check all other platoon members and obstacles
        all_actors = [(p.position, p.speed, p.heading_rad, p.id) for p in snapshot.platoon if p.is_alive and p.id != ego.id]
        all_actors += [(obs.position, obs.speed, obs.heading_rad, obs.id) for obs in snapshot.obstacles]

        for pos, spd, hd, actor_id in all_actors:
            dx = pos.x - target_pos.x
            dy = pos.y - target_pos.y
            lon = dx * cos_h + dy * sin_h
            lat = -dx * sin_h + dy * cos_h

            # A. Swept path envelope verification (from current lane to target lane):
            # Check intermediate lateral positions at 33% and 66% shift
            for alpha in (0.33, 0.66):
                inter_x = ego.position.x - (alpha * target_lat_offset) * sin_h
                inter_y = ego.position.y + (alpha * target_lat_offset) * cos_h
                sweep_dist = math.hypot(pos.x - inter_x, pos.y - inter_y)
                if sweep_dist < 2.8:
                    return False, sweep_dist, f"Swept evasion path intersects actor #{actor_id} (dist={sweep_dist:.1f}m)"

            # B. Target lane corridor occupancy (|lat| < 1.85m):
            if abs(lat) < 1.85:
                longitudinal_gap = abs(lon) - 4.7
                if longitudinal_gap < min_clearance:
                    min_clearance = max(0.0, longitudinal_gap)

                # Check if longitudinal gap is below required escape threshold
                if abs(lon) < EVASION_LANE_CHANGE_CLEARANCE_M:
                    return False, min_clearance, f"Target lane occupied by actor #{actor_id} (lon={lon:.1f}m)"

                # Check lag vehicle approaching in target lane from behind
                if lon < 0:
                    closing = spd - ego.speed
                    if closing > 1.0:
                        ttc = abs(lon) / closing
                        if ttc < 3.0:
                            return False, min_clearance, f"Fast vehicle #{actor_id} approaching in target lane (TTC={ttc:.1f}s)"

        return True, min_clearance, "CLEAR"

    def validate_lane_change(
        self,
        ego: VehicleSnapshot,
        direction: str,  # "LEFT" or "RIGHT"
        snapshot: WorldSnapshot,
        carla_world: Optional[Any] = None,
    ) -> EvasionCandidate:
        """Validate feasibility, road legality, traffic-signal compliance, and trajectory safety."""
        action_name = "CHANGE_LEFT" if direction == "LEFT" else "CHANGE_RIGHT"
        lat_offset = -EVASION_LATERAL_OFFSET_M if direction == "LEFT" else EVASION_LATERAL_OFFSET_M

        # 1. Red Light Stop Line Constraint:
        # A vehicle stopped at a red light must NOT execute a lane change that enters an intersection illegally.
        if snapshot.traffic_light.is_hard_stop:
            tl_dist = snapshot.traffic_light.distance
            # If right at the stop line (tl_dist < 4.0m), a lane change into an intersection turn is forbidden
            if tl_dist < 3.5:
                return EvasionCandidate(
                    action=action_name,
                    direction=direction,
                    is_valid=False,
                    predicted_collision=True,
                    rejection_reason="Red-light stop line prevents lateral intersection intrusion",
                )

        # 2. CARLA Map / Topology verification (if live map is accessible)
        target_lane_id = None
        if carla_world and hasattr(carla_world, "get_map"):
            try:
                carla_map = carla_world.get_map()
                import carla
                ego_loc = carla.Location(x=ego.position.x, y=ego.position.y, z=0.0)
                curr_wp = carla_map.get_waypoint(ego_loc, project_to_road=True)
                if curr_wp is not None:
                    candidate_wp = curr_wp.get_left_lane() if direction == "LEFT" else curr_wp.get_right_lane()
                    if candidate_wp is None or candidate_wp.lane_type != carla.LaneType.Driving:
                        return EvasionCandidate(
                            action=action_name,
                            direction=direction,
                            is_valid=False,
                            predicted_collision=False,
                            rejection_reason=f"No legal driving lane exists to the {direction.lower()}",
                        )
                    # Check opposite direction driving lane
                    if (curr_wp.lane_id * candidate_wp.lane_id) < 0 and curr_wp.road_id == candidate_wp.road_id:
                        # On two-lane two-way roads, adjacent lane might be oncoming
                        return EvasionCandidate(
                            action=action_name,
                            direction=direction,
                            is_valid=False,
                            predicted_collision=True,
                            rejection_reason=f"Adjacent {direction.lower()} lane is oncoming traffic",
                        )
                    target_lane_id = candidate_wp.lane_id
            except Exception:
                pass

        # 3. Dynamic target lane clearance check
        is_clear, min_clear, reason = self._check_target_lane_clearance(ego, lat_offset, snapshot)
        if not is_clear:
            return EvasionCandidate(
                action=action_name,
                direction=direction,
                target_lane_id=target_lane_id,
                is_valid=False,
                predicted_collision=True,
                min_predicted_clearance_m=min_clear,
                rejection_reason=reason,
            )

        return EvasionCandidate(
            action=action_name,
            direction=direction,
            target_lane_id=target_lane_id,
            is_valid=True,
            predicted_collision=False,
            min_predicted_clearance_m=min_clear,
            rejection_reason="",
        )

    def evaluate_candidates(
        self,
        ego: VehicleSnapshot,
        snapshot: WorldSnapshot,
        carla_world: Optional[Any] = None,
    ) -> List[EvasionCandidate]:
        """Generate and evaluate full set of candidate evasion maneuvers."""
        left_cand = self.validate_lane_change(ego, "LEFT", snapshot, carla_world)
        right_cand = self.validate_lane_change(ego, "RIGHT", snapshot, carla_world)
        maintain_cand = EvasionCandidate(
            action="CRUISE" if ego.speed > 0.5 else "SAFE_HOLD",
            direction="NONE",
            is_valid=True,
            predicted_collision=False,
            min_predicted_clearance_m=ego.predecessor_gap,
            rejection_reason="",
        )
        return [left_cand, right_cand, maintain_cand]

    def step(
        self,
        ego_idx: int,
        rear_threat: RearThreatAssessment,
        snapshot: WorldSnapshot,
        current_time: float,
        carla_world: Optional[Any] = None,
    ) -> Tuple[str, EvasionState, float, List[EvasionCandidate]]:
        """Advance evasion state machine for the specified vehicle.
        
        Returns:
            recommended_action: Tactical action string (e.g. CRUISE, CHANGE_LEFT, CHANGE_RIGHT, DECELERATE)
            current_state: EvasionState
            target_lateral_offset: float (meters from nominal lane centerline)
            candidates: List of evaluated candidates
        """
        ego = snapshot.platoon[ego_idx]
        tracker = self.get_or_create_tracker(ego.id, ego.role)
        state = tracker.state
        recommended_action = "CRUISE"
        candidates: List[EvasionCandidate] = []

        # ── STATE 1: NORMAL ──────────────────────────────────────────────────────────
        if state == EvasionState.NORMAL:
            tracker.target_lateral_offset_m = 0.0
            if rear_threat.is_actionable:
                tracker.active_threat_id = rear_threat.threat_id
                tracker.transition_to(
                    EvasionState.REAR_RISK_DETECTED,
                    current_time,
                    f"Rear threat #{rear_threat.threat_id} detected with risk {rear_threat.risk_level}",
                )
                state = EvasionState.REAR_RISK_DETECTED
            else:
                recommended_action = "CRUISE"

        # ── STATE 2: REAR_RISK_DETECTED ──────────────────────────────────────────────
        if state == EvasionState.REAR_RISK_DETECTED:
            tracker.transition_to(
                EvasionState.EVASION_EVALUATION,
                current_time,
                "Evaluating candidate lateral evasion escape routes",
            )
            state = EvasionState.EVASION_EVALUATION

        # ── STATE 3: EVASION_EVALUATION ──────────────────────────────────────────────
        if state == EvasionState.EVASION_EVALUATION:
            candidates = self.evaluate_candidates(ego, snapshot, carla_world)
            tracker.last_candidates = candidates

            # Select safest candidate: prefer Left if legal, else Right
            left_valid = candidates[0].is_valid
            right_valid = candidates[1].is_valid

            if left_valid:
                tracker.evasion_direction = "LEFT"
                tracker.target_lateral_offset_m = -EVASION_LATERAL_OFFSET_M
                tracker.transition_to(
                    EvasionState.EVASION_PREPARATION,
                    current_time,
                    "Left lane escape route validated safe",
                )
                state = EvasionState.EVASION_PREPARATION
            elif right_valid:
                tracker.evasion_direction = "RIGHT"
                tracker.target_lateral_offset_m = EVASION_LATERAL_OFFSET_M
                tracker.transition_to(
                    EvasionState.EVASION_PREPARATION,
                    current_time,
                    "Right lane escape route validated safe",
                )
                state = EvasionState.EVASION_PREPARATION
            else:
                tracker.evasion_direction = None
                tracker.target_lateral_offset_m = 0.0
                tracker.transition_to(
                    EvasionState.EMERGENCY_NO_SAFE_ESCAPE,
                    current_time,
                    "All lateral escape lanes blocked or constrained; holding position",
                )
                state = EvasionState.EMERGENCY_NO_SAFE_ESCAPE

        # ── STATE 4: EVASION_PREPARATION ─────────────────────────────────────────────
        if state == EvasionState.EVASION_PREPARATION:
            next_state = EvasionState.EVADING_LEFT if tracker.evasion_direction == "LEFT" else EvasionState.EVADING_RIGHT
            tracker.transition_to(next_state, current_time, f"Initiating dynamic evasion {tracker.evasion_direction}")
            recommended_action = "CHANGE_LEFT" if tracker.evasion_direction == "LEFT" else "CHANGE_RIGHT"
            state = next_state

        # ── STATE 5: EVADING_LEFT / EVADING_RIGHT ────────────────────────────────────
        elif state in (EvasionState.EVADING_LEFT, EvasionState.EVADING_RIGHT):
            recommended_action = "CHANGE_LEFT" if state == EvasionState.EVADING_LEFT else "CHANGE_RIGHT"
            # Maneuver completion condition:
            # Completed after allocated evasion duration (e.g. 2.0s)
            time_in_evasion = current_time - tracker.state_since_sec
            if time_in_evasion >= 2.0:
                tracker.transition_to(
                    EvasionState.SAFE_HOLD,
                    current_time,
                    f"Evasion to {tracker.evasion_direction} complete; holding safe position in adjacent lane",
                )
                state = EvasionState.SAFE_HOLD

        # ── STATE 6: SAFE_HOLD ───────────────────────────────────────────────────────
        elif state == EvasionState.SAFE_HOLD:
            recommended_action = "CRUISE"
            # Check if rear threat has passed:
            is_threat_gone = (
                rear_threat.threat_passed or
                rear_threat.risk_level == "NORMAL" or
                rear_threat.threat_id != tracker.active_threat_id
            )
            if is_threat_gone:
                if tracker.threat_passed_timer_sec == 0.0:
                    tracker.threat_passed_timer_sec = current_time
                if (current_time - tracker.threat_passed_timer_sec >= 1.0) or rear_threat.threat_passed:
                    tracker.transition_to(
                        EvasionState.WAITING_FOR_SAFE_REJOIN,
                        current_time,
                        "Threat passed; monitoring original platoon lane for safe rejoin opening",
                    )
            else:
                tracker.threat_passed_timer_sec = 0.0

        # ── STATE 7: WAITING_FOR_SAFE_REJOIN ─────────────────────────────────────────
        elif state == EvasionState.WAITING_FOR_SAFE_REJOIN:
            recommended_action = "CRUISE"
            # Verify original lane centerline (offset 0.0) is clear
            is_orig_clear, gap_avail, _ = self._check_target_lane_clearance(
                ego, -tracker.target_lateral_offset_m, snapshot
            )
            if is_orig_clear and gap_avail >= REJOIN_SAFE_GAP_M:
                tracker.transition_to(
                    EvasionState.REJOINING,
                    current_time,
                    f"Safe gap of {gap_avail:.1f}m confirmed in platoon lane; executing rejoin",
                )

        # ── STATE 8: REJOINING ───────────────────────────────────────────────────────
        elif state == EvasionState.REJOINING:
            # Return lane change: if previously evaded LEFT, now CHANGE_RIGHT to return
            recommended_action = "CHANGE_RIGHT" if tracker.evasion_direction == "LEFT" else "CHANGE_LEFT"
            time_in_rejoin = current_time - tracker.state_since_sec
            if time_in_rejoin >= 2.0:
                tracker.target_lateral_offset_m = 0.0
                tracker.transition_to(
                    EvasionState.RECOVERY,
                    current_time,
                    "Rejoin maneuver complete; realigning with platoon formation",
                )

        # ── STATE 9: RECOVERY ────────────────────────────────────────────────────────
        elif state == EvasionState.RECOVERY:
            recommended_action = "CRUISE"
            tracker.evasion_direction = None
            tracker.active_threat_id = None
            tracker.target_lateral_offset_m = 0.0
            tracker.transition_to(
                EvasionState.NORMAL,
                current_time,
                "Nominal platoon formation restored",
            )

        # ── STATE 10: EMERGENCY_NO_SAFE_ESCAPE ───────────────────────────────────────
        elif state == EvasionState.EMERGENCY_NO_SAFE_ESCAPE:
            # Trapped at red light or between blocked lanes: hold safest feasible control
            recommended_action = "DECELERATE" if ego.speed > 0.5 else "CRUISE"
            if rear_threat.risk_level == "NORMAL" or rear_threat.threat_passed:
                tracker.transition_to(
                    EvasionState.NORMAL,
                    current_time,
                    "Rear threat dissipated or stopped; exiting emergency escape trap",
                )

        return recommended_action, tracker.state, tracker.target_lateral_offset_m, candidates
