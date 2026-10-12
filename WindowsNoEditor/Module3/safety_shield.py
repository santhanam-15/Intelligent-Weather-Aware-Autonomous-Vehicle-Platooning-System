# WindowsNoEditor/Module3/safety_shield.py
"""Safety Shield module for Module 3.

Arbitrates proposed tactical actions against World B risk assessments,
coordinates rear-impact prediction and emergency evasion state machines,
handles shadow and active operating modes, manages timeouts and fallback safety,
and returns validated actions alongside auxiliary emergency brake flags.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

from Module3.config import (
    MODULE3_ENABLED,
    MODULE3_MODE,
    MAX_EXECUTION_BUDGET_MS,
    ENABLE_REAR_EVASION,
    REJOIN_SAFE_GAP_M,
)
from Module3.snapshot import WorldSnapshot, build_world_snapshot
from Module3.twin_world import TwinWorld
from Module3.risk_evaluator import RiskEvaluator, RiskAssessment
from Module3.decision_report import VehicleSafetyDecision, SafetyCycleReport
from Module3.rear_threat import RearThreatPredictor, RearThreatAssessment
from Module3.evasion_state_machine import EvasionStateMachine, EvasionState, EvasionCandidate
from Module3.platoon_coordinator import PlatoonEvasionCoordinator
from Module3.logger import Module3DecisionLogger

logger = logging.getLogger("Module3SafetyShield")


class SafetyShield:
    """Action validation and arbitration engine."""

    def __init__(
        self,
        mode: str = MODULE3_MODE,
        enabled: bool = MODULE3_ENABLED,
        budget_ms: float = MAX_EXECUTION_BUDGET_MS,
    ):
        self.mode = mode.upper()  # "DISABLED", "SHADOW", "ACTIVE"
        self.enabled = bool(enabled and self.mode != "DISABLED")
        self.budget_ms = float(budget_ms)
        self.twin = TwinWorld()
        self.evaluator = RiskEvaluator()
        self.rear_predictor = RearThreatPredictor()
        self.evasion_sm = EvasionStateMachine()
        self.coordinator = PlatoonEvasionCoordinator()
        self.logger = Module3DecisionLogger.get_instance()
        self.latest_lateral_offsets: Dict[int, float] = {}
        self._prev_states: Dict[int, str] = {}
        self._decision_seq: int = 0
        self.last_report: Optional[SafetyCycleReport] = None

    def arbitrate(
        self,
        proposed_actions: List[str],
        snapshot: WorldSnapshot,
    ) -> Tuple[List[str], List[bool], SafetyCycleReport]:
        """Arbitrate proposed actions against World B predictions and rear threat models.
        
        Returns:
            validated_actions: List[str] matching platoon size (strictly within 5-action MAPPO set)
            emergency_flags: List[bool] auxiliary full-brake flags per vehicle
            report: SafetyCycleReport containing structured diagnostics
        """
        t_start = time.perf_counter()
        num_vehicles = len(snapshot.platoon) if snapshot.platoon else len(proposed_actions)

        # Fast path if disabled
        if not self.enabled or self.mode == "DISABLED":
            decisions = [
                VehicleSafetyDecision(
                    vehicle_id=snapshot.platoon[i].id if (snapshot.platoon and i < len(snapshot.platoon)) else i,
                    role="Leader" if i == 0 else f"Follower {i}",
                    original_action=proposed_actions[i],
                    validated_action=proposed_actions[i],
                    emergency_brake=False,
                    is_safe=True,
                    is_critical=False,
                    action_modified=False,
                    risk_reasons=[],
                    min_distance_m=80.0,
                    min_ttc_s=float("inf"),
                    stopping_dist_m=0.0,
                    headway_margin_m=80.0,
                )
                for i in range(len(proposed_actions))
            ]
            rep = SafetyCycleReport(
                timestamp=snapshot.timestamp,
                frame=snapshot.frame,
                mode="DISABLED",
                execution_time_ms=0.0,
                budget_exceeded=False,
                decisions=decisions,
            )
            self.last_report = rep
            return list(proposed_actions), [False] * len(proposed_actions), rep

        # Fallback on invalid snapshot
        if not snapshot.is_valid:
            logger.warning("[MODULE 3] Invalid snapshot received: %s. Invoking fallback.", snapshot.error_message)
            fallback_act = "DECELERATE" if self.mode == "ACTIVE" else None
            out_actions = [fallback_act or act for act in proposed_actions]
            decisions = [
                VehicleSafetyDecision(
                    vehicle_id=snapshot.platoon[i].id if (snapshot.platoon and i < len(snapshot.platoon)) else i,
                    role="Leader" if i == 0 else f"Follower {i}",
                    original_action=proposed_actions[i],
                    validated_action=out_actions[i],
                    emergency_brake=False,
                    is_safe=False,
                    is_critical=True,
                    action_modified=(out_actions[i] != proposed_actions[i]),
                    risk_reasons=["INVALID_SNAPSHOT_FALLBACK"],
                    min_distance_m=0.0,
                    min_ttc_s=0.0,
                    stopping_dist_m=0.0,
                    headway_margin_m=0.0,
                )
                for i in range(len(proposed_actions))
            ]
            rep = SafetyCycleReport(
                timestamp=snapshot.timestamp,
                frame=snapshot.frame,
                mode=self.mode,
                execution_time_ms=0.0,
                budget_exceeded=False,
                decisions=decisions,
                snapshot_valid=False,
            )
            self.last_report = rep
            return out_actions, [False] * len(proposed_actions), rep

        # ── 1. REAR THREAT DETECTION & DYNAMIC EVASION STATE MACHINE ──
        rear_threats = self.rear_predictor.detect_all_rear_threats(snapshot) if ENABLE_REAR_EVASION else {}
        candidate_evasion_actions: List[str] = []
        target_lat_offsets: Dict[int, float] = {}

        for idx in range(num_vehicles):
            veh = snapshot.platoon[idx]
            threat = rear_threats.get(veh.id, None)
            if threat is None:
                threat = RearThreatAssessment(
                    ego_id=veh.id,
                    ego_role=veh.role,
                    threat_id=None,
                    risk_level="NORMAL",
                    rear_distance_m=80.0,
                    closing_speed_mps=0.0,
                    rear_ttc_s=float("inf"),
                    npc_speed_mps=0.0,
                    npc_stopping_dist_m=0.0,
                    ego_speed_mps=veh.speed,
                    is_in_lane=False,
                    is_converging=False,
                )

            ev_act, ev_state, lat_offset, cands = self.evasion_sm.step(
                idx, threat, snapshot, snapshot.timestamp
            )
            candidate_evasion_actions.append(ev_act)
            target_lat_offsets[veh.id] = lat_offset

            # Log risk assessment & candidate evaluations to dedicated terminal logger
            prev_st = self._prev_states.get(veh.id, "NORMAL")
            curr_st = ev_state.value
            if curr_st != prev_st:
                self.logger.log_state_transition(
                    snapshot.timestamp, veh.id, prev_st, curr_st,
                    self.evasion_sm.trackers[veh.id].last_transition_reason
                )
                self._prev_states[veh.id] = curr_st

            self.logger.log_risk_assessment(
                snapshot.timestamp, veh.id, curr_st, threat.risk_level,
                threat.rear_ttc_s, threat.rear_distance_m, threat.closing_speed_mps
            )

            for cand in cands:
                self.logger.log_candidate_validation(
                    snapshot.timestamp, veh.id, cand.action, cand.target_lane_id,
                    cand.is_valid, cand.rejection_reason
                )

            if ev_state == EvasionState.EMERGENCY_NO_SAFE_ESCAPE:
                self.logger.log_emergency(
                    snapshot.timestamp, veh.id,
                    "All escape lanes blocked or constrained", "HOLD_CURRENT_LANE"
                )
            elif ev_state in (EvasionState.WAITING_FOR_SAFE_REJOIN, EvasionState.REJOINING):
                dec = "APPROVED" if ev_state == EvasionState.REJOINING else "WAITING_GAP"
                self.logger.log_rejoin_status(
                    snapshot.timestamp, veh.id, curr_st, REJOIN_SAFE_GAP_M, dec
                )

        # ── 2. PLATOON MULTI-VEHICLE COORDINATION ──
        # Ensure followers do not copycat lane changes and calculate isolated following
        coordinated_evasion_actions, _ = self.coordinator.coordinate_platoon(
            snapshot, self.evasion_sm.trackers, candidate_evasion_actions
        )

        # ── 3. FORWARD TRAJECTORY PREDICTION (WORLD B) ──
        # Predict forward futures under proposed actions
        self._decision_seq += 1
        decision_id = f"M3-T{snapshot.timestamp:.2f}-F{snapshot.frame}-{self._decision_seq:04d}"

        t_pred_start = time.perf_counter()
        actions_to_simulate = [
            coordinated_evasion_actions[i] if self.evasion_sm.trackers[snapshot.platoon[i].id].state not in (EvasionState.NORMAL, EvasionState.RECOVERY) else proposed_actions[i]
            for i in range(num_vehicles)
        ]
        platoon_trajs, obstacle_trajs = self.twin.simulate_future(snapshot, actions_to_simulate)
        t_pred_end = time.perf_counter()
        pred_duration_ms = (t_pred_end - t_pred_start) * 1000.0

        validated_actions: List[str] = []
        emergency_flags: List[bool] = []
        decisions: List[VehicleSafetyDecision] = []

        # ── 4. EVALUATE RISKS & ARBITRATE ACTIONS ──
        t_eval_start = time.perf_counter()
        for idx in range(num_vehicles):
            orig_act = proposed_actions[idx]
            veh = snapshot.platoon[idx]
            role = veh.role
            tracker = self.evasion_sm.trackers.get(veh.id)
            ev_state_str = tracker.state.value if tracker else "NORMAL"
            threat = rear_threats.get(veh.id)

            # Evaluate forward collision risk
            risk = self.evaluator.evaluate_vehicle_risk(
                idx, actions_to_simulate[idx], snapshot, platoon_trajs, obstacle_trajs
            )

            val_act = orig_act
            emerg = False

            if risk.is_critical:
                emerg = True
                val_act = "DECELERATE"
            elif not risk.is_safe:
                if orig_act == "ACCELERATE":
                    cruise_trajs, _ = self.twin.simulate_future(
                        snapshot, [("CRUISE" if i == idx else actions_to_simulate[i]) for i in range(num_vehicles)]
                    )
                    c_risk = self.evaluator.evaluate_vehicle_risk(
                        idx, "CRUISE", snapshot, cruise_trajs, obstacle_trajs
                    )
                    val_act = "CRUISE" if c_risk.is_safe else "DECELERATE"
                elif orig_act in ("CRUISE", "CHANGE_LEFT", "CHANGE_RIGHT"):
                    val_act = "DECELERATE"
            elif tracker and tracker.state in (EvasionState.EVASION_PREPARATION, EvasionState.EVADING_LEFT, EvasionState.EVADING_RIGHT, EvasionState.REJOINING):
                # Evasion maneuver is active and verified forward-safe
                val_act = coordinated_evasion_actions[idx]

            # In SHADOW mode, we compute the verdict but do NOT alter actual vehicle actions
            action_to_dispatch = orig_act if self.mode == "SHADOW" else val_act
            flag_to_dispatch = False if self.mode == "SHADOW" else emerg

            validated_actions.append(action_to_dispatch)
            emergency_flags.append(flag_to_dispatch)

            # Log arbitration record to dedicated decision logger
            self.logger.log_action_arbitration(
                snapshot.timestamp, veh.id, orig_act, val_act, action_to_dispatch, emergency_brake=emerg
            )

            # If a lane change action was proposed, log dedicated overtake/lane-change decision
            if orig_act in ("CHANGE_LEFT", "CHANGE_RIGHT"):
                dir_str = "LEFT" if orig_act == "CHANGE_LEFT" else "RIGHT"
                dec_str = "APPROVED" if risk.is_safe else ("REJECTED_SHADOW_OBSERVED" if self.mode == "SHADOW" else "REJECTED_ACTIVE_ENFORCED")
                rsn_str = ", ".join(risk.risk_reasons) if risk.risk_reasons else "CLEAR"
                self.logger.log_overtake_decision(
                    sim_time_sec=snapshot.timestamp,
                    actor_id=veh.id,
                    role=role,
                    event_tag="TARGET_LANE_SAFETY_EVALUATION",
                    target_lane=dir_str,
                    gap_front=risk.min_predicted_distance,
                    gap_rear=80.0,
                    decision=dec_str,
                    reason=rsn_str,
                )

            decisions.append(
                VehicleSafetyDecision(
                    vehicle_id=veh.id,
                    role=role,
                    original_action=orig_act,
                    validated_action=val_act,
                    emergency_brake=emerg,
                    is_safe=risk.is_safe,
                    is_critical=risk.is_critical,
                    action_modified=(val_act != orig_act or emerg),
                    risk_reasons=risk.risk_reasons,
                    min_distance_m=risk.min_predicted_distance,
                    min_ttc_s=risk.min_ttc,
                    stopping_dist_m=risk.stopping_distance_required,
                    headway_margin_m=risk.headway_margin,
                    rear_risk_level=threat.risk_level if threat else "NORMAL",
                    rear_ttc_s=threat.rear_ttc_s if threat else float("inf"),
                    evasion_state=ev_state_str,
                    lateral_offset_m=target_lat_offsets.get(veh.id, 0.0),
                )
            )

        t_eval_end = time.perf_counter()
        eval_duration_ms = (t_eval_end - t_eval_start) * 1000.0
        self.latest_lateral_offsets = target_lat_offsets

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        budget_exceeded = (t_elapsed_ms > self.budget_ms)

        # Log complete pipeline trace: SNAPSHOT -> PREDICTION -> RISK EVALUATION -> SHIELD -> MODULE 2 COMMAND
        self.logger.log_pipeline_trace(
            decision_id=decision_id,
            timestamp=snapshot.timestamp,
            frame=snapshot.frame,
            snapshot_valid=True,
            n_platoon=num_vehicles,
            n_obstacles=len(snapshot.obstacles),
            pred_duration_ms=pred_duration_ms,
            eval_duration_ms=eval_duration_ms,
            total_duration_ms=t_elapsed_ms,
            budget_exceeded=budget_exceeded,
            mode=self.mode,
            proposed_actions=list(proposed_actions),
            selected_actions=validated_actions,
            emergency_flags=emergency_flags,
            delivered_actions=validated_actions,
            lateral_offsets=target_lat_offsets,
        )

        rep = SafetyCycleReport(
            timestamp=snapshot.timestamp,
            frame=snapshot.frame,
            mode=self.mode,
            execution_time_ms=t_elapsed_ms,
            budget_exceeded=budget_exceeded,
            decisions=decisions,
            weather_preset=snapshot.weather_preset,
            visibility_m=snapshot.visibility_m,
            friction=snapshot.nominal_friction,
            snapshot_valid=True,
        )
        self.last_report = rep
        return validated_actions, emergency_flags, rep
