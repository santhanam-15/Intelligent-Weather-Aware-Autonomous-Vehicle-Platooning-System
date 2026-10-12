# WindowsNoEditor/Module3/decision_report.py
"""Structured decision records and diagnostic logging for Module 3 Twin-World Safety Projection."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


class VehicleSafetyDecision:
    """Individual vehicle safety arbitration decision."""
    __slots__ = (
        "vehicle_id", "role", "original_action", "validated_action",
        "emergency_brake", "is_safe", "is_critical", "action_modified",
        "risk_reasons", "min_distance_m", "min_ttc_s", "stopping_dist_m",
        "headway_margin_m", "rear_risk_level", "rear_ttc_s", "evasion_state",
        "lateral_offset_m"
    )

    def __init__(
        self,
        vehicle_id: int,
        role: str,
        original_action: str,
        validated_action: str,
        emergency_brake: bool,
        is_safe: bool,
        is_critical: bool,
        action_modified: bool,
        risk_reasons: List[str],
        min_distance_m: float,
        min_ttc_s: float,
        stopping_dist_m: float,
        headway_margin_m: float,
        rear_risk_level: str = "NORMAL",
        rear_ttc_s: float = float("inf"),
        evasion_state: str = "NORMAL",
        lateral_offset_m: float = 0.0,
    ):
        self.vehicle_id = int(vehicle_id)
        self.role = str(role)
        self.original_action = str(original_action)
        self.validated_action = str(validated_action)
        self.emergency_brake = bool(emergency_brake)
        self.is_safe = bool(is_safe)
        self.is_critical = bool(is_critical)
        self.action_modified = bool(action_modified)
        self.risk_reasons = list(risk_reasons)
        self.min_distance_m = float(min_distance_m)
        self.min_ttc_s = float(min_ttc_s)
        self.stopping_dist_m = float(stopping_dist_m)
        self.headway_margin_m = float(headway_margin_m)
        self.rear_risk_level = str(rear_risk_level)
        self.rear_ttc_s = float(rear_ttc_s)
        self.evasion_state = str(evasion_state)
        self.lateral_offset_m = float(lateral_offset_m)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vehicle_id": self.vehicle_id,
            "role": self.role,
            "original_action": self.original_action,
            "validated_action": self.validated_action,
            "emergency_brake": self.emergency_brake,
            "is_safe": self.is_safe,
            "is_critical": self.is_critical,
            "action_modified": self.action_modified,
            "risk_reasons": self.risk_reasons,
            "min_distance_m": round(self.min_distance_m, 2),
            "min_ttc_s": round(self.min_ttc_s, 2) if self.min_ttc_s < 999.0 else "inf",
            "stopping_dist_m": round(self.stopping_dist_m, 2),
            "headway_margin_m": round(self.headway_margin_m, 2),
            "rear_risk_level": self.rear_risk_level,
            "rear_ttc_s": round(self.rear_ttc_s, 2) if self.rear_ttc_s < 999.0 else "inf",
            "evasion_state": self.evasion_state,
            "lateral_offset_m": round(self.lateral_offset_m, 2),
        }


class SafetyCycleReport:
    """Cycle-wide safety report capturing all decisions, timing, and environmental context."""
    __slots__ = (
        "timestamp", "frame", "mode", "execution_time_ms", "budget_exceeded",
        "decisions", "weather_preset", "visibility_m", "friction", "snapshot_valid"
    )

    def __init__(
        self,
        timestamp: float,
        frame: int,
        mode: str,
        execution_time_ms: float,
        budget_exceeded: bool,
        decisions: List[VehicleSafetyDecision],
        weather_preset: str = "CLEAR",
        visibility_m: float = 120.0,
        friction: float = 1.0,
        snapshot_valid: bool = True,
    ):
        self.timestamp = float(timestamp)
        self.frame = int(frame)
        self.mode = str(mode)
        self.execution_time_ms = float(execution_time_ms)
        self.budget_exceeded = bool(budget_exceeded)
        self.decisions = list(decisions)
        self.weather_preset = str(weather_preset)
        self.visibility_m = float(visibility_m)
        self.friction = float(friction)
        self.snapshot_valid = bool(snapshot_valid)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 3),
            "frame": self.frame,
            "mode": self.mode,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "budget_exceeded": self.budget_exceeded,
            "weather": self.weather_preset,
            "visibility_m": round(self.visibility_m, 1),
            "friction": round(self.friction, 2),
            "snapshot_valid": self.snapshot_valid,
            "decisions": [d.to_dict() for d in self.decisions],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

