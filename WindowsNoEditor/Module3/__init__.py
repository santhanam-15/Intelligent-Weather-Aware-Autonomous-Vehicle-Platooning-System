# WindowsNoEditor/Module3/__init__.py
"""Module 3: Twin-World Predictive Safety Projection System for CARLA Platooning."""

from Module3.config import MODULE3_ENABLED, MODULE3_MODE
from Module3.snapshot import WorldSnapshot, VehicleSnapshot, Vector2D, build_world_snapshot
from Module3.twin_world import TwinWorld, PredictedState
from Module3.risk_evaluator import RiskEvaluator, RiskAssessment, compute_ttc, compute_stopping_distance
from Module3.safety_shield import SafetyShield
from Module3.decision_report import VehicleSafetyDecision, SafetyCycleReport
from Module3.rear_threat import RearThreatAssessment, RearThreatPredictor
from Module3.evasion_state_machine import EvasionState, EvasionCandidate, EvasionStateMachine
from Module3.platoon_coordinator import PlatoonEvasionCoordinator
from Module3.logger import Module3DecisionLogger

__all__ = [
    "MODULE3_ENABLED",
    "MODULE3_MODE",
    "WorldSnapshot",
    "VehicleSnapshot",
    "Vector2D",
    "build_world_snapshot",
    "TwinWorld",
    "PredictedState",
    "RiskEvaluator",
    "RiskAssessment",
    "compute_ttc",
    "compute_stopping_distance",
    "SafetyShield",
    "VehicleSafetyDecision",
    "SafetyCycleReport",
    "RearThreatAssessment",
    "RearThreatPredictor",
    "EvasionState",
    "EvasionCandidate",
    "EvasionStateMachine",
    "PlatoonEvasionCoordinator",
    "Module3DecisionLogger",
]
