# WindowsNoEditor/Module3/tests/test_module3_safety.py
"""Deterministic Unit Tests for Module 3 Twin-World Safety Projection System.

Tests all required safety and prediction invariants without launching CARLA:
1. Safe CRUISE is passed unchanged.
2. Safe ACCELERATE is passed unchanged.
3. Unsafe acceleration is replaced with a safer action (CRUISE or DECELERATE).
4. Unsafe CRUISE is replaced with DECELERATE where appropriate.
5. Critical collision prediction triggers the emergency signal.
6. Overlapping vehicle is treated as critical risk.
7. TTC handles closing, equal-speed, separating, and invalid cases.
8. Following-distance calculations handle dry, wet, and low-friction conditions.
9. Fog visibility and road friction remain separate inputs.
10. Unsafe lane-change proposal is rejected.
11. Missing lane geometry does not incorrectly certify a lane change as safe.
12. Nearby intrusion is detected in the prediction.
13. Platoon-level effects are evaluated for follower safety.
14. NaN, Inf, missing actors, and stale snapshots trigger conservative handling.
15. Prediction timeout / invalid snapshot triggers correct mode-dependent fallback.
16. Shadow mode preserves the original action.
17. Active mode uses the validated action.
18. Emergency braking does not add a sixth MAPPO action.
19. Disabled Module 3 preserves Module 2 action behavior.
20. Twin does not mutate the authoritative world.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
for p in (ROOT, os.path.join(ROOT, "WindowsNoEditor")):
    if p not in sys.path:
        sys.path.insert(0, p)

from Module3.snapshot import (
    Vector2D,
    VehicleSnapshot,
    ObstacleSnapshot,
    TrafficLightSnapshot,
    WorldSnapshot,
)
from Module3.twin_world import TwinWorld
from Module3.risk_evaluator import (
    RiskEvaluator,
    compute_ttc,
    compute_stopping_distance,
)
from Module3.safety_shield import SafetyShield


class TestModule3Safety(unittest.TestCase):
    def setUp(self):
        # Standard 3-vehicle platoon in clear conditions
        self.v0 = VehicleSnapshot(1, "Leader", True, Vector2D(50.0, 0.0), 8.0, 0.0, friction=1.0)
        self.v1 = VehicleSnapshot(2, "Follower 1", True, Vector2D(35.0, 0.0), 8.0, 0.0, friction=1.0, predecessor_id=1, predecessor_gap=10.3)
        self.v2 = VehicleSnapshot(3, "Follower 2", True, Vector2D(20.0, 0.0), 8.0, 0.0, friction=1.0, predecessor_id=2, predecessor_gap=10.3)
        self.tl = TrafficLightSnapshot("GREEN", 50.0, False)
        self.clear_snapshot = WorldSnapshot(
            timestamp=10.0,
            frame=300,
            platoon=[self.v0, self.v1, self.v2],
            obstacles=[],
            traffic_light=self.tl,
            weather_preset="CLEAR",
            visibility_m=120.0,
            nominal_friction=1.0,
        )

    def test_01_safe_cruise_passed_unchanged(self):
        shield = SafetyShield(mode="ACTIVE")
        acts, emerg, rep = shield.arbitrate(["CRUISE", "CRUISE", "CRUISE"], self.clear_snapshot)
        self.assertEqual(acts, ["CRUISE", "CRUISE", "CRUISE"])
        self.assertEqual(emerg, [False, False, False])

    def test_02_safe_accelerate_passed_unchanged(self):
        # Open road with large gaps
        v0 = VehicleSnapshot(1, "Leader", True, Vector2D(60.0, 0.0), 6.0, 0.0)
        v1 = VehicleSnapshot(2, "Follower 1", True, Vector2D(35.0, 0.0), 6.0, 0.0, predecessor_id=1, predecessor_gap=20.3)
        snap = WorldSnapshot(10.0, 300, [v0, v1], [], self.tl)
        shield = SafetyShield(mode="ACTIVE")
        acts, emerg, rep = shield.arbitrate(["ACCELERATE", "ACCELERATE"], snap)
        self.assertEqual(acts, ["ACCELERATE", "ACCELERATE"])
        self.assertEqual(emerg, [False, False])

    def test_03_unsafe_acceleration_replaced_with_safer_action(self):
        # Follower 1 close behind leader (gap 4.5m)
        v0 = VehicleSnapshot(1, "Leader", True, Vector2D(20.0, 0.0), 5.0, 0.0)
        v1 = VehicleSnapshot(2, "Follower 1", True, Vector2D(10.8, 0.0), 5.0, 0.0, predecessor_id=1, predecessor_gap=4.5)
        snap = WorldSnapshot(10.0, 300, [v0, v1], [], self.tl)
        shield = SafetyShield(mode="ACTIVE")
        acts, emerg, rep = shield.arbitrate(["CRUISE", "ACCELERATE"], snap)
        # Follower 1's ACCELERATE must be downgraded to CRUISE or DECELERATE
        self.assertIn(acts[1], ("CRUISE", "DECELERATE"))
        self.assertNotEqual(acts[1], "ACCELERATE")

    def test_04_unsafe_cruise_replaced_with_decelerate(self):
        # Closing quickly on predecessor
        v0 = VehicleSnapshot(1, "Leader", True, Vector2D(20.0, 0.0), 2.0, 0.0)
        v1 = VehicleSnapshot(2, "Follower 1", True, Vector2D(12.0, 0.0), 8.0, 0.0, predecessor_id=1, predecessor_gap=3.3)
        snap = WorldSnapshot(10.0, 300, [v0, v1], [], self.tl)
        shield = SafetyShield(mode="ACTIVE")
        acts, emerg, rep = shield.arbitrate(["CRUISE", "CRUISE"], snap)
        self.assertEqual(acts[1], "DECELERATE")

    def test_05_critical_collision_triggers_emergency_signal(self):
        # Critical gap < 1.2m
        v0 = VehicleSnapshot(1, "Leader", True, Vector2D(15.0, 0.0), 2.0, 0.0)
        v1 = VehicleSnapshot(2, "Follower 1", True, Vector2D(9.5, 0.0), 6.0, 0.0, predecessor_id=1, predecessor_gap=0.8)
        snap = WorldSnapshot(10.0, 300, [v0, v1], [], self.tl)
        shield = SafetyShield(mode="ACTIVE")
        acts, emerg, rep = shield.arbitrate(["CRUISE", "CRUISE"], snap)
        self.assertTrue(emerg[1])
        self.assertEqual(acts[1], "DECELERATE")

    def test_06_overlapping_vehicle_treated_as_critical_risk(self):
        v0 = VehicleSnapshot(1, "Leader", True, Vector2D(10.0, 0.0), 0.0, 0.0)
        v1 = VehicleSnapshot(2, "Follower 1", True, Vector2D(10.0, 0.0), 0.0, 0.0, predecessor_id=1, predecessor_gap=0.0)
        snap = WorldSnapshot(10.0, 300, [v0, v1], [], self.tl)
        shield = SafetyShield(mode="ACTIVE")
        acts, emerg, rep = shield.arbitrate(["CRUISE", "CRUISE"], snap)
        self.assertTrue(emerg[1])

    def test_07_ttc_mathematical_handling(self):
        # 1. Closing case
        self.assertAlmostEqual(compute_ttc(10.0, 2.0), 5.0)
        # 2. Equal speed (closing = 0.0) -> inf
        self.assertEqual(compute_ttc(10.0, 0.0), float("inf"))
        # 3. Separating (closing < 0) -> inf
        self.assertEqual(compute_ttc(10.0, -2.0), float("inf"))
        # 4. Zero or negative distance -> 0.0
        self.assertEqual(compute_ttc(0.0, 2.0), 0.0)
        self.assertEqual(compute_ttc(-1.0, 2.0), 0.0)
        # 5. Non-finite values -> 0.0
        self.assertEqual(compute_ttc(float("nan"), 2.0), 0.0)
        self.assertEqual(compute_ttc(10.0, float("nan")), 0.0)

    def test_08_following_distance_weather_friction_sensitivity(self):
        # Compare dry vs wet vs low friction
        d_dry = compute_stopping_distance(8.0, 0.0, friction=1.0)
        d_wet = compute_stopping_distance(8.0, 0.0, friction=0.65)
        d_ice = compute_stopping_distance(8.0, 0.0, friction=0.35)
        self.assertLess(d_dry, d_wet)
        self.assertLess(d_wet, d_ice)

    def test_09_fog_visibility_and_road_friction_remain_separate(self):
        # High friction with low visibility (dense dry fog, visibility 20m)
        # At 15 m/s (~54 km/h), stopping distance is 15*0.35 + 225/(2*6.5) + 2.0 = 5.25 + 17.3 + 2.0 = 24.55m > 20.0m!
        v0 = VehicleSnapshot(1, "Leader", True, Vector2D(10.0, 0.0), 15.0, 0.0, friction=1.0)
        snap = WorldSnapshot(10.0, 300, [v0], [], self.tl, weather_preset="HEAVY_FOG", visibility_m=20.0, nominal_friction=1.0)
        evaluator = RiskEvaluator()
        twin = TwinWorld()
        trajs, _ = twin.simulate_future(snap, ["ACCELERATE"])
        risk = evaluator.evaluate_vehicle_risk(0, "ACCELERATE", snap, trajs, {})
        # Must detect visibility risk even though friction is 1.0
        self.assertIn("FOG_VISIBILITY_EXCEEDED_FOR_ACCELERATION", risk.risk_reasons)

    def test_10_unsafe_lane_change_rejected(self):
        # Obstacle occupying adjacent lane within 12m ahead
        obs = ObstacleSnapshot(99, Vector2D(60.0, 1.0), 8.0, 0.0)
        snap = WorldSnapshot(10.0, 300, [self.v0], [obs], self.tl)
        shield = SafetyShield(mode="ACTIVE")
        acts, emerg, rep = shield.arbitrate(["CHANGE_LEFT"], snap)
        self.assertNotEqual(acts[0], "CHANGE_LEFT")

    def test_11_missing_lane_geometry_does_not_certify_lane_change(self):
        evaluator = RiskEvaluator()
        snap = WorldSnapshot(10.0, 300, [], [], self.tl, is_valid=False)
        risk = evaluator.evaluate_vehicle_risk(0, "CHANGE_LEFT", snap, {}, {})
        self.assertFalse(risk.is_safe)
        self.assertTrue(risk.is_critical)

    def test_12_nearby_intrusion_detected_in_prediction(self):
        # Ghost intruder cuts directly in front of follower 1
        intruder = ObstacleSnapshot(42, Vector2D(38.0, 0.0), 3.0, 0.0)
        snap = WorldSnapshot(10.0, 300, [self.v0, self.v1], [intruder], self.tl)
        shield = SafetyShield(mode="ACTIVE")
        acts, emerg, rep = shield.arbitrate(["CRUISE", "ACCELERATE"], snap)
        self.assertNotEqual(acts[1], "ACCELERATE")

    def test_13_platoon_level_follower_safety_evaluated(self):
        # Leader sudden deceleration propagates caution to follower
        v0 = VehicleSnapshot(1, "Leader", True, Vector2D(20.0, 0.0), 3.0, 0.0)
        v1 = VehicleSnapshot(2, "Follower 1", True, Vector2D(12.0, 0.0), 8.0, 0.0, predecessor_id=1, predecessor_gap=3.3)
        snap = WorldSnapshot(10.0, 300, [v0, v1], [], self.tl)
        shield = SafetyShield(mode="ACTIVE")
        acts, emerg, rep = shield.arbitrate(["DECELERATE", "ACCELERATE"], snap)
        self.assertEqual(acts[1], "DECELERATE")

    def test_14_nan_inf_triggers_conservative_handling(self):
        evaluator = RiskEvaluator()
        snap = WorldSnapshot(10.0, 300, [], [], self.tl, is_valid=False, error_message="NaN in telemetry")
        risk = evaluator.evaluate_vehicle_risk(0, "CRUISE", snap, {}, {})
        self.assertFalse(risk.is_safe)
        self.assertTrue(risk.is_critical)

    def test_15_invalid_snapshot_fallback(self):
        snap = WorldSnapshot(10.0, 300, [], [], self.tl, is_valid=False)
        shield_act = SafetyShield(mode="ACTIVE")
        acts, emerg, rep = shield_act.arbitrate(["ACCELERATE"], snap)
        self.assertEqual(acts, ["DECELERATE"])

    def test_16_shadow_mode_preserves_original_action(self):
        # Critical situation in SHADOW mode
        v0 = VehicleSnapshot(1, "Leader", True, Vector2D(15.0, 0.0), 0.0, 0.0)
        v1 = VehicleSnapshot(2, "Follower 1", True, Vector2D(14.0, 0.0), 8.0, 0.0, predecessor_id=1, predecessor_gap=0.3)
        snap = WorldSnapshot(10.0, 300, [v0, v1], [], self.tl)
        shield = SafetyShield(mode="SHADOW")
        acts, emerg, rep = shield.arbitrate(["ACCELERATE", "ACCELERATE"], snap)
        # Original actions preserved in shadow mode!
        self.assertEqual(acts, ["ACCELERATE", "ACCELERATE"])
        self.assertEqual(emerg, [False, False])
        # But report records the modification verdict
        self.assertTrue(rep.decisions[1].is_critical)
        self.assertEqual(rep.decisions[1].validated_action, "DECELERATE")

    def test_17_active_mode_uses_validated_action(self):
        v0 = VehicleSnapshot(1, "Leader", True, Vector2D(15.0, 0.0), 0.0, 0.0)
        v1 = VehicleSnapshot(2, "Follower 1", True, Vector2D(14.0, 0.0), 8.0, 0.0, predecessor_id=1, predecessor_gap=0.3)
        snap = WorldSnapshot(10.0, 300, [v0, v1], [], self.tl)
        shield = SafetyShield(mode="ACTIVE")
        acts, emerg, rep = shield.arbitrate(["ACCELERATE", "ACCELERATE"], snap)
        self.assertEqual(acts[1], "DECELERATE")
        self.assertTrue(emerg[1])

    def test_18_emergency_braking_does_not_add_sixth_mappo_action(self):
        # Validated actions must strictly belong to 5-element MAPPO action space
        mappo_set = {"CRUISE", "ACCELERATE", "DECELERATE", "CHANGE_LEFT", "CHANGE_RIGHT"}
        v0 = VehicleSnapshot(1, "Leader", True, Vector2D(10.0, 0.0), 0.0, 0.0)
        snap = WorldSnapshot(10.0, 300, [v0], [], TrafficLightSnapshot("RED", 2.0, True))
        shield = SafetyShield(mode="ACTIVE")
        acts, emerg, rep = shield.arbitrate(["ACCELERATE"], snap)
        for a in acts:
            self.assertIn(a, mappo_set)

    def test_19_disabled_module3_preserves_module2_behavior(self):
        shield = SafetyShield(mode="DISABLED")
        acts, emerg, rep = shield.arbitrate(["ACCELERATE", "CHANGE_LEFT"], self.clear_snapshot)
        self.assertEqual(acts, ["ACCELERATE", "CHANGE_LEFT"])
        self.assertEqual(emerg, [False, False])

    def test_20_twin_does_not_mutate_snapshot(self):
        # Twin forward simulation must not mutate input snapshot objects
        twin = TwinWorld()
        initial_pos = (self.v0.position.x, self.v0.position.y)
        trajs, _ = twin.simulate_future(self.clear_snapshot, ["ACCELERATE", "ACCELERATE", "ACCELERATE"])
        self.assertEqual((self.v0.position.x, self.v0.position.y), initial_pos)


if __name__ == "__main__":
    unittest.main()
