# WindowsNoEditor/Module2/tests/test_longitudinal_controller.py
"""Unit tests for the isolated longitudinal controller correction in MissionPlatoonController.

Verifies:
1. Commanded stop requests force zero throttle and positive brake hold.
2. Cruising speed produces smooth feedforward-plus-proportional throttle.
3. Overspeed within the deadband coasts naturally with zero throttle and zero brake.
4. Overspeed beyond the deadband engages proportional braking.
5. Strict mutual exclusion: throttle and brake never overlap across the operating range.
6. Standstill breakout is inhibited under stop conditions, obstacles, or safety overrides.
7. Emergency predecessor gap triggers full brake override.
"""

import os
import sys
import unittest

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
for path in (ROOT, os.path.join(ROOT, "WindowsNoEditor")):
    if path not in sys.path:
        sys.path.insert(0, path)

from Module2.mission_controller import MissionPlatoonController


class MockVehicle:
    def __init__(self, is_alive=True):
        self.is_alive = is_alive


class TestLongitudinalController(unittest.TestCase):
    def setUp(self):
        # We can test _compute_longitudinal_control unbound or bound to an instance
        self.ctrl = MissionPlatoonController.__new__(MissionPlatoonController)

    def test_standstill_stop_request_forces_zero_throttle_and_brake_hold(self):
        """Test 1: Commanded stop requests (target_speed <= 0.05) must force throttle=0.0 and brake > 0."""
        # Stationary vehicle commanded to stop
        throttle, brake = self.ctrl._compute_longitudinal_control(
            target_speed=0.0,
            current_speed=0.0,
        )
        self.assertEqual(throttle, 0.0)
        self.assertGreaterEqual(brake, 0.5)

        # Moving vehicle commanded to stop (e.g. red light or collision hazard)
        throttle_moving, brake_moving = self.ctrl._compute_longitudinal_control(
            target_speed=0.0,
            current_speed=8.0,
        )
        self.assertEqual(throttle_moving, 0.0)
        self.assertEqual(brake_moving, 1.0)

        # Target speed near zero threshold
        throttle_edge, brake_edge = self.ctrl._compute_longitudinal_control(
            target_speed=0.04,
            current_speed=2.0,
        )
        self.assertEqual(throttle_edge, 0.0)
        self.assertEqual(brake_edge, 1.0)

    def test_cruising_speed_produces_smooth_feedforward_and_proportional_throttle(self):
        """Test 2: Cruising speed tracking applies feedforward + proportional throttle."""
        target_speed = 8.33  # ~30 km/h
        current_speed = 8.0   # small tracking error = 0.33 m/s

        throttle, brake = self.ctrl._compute_longitudinal_control(
            target_speed=target_speed,
            current_speed=current_speed,
        )

        expected_k_ff = min(0.38, max(0.0, 0.038 * target_speed))
        expected_error = target_speed - current_speed
        expected_throttle = min(0.65, max(0.0, expected_k_ff + 0.24 * expected_error))

        self.assertAlmostEqual(throttle, expected_throttle, places=4)
        self.assertEqual(brake, 0.0)
        self.assertGreater(throttle, 0.2)
        self.assertLessEqual(throttle, 0.65)

    def test_overspeed_deadband_coasts_without_oscillating_brake(self):
        """Test 3: Overspeed within deadband (0 to 0.35 m/s overspeed) must coast (throttle=0, brake=0)."""
        target_speed = 8.33
        # Slight overspeed of 0.15 m/s (inside 0.35 m/s deadband)
        current_speed = target_speed + 0.15

        throttle, brake = self.ctrl._compute_longitudinal_control(
            target_speed=target_speed,
            current_speed=current_speed,
        )
        self.assertEqual(throttle, 0.0)
        self.assertEqual(brake, 0.0)

        # At the exact boundary of the deadband
        current_speed_edge = target_speed + 0.34
        throttle_edge, brake_edge = self.ctrl._compute_longitudinal_control(
            target_speed=target_speed,
            current_speed=current_speed_edge,
        )
        self.assertEqual(throttle_edge, 0.0)
        self.assertEqual(brake_edge, 0.0)

    def test_overspeed_beyond_deadband_engages_proportional_braking(self):
        """Test 4: Overspeed beyond deadband (>0.35 m/s overspeed) engages smooth proportional braking."""
        target_speed = 8.33
        # Overspeed of 1.0 m/s (beyond 0.35 m/s deadband)
        current_speed = target_speed + 1.0

        throttle, brake = self.ctrl._compute_longitudinal_control(
            target_speed=target_speed,
            current_speed=current_speed,
        )
        self.assertEqual(throttle, 0.0)
        self.assertGreater(brake, 0.0)
        # Expected brake: -0.42 * (error + overspeed_deadband) = -0.42 * (-1.0 + 0.35) = -0.42 * (-0.65) = 0.273
        self.assertAlmostEqual(brake, 0.42 * (1.0 - 0.35), places=3)

    def test_throttle_and_brake_never_overlap(self):
        """Test 5: Exhaustive grid test verifying throttle and brake never simultaneously exceed zero."""
        targets = [0.0, 1.0, 5.0, 8.33, 12.0, 15.0]
        speeds = [0.0, 0.2, 0.8, 3.0, 5.0, 8.33, 9.0, 12.0, 16.0]
        light_dists = [4.0, 10.0, 50.0]
        pred_gaps = [2.5, 4.2, 6.5, 20.0]

        for target in targets:
            for speed in speeds:
                for l_dist in light_dists:
                    for p_gap in pred_gaps:
                        throttle, brake = self.ctrl._compute_longitudinal_control(
                            target_speed=target,
                            current_speed=speed,
                            hard_stop=(l_dist < 8.0),
                            light_distance=l_dist,
                            predecessor=MockVehicle(True),
                            predecessor_clearance=p_gap,
                        )
                        # Neither value may be negative
                        self.assertGreaterEqual(throttle, 0.0)
                        self.assertGreaterEqual(brake, 0.0)
                        # When brake is active (>0.01), throttle MUST be exactly 0.0
                        if brake > 0.01:
                            self.assertEqual(
                                throttle,
                                0.0,
                                "Throttle and brake overlapped! target={}, speed={}, thr={}, brk={}".format(
                                    target, speed, throttle, brake
                                ),
                            )

    def test_standstill_breakout_inhibited_during_stop_conditions(self):
        """Test 6: Standstill breakout throttle is strictly inhibited if any stop or hazard condition exists."""
        # Baseline: Target speed under normal driving conditions has normal throttle + breakout
        thr_norm, _ = self.ctrl._compute_longitudinal_control(
            target_speed=3.0,
            current_speed=0.1,
            hard_stop=False,
            light_distance=50.0,
            obstacle=None,
            obstacle_clearance=80.0,
            predecessor=None,
            predecessor_clearance=80.0,
            override="NORMAL",
        )
        # Normal feedforward + proportional: k_ff=0.114, k_p*error = 0.24*2.9 = 0.696 -> clamped to 0.65
        self.assertGreaterEqual(thr_norm, 0.38)

        # Condition A: Breakout is inhibited when target_speed <= 0.05 (e.g. at red light / stop)
        thr_stop, brk_stop = self.ctrl._compute_longitudinal_control(
            target_speed=0.0,
            current_speed=0.1,
            hard_stop=True,
            light_distance=2.0,
        )
        self.assertEqual(thr_stop, 0.0)
        self.assertGreaterEqual(brk_stop, 0.5)

        # Condition B: Breakout is inhibited when hard_stop=True, obstacle < 6.0m, or predecessor < 6.0m
        # With target_speed=1.0 (error=0.9 m/s, so normal thr = 0.038*1.0 + 0.24*0.9 = 0.254 < 0.38)
        # In clear conditions, error < 1.0 does not trigger breakout
        # Let's test with target_speed=2.5, current_speed=0.1 (error=2.4)
        # Normal thr would be 0.038*2.5 + 0.24*2.4 = 0.095 + 0.576 = 0.671 -> 0.65.
        # But if target_speed=1.5, current_speed=0.1 (error=1.4):
        # k_ff = 0.038 * 1.5 = 0.057, k_p * 1.4 = 0.336 -> base throttle = 0.393
        # If target_speed=1.2, current_speed=0.1 (error=1.1 > 1.0):
        # base throttle = 0.038*1.2 + 0.24*1.1 = 0.0456 + 0.264 = 0.3096 < 0.38.
        # If is_breakout_safe was True, it would boost to 0.38.
        # But target_speed must be >= 2.5 for breakout, preventing breakout when target_speed is low!
        thr_low_target, _ = self.ctrl._compute_longitudinal_control(
            target_speed=1.2,
            current_speed=0.1,
        )
        self.assertLess(thr_low_target, 0.38)

        # Condition C: When target_speed=2.5, error=2.4:
        # Base throttle = 0.095 + 0.576 = 0.65.
        # If hard_stop and light_distance < 8.0, and speed > 0.3, brake engages and throttle is zeroed.
        # If predecessor_clearance < 3.8m, brake=1.0 and throttle is zeroed:
        pred_close = MockVehicle(True)
        thr_pred, brk_pred = self.ctrl._compute_longitudinal_control(
            target_speed=5.0,
            current_speed=0.1,
            predecessor=pred_close,
            predecessor_clearance=3.5,
        )
        self.assertEqual(thr_pred, 0.0)
        self.assertEqual(brk_pred, 1.0)

        # Condition D: Obstacle clearance < 3.0m forces full brake and zero throttle
        obs_close = MockVehicle(True)
        thr_obs, brk_obs = self.ctrl._compute_longitudinal_control(
            target_speed=5.0,
            current_speed=0.1,
            obstacle=obs_close,
            obstacle_clearance=2.5,
        )
        self.assertEqual(thr_obs, 0.0)
        self.assertEqual(brk_obs, 1.0)

        # Condition E: Safety override or active stop command never produces breakout throttle
        thr_override_stop, brk_override_stop = self.ctrl._compute_longitudinal_control(
            target_speed=0.0,
            current_speed=0.0,
            override="COLLISION_RISK",
        )
        self.assertEqual(thr_override_stop, 0.0)
        self.assertGreaterEqual(brk_override_stop, 0.5)

    def test_emergency_predecessor_gap_forces_full_brake(self):
        """Test 7: Critical predecessor gap (< 3.8m) commands emergency full brake (1.0)."""
        pred = MockVehicle(True)
        throttle, brake = self.ctrl._compute_longitudinal_control(
            target_speed=8.0,
            current_speed=5.0,
            predecessor=pred,
            predecessor_clearance=3.2,
        )
        self.assertEqual(throttle, 0.0)
        self.assertEqual(brake, 1.0)


if __name__ == "__main__":
    unittest.main()
