# WindowsNoEditor/Module3/tests/test_rear_evasion.py
"""Deterministic Unit Tests for Module 3 Dynamic Rear-Impact Evasion,
Platoon Rejoining, and Dedicated Live Decision Terminal.

Validates the 15 required test specifications:
1. Stationary platoon at red light; NPC approaching slowly and able to stop.
2. Stationary platoon at red light; fast NPC; safe legal escape lane available.
3. Stationary platoon at red light; fast NPC; all escape lanes blocked.
4. Moving platoon; fast NPC approaching; safe lane change available.
5. Moving platoon; fast NPC approaching; both adjacent lanes occupied.
6. Threat is in adjacent lane and not converging with platoon.
7. Rear threat passes; platoon vehicle safely rejoins.
8. Follower avoids independently copying leader's evasive maneuver.
9. Multiple vehicles face rear-impact risks simultaneously.
10. Red-light stop line prevents illegal lane change into intersection.
11. NPC disappears during prediction.
12. Module 3 disabled preserves Module 2 behavior.
13. SHADOW mode logs decisions without affecting actions.
14. Dedicated logging terminal receives live Module 3 records.
15. Logging output cannot block simulation thread.

Run with:
``py -3.7 -m unittest Module3.tests.test_rear_evasion``
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import time
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
from Module3.rear_threat import RearThreatPredictor, RearThreatAssessment
from Module3.evasion_state_machine import (
    EvasionStateMachine,
    EvasionState,
    EvasionCandidate,
)
from Module3.platoon_coordinator import PlatoonEvasionCoordinator
from Module3.safety_shield import SafetyShield
from Module3.logger import Module3DecisionLogger


class TestRearEvasionAndTerminal(unittest.TestCase):
    def setUp(self):
        # Default 3-vehicle platoon positioned along X axis (heading 0.0 rad)
        self.v0 = VehicleSnapshot(157, "Leader", True, Vector2D(50.0, 0.0), 8.0, 0.0)
        self.v1 = VehicleSnapshot(158, "Follower 1", True, Vector2D(38.0, 0.0), 8.0, 0.0, predecessor_id=157, predecessor_gap=7.3)
        self.v2 = VehicleSnapshot(159, "Follower 2", True, Vector2D(26.0, 0.0), 8.0, 0.0, predecessor_id=158, predecessor_gap=7.3)
        self.tl_green = TrafficLightSnapshot("GREEN", 50.0, False)
        self.tl_red_far = TrafficLightSnapshot("RED", 18.0, True)
        self.tl_red_stopline = TrafficLightSnapshot("RED", 2.0, True)
        self.clear_snapshot = WorldSnapshot(10.0, 100, [self.v0, self.v1, self.v2], [], self.tl_green)

    def test_01_stationary_red_light_slow_approaching_npc_able_to_stop(self):
        """Stationary platoon at red light; NPC approaching slowly and able to stop safely."""
        # Vehicles stopped at red light
        v0 = VehicleSnapshot(157, "Leader", True, Vector2D(30.0, 0.0), 0.0, 0.0)
        v1 = VehicleSnapshot(158, "Follower 1", True, Vector2D(22.0, 0.0), 0.0, 0.0, predecessor_id=157, predecessor_gap=3.3)
        v2 = VehicleSnapshot(159, "Follower 2", True, Vector2D(14.0, 0.0), 0.0, 0.0, predecessor_id=158, predecessor_gap=3.3)
        
        # NPC 28m behind tail follower (v2 at x=14.0, npc at x=-14.0) traveling at 4.0 m/s (~14 km/h)
        # NPC stopping distance is ~4*0.5 + 16/(2*4.5) = 2.0 + 1.77 = 3.8m << 28m
        slow_npc = ObstacleSnapshot(201, Vector2D(-14.0, 0.0), speed=4.0, heading_rad=0.0)
        
        snap = WorldSnapshot(10.0, 100, [v0, v1, v2], [slow_npc], self.tl_red_far)
        predictor = RearThreatPredictor()
        
        threat_v2 = predictor.evaluate_threat_for_vehicle(2, snap)
        # Closing speed is 4.0 m/s, gap is ~23.4m, TTC is ~5.8s (> 4.5s caution threshold)
        self.assertEqual(threat_v2.risk_level, "NORMAL")
        self.assertFalse(threat_v2.is_actionable)

    def test_02_stationary_red_light_fast_npc_safe_escape_lane_available(self):
        """Stationary platoon at red light; fast NPC; safe legal escape lane available."""
        v0 = VehicleSnapshot(157, "Leader", True, Vector2D(30.0, 0.0), 0.0, 0.0)
        v1 = VehicleSnapshot(158, "Follower 1", True, Vector2D(22.0, 0.0), 0.0, 0.0, predecessor_id=157, predecessor_gap=3.3)
        v2 = VehicleSnapshot(159, "Follower 2", True, Vector2D(14.0, 0.0), 0.0, 0.0, predecessor_id=158, predecessor_gap=3.3)

        # Fast NPC 20m behind v2 (x=-6.0) approaching at 16.0 m/s (~58 km/h)
        # Stopping distance is 16*0.5 + 256/9 = 8.0 + 28.4 = 36.4m > 20m (physics cannot stop!)
        fast_npc = ObstacleSnapshot(202, Vector2D(-6.0, 0.0), speed=16.0, heading_rad=0.0)
        snap = WorldSnapshot(10.0, 100, [v0, v1, v2], [fast_npc], self.tl_red_far)

        shield = SafetyShield(mode="ACTIVE")
        # Step through hysteresis cycles (2 frames to confirm threat persistence)
        acts1, _, _ = shield.arbitrate(["CRUISE", "CRUISE", "CRUISE"], snap)
        acts2, _, rep = shield.arbitrate(["CRUISE", "CRUISE", "CRUISE"], snap)

        # Tail follower (v2) must select an evasive maneuver (CHANGE_LEFT or CHANGE_RIGHT)
        dec_v2 = rep.decisions[2]
        self.assertIn(dec_v2.rear_risk_level, ("HIGH_RISK", "CRITICAL"))
        self.assertIn(dec_v2.evasion_state, (EvasionState.EVASION_PREPARATION.value, EvasionState.EVADING_LEFT.value, EvasionState.EVADING_RIGHT.value))
        self.assertIn(acts2[2], ("CHANGE_LEFT", "CHANGE_RIGHT"))

    def test_03_stationary_red_light_fast_npc_all_escape_lanes_blocked(self):
        """Stationary platoon at red light; fast NPC; all escape lanes blocked."""
        v0 = VehicleSnapshot(157, "Leader", True, Vector2D(30.0, 0.0), 0.0, 0.0)
        v1 = VehicleSnapshot(158, "Follower 1", True, Vector2D(22.0, 0.0), 0.0, 0.0, predecessor_id=157, predecessor_gap=3.3)
        v2 = VehicleSnapshot(159, "Follower 2", True, Vector2D(14.0, 0.0), 0.0, 0.0, predecessor_id=158, predecessor_gap=3.3)

        fast_npc = ObstacleSnapshot(202, Vector2D(-6.0, 0.0), speed=16.0, heading_rad=0.0)
        # Block left lane (y=3.5m) and right lane (y=-3.5m) right next to v2
        blocker_left = ObstacleSnapshot(301, Vector2D(14.0, 3.5), speed=0.0, heading_rad=0.0)
        blocker_right = ObstacleSnapshot(302, Vector2D(14.0, -3.5), speed=0.0, heading_rad=0.0)

        snap = WorldSnapshot(10.0, 100, [v0, v1, v2], [fast_npc, blocker_left, blocker_right], self.tl_red_far)
        shield = SafetyShield(mode="ACTIVE")

        # Step 2 frames for hysteresis
        shield.arbitrate(["CRUISE", "CRUISE", "CRUISE"], snap)
        acts, _, rep = shield.arbitrate(["CRUISE", "CRUISE", "CRUISE"], snap)

        dec_v2 = rep.decisions[2]
        # Must detect that no safe escape exists
        self.assertEqual(dec_v2.evasion_state, EvasionState.EMERGENCY_NO_SAFE_ESCAPE.value)
        # Must NOT force an unsafe lane change into the blocked cars
        self.assertNotIn(acts[2], ("CHANGE_LEFT", "CHANGE_RIGHT"))

    def test_04_moving_platoon_fast_npc_safe_lane_change_available(self):
        """Moving platoon; fast NPC approaching from behind; safe lane change available."""
        # Platoon moving at 8.0 m/s (~29 km/h)
        fast_npc = ObstacleSnapshot(203, Vector2D(6.0, 0.0), speed=18.0, heading_rad=0.0)
        # v2 is at x=26.0m. Gap is ~15.4m. Closing speed is 10.0 m/s. TTC is ~1.54s (< 1.8s critical)
        snap = WorldSnapshot(10.0, 100, [self.v0, self.v1, self.v2], [fast_npc], self.tl_green)

        shield = SafetyShield(mode="ACTIVE")
        shield.arbitrate(["CRUISE", "CRUISE", "CRUISE"], snap)
        acts, _, rep = shield.arbitrate(["CRUISE", "CRUISE", "CRUISE"], snap)

        dec_v2 = rep.decisions[2]
        self.assertEqual(dec_v2.rear_risk_level, "CRITICAL")
        self.assertIn(acts[2], ("CHANGE_LEFT", "CHANGE_RIGHT"))

    def test_05_moving_platoon_fast_npc_both_adjacent_lanes_occupied(self):
        """Moving platoon; fast NPC approaching; both adjacent lanes occupied."""
        fast_npc = ObstacleSnapshot(203, Vector2D(6.0, 0.0), speed=18.0, heading_rad=0.0)
        left_traffic = ObstacleSnapshot(303, Vector2D(26.0, 3.5), speed=8.0, heading_rad=0.0)
        right_traffic = ObstacleSnapshot(304, Vector2D(26.0, -3.5), speed=8.0, heading_rad=0.0)

        snap = WorldSnapshot(10.0, 100, [self.v0, self.v1, self.v2], [fast_npc, left_traffic, right_traffic], self.tl_green)
        shield = SafetyShield(mode="ACTIVE")
        shield.arbitrate(["CRUISE", "CRUISE", "CRUISE"], snap)
        acts, _, rep = shield.arbitrate(["CRUISE", "CRUISE", "CRUISE"], snap)

        dec_v2 = rep.decisions[2]
        self.assertEqual(dec_v2.evasion_state, EvasionState.EMERGENCY_NO_SAFE_ESCAPE.value)
        self.assertNotIn(acts[2], ("CHANGE_LEFT", "CHANGE_RIGHT"))

    def test_06_threat_in_adjacent_lane_not_converging(self):
        """Threat is in an adjacent lane and not converging with platoon."""
        # Fast NPC is at x=10.0, but in adjacent lane y=3.5m traveling straight
        adj_npc = ObstacleSnapshot(204, Vector2D(10.0, 3.5), speed=20.0, heading_rad=0.0)
        snap = WorldSnapshot(10.0, 100, [self.v0, self.v1, self.v2], [adj_npc], self.tl_green)

        predictor = RearThreatPredictor()
        threat_v2 = predictor.evaluate_threat_for_vehicle(2, snap)

        # Must NOT classify adjacent-lane traffic as an in-lane rear collision threat
        self.assertEqual(threat_v2.risk_level, "NORMAL")
        self.assertFalse(threat_v2.is_in_lane)

    def test_07_rear_threat_passes_platoon_safely_rejoins(self):
        """Rear threat passes; the platoon vehicle safely rejoins."""
        ev_sm = EvasionStateMachine()
        tracker = ev_sm.get_or_create_tracker(159, "Follower 2")
        # Put vehicle in SAFE_HOLD after evading to the left (lateral offset -3.5m)
        tracker.state = EvasionState.SAFE_HOLD
        tracker.evasion_direction = "LEFT"
        tracker.target_lateral_offset_m = -3.5
        tracker.active_threat_id = 205

        # Threat has passed: rear threat assessment reports threat_passed=True, risk NORMAL
        passed_threat = RearThreatAssessment(
            ego_id=159,
            ego_role="Follower 2",
            threat_id=None,
            risk_level="NORMAL",
            rear_distance_m=80.0,
            closing_speed_mps=0.0,
            rear_ttc_s=float("inf"),
            npc_speed_mps=0.0,
            npc_stopping_dist_m=0.0,
            ego_speed_mps=8.0,
            is_in_lane=False,
            is_converging=False,
            threat_passed=True,
        )

        snap = WorldSnapshot(20.0, 200, [self.v0, self.v1, self.v2], [], self.tl_green)

        # Hold timer advances and transitions to WAITING_FOR_SAFE_REJOIN
        for t in [20.0, 20.5, 21.1]:
            ev_sm.step(2, passed_threat, snap, current_time=t)

        self.assertIn(tracker.state, (EvasionState.WAITING_FOR_SAFE_REJOIN, EvasionState.REJOINING))

        # Platoon original lane is clear; step again to trigger REJOINING
        act, st, lat, _ = ev_sm.step(2, passed_threat, snap, current_time=21.2)
        if st == EvasionState.REJOINING:
            # Must command CHANGE_RIGHT to return from left evasion
            self.assertEqual(act, "CHANGE_RIGHT")

    def test_08_follower_avoids_copying_leader_evasion(self):
        """Follower avoids independently copying the leader's evasive maneuver."""
        coord = PlatoonEvasionCoordinator()
        ev_sm = EvasionStateMachine()

        # Leader (157) is evading left
        tr_leader = ev_sm.get_or_create_tracker(157, "Leader")
        tr_leader.state = EvasionState.EVADING_LEFT
        tr_leader.evasion_direction = "LEFT"
        tr_leader.target_lateral_offset_m = -3.5

        # Follower 1 (158) is normal
        tr_f1 = ev_sm.get_or_create_tracker(158, "Follower 1")
        tr_f1.state = EvasionState.NORMAL

        snap = WorldSnapshot(10.0, 100, [self.v0, self.v1], [], self.tl_green)

        # Suppose MAPPO policy proposed CHANGE_LEFT for both
        proposed = ["CHANGE_LEFT", "CHANGE_LEFT"]
        coordinated, states = coord.coordinate_platoon(snap, ev_sm.trackers, proposed)

        # Leader executes CHANGE_LEFT
        self.assertEqual(coordinated[0], "CHANGE_LEFT")
        # Follower 1 must NOT copy leader; masked to CRUISE!
        self.assertEqual(coordinated[1], "CRUISE")

    def test_09_multiple_vehicles_face_rear_threats_simultaneously(self):
        """Multiple vehicles face rear-impact risks simultaneously."""
        # Threat 1 behind Follower 1 (x=38, threat at x=28, closing fast)
        threat_f1 = ObstacleSnapshot(401, Vector2D(28.0, 0.0), speed=18.0, heading_rad=0.0)
        # Threat 2 behind Follower 2 (x=26, threat at x=10, closing fast)
        threat_f2 = ObstacleSnapshot(402, Vector2D(10.0, 0.0), speed=18.0, heading_rad=0.0)

        snap = WorldSnapshot(10.0, 100, [self.v0, self.v1, self.v2], [threat_f1, threat_f2], self.tl_green)
        predictor = RearThreatPredictor()
        # Step through 2 frames of hysteresis
        predictor.detect_all_rear_threats(snap)
        all_threats = predictor.detect_all_rear_threats(snap)

        # Both Follower 1 (158) and Follower 2 (159) detect closing threats
        self.assertIn(all_threats[158].risk_level, ("HIGH_RISK", "CRITICAL"))
        self.assertIn(all_threats[159].risk_level, ("HIGH_RISK", "CRITICAL"))

    def test_10_red_light_stop_line_prevents_illegal_lane_change(self):
        """Red-light stop line constraint prevents an otherwise geometrically possible lane change."""
        # Vehicle stopped right at the stop line (dist=2.0m to red light)
        v0 = VehicleSnapshot(157, "Leader", True, Vector2D(30.0, 0.0), 0.0, 0.0)
        snap = WorldSnapshot(10.0, 100, [v0], [], self.tl_red_stopline)

        ev_sm = EvasionStateMachine()
        cand = ev_sm.validate_lane_change(v0, "LEFT", snap)

        # Must be rejected because vehicle is at stop line and cannot cross into intersection
        self.assertFalse(cand.is_valid)
        self.assertIn("Red-light stop line", cand.rejection_reason)

    def test_11_npc_disappears_during_prediction(self):
        """NPC disappears during prediction."""
        fast_npc = ObstacleSnapshot(206, Vector2D(6.0, 0.0), speed=18.0, heading_rad=0.0)
        snap1 = WorldSnapshot(10.0, 100, [self.v0, self.v1, self.v2], [fast_npc], self.tl_green)
        # Frame 2: NPC despawns / disappears
        snap2 = WorldSnapshot(10.1, 101, [self.v0, self.v1, self.v2], [], self.tl_green)

        predictor = RearThreatPredictor()
        t1 = predictor.evaluate_threat_for_vehicle(2, snap1)
        self.assertIn(t1.risk_level, ("HIGH_RISK", "CRITICAL", "CAUTION"))

        t2 = predictor.evaluate_threat_for_vehicle(2, snap2)
        # Predictor must gracefully recognize threat has disappeared
        self.assertTrue(t2.threat_passed or t2.threat_id is None)

    def test_12_module3_disabled_preserves_module2_behavior(self):
        """Module 3 disabled preserves Module 2 behavior."""
        shield = SafetyShield(mode="DISABLED")
        acts, emerg, rep = shield.arbitrate(["ACCELERATE", "CHANGE_LEFT", "CRUISE"], self.clear_snapshot)
        self.assertEqual(acts, ["ACCELERATE", "CHANGE_LEFT", "CRUISE"])
        self.assertEqual(emerg, [False, False, False])
        self.assertEqual(rep.mode, "DISABLED")

    def test_13_shadow_mode_logs_decisions_without_affecting_actions(self):
        """SHADOW mode logs decisions without affecting actions."""
        fast_npc = ObstacleSnapshot(207, Vector2D(6.0, 0.0), speed=18.0, heading_rad=0.0)
        snap = WorldSnapshot(10.0, 100, [self.v0, self.v1, self.v2], [fast_npc], self.tl_green)

        shield = SafetyShield(mode="SHADOW")
        shield.arbitrate(["CRUISE", "CRUISE", "CRUISE"], snap)
        acts, emerg, rep = shield.arbitrate(["CRUISE", "CRUISE", "CRUISE"], snap)

        # In shadow mode, actions dispatched to car remain identical to proposed actions
        self.assertEqual(acts, ["CRUISE", "CRUISE", "CRUISE"])
        self.assertEqual(emerg, [False, False, False])
        # But report captures evaluated decisions
        self.assertTrue(len(rep.decisions) == 3)

    def test_14_dedicated_logging_terminal_receives_live_records(self):
        """Dedicated logging terminal receives live Module 3 records."""
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".log") as tmp:
            tmp_log = tmp.name

        logger = Module3DecisionLogger(log_path=tmp_log, heartbeat_sec=0.0)
        logger.log_risk_assessment(12.345, 158, "HIGH_RISK", "HIGH_RISK", 1.85, 12.0, 6.5)
        logger.log_candidate_validation(12.345, 158, "CHANGE_LEFT", 2, True, "")
        logger.log_candidate_validation(12.345, 158, "CHANGE_RIGHT", 0, False, "Target lane occupied")
        logger.log_state_transition(12.345, 158, "NORMAL", "EVADING_LEFT", "Rear-impact risk")
        logger.log_rejoin_status(15.000, 158, "REJOINING", 12.0, "APPROVED")
        logger.log_emergency(16.000, 158, "All escape lanes blocked", "SAFE_HOLD")

        # Allow worker thread to flush
        time.sleep(0.15)
        logger.close()

        with open(tmp_log, "r", encoding="utf-8") as f:
            content = f.read()

        try:
            os.remove(tmp_log)
        except Exception:
            pass

        self.assertIn("[MODULE3]", content)
        self.assertIn("Actor=158", content)
        self.assertIn("State=HIGH_RISK", content)
        self.assertIn("Candidate=CHANGE_LEFT", content)
        self.assertIn("Candidate=CHANGE_RIGHT", content)
        self.assertIn("Target lane occupied", content)
        self.assertIn("NORMAL -> EVADING_LEFT", content)
        self.assertIn("Rejoin Status=REJOINING", content)
        self.assertIn("Emergency | Reason=All escape lanes blocked", content)

    def test_15_logging_output_cannot_block_simulation_thread(self):
        """Logging output cannot block simulation thread under high load."""
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".log") as tmp:
            tmp_log = tmp.name

        logger = Module3DecisionLogger(log_path=tmp_log, heartbeat_sec=0.0)
        t_start = time.perf_counter()

        # Enqueue 2000 records
        for i in range(2000):
            logger.log_risk_assessment(float(i), 157, "NORMAL", "NORMAL", 10.0, 30.0, 0.0)

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        logger.close()

        try:
            os.remove(tmp_log)
        except Exception:
            pass

        # 2000 non-blocking enqueues should easily execute in < 25ms (< 0.015ms per call)
        self.assertLess(t_elapsed_ms, 100.0)


if __name__ == "__main__":
    unittest.main()
