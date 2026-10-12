# WindowsNoEditor/Module2/tests/test_leader_failover_and_consoles.py
"""Deterministic Unit and Integration Tests for Leader Failover State Machine and Logging Consoles.

Covers the complete matrix of 16 requirements:
1. Normal stop at red light does not trigger promotion.
2. Normal stop at destination / stop sign does not trigger promotion.
3. Temporary obstacle ahead (pedestrian/vehicle crossing) does not trigger promotion.
4. Safety shield intervention (emergency brake) does not trigger promotion.
5. Destroyed / dead leader triggers immediate promotion.
6. Collision immobilized leader triggers promotion after collision timeout.
7. Unexplained prolonged immobilization transitions through state machine to promotion.
8. Resuming motion during suspect/recovery cancels failover and returns to ACTIVE.
9. Ineligible / dead follower candidate is skipped in favor of the next valid follower.
10. Promotion happens exactly once per failure event without duplication or oscillation.
11. Followers re-target the newly promoted leader (predecessor map rewired).
12. Promoted leader preserves route continuity from current position, not resetting to 0.
13. Failed leader is recorded as disabled and treated as obstacle in control and snapshot.
14. When no follower candidate is eligible, system holds safe state (NO_ELIGIBLE_LEADER).
15. Dedicated MAPPO and Module 3 log channels receive distinct events and startup banners.
16. Loggers flush and shut down cleanly without leaving orphaned resources.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import time
import unittest

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PYTHON_API = os.path.join(ROOT, "WindowsNoEditor", "PythonAPI")
for path in (ROOT, PYTHON_API, os.path.join(ROOT, "WindowsNoEditor")):
    if path not in sys.path:
        sys.path.insert(0, path)

import carla

from Module2.mission_controller import (
    MissionPlatoonController,
    LeaderHealthState,
    PlatoonState,
    _clamp,
)
from Module2.logger import MAPPODecisionLogger
from Module3.logger import Module3DecisionLogger
from Module3.snapshot import build_world_snapshot
from Module3.safety_shield import SafetyShield


class MockLocation(object):
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def distance(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)


class MockRotation(object):
    def __init__(self, yaw=0.0):
        self.yaw = float(yaw)


class MockTransform(object):
    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        self.location = MockLocation(x, y)
        self.rotation = MockRotation(yaw)

    def get_forward_vector(self):
        rad = math.radians(self.rotation.yaw)
        return MockLocation(math.cos(rad), math.sin(rad))


class MockBoundingBox(object):
    def __init__(self, extent_x=2.2, extent_y=1.0, extent_z=0.8):
        self.extent = MockLocation(extent_x, extent_y, extent_z)
        self.location = MockLocation(0.0, 0.0, 0.0)


class MockWaypoint(object):
    def __init__(self, x, y, yaw=0.0, is_junction=False):
        self.transform = MockTransform(x, y, yaw)
        self.is_junction = is_junction
        self.road_id = 1
        self.lane_id = 1
        self._next = []

    def next(self, spacing):
        return list(self._next)


class MockMap(object):
    def __init__(self, waypoints):
        self.waypoints = list(waypoints)

    def get_waypoint(self, location, **kwargs):
        if not self.waypoints:
            return None
        return min(self.waypoints, key=lambda wp: wp.transform.location.distance(location))


class MockTrafficLight(object):
    def __init__(self, state=carla.TrafficLightState.Red, x=70.0, y=0.0):
        self.state = state
        self.transform = MockTransform(x, y)

    def get_state(self):
        return self.state

    def get_location(self):
        return self.transform.location

    def get_transform(self):
        return self.transform


class MockActor(object):
    def __init__(self, actor_id, x=0.0, y=0.0, yaw=0.0, speed_mps=0.0):
        self.id = int(actor_id)
        self.transform = MockTransform(x, y, yaw)
        self.is_alive = True
        self.speed_mps = float(speed_mps)
        self.last_control = None
        self.traffic_light = None
        self.bounding_box = MockBoundingBox()

    def get_location(self):
        return self.transform.location

    def get_transform(self):
        return self.transform

    def get_velocity(self):
        rad = math.radians(self.transform.rotation.yaw)
        return MockLocation(self.speed_mps * math.cos(rad), self.speed_mps * math.sin(rad))

    def apply_control(self, ctrl):
        self.last_control = ctrl

    def get_control(self):
        if self.last_control is None:
            return carla.VehicleControl()
        return self.last_control

    def get_traffic_light(self):
        return self.traffic_light

    def get_traffic_light_state(self):
        return self.traffic_light.get_state() if self.traffic_light else None

    def is_at_traffic_light(self):
        return self.traffic_light is not None


class MockActorList(object):
    def __init__(self, actors):
        self._actors = list(actors)

    def filter(self, pattern):
        return [a for a in self._actors if a.is_alive]


class MockFrameSnapshot(object):
    def __init__(self, frame=100):
        self.frame = int(frame)


class MockWorld(object):
    def __init__(self, map_obj, actors):
        self._map = map_obj
        self._actors = list(actors)

    def get_map(self):
        return self._map

    def get_actors(self):
        return MockActorList(self._actors)

    def get_snapshot(self):
        return MockFrameSnapshot()


def make_test_route(start_x=0.0, end_x=200.0, step=2.0):
    wps = []
    x = start_x
    while x <= end_x:
        wps.append(MockWaypoint(x=x, y=0.0, yaw=0.0))
        x += step
    for i in range(len(wps) - 1):
        wps[i]._next = [wps[i + 1]]
    return wps


class TestLeaderFailoverStateMachine(unittest.TestCase):
    def setUp(self):
        self.wps = make_test_route(0.0, 200.0, 2.0)
        self.mock_map = MockMap(self.wps)

        # 3 Platoon vehicles: Lead at x=60, F1 at x=40, F2 at x=20
        self.lead = MockActor(101, x=60.0, y=0.0, speed_mps=8.0)
        self.f1 = MockActor(102, x=40.0, y=0.0, speed_mps=8.0)
        self.f2 = MockActor(103, x=20.0, y=0.0, speed_mps=8.0)

        self.mock_world = MockWorld(self.mock_map, [self.lead, self.f1, self.f2])
        self.controller = MissionPlatoonController(
            self.mock_world, [self.lead, self.f1, self.f2], cruise_speed_kmh=28.0
        )
        self.controller.route.route = self.wps
        self.controller.route.destination = self.wps[-1].transform.location

    def test_01_red_light_normal_stop_no_promotion(self):
        """Leader stopped at red traffic light preserves leadership and does not promote."""
        self.lead.speed_mps = 0.0
        self.lead.traffic_light = MockTrafficLight(carla.TrafficLightState.Red, x=70.0, y=0.0)

        # Simulate 15 seconds at red light (well beyond 8s timeout)
        for t in range(1, 16):
            self.controller._check_leader_status(float(t))

        self.assertEqual(self.controller.active_leader_index, 0)
        self.assertEqual(self.controller.leader_health_state, LeaderHealthState.ACTIVE)
        self.assertEqual(self.controller.promotion_events_count, 0)

    def test_02_destination_normal_stop_no_promotion(self):
        """Leader stopping at destination does not trigger leader promotion."""
        self.lead.speed_mps = 0.0
        # Place near destination
        dest = self.controller.route.destination
        self.lead.transform.location = MockLocation(dest.x - 2.0, dest.y)
        self.controller.route.cursors[self.lead.id] = len(self.wps) - 2

        for t in range(1, 12):
            self.controller._check_leader_status(float(t))

        self.assertEqual(self.controller.active_leader_index, 0)
        self.assertEqual(self.controller.leader_health_state, LeaderHealthState.ACTIVE)
        self.assertEqual(self.controller.promotion_events_count, 0)

    def test_03_temporary_obstacle_crossing_no_promotion(self):
        """Leader waiting for a crossing pedestrian/vehicle ahead does not trigger promotion."""
        self.lead.speed_mps = 0.0
        self.controller.route.cursors[self.lead.id] = 30  # align route cursor with lead at x=60
        ped = MockActor(999, x=65.0, y=0.0, speed_mps=1.0)
        self.mock_world._actors.append(ped)

        for t in range(1, 12):
            self.controller._check_leader_status(float(t))

        self.assertEqual(self.controller.active_leader_index, 0)
        self.assertEqual(self.controller.leader_health_state, LeaderHealthState.ACTIVE)
        self.assertEqual(self.controller.promotion_events_count, 0)

    def test_04_safety_shield_intervention_no_promotion(self):
        """Safety shield emergency brake intervention does not trigger promotion."""
        self.lead.speed_mps = 0.0
        self.controller.emergency_brakes = [True, False, False]

        for t in range(1, 10):
            self.controller._check_leader_status(float(t))

        self.assertEqual(self.controller.active_leader_index, 0)
        self.assertEqual(self.controller.promotion_events_count, 0)

    def test_05_destroyed_leader_immediate_promotion(self):
        """Destroyed leader actor triggers immediate promotion to next valid follower."""
        self.lead.is_alive = False

        self.controller._check_leader_status(current_time=1.0)

        self.assertEqual(self.controller.active_leader_index, 1)
        self.assertEqual(self.controller.leader_health_state, LeaderHealthState.NEW_LEADER_ACTIVE)
        self.assertIn(101, self.controller.disabled_vehicles)
        self.assertEqual(self.controller.promotion_events_count, 1)
        self.assertEqual(self.controller.get_active_leader().id, 102)

    def test_06_collision_immobilized_leader_promotion(self):
        """Leader involved in collision and remaining stationary triggers promotion."""
        self.lead.speed_mps = 0.0
        self.controller.record_collision(self.lead.id, intensity=100.0, current_time=2.0)

        # First standstill detection at t=4.0
        self.controller._check_leader_status(current_time=4.0)
        self.assertEqual(self.controller.active_leader_index, 0)

        # At t=6.0 (duration 2.0s < 5.0s col_timeout):
        self.controller._check_leader_status(current_time=6.0)
        self.assertEqual(self.controller.active_leader_index, 0)

        # After collision timeout (t=9.5 -> duration 5.5s >= 5.0s col_timeout):
        self.controller._check_leader_status(current_time=9.5)
        self.assertEqual(self.controller.active_leader_index, 1)
        self.assertEqual(self.controller.leader_health_state, LeaderHealthState.NEW_LEADER_ACTIVE)
        self.assertIn(101, self.controller.disabled_vehicles)

    def test_07_unexplained_prolonged_immobilization_state_transitions(self):
        """Prolonged unexplainable standstill transitions through explicit states to promotion."""
        self.lead.speed_mps = 0.0

        # At t=1.0s: ACTIVE
        self.controller._check_leader_status(1.0)
        self.assertEqual(self.controller.leader_health_state, LeaderHealthState.ACTIVE)

        # At t=4.5s (> suspect timeout 3.0s): SUSPECTED_FAILURE -> RECOVERY_ATTEMPT
        self.controller._check_leader_status(4.5)
        self.assertIn(
            self.controller.leader_health_state,
            (LeaderHealthState.SUSPECTED_FAILURE, LeaderHealthState.RECOVERY_ATTEMPT),
        )

        # At t=9.5s (> failure timeout 8.0s): FAILED -> PROMOTION_PENDING -> NEW_LEADER_ACTIVE
        self.controller._check_leader_status(9.5)
        self.assertEqual(self.controller.active_leader_index, 1)
        self.assertEqual(self.controller.leader_health_state, LeaderHealthState.NEW_LEADER_ACTIVE)

    def test_08_recovery_cancellation(self):
        """Leader moving again during recovery cancels promotion and returns to ACTIVE."""
        self.lead.speed_mps = 0.0
        self.controller._check_leader_status(1.0)
        self.controller._check_leader_status(4.5)  # enters RECOVERY_ATTEMPT

        # Leader manages to move
        self.lead.speed_mps = 1.5
        self.controller._check_leader_status(5.0)

        self.assertEqual(self.controller.leader_health_state, LeaderHealthState.ACTIVE)
        self.assertEqual(self.controller.active_leader_index, 0)
        self.assertEqual(self.controller.promotion_events_count, 0)

    def test_09_ineligible_candidate_skipped(self):
        """If Follower 1 is also destroyed, failover skips to Follower 2."""
        self.lead.is_alive = False
        self.f1.is_alive = False  # Candidate 1 is dead

        self.controller._check_leader_status(1.0)

        # Skips index 1 (102) and promotes index 2 (103)
        self.assertEqual(self.controller.active_leader_index, 2)
        self.assertEqual(self.controller.get_active_leader().id, 103)
        self.assertEqual(self.controller.leader_health_state, LeaderHealthState.NEW_LEADER_ACTIVE)

    def test_10_promotion_happens_exactly_once(self):
        """Repeated status checks after promotion do not duplicate failover or cycle leaders."""
        self.lead.is_alive = False
        self.controller._check_leader_status(1.0)
        self.assertEqual(self.controller.promotion_events_count, 1)

        # Additional ticks
        for t in range(2, 10):
            self.controller._check_leader_status(float(t))

        self.assertEqual(self.controller.promotion_events_count, 1)
        self.assertEqual(self.controller.active_leader_index, 1)

    def test_11_followers_retarget_new_leader(self):
        """Followers re-target the newly promoted leader via predecessor map."""
        self.lead.is_alive = False
        self.controller._check_leader_status(1.0)

        # Promoted leader (index 1) has predecessor None
        self.assertIsNone(self.controller.predecessor_map[1])
        # Follower 2 (index 2) now tracks index 1 (new leader)
        self.assertEqual(self.controller.predecessor_map[2], 1)

    def test_12_route_continuity_preserved(self):
        """Promoted leader preserves its current route progress rather than restarting at wp 0."""
        # Set Follower 1 cursor at waypoint 20
        self.controller.route.cursors[self.f1.id] = 20
        self.lead.is_alive = False

        self.controller._check_leader_status(1.0)

        new_cursor = self.controller.route.cursors.get(self.f1.id)
        self.assertGreaterEqual(new_cursor, 20)

    def test_13_failed_leader_treated_as_obstacle(self):
        """Disabled leader remaining on road is included in disabled_vehicles and snapshot obstacles."""
        self.lead.speed_mps = 0.0
        self.controller._set_leader_health_state(LeaderHealthState.FAILED, 1.0, "IMMOBILIZED")
        self.controller._handle_leader_promotion(1.0, "IMMOBILIZED")

        self.assertIn(self.lead.id, self.controller.disabled_vehicles)

        snap = build_world_snapshot(
            world=self.mock_world,
            platoon=[self.lead, self.f1, self.f2],
            current_time=1.0,
            controller=self.controller,
        )

        # Disabled leader should appear as an obstacle
        obs_ids = [obs.id for obs in snap.obstacles]
        self.assertIn(self.lead.id, obs_ids)

    def test_14_no_eligible_candidate_safe_state(self):
        """If all followers are dead/destroyed, system enters safe NO_ELIGIBLE_LEADER state."""
        self.lead.is_alive = False
        self.f1.is_alive = False
        self.f2.is_alive = False

        self.controller._check_leader_status(1.0)

        self.assertEqual(self.controller.leader_health_state, LeaderHealthState.NO_ELIGIBLE_LEADER)
        self.assertIsNone(self.controller.get_active_leader())


class TestLoggingChannelsAndBanners(unittest.TestCase):
    def test_15_separate_log_channels_and_startup_messages(self):
        """MAPPO and Module 3 write to distinct files with clear startup messages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mappo_log = os.path.join(tmpdir, "mappo_test.log")
            m3_log = os.path.join(tmpdir, "m3_test.log")

            mappo_logger = MAPPODecisionLogger(log_path=mappo_log)
            m3_logger = Module3DecisionLogger(log_path=m3_log)

            mappo_logger.log_startup(pid=12345, mode="EVALUATE", leader_id=101)
            mappo_logger.log_initialization(
                ckpt_path="checkpoint.pt",
                ckpt_hash="abc123",
                obs_dim=32,
                action_dim=5,
                update_step=42,
                is_deterministic=True,
                actor_loaded=True,
                critic_loaded=True,
            )

            m3_logger.log_startup(pid=12345, mode="SHADOW", leader_id=101)
            m3_logger.log_initialization(
                mode="SHADOW",
                enabled=True,
                horizon_sec=2.5,
                dt_sec=0.1,
                budget_ms=15.0,
            )

            # Give worker threads time to flush
            time.sleep(0.3)
            mappo_logger.close()
            m3_logger.close()

            with open(mappo_log, "r", encoding="utf-8") as f:
                mappo_content = f.read()

            with open(m3_log, "r", encoding="utf-8") as f:
                m3_content = f.read()

            self.assertIn("[MAPPO CONSOLE] Dedicated MAPPO logging window connected.", mappo_content)
            self.assertIn("[MAPPO STARTUP SUMMARY]", mappo_content)
            self.assertIn("Authoritative Leader ID: 101", mappo_content)

            self.assertIn("[MODULE 3 CONSOLE] Dedicated Module 3 logging window connected.", m3_content)
            self.assertIn("[MODULE 3 STARTUP SUMMARY]", m3_content)
            self.assertIn("Operating Mode:          SHADOW", m3_content)

    def test_16_clean_shutdown_and_non_blocking_behavior(self):
        """Loggers enqueue and shutdown cleanly without raising errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_log = os.path.join(tmpdir, "stress.log")
            logger = MAPPODecisionLogger(log_path=test_log)

            for i in range(50):
                logger.log_tactical_cycle(
                    sim_time_sec=float(i),
                    step=i,
                    mode="EVALUATE",
                    vehicle_records=[{
                        "id": 1,
                        "role": "Leader",
                        "proposed_action": "CRUISE",
                        "shield_action": "CRUISE",
                        "shield_mode": "SHADOW",
                        "executed_action": "CRUISE",
                        "reason": "NOMINAL",
                        "speed_mps": 8.0,
                        "target_mps": 8.0,
                        "gap_m": 0.0,
                        "desired_gap_m": 0.0,
                        "cursor": i,
                        "total_wps": 100,
                    }],
                )

            time.sleep(0.2)
            logger.close()
            self.assertTrue(os.path.exists(test_log))


if __name__ == "__main__":
    unittest.main()
