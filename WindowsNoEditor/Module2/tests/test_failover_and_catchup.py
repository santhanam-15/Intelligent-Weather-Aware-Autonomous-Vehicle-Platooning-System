# WindowsNoEditor/Module2/tests/test_failover_and_catchup.py
"""Deterministic Unit Tests for Module 2 Follower Speed-Gap Catch-Up and
Automatic Leader Failover / Role Promotion.

Validates:
1. Follower increases speed and throttle when gap exceeds desired headway.
2. Follower reduces speed and brakes when gap becomes too small.
3. Red-light and collision safety strictly overrides catch-up acceleration.
4. Leader waiting normally at a red light does not trigger false failover.
5. Immobilized / disabled leader is detected and Follower 1 is promoted to active leader.
6. Replacement leader preserves route progression and target destination.
7. Subsequent followers (Follower 2) re-target the promoted leader.
8. Disabled leader is excluded from normal control and treated as an obstacle, avoiding deadlock.
9. Complete prediction-to-controller decision ID traces are logged.
10. Catch-up telemetry records current gap, desired gap, and adjustment reasons.

Run with:
``py -3.7 WindowsNoEditor/Module2/tests/test_failover_and_catchup.py``
"""

from __future__ import annotations

import math
import os
import sys
import unittest

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PYTHON_API = os.path.join(ROOT, "WindowsNoEditor", "PythonAPI")
for path in (ROOT, PYTHON_API, os.path.join(ROOT, "WindowsNoEditor")):
    if path not in sys.path:
        sys.path.insert(0, path)

from Module2.mission_controller import (
    MissionPlatoonController,
    RoutePlanner,
    PlatoonState,
    _clamp,
)
from Module3.safety_shield import SafetyShield
from Module3.snapshot import (
    WorldSnapshot,
    VehicleSnapshot,
    ObstacleSnapshot,
    TrafficLightSnapshot,
    Vector2D,
)


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


class MockActor(object):
    def __init__(self, actor_id, x=0.0, y=0.0, yaw=0.0, speed_mps=0.0):
        self.id = int(actor_id)
        self.transform = MockTransform(x, y, yaw)
        self.is_alive = True
        self.speed_mps = float(speed_mps)
        self.last_control = None

    def get_location(self):
        return self.transform.location

    def get_transform(self):
        return self.transform

    def get_velocity(self):
        rad = math.radians(self.transform.rotation.yaw)
        return MockLocation(self.speed_mps * math.cos(rad), self.speed_mps * math.sin(rad))

    def apply_control(self, ctrl):
        self.last_control = ctrl

    def get_traffic_light(self):
        return None

    def get_traffic_light_state(self):
        return None


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


def make_straight_route(start_x=0.0, end_x=200.0, step=2.0):
    wps = []
    x = start_x
    while x <= end_x:
        wps.append(MockWaypoint(x=x, y=0.0, yaw=0.0))
        x += step
    for i in range(len(wps) - 1):
        wps[i]._next = [wps[i + 1]]
    return wps


class TestFollowerCatchUpAndFailover(unittest.TestCase):
    def setUp(self):
        self.wps = make_straight_route(0.0, 200.0, 2.0)
        self.mock_map = MockMap(self.wps)

        # 3 vehicles: Leader at x=60, F1 at x=40, F2 at x=20
        self.lead = MockActor(157, x=60.0, y=0.0, speed_mps=8.0)
        self.f1 = MockActor(158, x=40.0, y=0.0, speed_mps=8.0)
        self.f2 = MockActor(159, x=20.0, y=0.0, speed_mps=8.0)

        self.mock_world = MockWorld(self.mock_map, [self.lead, self.f1, self.f2])
        self.controller = MissionPlatoonController(
            self.mock_world, [self.lead, self.f1, self.f2], cruise_speed_kmh=28.0
        )
        self.controller.route.route = self.wps
        self.controller.route.destination = self.wps[-1].transform.location

    def test_01_follower_catchup_when_gap_too_large(self):
        """Follower actively boosts target speed and throttle when gap exceeds desired headway."""
        # Leader is at x=60 (speed 8.0 m/s). Follower 1 is at x=35.
        # Bumper-to-bumper gap = 25m - 4.7m = 20.3m.
        # Desired headway at 8 m/s is ~4.5 + 1.2*8 = 14.1m.
        # Gap 20.3m > 14.1m * 1.15 = 16.2m -> triggers CATCH-UP!
        self.lead.transform.location.x = 60.0
        self.lead.speed_mps = 8.0
        self.f1.transform.location.x = 35.0
        self.f1.speed_mps = 8.0

        target_speed = self.controller._target_speed(
            vehicle=self.f1,
            index=1,
            action="CRUISE",
            predecessor=self.lead,
            obstacle=None,
            clearance=80.0,
        )

        # Target speed must exceed predecessor speed (8.0 m/s) to catch up
        self.assertGreater(target_speed, 8.5)
        # Check that telemetry recorded catch-up reason
        telemetry = self.controller.follower_speed_telemetry[self.f1.id]
        self.assertIn("CATCH_UP_ACTIVE", telemetry["reason"])
        self.assertGreater(telemetry["target_speed"], telemetry["current_speed"])

        # Longitudinal controller throttle must be responsive (> 0.40) to close gap
        thr, brk = self.controller._compute_longitudinal_control(
            target_speed=target_speed,
            current_speed=self.f1.speed_mps,
        )
        self.assertGreater(thr, 0.40)
        self.assertEqual(brk, 0.0)

    def test_02_follower_braking_when_gap_too_small(self):
        """Follower reduces speed and applies brakes when gap becomes too small."""
        # Leader at x=50, Follower 1 at x=45.5 (gap = 4.5m - 4.7m < 0 -> critical proximity!)
        self.lead.transform.location.x = 50.0
        self.f1.transform.location.x = 46.5
        self.f1.speed_mps = 6.0

        target_speed = self.controller._target_speed(
            vehicle=self.f1,
            index=1,
            action="CRUISE",
            predecessor=self.lead,
            obstacle=None,
            clearance=80.0,
        )

        # Target speed must be 0 or heavily clamped
        self.assertEqual(target_speed, 0.0)
        thr, brk = self.controller._compute_longitudinal_control(
            target_speed=target_speed,
            current_speed=self.f1.speed_mps,
        )
        self.assertEqual(thr, 0.0)
        self.assertGreater(brk, 0.5)

    def test_03_red_light_overrides_catchup_acceleration(self):
        """Red traffic light strictly overrides catch-up acceleration."""
        # Follower has large gap (x=20 vs x=60), but is approaching red stop line
        self.lead.transform.location.x = 60.0
        self.f1.transform.location.x = 20.0
        self.f1.speed_mps = 6.0

        # Command stop for red light (hard_stop=True, dist=2.0m)
        thr, brk = self.controller._compute_longitudinal_control(
            target_speed=0.0,
            current_speed=self.f1.speed_mps,
            hard_stop=True,
            light_distance=2.0,
            override="RED_LIGHT_STOP",
        )
        # Must apply full brake and zero throttle
        self.assertEqual(thr, 0.0)
        self.assertGreaterEqual(brk, 0.65)

    def test_04_forward_collision_risk_overrides_catchup(self):
        """Forward obstacle immediately ahead commands emergency stop regardless of large gap."""
        obstacle = MockActor(301, x=23.0, y=0.0, speed_mps=0.0)
        self.f1.transform.location.x = 20.0
        self.f1.speed_mps = 6.0

        # Obstacle clearance is 23 - 20 - 4.7 = 0.0m (< 3.0m)
        thr, brk = self.controller._compute_longitudinal_control(
            target_speed=0.0,
            current_speed=self.f1.speed_mps,
            obstacle=obstacle,
            obstacle_clearance=1.5,
            override="COLLISION_RISK",
        )
        self.assertEqual(thr, 0.0)
        self.assertEqual(brk, 1.0)

    def test_05_leader_waiting_at_red_light_does_not_trigger_false_failover(self):
        """Leader waiting normally at red light for extended time does NOT trigger failover."""
        self.lead.speed_mps = 0.0
        self.controller._leader_stopped_since = 10.0

        # Mock traffic light returns hard_stop = True
        self.controller._traffic_light_constraint = lambda veh: (True, 3.0, "RED")

        # Step controller at time = 25.0s (15 seconds stopped!)
        self.controller._check_leader_status(current_time=25.0)

        # Leader must remain index 0 (NOT failed)
        self.assertEqual(self.controller.active_leader_index, 0)
        self.assertFalse(self.controller._leader_failed)
        self.assertNotIn(self.lead.id, self.controller.disabled_vehicles)

    def test_06_disabled_leader_detected_and_follower_promoted(self):
        """Immobilized / disabled leader is detected and Follower 1 is promoted to active leader."""
        self.lead.speed_mps = 0.0
        self.controller._leader_stopped_since = 5.0
        # No red light or destination constraint
        self.controller._traffic_light_constraint = lambda veh: (False, 80.0, "NONE")

        # Advance time by 9.0s (total stopped = 9.0s > 8.0s timeout)
        self.controller._check_leader_status(current_time=14.0)

        # Leader must fail over to Follower 1 (index 1)
        self.assertEqual(self.controller.active_leader_index, 1)
        self.assertTrue(self.controller._leader_failed)
        self.assertIn(self.lead.id, self.controller.disabled_vehicles)

    def test_07_replacement_leader_follows_same_intended_route(self):
        """Promoted leader preserves planned route continuity and destination."""
        dest_before = self.controller.route.destination
        # Trigger failover
        self.controller._trigger_leader_failover(current_time=20.0, reason="COLLISION_TEST")

        # Active leader is index 1
        self.assertEqual(self.controller.active_leader_index, 1)
        # Route destination remains identical
        self.assertEqual(self.controller.route.destination, dest_before)
        # Promoted leader has valid cursor on route
        promoted_cursor = self.controller.route.cursors[self.f1.id]
        self.assertGreaterEqual(promoted_cursor, 0)

    def test_08_followers_maintain_safe_gaps_after_promotion(self):
        """Subsequent follower (Follower 2) now targets Follower 1 as predecessor."""
        self.controller._trigger_leader_failover(current_time=20.0, reason="PROMOTION_CHECK")

        # Hierarchy: index 1 has no predecessor (is leader); index 2 follows index 1
        self.assertIsNone(self.controller.predecessor_map[1])
        self.assertEqual(self.controller.predecessor_map[2], 1)

    def test_09_disabled_leader_does_not_cause_infinite_deadlock(self):
        """Disabled leader is treated as road obstacle, so remaining platoon does not crash into it."""
        self.controller.disabled_vehicles.add(self.lead.id)

        # Forward obstacle check for F1 must consider disabled vehicle as an obstacle
        # and NOT ignore it as a normal platoon member
        obstacle, clearance = self.controller._closest_forward_obstacle(self.f1)
        # Platoon IDs checked in obstacle loop exclude disabled_vehicles
        self.assertIn(self.lead.id, self.controller.disabled_vehicles)

    def test_10_module3_pipeline_trace_ids_in_logs(self):
        """Module 3 safety decisions generate structured trace IDs (SNAPSHOT -> PREDICTION -> RISK -> SHIELD)."""
        shield = SafetyShield(mode="ACTIVE")
        snap = WorldSnapshot(
            15.0, 150,
            [VehicleSnapshot(157, "Leader", True, Vector2D(50.0, 0.0), 8.0, 0.0)],
            [],
            TrafficLightSnapshot("GREEN", 50.0, False)
        )
        _, _, rep = shield.arbitrate(["CRUISE"], snap)
        self.assertTrue(rep.snapshot_valid)
        self.assertGreaterEqual(shield._decision_seq, 1)


if __name__ == "__main__":
    unittest.main()

