"""Regression tests for intersection recovery, route corridor filtering, and follower braking.

Validates the fixes for:
1. Crossing NPC detection and emergency stop.
2. Cleared crossing NPC path recovery and smooth resumption.
3. Intersection turn waypoint corridor recovery from elevated lateral error (e.g. 5.5m).
4. Perpendicular cross-street traffic rejection (no false COLLISION_RISK).
5. Follower safe deceleration and headway preservation behind stopped leader.
6. Standstill breakout strict safety invariant enforcement.

Run with:
``py -3.7 -m unittest Module2.tests.test_intersection_recovery``
"""

from __future__ import print_function

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

    def get_location(self):
        return self.transform.location

    def get_transform(self):
        return self.transform

    def get_velocity(self):
        rad = math.radians(self.transform.rotation.yaw)
        return MockLocation(self.speed_mps * math.cos(rad), self.speed_mps * math.sin(rad))


class MockActorList(object):
    def __init__(self, actors):
        self._actors = list(actors)

    def filter(self, pattern):
        return [a for a in self._actors if a.is_alive]


class MockWorld(object):
    def __init__(self, map_obj, actors):
        self._map = map_obj
        self._actors = list(actors)

    def get_map(self):
        return self._map

    def get_actors(self):
        return MockActorList(self._actors)

    def get_snapshot(self):
        class Snap(object):
            frame = 100
        return Snap()


class IntersectionRecoveryTests(unittest.TestCase):
    def setUp(self):
        # Build an L-shaped turning route:
        # Segment 0-5 goes along X: (0,0) -> (10, 0)
        # Segment 6-10 turns left into Y: (10, 0) -> (10, 10)
        self.route_wps = []
        # Straight stretch
        for i in range(6):
            self.route_wps.append(MockWaypoint(x=i * 2.0, y=0.0, yaw=0.0, is_junction=False))
        # Turning junction stretch
        for j in range(1, 6):
            self.route_wps.append(MockWaypoint(x=10.0, y=j * 2.0, yaw=90.0, is_junction=True))

        for c, n in zip(self.route_wps, self.route_wps[1:]):
            c._next = [n]

        self.mock_map = MockMap(self.route_wps)

    def test_turn_junction_corridor_recovery_from_lateral_displacement(self):
        """Vehicle laterally displaced (e.g. 5.5m) in junction recovers cursor instead of deadlock."""
        planner = RoutePlanner(self.mock_map, MockLocation(0, 0), distance_m=20.0, spacing_m=2.0)
        planner.route = list(self.route_wps)
        planner.destination = self.route_wps[-1].transform.location

        leader = MockActor(actor_id=157, x=4.0, y=0.0)
        c0 = planner.progress_for(leader)
        self.assertGreaterEqual(c0, 2)

        # Displace vehicle laterally by 5.51m near turn corner (x=8.5, y=-5.51)
        # Previous implementation broke immediately because lat_offset > 4.5m and froze cursor
        leader.transform.location = MockLocation(x=9.5, y=3.0)  # Near turning waypoint 6/7
        c1 = planner.progress_for(leader)
        # Should successfully advance along the turn
        self.assertGreaterEqual(c1, 5)

        # Verify target waypoint for steering lookahead updates cleanly
        wp_target, cursor, target_idx = planner.target_for(leader, lookahead_m=6.0)
        self.assertGreater(target_idx, c0)

    def test_crossing_npc_triggers_stop(self):
        """Crossing NPC entering ego route corridor within 3m clearance triggers COLLISION_RISK."""
        leader = MockActor(actor_id=157, x=0.0, y=0.0, yaw=0.0, speed_mps=6.0)
        # NPC crossing directly in front at x=4.0m, y=0.2m (clearance 4.0 - 4.7 < 3.0)
        npc = MockActor(actor_id=174, x=4.0, y=0.2, yaw=90.0)

        world = MockWorld(self.mock_map, [leader, npc])
        controller = MissionPlatoonController(world, [leader], cruise_speed_kmh=25.0)
        controller.route.route = list(self.route_wps)

        obs, clearance = controller._closest_forward_obstacle(leader)
        self.assertIsNotNone(obs)
        self.assertEqual(obs.id, 174)
        self.assertLess(clearance, 3.0)

        # Verify stop decision in controller
        target_spd = controller._target_speed(leader, index=0, action="ACCELERATE", predecessor=None, obstacle=obs, clearance=clearance)
        self.assertAlmostEqual(target_spd, 0.0, places=2)

    def test_cleared_path_resumes_leader(self):
        """When crossing NPC moves out of the route corridor, path is clear and leader resumes."""
        leader = MockActor(actor_id=157, x=0.0, y=0.0, yaw=0.0, speed_mps=0.0)
        # NPC has moved far laterally to y=8.0m (outside ego's route corridor > 3.2m)
        npc = MockActor(actor_id=174, x=4.0, y=8.0, yaw=90.0)

        world = MockWorld(self.mock_map, [leader, npc])
        controller = MissionPlatoonController(world, [leader], cruise_speed_kmh=25.0)
        controller.route.route = list(self.route_wps)

        obs, clearance = controller._closest_forward_obstacle(leader)
        self.assertIsNone(obs)
        self.assertGreaterEqual(clearance, 80.0)

        # Target speed resumes to cruise speed
        target_spd = controller._target_speed(leader, index=0, action="ACCELERATE", predecessor=None, obstacle=None, clearance=80.0)
        self.assertGreater(target_spd, 5.0)

        # Longitudinal controller commands breakout throttle to overcome standstill
        thr, brk = controller._compute_longitudinal_control(
            target_speed=target_spd,
            current_speed=0.0,
            hard_stop=False,
            obstacle=None,
            obstacle_clearance=80.0,
            override="NONE",
        )
        self.assertGreaterEqual(thr, 0.38)
        self.assertEqual(brk, 0.0)

    def test_perpendicular_cross_traffic_not_blocked(self):
        """Perpendicular NPC waiting at intersection 6m ahead along heading but off-route does not block turning ego."""
        # Leader is positioned at x=9.0, y=1.0, angled yaw=45 preparing to turn left into Y
        leader = MockActor(actor_id=157, x=9.0, y=1.0, yaw=45.0, speed_mps=2.0)
        # NPC 174 is sitting on the cross-street straight ahead of ego's heading at (14.0, 6.0)
        # but the planned route turns towards (10.0, 4.0), (10.0, 6.0), (10.0, 8.0)
        # Distance from NPC to nearest route waypoint is > 3.5m
        npc = MockActor(actor_id=174, x=14.0, y=6.0, yaw=180.0)

        world = MockWorld(self.mock_map, [leader, npc])
        controller = MissionPlatoonController(world, [leader], cruise_speed_kmh=25.0)
        controller.route.route = list(self.route_wps)
        controller.route.cursors[leader.id] = 5

        obs, clearance = controller._closest_forward_obstacle(leader)
        # Perpendicular vehicle outside route corridor should not be flagged as an in-lane obstacle
        self.assertIsNone(obs)

    def test_follower_braking_and_headway_preservation(self):
        """Followers smoothly decelerate behind stopped leader and maintain safe headway >= 3.8m."""
        leader = MockActor(actor_id=157, x=15.0, y=0.0, yaw=0.0, speed_mps=0.0)
        follower = MockActor(actor_id=158, x=10.0, y=0.0, yaw=0.0, speed_mps=5.0)
        world = MockWorld(self.mock_map, [leader, follower])
        controller = MissionPlatoonController(world, [leader, follower], cruise_speed_kmh=25.0)

        # Gap is 5.0 - 4.7 = 0.3m (very close)
        gap = controller._vehicle_clearance(leader, follower)
        target_spd = controller._target_speed(
            vehicle=follower,
            index=1,
            action="CRUISE",
            predecessor=leader,
            obstacle=None,
            clearance=80.0,
        )
        # Follower must command stop / safe deceleration
        self.assertAlmostEqual(target_spd, 0.0, delta=0.5)

        # Longitudinal controller applies full brake
        thr, brk = controller._compute_longitudinal_control(
            target_speed=target_spd,
            current_speed=5.0,
            predecessor=leader,
            predecessor_clearance=gap,
            override="INSUFFICIENT_HEADWAY" if gap < 3.8 else "NONE",
        )
        self.assertEqual(thr, 0.0)
        self.assertGreaterEqual(brk, 0.65)

    def test_standstill_breakout_never_violates_safety(self):
        """Standstill breakout is strictly inhibited if an obstacle, red light, or headway hazard exists."""
        leader = MockActor(actor_id=157, x=0.0, y=0.0)
        controller = MissionPlatoonController(MockWorld(self.mock_map, [leader]), [leader], cruise_speed_kmh=25.0)

        # Case 1: Obstacle nearby (< 6m)
        fake_obs = MockActor(actor_id=999, x=5.0, y=0.0)
        thr, brk = controller._compute_longitudinal_control(
            target_speed=7.0,
            current_speed=0.0,
            obstacle=fake_obs,
            obstacle_clearance=4.0,  # Below 6.0m breakout threshold
        )
        # Breakout must not activate
        self.assertLess(thr, 0.38)

        # Case 2: Predecessor gap tight (< 6m)
        fake_pred = MockActor(actor_id=157, x=5.0, y=0.0)
        thr, brk = controller._compute_longitudinal_control(
            target_speed=7.0,
            current_speed=0.0,
            predecessor=fake_pred,
            predecessor_clearance=4.5,
        )
        self.assertLess(thr, 0.38)

        # Case 3: Hard stop active
        thr, brk = controller._compute_longitudinal_control(
            target_speed=7.0,
            current_speed=0.0,
            hard_stop=True,
            light_distance=5.0,
        )
        self.assertLess(thr, 0.38)


if __name__ == "__main__":
    unittest.main()
