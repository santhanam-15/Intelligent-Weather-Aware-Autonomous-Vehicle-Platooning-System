"""Regression test suite for promoted leader control authority and forward motion.

Covers:
1. Normal driving with no leader failure.
2. Leader disabled after crash / collision immobilization.
3. Candidate follower promoted to ACTIVE LEADER.
4. Old leader disabled (brake=1.0, handbrake=True, throttle=0.0).
5. Promoted leader has valid route and target speed > 0.
6. Promoted leader outputs positive throttle and zero brake when forward lane is clear.
7. Promoted leader with real forward obstacle (< 2.5m) outputs full brake.
8. Promoted leader with red traffic light outputs full brake.
9. Follower 2 tracks promoted leader (Follower 1) instead of disabled leader.
10. UI/telemetry displays correct role, gap ('--' for leader), and no false collision tag.
11. Module 3 evaluates promoted leader as leader and followers relative to new predecessor.
12. Repeated ticks demonstrate steady forward progress (speed increases or reaches cruise).
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
    LeaderHealthState,
    _speed_mps,
    _clamp,
)
from PythonAPI.autopilot_platoon_window import platoon_rows
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


class MockBoundingBox(object):
    def __init__(self, extent_x=2.4, extent_y=1.0):
        self.extent = MockLocation(extent_x, extent_y, 0.8)


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


class MockCarlaVehicle(object):
    def __init__(self, actor_id, x=0.0, y=0.0, yaw=0.0, speed_mps=0.0):
        self.id = int(actor_id)
        self.is_alive = True
        self.transform = MockTransform(x, y, yaw)
        self.speed_mps = float(speed_mps)
        self.applied_control = None
        self.bounding_box = MockBoundingBox()
        self.type_id = "vehicle.lincoln.mkz_2020"

    def get_location(self):
        return self.transform.location

    def get_transform(self):
        return self.transform

    def get_velocity(self):
        rad = math.radians(self.transform.rotation.yaw)
        return MockLocation(self.speed_mps * math.cos(rad), self.speed_mps * math.sin(rad))

    def get_forward_vector(self):
        return self.transform.get_forward_vector()

    def get_traffic_light(self):
        return None

    def get_traffic_light_state(self):
        return None

    def get_world(self):
        return type("MockW", (), {"get_snapshot": lambda s: type("Snap", (), {"frame": 100})()})()

    def apply_control(self, ctrl):
        self.applied_control = ctrl


class MockWorld(object):
    def __init__(self, mock_map, actors):
        self.mock_map = mock_map
        self.actors = list(actors)

    def get_map(self):
        return self.mock_map

    def get_actors(self):
        return type("ActorList", (), {
            "filter": lambda s, pat: [a for a in self.actors if a.is_alive]
        })()

    def get_snapshot(self):
        return type("Snap", (), {"frame": 100})()


def make_test_route(start_x=0.0, end_x=200.0, step=2.0):
    wps = []
    x = start_x
    while x <= end_x:
        wps.append(MockWaypoint(x=x, y=0.0, yaw=0.0))
        x += step
    for i in range(len(wps) - 1):
        wps[i]._next = [wps[i + 1]]
    return wps


class PromotedLeaderControlTests(unittest.TestCase):
    def setUp(self):
        # 3 Platoon vehicles along straight line X:
        # Leader (id=300) at x=20.0, y=0.0
        # Follower 1 (id=301) at x=13.0, y=0.0
        # Follower 2 (id=302) at x=6.0, y=0.0
        self.lead = MockCarlaVehicle(300, x=20.0, y=0.0, yaw=0.0, speed_mps=5.0)
        self.f1 = MockCarlaVehicle(301, x=13.0, y=0.0, yaw=0.0, speed_mps=5.0)
        self.f2 = MockCarlaVehicle(302, x=6.0, y=0.0, yaw=0.0, speed_mps=5.0)
        self.platoon = [self.lead, self.f1, self.f2]

        self.wps = make_test_route(0.0, 300.0, 2.0)
        self.mock_map = MockMap(self.wps)
        self.world = MockWorld(self.mock_map, self.platoon)

        self.controller = MissionPlatoonController(
            self.world,
            self.platoon,
            cruise_speed_kmh=25.0,  # ~6.94 m/s
        )
        self.controller.route.route = list(self.wps)
        for v in self.platoon:
            self.controller.route.cursors[v.id] = self.controller.route.progress_for(v)

    def test_01_normal_driving_nominal_state(self):
        """1. Normal driving with no leader failure."""
        self.assertEqual(self.controller.active_leader_index, 0)
        self.assertEqual(self.controller.leader_health_state, LeaderHealthState.ACTIVE)
        self.assertEqual(self.controller.predecessor_map[1], 0)
        self.assertEqual(self.controller.predecessor_map[2], 1)

    def test_02_and_03_leader_failure_and_follower_promotion(self):
        """2 & 3. Leader disabled after crash and follower 1 promoted to ACTIVE LEADER."""
        # Leader crashes into tree off-road at x=20.0, y=-2.5
        self.lead.transform.location.y = -2.5
        self.lead.speed_mps = 0.0
        self.controller.record_collision(self.lead.id, intensity=500.0, current_time=1.0)
        self.controller._leader_stopped_since = 1.0

        # Advance beyond collision immobilization threshold
        self.controller._check_leader_status(current_time=7.5)

        self.assertEqual(self.controller.active_leader_index, 1)
        self.assertEqual(self.controller.leader_health_state, LeaderHealthState.NEW_LEADER_ACTIVE)
        self.assertIn(self.lead.id, self.controller.disabled_vehicles)

    def test_04_old_leader_full_hold_brake(self):
        """4. Old leader disabled (brake=1.0, handbrake=True, throttle=0.0)."""
        self.controller._handle_leader_promotion(current_time=5.0, reason="COLLISION_TEST")
        self.controller.tick(current_time=5.1)

        self.assertIsNotNone(self.lead.applied_control)
        self.assertEqual(self.lead.applied_control.throttle, 0.0)
        self.assertEqual(self.lead.applied_control.brake, 1.0)
        self.assertTrue(self.lead.applied_control.hand_brake)

    def test_05_promoted_leader_route_and_target_speed_valid(self):
        """5. Promoted leader has valid route and target speed > 0."""
        self.controller._handle_leader_promotion(current_time=5.0, reason="COLLISION_TEST")
        f1_cursor = self.controller.route.cursors[self.f1.id]
        self.assertGreaterEqual(f1_cursor, 0)
        self.assertLess(f1_cursor, len(self.controller.route.route))

        tgt_spd = self.controller._target_speed(
            self.f1, index=1, action="ACCELERATE", predecessor=None, obstacle=None, clearance=80.0
        )
        self.assertGreater(tgt_spd, 3.0)

    def test_06_promoted_leader_moves_forward_positive_throttle_zero_brake(self):
        """6. Promoted leader outputs positive throttle and zero brake when forward lane is clear."""
        # Old leader crashed on curb at y=-2.5m, ego is at (13.0, 0.0), speed 0.0 m/s
        self.lead.transform.location.y = -2.5
        self.lead.speed_mps = 0.0
        self.f1.speed_mps = 0.0

        self.controller._handle_leader_promotion(current_time=5.0, reason="COLLISION_TEST")
        self.controller.set_tactical_actions(["STOP", "ACCELERATE", "ACCELERATE"])

        # Tick controller
        self.controller.tick(current_time=5.1)

        ctrl = self.f1.applied_control
        self.assertIsNotNone(ctrl)
        self.assertGreater(ctrl.throttle, 0.20, "Promoted leader must receive positive acceleration throttle")
        self.assertEqual(ctrl.brake, 0.0, "Promoted leader must not be commanded to brake when lane is clear")

    def test_07_promoted_leader_real_forward_obstacle_stops(self):
        """7. Promoted leader with real forward obstacle in lane (< 2.5m) outputs full brake."""
        self.f1.speed_mps = 0.0
        self.controller._handle_leader_promotion(current_time=5.0, reason="COLLISION_TEST")

        # Spawn obstacle in ego's lane 2.0m ahead: (15.0, 0.0)
        obs = MockCarlaVehicle(999, x=15.0, y=0.0, yaw=0.0, speed_mps=0.0)
        self.world.actors.append(obs)

        self.controller.tick(current_time=5.1)
        ctrl = self.f1.applied_control
        self.assertEqual(ctrl.throttle, 0.0)
        self.assertEqual(ctrl.brake, 1.0)

    def test_08_promoted_leader_red_light_stops(self):
        """8. Promoted leader with red traffic light outputs full brake."""
        self.f1.speed_mps = 2.0
        self.controller._handle_leader_promotion(current_time=5.0, reason="COLLISION_TEST")
        self.controller._traffic_light_constraint = lambda veh: (True, 2.5, "RED")

        self.controller.tick(current_time=5.1)
        ctrl = self.f1.applied_control
        self.assertEqual(ctrl.throttle, 0.0)
        self.assertEqual(ctrl.brake, 1.0)

    def test_09_follower2_tracks_promoted_leader(self):
        """9. Follower 2 tracks promoted leader (Follower 1) instead of disabled leader."""
        self.controller._handle_leader_promotion(current_time=5.0, reason="COLLISION_TEST")
        self.assertEqual(self.controller.predecessor_map[2], 1)
        self.assertIsNone(self.controller.predecessor_map[1])

    def test_10_ui_telemetry_gap_and_overrides(self):
        """10. UI/telemetry displays correct role, gap ('--' for leader), and no false collision tag."""
        self.lead.transform.location.y = -2.5
        self.controller._handle_leader_promotion(current_time=5.0, reason="COLLISION_TEST")

        rows = platoon_rows(
            self.platoon,
            active_leader_index=self.controller.active_leader_index,
            disabled_ids=self.controller.disabled_vehicles,
        )

        # Row 0: Disabled leader
        self.assertEqual(rows[0]["role"], "DISABLED LEADER")
        # Row 1: Active leader
        self.assertEqual(rows[1]["role"], "ACTIVE LEADER")
        self.assertIsNone(rows[1]["gap"], "Active leader gap must be None (rendered as --)")
        # Row 2: Follower 2
        self.assertEqual(rows[2]["role"], "FOLLOWER 2")
        self.assertIsNotNone(rows[2]["gap"], "Follower 2 must have valid gap to active leader")

    def test_11_module3_safety_evaluation_for_promoted_leader(self):
        """11. Module 3 evaluates promoted leader as leader and followers relative to new predecessor."""
        shield = SafetyShield(mode="SHADOW", enabled=True)
        # Snapshot where vehicle 0 is disabled, vehicle 1 is active leader
        ego_states = [
            VehicleSnapshot(vehicle_id=300, role="Disabled Leader", is_alive=True, position=Vector2D(20, -2.5), speed=0.0, heading_rad=0.0, length=4.5, width=2.0, friction=0.8, predecessor_id=None, predecessor_gap=80.0),
            VehicleSnapshot(vehicle_id=301, role="Active Leader", is_alive=True, position=Vector2D(13, 0), speed=0.0, heading_rad=0.0, length=4.5, width=2.0, friction=0.8, predecessor_id=None, predecessor_gap=80.0),
            VehicleSnapshot(vehicle_id=302, role="Follower 2", is_alive=True, position=Vector2D(6, 0), speed=0.0, heading_rad=0.0, length=4.5, width=2.0, friction=0.8, predecessor_id=301, predecessor_gap=7.0),
        ]
        snap = WorldSnapshot(
            timestamp=5.0,
            frame=100,
            platoon=ego_states,
            obstacles=[],
            traffic_light=TrafficLightSnapshot(state="GREEN", distance=80.0, is_hard_stop=False),
            weather_preset="ClearNoon",
            visibility_m=100.0,
            nominal_friction=0.8,
            is_valid=True,
        )

        actions, emerg, report = shield.arbitrate(
            proposed_actions=["STOP", "ACCELERATE", "ACCELERATE"],
            snapshot=snap,
        )
        self.assertEqual(actions[1], "ACCELERATE")
        self.assertFalse(emerg[1], "Promoted leader must pass safety shield without emergency brake")

    def test_12_repeated_ticks_demonstrate_acceleration_and_forward_motion(self):
        """12. Repeated ticks demonstrate steady forward progress (positive throttle maintained)."""
        self.lead.transform.location.y = -2.5
        self.lead.speed_mps = 0.0
        self.f1.speed_mps = 0.0

        self.controller._handle_leader_promotion(current_time=5.0, reason="COLLISION_TEST")
        self.controller.set_tactical_actions(["STOP", "ACCELERATE", "ACCELERATE"])

        # Run 5 consecutive ticks simulating vehicle accelerating
        simulated_speeds = [0.0, 0.4, 1.0, 1.8, 2.5]
        for step, spd in enumerate(simulated_speeds):
            self.f1.speed_mps = spd
            self.f1.transform.location.x += spd * 0.033
            self.controller.tick(current_time=5.0 + step * 0.033)
            ctrl = self.f1.applied_control
            self.assertGreater(ctrl.throttle, 0.15, f"Step {step}: throttle must remain positive")
            self.assertEqual(ctrl.brake, 0.0, f"Step {step}: brake must remain 0.0")


if __name__ == "__main__":
    unittest.main()
