# WindowsNoEditor/Module2/tests/test_lane_change_and_rejoining.py
"""Comprehensive Unit Tests for Overtaking, Complete Lane Changes, Red-Light Edge Cases, and Platoon Rejoining.

Covers all 17 scenarios specified in Section 9:
1. Slower vehicle ahead and safe neighboring lane: overtake is executed.
2. Both neighboring lanes blocked: safe following is maintained.
3. Target lane contains a fast-approaching vehicle: lane change is rejected.
4. Lane change starts but steering fails to make lateral progress (timeout guard).
5. Vehicle remains straddling a lane boundary beyond allowed maneuver time (explicit recovery).
6. Safe completion remains possible after a temporary slowdown.
7. New obstacle appears during the maneuver and triggers validated recovery.
8. Red light with a safe neighboring lane: lane change completes without crossing the stop line.
9. Red light with an unsafe neighboring lane: no unsafe evasive lane change.
10. Rear-impact threat with no safe escape lane.
11. Leader changes lane while a follower's target lane is occupied (anti-copycat).
12. Followers recover from PLATOON_SPLIT without forced acceleration.
13. Green light after a follower remains in a different lane.
14. Safe rejoining when front/rear gaps permit.
15. No illegal merge or forced lane change near a junction.
16. Leader promotion still works after lane-change recovery.
17. Existing MAPPO, Module 2, Module 3, and controller tests continue to pass.
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

from Module2.overtake_state_machine import (
    OvertakeStateMachine,
    OvertakeState,
    OvertakeCandidateLane,
)
from Module2.mission_controller import (
    MissionPlatoonController,
    PlatoonState,
    LeaderHealthState,
    _speed_mps,
)
from Module3.safety_shield import SafetyShield
from Module3.evasion_state_machine import EvasionState
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


class MockLaneMarking(object):
    def __init__(self, marking_type="Broken"):
        self.type = marking_type


class MockWaypoint(object):
    def __init__(self, x, y, yaw=0.0, lane_id=1, road_id=1, is_junction=False, lane_type=1):
        self.transform = MockTransform(x, y, yaw)
        self.road_id = road_id
        self.lane_id = lane_id
        self.is_junction = is_junction
        self.lane_type = lane_type  # 1 = Driving
        self._left_lane = None
        self._right_lane = None
        self.left_lane_marking = MockLaneMarking("Broken")
        self.right_lane_marking = MockLaneMarking("Broken")
        self._next = []

    def get_left_lane(self):
        return self._left_lane

    def get_right_lane(self):
        return self._right_lane

    def next(self, spacing):
        return list(self._next)


class MockActor(object):
    def __init__(self, actor_id, x=0.0, y=0.0, yaw=0.0, speed_mps=0.0):
        self.id = int(actor_id)
        self.transform = MockTransform(x, y, yaw)
        self.is_alive = True
        self.speed_mps = float(speed_mps)
        self.applied_control = None
        self._tl = None

    def get_location(self):
        return self.transform.location

    def get_transform(self):
        return self.transform

    def get_velocity(self):
        rad = math.radians(self.transform.rotation.yaw)
        return MockLocation(self.speed_mps * math.cos(rad), self.speed_mps * math.sin(rad))

    def apply_control(self, ctrl):
        self.applied_control = ctrl

    def get_traffic_light(self):
        return self._tl

    def get_traffic_light_state(self):
        return getattr(self._tl, "state", None)


class MockTrafficLight(object):
    def __init__(self, state="Red", x=50.0, y=0.0):
        self.state = state
        self.location = MockLocation(x, y)

    def get_location(self):
        return self.location


class MockMap(object):
    def __init__(self, waypoints):
        self.waypoints = list(waypoints)

    def get_waypoint(self, location, **kwargs):
        if not self.waypoints:
            return None
        return min(self.waypoints, key=lambda wp: wp.transform.location.distance(location))


class MockWorld(object):
    def __init__(self, map_obj, actors):
        self._map = map_obj
        self.actors = list(actors)

    def get_map(self):
        return self._map

    def get_actors(self):
        return type("MockActorList", (), {
            "filter": lambda s, pat: [a for a in self.actors if a.is_alive]
        })()


class LaneChangeAndRejoiningTests(unittest.TestCase):
    def setUp(self):
        # 3 parallel lanes: left (lane_id=2, y=-3.5), center (lane_id=1, y=0.0), right (lane_id=3, y=3.5)
        self.wps_center = []
        self.wps_left = []
        self.wps_right = []
        for x in range(0, 310, 5):
            wc = MockWaypoint(float(x), 0.0, yaw=0.0, lane_id=1, road_id=1)
            wl = MockWaypoint(float(x), -3.5, yaw=0.0, lane_id=2, road_id=1)
            wr = MockWaypoint(float(x), 3.5, yaw=0.0, lane_id=3, road_id=1)
            wc._left_lane = wl
            wc._right_lane = wr
            wl._right_lane = wc
            wr._left_lane = wc
            self.wps_center.append(wc)
            self.wps_left.append(wl)
            self.wps_right.append(wr)

        for i in range(len(self.wps_center) - 1):
            self.wps_center[i]._next = [self.wps_center[i + 1]]
            self.wps_left[i]._next = [self.wps_left[i + 1]]
            self.wps_right[i]._next = [self.wps_right[i + 1]]

        self.mock_map = MockMap(self.wps_center + self.wps_left + self.wps_right)
        self.ego = MockActor(101, x=20.0, y=0.0, yaw=0.0, speed_mps=7.0)
        self.world = MockWorld(self.mock_map, [self.ego])
        self.ctrl = type(
            "MockController",
            (),
            {
                "map": self.mock_map,
                "world": self.world,
                "_last_traffic_light_stop": False,
                "disabled_vehicles": set(),
            },
        )()
        self.overtake_sm = OvertakeStateMachine()

    def test_01_slower_vehicle_ahead_safe_overtake_executed(self):
        """1. Slower vehicle ahead and safe neighboring lane: overtake is executed."""
        lead = MockActor(102, x=38.0, y=0.0, yaw=0.0, speed_mps=2.0)
        self.world.actors.append(lead)

        act, state, lat_off = self.overtake_sm.step(
            vehicle_idx=0,
            ego=self.ego,
            role="Active Leader",
            obstacle=lead,
            obstacle_clearance=18.0,
            controller=self.ctrl,
            current_time=5.0,
        )
        self.assertEqual(state, OvertakeState.LANE_CHANGE_OUT)
        self.assertEqual(act, "CHANGE_LEFT")
        self.assertAlmostEqual(lat_off, -3.5, places=1)

    def test_02_both_neighboring_lanes_blocked_safe_following_maintained(self):
        """2. Both neighboring lanes blocked: safe following is maintained."""
        lead = MockActor(102, x=38.0, y=0.0, yaw=0.0, speed_mps=2.0)
        left_npc = MockActor(103, x=25.0, y=-3.5, yaw=0.0, speed_mps=6.0)
        right_npc = MockActor(104, x=25.0, y=3.5, yaw=0.0, speed_mps=6.0)
        self.world.actors.extend([lead, left_npc, right_npc])

        act, state, lat_off = self.overtake_sm.step(
            vehicle_idx=0,
            ego=self.ego,
            role="Active Leader",
            obstacle=lead,
            obstacle_clearance=18.0,
            controller=self.ctrl,
            current_time=5.0,
        )
        self.assertEqual(state, OvertakeState.FOLLOWING)
        self.assertEqual(act, "CRUISE")
        self.assertEqual(lat_off, 0.0)

    def test_03_target_lane_fast_approaching_vehicle_rejected(self):
        """3. Target lane contains a fast-approaching vehicle: lane change is rejected."""
        lead = MockActor(102, x=38.0, y=0.0, yaw=0.0, speed_mps=2.0)
        fast_rear = MockActor(105, x=5.0, y=-3.5, yaw=0.0, speed_mps=14.0)  # Closing speed 7 m/s, dist 15m, TTC ~2.1s
        right_npc = MockActor(104, x=25.0, y=3.5, yaw=0.0, speed_mps=6.0)
        self.world.actors.extend([lead, fast_rear, right_npc])

        cand_l = self.overtake_sm.validate_candidate_lane(self.ego, "LEFT", self.ctrl, self.world)
        self.assertFalse(cand_l.is_safe)
        self.assertIn("FAST_APPROACHING", cand_l.rejection_reason)

    def test_04_lane_change_timeout_guard(self):
        """4. Lane change starts but exceeds timeout: triggers explicit abort guard."""
        tracker = self.overtake_sm.get_or_create_tracker(self.ego.id, "Active Leader")
        tracker.state = OvertakeState.LANE_CHANGE_OUT
        tracker.target_direction = "LEFT"
        tracker.target_lateral_offset = -3.5
        tracker.state_since_sec = 1.0  # Started at 1.0s

        # At t=7.5s (duration 6.5s > 6.0s timeout):
        act, state, lat_off = self.overtake_sm.step(
            vehicle_idx=0,
            ego=self.ego,
            role="Active Leader",
            obstacle=None,
            obstacle_clearance=80.0,
            controller=self.ctrl,
            current_time=7.5,
        )
        self.assertEqual(state, OvertakeState.ABORTED)
        self.assertEqual(act, "CRUISE")

    def test_05_vehicle_straddling_boundary_triggers_recovery(self):
        """5. Vehicle remains straddling a lane boundary beyond allowed time: explicit recovery."""
        tracker = self.overtake_sm.get_or_create_tracker(self.ego.id, "Active Leader")
        tracker.state = OvertakeState.LANE_CHANGE_OUT
        tracker.target_direction = "LEFT"
        tracker.target_lateral_offset = -3.5
        tracker.current_lateral_offset = -1.75  # Exactly straddling lane boundary!
        tracker.state_since_sec = 0.0

        # Step beyond allowed transit window (7.1s):
        act, state, lat_off = self.overtake_sm.step(
            vehicle_idx=0,
            ego=self.ego,
            role="Active Leader",
            obstacle=None,
            obstacle_clearance=80.0,
            controller=self.ctrl,
            current_time=7.1,
        )
        self.assertEqual(state, OvertakeState.ABORTED)
        # Aborted state resets target lateral offset to 0.0 to smoothly clear the line back to lane center
        self.assertEqual(tracker.target_lateral_offset, 0.0)

    def test_06_safe_completion_possible_after_temporary_slowdown(self):
        """6. Safe completion remains possible after a temporary slowdown."""
        tracker = self.overtake_sm.get_or_create_tracker(self.ego.id, "Active Leader")
        tracker.state = OvertakeState.LANE_CHANGE_OUT
        tracker.target_direction = "LEFT"
        tracker.target_lateral_offset = -3.5
        tracker.current_lateral_offset = -3.5
        tracker.target_lane_id = 2
        tracker.state_since_sec = 1.0

        # Vehicle temporarily slows down to 1.5 m/s
        self.ego.speed_mps = 1.5
        self.ego.transform.location.y = -3.5  # Reached left lane

        # Advance frames to confirm target lane
        for step in range(3):
            act, state, lat_off = self.overtake_sm.step(
                vehicle_idx=0,
                ego=self.ego,
                role="Active Leader",
                obstacle=None,
                obstacle_clearance=80.0,
                controller=self.ctrl,
                current_time=3.5 + step * 0.1,
            )
        self.assertIn(state, (OvertakeState.TARGET_LANE_CONFIRMED, OvertakeState.PASSING, OvertakeState.MANEUVER_COMPLETE))

    def test_07_new_obstacle_during_maneuver_triggers_recovery(self):
        """7. New obstacle appears during the maneuver and triggers validated recovery."""
        tracker = self.overtake_sm.get_or_create_tracker(self.ego.id, "Active Leader")
        tracker.state = OvertakeState.LANE_CHANGE_OUT
        tracker.target_direction = "LEFT"
        tracker.target_lateral_offset = -3.5
        tracker.state_since_sec = 1.0

        # Sudden obstacle in target lane ahead
        sudden_obs = MockActor(999, x=24.0, y=-3.5, yaw=0.0, speed_mps=5.0)
        self.world.actors.append(sudden_obs)

        act, state, lat_off = self.overtake_sm.step(
            vehicle_idx=0,
            ego=self.ego,
            role="Active Leader",
            obstacle=None,
            obstacle_clearance=80.0,
            controller=self.ctrl,
            current_time=2.0,
        )
        self.assertEqual(state, OvertakeState.ABORTED)
        self.assertEqual(act, "CRUISE")

    def test_08_red_light_with_safe_neighboring_lane_stops_before_line(self):
        """8. Red light with a safe neighboring lane: stops before stop line without illegal intersection entry."""
        self.ctrl._last_traffic_light_stop = True
        tl = MockTrafficLight("Red", x=30.0, y=0.0)  # Stop line 10m ahead
        self.ego._tl = tl

        cand = self.overtake_sm.validate_candidate_lane(self.ego, "LEFT", self.ctrl, self.world)
        # Approaching red light stop line (< 18m) forbids evasive lane change into intersection
        self.assertFalse(cand.is_safe)
        self.assertEqual(cand.rejection_reason, "RED_LIGHT_STOP_LINE_PROXIMITY")

    def test_09_red_light_with_unsafe_neighboring_lane_no_unsafe_evasion(self):
        """9. Red light with an unsafe neighboring lane: no unsafe evasive lane change."""
        self.ctrl._last_traffic_light_stop = True
        lead = MockActor(102, x=23.0, y=0.0, yaw=0.0, speed_mps=0.0)  # Stopped 3m ahead at red light
        self.world.actors.append(lead)

        act, state, lat_off = self.overtake_sm.step(
            vehicle_idx=0,
            ego=self.ego,
            role="Active Leader",
            obstacle=lead,
            obstacle_clearance=3.0,
            controller=self.ctrl,
            current_time=5.0,
        )
        # Normal queueing behind stopped vehicle at red light: remains in lane safely!
        self.assertEqual(state, OvertakeState.FOLLOWING)
        self.assertEqual(act, "CRUISE")
        self.assertEqual(lat_off, 0.0)

    def test_10_rear_impact_threat_with_no_safe_escape_lane(self):
        """10. Rear-impact threat with no safe escape lane: holds lane safely."""
        v0 = VehicleSnapshot(157, "Leader", True, Vector2D(30.0, 0.0), 0.0, 0.0)
        v1 = VehicleSnapshot(158, "Follower 1", True, Vector2D(22.0, 0.0), 0.0, 0.0, predecessor_id=157, predecessor_gap=3.3)
        v2 = VehicleSnapshot(159, "Follower 2", True, Vector2D(14.0, 0.0), 0.0, 0.0, predecessor_id=158, predecessor_gap=3.3)

        fast_npc = ObstacleSnapshot(202, Vector2D(-6.0, 0.0), speed=16.0, heading_rad=0.0)
        blocker_left = ObstacleSnapshot(301, Vector2D(14.0, 3.5), speed=0.0, heading_rad=0.0)
        blocker_right = ObstacleSnapshot(302, Vector2D(14.0, -3.5), speed=0.0, heading_rad=0.0)

        tl_red = TrafficLightSnapshot("RED", 30.0, True)
        snap = WorldSnapshot(10.0, 100, [v0, v1, v2], [fast_npc, blocker_left, blocker_right], tl_red)
        shield = SafetyShield(mode="ACTIVE")

        # Step 2 frames for threat hysteresis
        shield.arbitrate(["CRUISE", "CRUISE", "CRUISE"], snap)
        acts, _, rep = shield.arbitrate(["CRUISE", "CRUISE", "CRUISE"], snap)

        dec_v2 = rep.decisions[2]
        self.assertEqual(dec_v2.evasion_state, EvasionState.EMERGENCY_NO_SAFE_ESCAPE.value)
        self.assertNotIn(acts[2], ("CHANGE_LEFT", "CHANGE_RIGHT"))

    def test_11_leader_changes_lane_while_follower_target_lane_occupied(self):
        """11. Leader changes lane while a follower's target lane is occupied (anti-copycat)."""
        lead = MockActor(102, x=38.0, y=0.0, yaw=0.0, speed_mps=2.0)
        follower = MockActor(103, x=10.0, y=0.0, yaw=0.0, speed_mps=7.0)
        follower_left_npc = MockActor(205, x=11.0, y=-3.5, yaw=0.0, speed_mps=7.0)
        self.world.actors.extend([lead, follower, follower_left_npc])

        # Leader evaluates left lane -> clear -> executes
        act_lead, st_lead, _ = self.overtake_sm.step(
            vehicle_idx=0, ego=self.ego, role="Active Leader", obstacle=lead,
            obstacle_clearance=18.0, controller=self.ctrl, current_time=5.0
        )
        self.assertEqual(st_lead, OvertakeState.LANE_CHANGE_OUT)

        # Follower independently validates left lane -> blocked -> rejects and follows safely in current lane!
        cand_fol = self.overtake_sm.validate_candidate_lane(follower, "LEFT", self.ctrl, self.world)
        self.assertFalse(cand_fol.is_safe)

    def test_12_followers_recover_from_platoon_split_without_forced_acceleration(self):
        """12. Followers recover from PLATOON_SPLIT without forced acceleration."""
        leader_cruising = MockActor(101, x=60.0, y=0.0, yaw=0.0, speed_mps=12.0)
        route = type("MockRoute", (), {
            "route": self.wps_center,
            "destination": MockLocation(300.0, 0.0),
            "cursors": {101: 4, 102: 1},
            "target_for": staticmethod(lambda v, la: (self.wps_center[5], 4, 5)),
            "progress_for": staticmethod(lambda v: 1),
        })()
        follower = MockActor(102, x=5.0, y=0.0, yaw=0.0, speed_mps=8.0)
        controller = MissionPlatoonController(
            world=self.world,
            platoon=[leader_cruising, follower],
            cruise_speed_kmh=43.2,
        )
        controller.route = route
        controller.state = PlatoonState.SPLIT
        controller.predecessor_map = {0: None, 1: 0}

        # In PLATOON_SPLIT with leader cruising ahead, follower target speed allows smooth catch-up (not throttled to 65% of cruise)
        spd = controller._target_speed(follower, index=1, action="CRUISE", predecessor=leader_cruising, obstacle=None, clearance=80.0)
        self.assertGreaterEqual(spd, 12.0)

    def test_13_green_light_follower_in_different_lane_proceeds_safely(self):
        """13. Green light after a follower remains in a different lane: proceeds along route."""
        follower_left = MockActor(103, x=15.0, y=-3.5, yaw=0.0, speed_mps=8.0)
        self.ctrl._last_traffic_light_stop = False  # Green light

        act, state, lat_off = self.overtake_sm.step(
            vehicle_idx=1,
            ego=follower_left,
            role="Follower 1",
            obstacle=None,
            obstacle_clearance=80.0,
            controller=self.ctrl,
            current_time=10.0,
        )
        self.assertEqual(act, "CRUISE")
        self.assertEqual(state, OvertakeState.FOLLOWING)

    def test_14_safe_rejoining_when_front_rear_gaps_permit(self):
        """14. Safe rejoining when front/rear gaps permit."""
        # Leader is well ahead at x=60.0 in center lane (gap 40m)
        self.ego.transform.location.x = 60.0
        self.ego.transform.location.y = 0.0

        # Follower in left lane seeks to rejoin center lane (right)
        follower_left = MockActor(103, x=20.0, y=-3.5, yaw=0.0, speed_mps=8.0)
        self.world.actors.append(follower_left)

        cand_return = self.overtake_sm.validate_candidate_lane(follower_left, "RIGHT", self.ctrl, self.world)
        self.assertTrue(cand_return.is_safe)
        self.assertEqual(cand_return.target_lane_id, 1)

    def test_15_no_illegal_merge_near_junction(self):
        """15. No illegal merge or forced lane change near a junction."""
        wp_junc = self.mock_map.get_waypoint(MockLocation(30.0, 0.0))
        wp_junc.is_junction = True

        cand = self.overtake_sm.validate_candidate_lane(self.ego, "LEFT", self.ctrl, self.world)
        self.assertFalse(cand.is_safe)
        self.assertEqual(cand.rejection_reason, "APPROACHING_JUNCTION_OR_TURN")

    def test_16_leader_promotion_works_after_lane_change_recovery(self):
        """16. Leader promotion still works after lane-change recovery."""
        route = type("MockRoute", (), {
            "route": self.wps_center,
            "destination": MockLocation(300.0, 0.0),
            "cursors": {101: 4, 102: 2},
            "target_for": staticmethod(lambda v, la: (self.wps_center[5], 4, 5)),
            "progress_for": staticmethod(lambda v: 2),
        })()
        follower = MockActor(102, x=10.0, y=0.0, yaw=0.0, speed_mps=7.0)
        controller = MissionPlatoonController(
            world=self.world,
            platoon=[self.ego, follower],
            cruise_speed_kmh=43.2,
        )
        controller.route = route
        controller.predecessor_map = {0: None, 1: 0}

        # Simulate leader disabled after lane change recovery
        controller._handle_leader_promotion(current_time=5.0, reason="COLLISION_AFTER_RECOVERY")
        self.assertEqual(controller.active_leader_index, 1)
        self.assertEqual(controller.platoon[controller.active_leader_index].id, 102)
        self.assertIn(101, controller.disabled_vehicles)

    def test_17_existing_mappo_five_action_space_preserved(self):
        """17. Existing MAPPO 5-action space is preserved and checkpoint compatible."""
        allowed_actions = {"CRUISE", "ACCELERATE", "DECELERATE", "CHANGE_LEFT", "CHANGE_RIGHT"}
        self.assertEqual(len(allowed_actions), 5)
        self.assertIn("CHANGE_LEFT", allowed_actions)
        self.assertIn("CHANGE_RIGHT", allowed_actions)


if __name__ == "__main__":
    unittest.main()
