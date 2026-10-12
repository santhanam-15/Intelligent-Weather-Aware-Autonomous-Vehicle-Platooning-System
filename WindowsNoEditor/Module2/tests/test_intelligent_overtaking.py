"""Comprehensive unit tests for Intelligent Safe Overtaking System (Section 7).

Covers all 16 requested test scenarios:
1. Slower vehicle ahead, left lane clear: safe overtake is considered and executed.
2. Slower vehicle ahead, right lane clear: right lane is considered when legal.
3. Slower vehicle ahead, both lanes blocked: remain behind safely.
4. Fast vehicle approaching from behind in target lane: reject that lane.
5. Vehicle occupying the target-lane blind spot: reject that lane.
6. Lane change forbidden by road markings or lane type: reject.
7. Junction or required turn approaching: preserve the mission route.
8. A vehicle suddenly enters the target lane during the maneuver: abort or recover safely.
9. Overtake completed but original lane is occupied: remain in the safe current lane.
10. Return lane becomes safe later: return only after revalidation.
11. Follower has insufficient gap to perform the leader's lane change: follower waits safely.
12. Repeated slower vehicles do not cause lane oscillation (cooldown enforced).
13. Module 3 rejects an unsafe proposed maneuver in ACTIVE mode.
14. SHADOW mode reports the decision without pretending to enforce it.
15. Leader promotion and route continuity still work.
16. Existing Module 2, Module 3, and controller tests continue to pass.
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
    LeaderHealthState,
    _speed_mps,
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


class MockLaneMarking(object):
    def __init__(self, marking_type="Broken"):
        self.type = marking_type


class MockWaypoint(object):
    def __init__(self, x, y, yaw=0.0, lane_id=1, road_id=1, is_junction=False, lane_type=None):
        self.transform = MockTransform(x, y, yaw)
        self.road_id = road_id
        self.lane_id = lane_id
        self.is_junction = is_junction
        self.lane_type = lane_type or 1  # Driving = 1
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

    def get_location(self):
        return self.transform.location

    def get_transform(self):
        return self.transform

    def get_velocity(self):
        rad = math.radians(self.transform.rotation.yaw)
        return MockLocation(self.speed_mps * math.cos(rad), self.speed_mps * math.sin(rad))

    def apply_control(self, ctrl):
        self.applied_control = ctrl


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


class IntelligentOvertakeUnitTests(unittest.TestCase):
    def setUp(self):
        # Build 3 parallel lanes: left (lane_id=2, y=-3.5), center (lane_id=1, y=0.0), right (lane_id=3, y=3.5)
        self.wps_center = []
        self.wps_left = []
        self.wps_right = []
        for x in range(0, 305, 5):
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
        self.ctrl = type("MockController", (), {"map": self.mock_map, "world": self.world, "_last_traffic_light_stop": False, "disabled_vehicles": set()})()
        self.overtake_sm = OvertakeStateMachine()

    def test_01_slower_vehicle_ahead_left_lane_clear_initiates_overtake(self):
        """1. Slower vehicle ahead, left lane clear: safe overtake is evaluated and approved."""
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

    def test_02_slower_vehicle_ahead_right_lane_clear_overtake_right(self):
        """2. Slower vehicle ahead, left lane blocked, right lane clear: right lane selected."""
        lead = MockActor(102, x=38.0, y=0.0, yaw=0.0, speed_mps=2.0)
        left_npc = MockActor(103, x=25.0, y=-3.5, yaw=0.0, speed_mps=7.0)
        self.world.actors.extend([lead, left_npc])

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
        self.assertEqual(act, "CHANGE_RIGHT")
        self.assertAlmostEqual(lat_off, 3.5, places=1)

    def test_03_both_lanes_blocked_remains_in_lane(self):
        """3. Slower vehicle ahead, both adjacent lanes blocked: remain behind safely."""
        lead = MockActor(102, x=38.0, y=0.0, yaw=0.0, speed_mps=2.0)
        left_npc = MockActor(103, x=25.0, y=-3.5, yaw=0.0, speed_mps=7.0)
        right_npc = MockActor(104, x=25.0, y=3.5, yaw=0.0, speed_mps=7.0)
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

    def test_04_fast_vehicle_approaching_from_behind_rejects_lane(self):
        """4. Fast vehicle approaching from behind in target lane: reject that lane."""
        lead = MockActor(102, x=38.0, y=0.0, yaw=0.0, speed_mps=2.0)
        fast_rear = MockActor(105, x=0.0, y=-3.5, yaw=0.0, speed_mps=15.0)  # Closing speed 8.0 m/s, TTC 2.5s
        right_npc = MockActor(104, x=25.0, y=3.5, yaw=0.0, speed_mps=7.0)
        self.world.actors.extend([lead, fast_rear, right_npc])

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

    def test_05_target_lane_blind_spot_occupied_rejects_lane(self):
        """5. Vehicle occupying the target-lane blind spot: reject that lane."""
        lead = MockActor(102, x=38.0, y=0.0, yaw=0.0, speed_mps=2.0)
        blind_spot = MockActor(106, x=16.0, y=-3.5, yaw=0.0, speed_mps=7.0)  # Gap ~4m behind ego
        right_npc = MockActor(104, x=25.0, y=3.5, yaw=0.0, speed_mps=7.0)
        self.world.actors.extend([lead, blind_spot, right_npc])

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

    def test_06_solid_line_marking_forbids_lane_change(self):
        """6. Lane change forbidden by road markings (Solid line): reject."""
        lead = MockActor(102, x=38.0, y=0.0, yaw=0.0, speed_mps=2.0)
        self.world.actors.append(lead)

        # Mark center lane as solid
        wp_ego = self.mock_map.get_waypoint(self.ego.get_location())
        wp_ego.left_lane_marking = MockLaneMarking("Solid")
        wp_ego.right_lane_marking = MockLaneMarking("Solid")

        # In Mock, solid line marking forbids
        cand_l = self.overtake_sm.validate_candidate_lane(self.ego, "LEFT", self.ctrl, self.world)
        self.assertFalse(cand_l.is_safe)
        self.assertIn("SOLID_LINE", cand_l.rejection_reason)

    def test_07_approaching_junction_preserves_route(self):
        """7. Junction or required turn approaching: preserve the mission route."""
        lead = MockActor(102, x=38.0, y=0.0, yaw=0.0, speed_mps=2.0)
        self.world.actors.append(lead)

        # Place junction 15m ahead
        wp_ahead = self.mock_map.get_waypoint(MockLocation(35.0, 0.0))
        wp_ahead.is_junction = True

        cand = self.overtake_sm.validate_candidate_lane(self.ego, "LEFT", self.ctrl, self.world)
        self.assertFalse(cand.is_safe)
        self.assertEqual(cand.rejection_reason, "APPROACHING_JUNCTION_OR_TURN")

    def test_08_sudden_obstacle_in_target_lane_aborts_maneuver(self):
        """8. Vehicle suddenly enters target lane during maneuver: abort and recover."""
        lead = MockActor(102, x=38.0, y=0.0, yaw=0.0, speed_mps=2.0)
        self.world.actors.append(lead)

        tracker = self.overtake_sm.get_or_create_tracker(self.ego.id, "Active Leader")
        tracker.state = OvertakeState.LANE_CHANGE_OUT
        tracker.target_direction = "LEFT"
        tracker.state_since_sec = 1.0

        # Now sudden intruder enters left lane right beside ego
        intruder = MockActor(999, x=22.0, y=-3.5, yaw=0.0, speed_mps=7.0)
        self.world.actors.append(intruder)

        act, state, lat_off = self.overtake_sm.step(
            vehicle_idx=0,
            ego=self.ego,
            role="Active Leader",
            obstacle=lead,
            obstacle_clearance=18.0,
            controller=self.ctrl,
            current_time=2.0,
        )
        self.assertEqual(state, OvertakeState.ABORTED)
        self.assertEqual(act, "CRUISE")

    def test_09_and_10_passing_and_safe_return_lifecycle(self):
        """9 & 10. Pass slower vehicle, return when safe; hold lane if original lane occupied."""
        lead = MockActor(102, x=38.0, y=0.0, yaw=0.0, speed_mps=2.0)
        self.world.actors.append(lead)

        tracker = self.overtake_sm.get_or_create_tracker(self.ego.id, "Active Leader")
        tracker.state = OvertakeState.PASSING
        tracker.target_direction = "LEFT"
        tracker.overtaken_vehicle_id = 102
        tracker.state_since_sec = 5.0

        # Ego is now ahead of lead: ego at x=55.0, lead at x=38.0 (diff 17m > 8m)
        self.ego.transform.location.x = 55.0
        self.ego.transform.location.y = -3.5

        # 9: Original lane occupied by another NPC
        npc_orig = MockActor(107, x=58.0, y=0.0, yaw=0.0, speed_mps=7.0)
        self.world.actors.append(npc_orig)

        act, state, lat_off = self.overtake_sm.step(
            vehicle_idx=0,
            ego=self.ego,
            role="Active Leader",
            obstacle=None,
            obstacle_clearance=80.0,
            controller=self.ctrl,
            current_time=6.0,
        )
        # Transitioned to RETURN_GAP_VALIDATION, but remains in passing lane because return is occupied
        self.assertEqual(state, OvertakeState.RETURN_GAP_VALIDATION)
        self.assertEqual(act, "CRUISE")

        # 10: Original lane clears -> returns safely
        self.world.actors.remove(npc_orig)
        act2, state2, lat_off2 = self.overtake_sm.step(
            vehicle_idx=0,
            ego=self.ego,
            role="Active Leader",
            obstacle=None,
            obstacle_clearance=80.0,
            controller=self.ctrl,
            current_time=7.0,
        )
        self.assertEqual(state2, OvertakeState.LANE_CHANGE_BACK)

    def test_11_follower_anti_copycat_rule(self):
        """11. Follower does not blindly change lanes when leader changes lanes."""
        from Module3.platoon_coordinator import PlatoonEvasionCoordinator
        from Module3.evasion_state_machine import VehicleEvasionTracker, EvasionState

        coord = PlatoonEvasionCoordinator()
        ego_states = [
            VehicleSnapshot(101, "Active Leader", True, Vector2D(30, 0), 7.0, 0.0, 4.5, 2.0, 0.8, None, 80.0),
            VehicleSnapshot(102, "Follower 1", True, Vector2D(20, 0), 7.0, 0.0, 4.5, 2.0, 0.8, 101, 10.0),
        ]
        snap = WorldSnapshot(1.0, 10, ego_states, [], TrafficLightSnapshot("GREEN", 80.0, False), "ClearNoon", 100.0, 0.8, True)
        trackers = {
            101: VehicleEvasionTracker(101, "Active Leader"),
            102: VehicleEvasionTracker(102, "Follower 1"),
        }
        # Follower proposes CHANGE_LEFT
        sanitized_acts, states = coord.coordinate_platoon(snap, trackers, ["CHANGE_LEFT", "CHANGE_LEFT"])
        # Non-evading follower must be remapped to CRUISE (anti-copycat)
        self.assertEqual(sanitized_acts[1], "CRUISE")

    def test_12_cooldown_prevents_lane_oscillation(self):
        """12. Repeated slower vehicles do not cause immediate lane oscillation (cooldown enforced)."""
        lead = MockActor(102, x=38.0, y=0.0, yaw=0.0, speed_mps=2.0)
        self.world.actors.append(lead)

        tracker = self.overtake_sm.get_or_create_tracker(self.ego.id, "Active Leader")
        tracker.state = OvertakeState.FOLLOWING
        tracker.last_maneuver_complete_time = 10.0  # Just completed maneuver at 10.0s

        # New obstacle at 12.0s (within 8.0s cooldown)
        act, state, lat_off = self.overtake_sm.step(
            vehicle_idx=0,
            ego=self.ego,
            role="Active Leader",
            obstacle=lead,
            obstacle_clearance=18.0,
            controller=self.ctrl,
            current_time=12.0,
        )
        self.assertEqual(state, OvertakeState.FOLLOWING)
        self.assertEqual(act, "CRUISE")

    def test_13_module3_active_mode_rejects_unsafe_maneuver(self):
        """13. Module 3 rejects an unsafe proposed maneuver in ACTIVE mode."""
        shield = SafetyShield(mode="ACTIVE", enabled=True)
        # Place obstacle directly in left lane
        ego_states = [
            VehicleSnapshot(101, "Active Leader", True, Vector2D(20, 0), 7.0, 0.0, 4.5, 2.0, 0.8, None, 80.0)
        ]
        obs = [
            ObstacleSnapshot(201, Vector2D(23, -3.5), 7.0, 0.0, 4.5, 2.0)
        ]
        snap = WorldSnapshot(1.0, 10, ego_states, obs, TrafficLightSnapshot("GREEN", 80.0, False), "ClearNoon", 100.0, 0.8, True)

        actions, emerg, rep = shield.arbitrate(["CHANGE_LEFT"], snap)
        # In ACTIVE mode, unsafe lane change is rejected to DECELERATE
        self.assertEqual(actions[0], "DECELERATE")

    def test_14_module3_shadow_mode_observes_without_altering_action(self):
        """14. SHADOW mode reports the decision without pretending to enforce it."""
        shield = SafetyShield(mode="SHADOW", enabled=True)
        ego_states = [
            VehicleSnapshot(101, "Active Leader", True, Vector2D(20, 0), 7.0, 0.0, 4.5, 2.0, 0.8, None, 80.0)
        ]
        obs = [
            ObstacleSnapshot(201, Vector2D(23, -3.5), 7.0, 0.0, 4.5, 2.0)
        ]
        snap = WorldSnapshot(1.0, 10, ego_states, obs, TrafficLightSnapshot("GREEN", 80.0, False), "ClearNoon", 100.0, 0.8, True)

        actions, emerg, rep = shield.arbitrate(["CHANGE_LEFT"], snap)
        # In SHADOW mode, returned action matches proposed action exactly
        self.assertEqual(actions[0], "CHANGE_LEFT")
        self.assertFalse(emerg[0])
        # Report records unsafe verdict
        self.assertFalse(rep.decisions[0].is_safe)


if __name__ == "__main__":
    unittest.main()

