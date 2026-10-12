"""CARLA-free deterministic tests for Module 2 control invariants.

Run with the CARLA-supported interpreter:
``py -3.7 -m unittest Module2.tests.test_module2_logic``
"""

from __future__ import print_function

import math
import os
import sys
import unittest

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PYTHON_API = os.path.join(ROOT, "PythonAPI")
for path in (ROOT, PYTHON_API):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import tempfile
import torch

from Module2.action.action_manager import ActionManager
from Module2.mappo.trainer import MAPPOTrainer
from Module2.mission_controller import MissionPlatoonController, RoutePlanner, _angle_error_deg
from Module2.module2_manager import Module2Manager
from Module2.reward.reward_manager import RewardManager


class FakeLocation(object):
    def __init__(self, x, y=0.0):
        self.x, self.y, self.z = x, y, 0.0

    def distance(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)


class FakeRotation(object):
    def __init__(self, yaw=0.0):
        self.yaw = yaw


class FakeTransform(object):
    def __init__(self, x, yaw=0.0, y=0.0):
        self.location = FakeLocation(x, y)
        self.rotation = FakeRotation(yaw)


class FakeWaypoint(object):
    def __init__(self, index, yaw=0.0, road=1, lane=1, x=None, y=0.0):
        self.index = index
        self.road_id, self.lane_id = road, lane
        pos_x = float(index) if x is None else float(x)
        self.transform = FakeTransform(pos_x, yaw, y=float(y))
        self._next = []

    def next(self, _spacing):
        return list(self._next)


class FakeMap(object):
    def __init__(self, waypoints):
        self.waypoints = waypoints

    def get_waypoint(self, _origin, **_kwargs):
        return self.waypoints[0]


class FakeVehicle(object):
    def __init__(self, vehicle_id, x, y=0.0):
        self.id = vehicle_id
        self._location = FakeLocation(x, y)

    def get_location(self):
        return self._location


class Module2LogicTests(unittest.TestCase):
    def test_action_persistence_blocks_one_tick_chatter(self):
        manager = ActionManager(2)
        self.assertEqual(manager.process_actions([0, 0], 0.0)[0], ["CRUISE", "CRUISE"])
        self.assertEqual(manager.process_actions([1, 2], 0.1)[0], ["CRUISE", "CRUISE"])
        self.assertEqual(manager.process_actions([1, 2], 0.2)[0], ["ACCELERATE", "DECELERATE"])

    def test_follower_lane_actions_are_masked(self):
        manager = ActionManager(2)
        actions, _ = manager.process_actions([3, 4], 0.0)
        self.assertEqual(actions, ["CHANGE_LEFT", "CRUISE"])

    def test_route_is_preplanned_and_vehicle_cursors_do_not_rewind(self):
        waypoints = [FakeWaypoint(i) for i in range(6)]
        for current, following in zip(waypoints, waypoints[1:]):
            current._next = [following]
        planner = RoutePlanner(FakeMap(waypoints), FakeLocation(0), distance_m=5.0, spacing_m=1.0)
        vehicle = FakeVehicle(4, 3.1)
        self.assertGreaterEqual(planner.progress_for(vehicle), 3)
        vehicle._location = FakeLocation(1.0)
        self.assertGreaterEqual(planner.progress_for(vehicle), 3)
        self.assertEqual(planner.destination.x, 5.0)

    def test_stationary_vehicle_cannot_jump_to_self_near_route_section(self):
        # Route points 0 and 5 occupy nearly the same space, modelling a
        # road loop/crossing.  A stationary CAV must remain at cursor zero.
        waypoints = [FakeWaypoint(i) for i in range(6)]
        waypoints[4].transform.location = FakeLocation(0.1)
        for current, following in zip(waypoints, waypoints[1:]):
            current._next = [following]
        planner = RoutePlanner(FakeMap(waypoints), FakeLocation(0), distance_m=5.0, spacing_m=1.0)
        vehicle = FakeVehicle(9, 0.0)
        for _ in range(10):
            self.assertEqual(planner.progress_for(vehicle), 0)

    def test_weather_headway_increases_when_friction_drops(self):
        rewards = RewardManager(None)
        dry = rewards.calculate_desired_headway(12.0, 0.0, 1.0)
        wet = rewards.calculate_desired_headway(12.0, 0.0, 0.4)
        self.assertGreater(wet, dry)

    def test_controller_headway_is_speed_and_friction_aware(self):
        controller = MissionPlatoonController.__new__(MissionPlatoonController)
        controller.friction_provider = lambda _vehicle: 1.0
        vehicle = type("Vehicle", (), {"get_velocity": lambda self: type("V", (), {"x": 15.0, "y": 0.0, "z": 0.0})()})()
        dry = controller._desired_headway(vehicle)
        controller.friction_provider = lambda _vehicle: 0.4
        wet = controller._desired_headway(vehicle)
        self.assertGreater(wet, dry)

    def test_route_angle_wrap_is_bounded(self):
        self.assertAlmostEqual(_angle_error_deg(179.0, -179.0), 2.0)
        self.assertAlmostEqual(_angle_error_deg(-179.0, 179.0), -2.0)

    def test_carla_lateral_convention_requires_negative_left_steer(self):
        # CARLA's keyboard controller maps left to decreasing steer (< 0).
        # In CARLA UE4 coordinates, a left-side waypoint has negative local_y.
        # Steer towards a left-side waypoint must therefore yield < 0.
        left_y, local_x = -2.0, 10.0
        alpha_left = math.atan2(left_y, local_x)
        steer_left = math.atan2(2.7 * math.sin(alpha_left), math.hypot(local_x, left_y)) * 1.8
        self.assertLess(steer_left, 0.0)

        # A right-side waypoint (positive local_y) must yield > 0.
        right_y = 2.0
        alpha_right = math.atan2(right_y, local_x)
        steer_right = math.atan2(2.7 * math.sin(alpha_right), math.hypot(local_x, right_y)) * 1.8
        self.assertGreater(steer_right, 0.0)

    def test_shared_actor_is_batched_and_follower_actions_are_masked(self):
        trainer = MAPPOTrainer(obs_dim=32, action_dim=5, global_state_dim=128)
        actions, log_probs = trainer.select_action(np.zeros((4, 32), dtype=np.float32))
        self.assertEqual(actions.shape, (4,))
        self.assertEqual(log_probs.shape, (4,))
        self.assertTrue(np.all(actions[1:] < 3))

    def test_vehicle_at_route_start_does_not_become_destination_complete(self):
        waypoints = [FakeWaypoint(i, x=float(i * 2.0)) for i in range(76)]
        for current, following in zip(waypoints, waypoints[1:]):
            current._next = [following]
        planner = RoutePlanner(FakeMap(waypoints), FakeLocation(0.0), distance_m=150.0, spacing_m=2.0)
        vehicle = FakeVehicle(101, 0.0, 0.0)

        # Vehicle at spawn must start at cursor 0 and progress 0.0
        cursor = planner.progress_for(vehicle)
        self.assertEqual(cursor, 0)
        self.assertAlmostEqual(planner.route_progress(vehicle), 0.0)
        self.assertLess(cursor, len(waypoints) - 3)

        # Repeated ticks at spawn must remain at cursor 0
        for _ in range(50):
            self.assertEqual(planner.progress_for(vehicle), 0)
            self.assertAlmostEqual(planner.route_progress(vehicle), 0.0)

    def test_cursor_advances_sequentially_and_never_rewinds(self):
        waypoints = [FakeWaypoint(i, x=float(i * 2.0)) for i in range(10)]
        for current, following in zip(waypoints, waypoints[1:]):
            current._next = [following]
        planner = RoutePlanner(FakeMap(waypoints), FakeLocation(0.0), distance_m=20.0, spacing_m=2.0)
        vehicle = FakeVehicle(102, 0.0, 0.0)

        # Spawn at 0.0 -> cursor 0
        self.assertEqual(planner.progress_for(vehicle), 0)

        # Move forward slightly (0.8m) -> before midpoint of wp0->wp1 (1.0m) -> cursor 0
        vehicle._location = FakeLocation(0.8, 0.0)
        self.assertEqual(planner.progress_for(vehicle), 0)

        # Past midpoint of wp0->wp1 (1.2m) -> cursor advances to 1
        vehicle._location = FakeLocation(1.2, 0.0)
        self.assertEqual(planner.progress_for(vehicle), 1)

        # In segment 1->2 (2.0m to 4.0m): at 2.5m -> before midpoint (3.0m) -> cursor 1
        vehicle._location = FakeLocation(2.5, 0.0)
        self.assertEqual(planner.progress_for(vehicle), 1)

        # Past midpoint of wp1->wp2 (3.2m) -> cursor advances to 2
        vehicle._location = FakeLocation(3.2, 0.0)
        self.assertEqual(planner.progress_for(vehicle), 2)

        # Past midpoint of wp2->wp3 (5.2m) -> cursor advances to 3
        vehicle._location = FakeLocation(5.2, 0.0)
        self.assertEqual(planner.progress_for(vehicle), 3)

        # Physical rewind/backtrack must NEVER decrease cursor
        vehicle._location = FakeLocation(0.0, 0.0)
        self.assertEqual(planner.progress_for(vehicle), 3)

    def test_cursor_cannot_jump_from_zero_to_final_waypoint_in_one_call(self):
        waypoints = [FakeWaypoint(i, x=float(i * 2.0)) for i in range(76)]
        for current, following in zip(waypoints, waypoints[1:]):
            current._next = [following]
        planner = RoutePlanner(FakeMap(waypoints), FakeLocation(0.0), distance_m=150.0, spacing_m=2.0)
        vehicle = FakeVehicle(103, 150.0, 0.0)

        # Even if vehicle is placed at the route end, a vehicle starting at cursor 0
        # cannot jump to waypoint 75 in a single call (bounded and corridor-checked).
        cursor = planner.progress_for(vehicle)
        self.assertLess(cursor, len(waypoints) - 3)
        self.assertLessEqual(cursor, 3)

    def test_geographically_close_future_loop_does_not_skip_ordered_route(self):
        # Build 76 waypoints where waypoint 70 loops back right next to waypoint 0
        waypoints = [FakeWaypoint(i, x=float(i * 2.0)) for i in range(76)]
        waypoints[70].transform.location = FakeLocation(0.1, 0.05)
        waypoints[71].transform.location = FakeLocation(0.2, 0.05)
        for current, following in zip(waypoints, waypoints[1:]):
            current._next = [following]
        planner = RoutePlanner(FakeMap(waypoints), FakeLocation(0.0), distance_m=150.0, spacing_m=2.0)
        vehicle = FakeVehicle(104, 0.0, 0.0)

        # Vehicle at spawn must NOT jump to geographically close future waypoints 70/71
        for _ in range(25):
            self.assertEqual(planner.progress_for(vehicle), 0)
            self.assertAlmostEqual(planner.route_progress(vehicle), 0.0)

    def test_turns_and_intersections_advance_smoothly_without_uncontrolled_jump(self):
        # Route goes East: (0,0), (2,0), (4,0), (6,0)
        # Then turns North at x=6: (6,2), (6,4), (6,6), (6,8)
        pts = [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0), (6.0, 0.0),
               (6.0, 2.0), (6.0, 4.0), (6.0, 6.0), (6.0, 8.0)]
        waypoints = [FakeWaypoint(i, x=p[0], y=p[1]) for i, p in enumerate(pts)]
        for current, following in zip(waypoints, waypoints[1:]):
            current._next = [following]
        planner = RoutePlanner(FakeMap(waypoints), FakeLocation(0.0, 0.0), distance_m=16.0, spacing_m=2.0)
        vehicle = FakeVehicle(105, 0.0, 0.0)

        # Step along East section
        self.assertEqual(planner.progress_for(vehicle), 0)
        vehicle._location = FakeLocation(1.2, 0.0)
        self.assertEqual(planner.progress_for(vehicle), 1)
        vehicle._location = FakeLocation(3.2, 0.0)
        self.assertEqual(planner.progress_for(vehicle), 2)
        vehicle._location = FakeLocation(5.2, 0.0)
        self.assertEqual(planner.progress_for(vehicle), 3)

        # Step around the 90-degree corner towards (6, 2)
        vehicle._location = FakeLocation(6.0, 1.2)
        self.assertEqual(planner.progress_for(vehicle), 4)

        # Step along North section
        vehicle._location = FakeLocation(6.0, 3.2)
        self.assertEqual(planner.progress_for(vehicle), 5)
        vehicle._location = FakeLocation(6.0, 5.2)
        self.assertEqual(planner.progress_for(vehicle), 6)

    def test_destination_complete_only_near_actual_end_of_route(self):
        waypoints = [FakeWaypoint(i, x=float(i * 2.0)) for i in range(76)]
        for current, following in zip(waypoints, waypoints[1:]):
            current._next = [following]
        planner = RoutePlanner(FakeMap(waypoints), FakeLocation(0.0), distance_m=150.0, spacing_m=2.0)
        vehicle = FakeVehicle(106, 0.0, 0.0)
        completion_threshold = len(waypoints) - 3  # 73

        # Drive vehicle along the route
        for i in range(72):
            vehicle._location = FakeLocation(float(i * 2.0 + 1.2), 0.0)
            cursor = planner.progress_for(vehicle)
            # Before waypoint 73, cannot trigger destination complete
            self.assertLess(cursor, completion_threshold)

        # Now reach waypoint 73 section (x = 73 * 2.0 + 1.2 = 147.2m)
        vehicle._location = FakeLocation(147.2, 0.0)
        cursor = planner.progress_for(vehicle)
        self.assertGreaterEqual(cursor, completion_threshold)
        self.assertGreaterEqual(planner.route_progress(vehicle), 0.95)

    def test_deterministic_action_selection(self):
        trainer = MAPPOTrainer(obs_dim=32, action_dim=5, global_state_dim=128)
        dummy_obs = np.ones((4, 32), dtype=np.float32)

        # Deterministic selection must yield identical actions every run
        det_actions_1, _ = trainer.select_action(dummy_obs, deterministic=True)
        det_actions_2, _ = trainer.select_action(dummy_obs, deterministic=True)
        np.testing.assert_array_equal(det_actions_1, det_actions_2)

        # Followers must still have their actions masked (0, 1, or 2)
        self.assertTrue(np.all(det_actions_1[1:] < 3))

    def test_checkpoint_dimension_mismatch_fails_safely(self):
        trainer = MAPPOTrainer(obs_dim=32, action_dim=5, global_state_dim=128)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            fake_ckpt_path = f.name
        try:
            # Create a 14-dim checkpoint (historical incompatible architecture)
            fake_checkpoint = {
                "actor": {"net.0.weight": torch.randn(64, 14), "net.0.bias": torch.randn(64)},
                "critic": {"net.0.weight": torch.randn(128, 56), "net.0.bias": torch.randn(128)},
                "actor_opt": {},
                "critic_opt": {},
                "update_step": 42,
            }
            torch.save(fake_checkpoint, fake_ckpt_path)

            # Loading must return False and NOT corrupt current 32-dim weights
            loaded = trainer.load_checkpoint(fake_ckpt_path)
            self.assertFalse(loaded)
            self.assertEqual(trainer.actor.net[0].in_features, 32)
        finally:
            if os.path.exists(fake_ckpt_path):
                os.remove(fake_ckpt_path)

    def test_checkpoint_save_and_load_roundtrip(self):
        trainer1 = MAPPOTrainer(obs_dim=32, action_dim=5, global_state_dim=128)
        trainer1.update_step = 17
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            trainer1.save_checkpoint(ckpt_path)

            trainer2 = MAPPOTrainer(obs_dim=32, action_dim=5, global_state_dim=128)
            loaded = trainer2.load_checkpoint(ckpt_path)
            self.assertTrue(loaded)
            self.assertEqual(trainer2.update_step, 17)
            # Verify weight equality
            p1 = list(trainer1.actor.parameters())[0]
            p2 = list(trainer2.actor.parameters())[0]
            self.assertTrue(torch.allclose(p1, p2))
        finally:
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)

    def test_eval_mode_requires_checkpoint_or_raises(self):
        # Point to a nonexistent checkpoint path in evaluate mode
        class DummyUncertainty(object):
            pass
        with tempfile.TemporaryDirectory() as tmpdir:
            from Module2 import config
            orig_dir = config.CHECKPOINT_DIR
            orig_path = MAPPOTrainer.CKPT_PATH
            try:
                config.CHECKPOINT_DIR = tmpdir
                MAPPOTrainer.CKPT_PATH = "nonexistent_ckpt.pt"
                with self.assertRaises(RuntimeError):
                    Module2Manager(DummyUncertainty(), num_vehicles=2, mode="evaluate")
            finally:
                config.CHECKPOINT_DIR = orig_dir
                MAPPOTrainer.CKPT_PATH = orig_path

    def test_tactical_actions_modulate_target_speed_and_control(self):
        class DummyController(object):
            cruise_speed_mps = 20.0 / 3.6
            state = "NORMAL"
            intruded_between = None
            def _friction(self, _v):
                return 1.0
            def _desired_headway(self, _v, _p):
                return 10.0
            _target_speed = MissionPlatoonController._target_speed

        dc = DummyController()
        spd_cruise = dc._target_speed(None, 0, "CRUISE", None, None, 50.0)
        spd_accel = dc._target_speed(None, 0, "ACCELERATE", None, None, 50.0)
        self.assertAlmostEqual(spd_accel - spd_cruise, 1.1, places=2)

        spd_decel = dc._target_speed(None, 0, "DECELERATE", None, None, 50.0)
        self.assertAlmostEqual(spd_decel - spd_cruise, -1.8, places=2)

        spd_brake = dc._target_speed(None, 0, "BRAKE", None, None, 50.0)
        self.assertAlmostEqual(spd_brake, 0.0, places=2)

    def test_route_dynamic_extension_appends_connected_waypoints(self):
        waypoints = [FakeWaypoint(i, x=float(i * 2.0)) for i in range(30)]
        for current, following in zip(waypoints, waypoints[1:]):
            current._next = [following]
        planner = RoutePlanner(FakeMap(waypoints), FakeLocation(0.0), distance_m=10.0, spacing_m=2.0)
        initial_count = len(planner.route)
        self.assertLess(initial_count, 15)

        # Extend route by another 20 meters
        added = planner.extend_route(distance_m=20.0)
        self.assertGreater(added, 5)
        self.assertEqual(len(planner.route), initial_count + added)
        self.assertEqual(planner.destination.x, planner.route[-1].transform.location.x)

    def test_traffic_light_ignores_light_behind_vehicle(self):
        class DummyMap(object):
            def get_waypoint(self, _loc, project_to_road=True):
                class DummyWP:
                    is_junction = False
                return DummyWP()

        class DummyTrafficLight(object):
            id = 501
            def get_location(self):
                return FakeLocation(5.0, 0.0)

        class DummyVehicle(object):
            id = 1
            def get_location(self):
                return FakeLocation(10.0, 0.0)  # Vehicle at x=10, light at x=5 (behind)
            def get_transform(self):
                class DummyTransform:
                    location = FakeLocation(10.0, 0.0)
                    def get_forward_vector(self):
                        class Vec:
                            x = 1.0; y = 0.0; z = 0.0
                        return Vec()
                return DummyTransform()
            def get_traffic_light(self):
                return DummyTrafficLight()
            def get_traffic_light_state(self):
                import carla
                return carla.TrafficLightState.Red
            def is_at_traffic_light(self):
                return False

        controller = MissionPlatoonController.__new__(MissionPlatoonController)
        controller.map = DummyMap()
        hard_stop, dist, label = controller._traffic_light_constraint(DummyVehicle())
        self.assertFalse(hard_stop)
        self.assertEqual(label, "CLEARED_LIGHT")

    def test_traffic_light_ignores_light_inside_junction(self):
        class DummyMap(object):
            def get_waypoint(self, _loc, project_to_road=True):
                class DummyWP:
                    is_junction = True  # Vehicle is inside an intersection
                return DummyWP()

        class DummyTrafficLight(object):
            id = 502
            def get_location(self):
                return FakeLocation(20.0, 0.0)

        class DummyVehicle(object):
            id = 1
            def get_location(self):
                return FakeLocation(10.0, 0.0)
            def get_transform(self):
                class DummyTransform:
                    location = FakeLocation(10.0, 0.0)
                    def get_forward_vector(self):
                        class Vec:
                            x = 1.0; y = 0.0; z = 0.0
                        return Vec()
                return DummyTransform()
            def get_traffic_light(self):
                return DummyTrafficLight()
            def get_traffic_light_state(self):
                import carla
                return carla.TrafficLightState.Red
            def is_at_traffic_light(self):
                return False

        controller = MissionPlatoonController.__new__(MissionPlatoonController)
        controller.map = DummyMap()
        hard_stop, dist, label = controller._traffic_light_constraint(DummyVehicle())
        self.assertFalse(hard_stop)
        self.assertEqual(label, "INSIDE_JUNCTION")

    def test_obstacle_corridor_ignores_adjacent_lane_traffic(self):
        class DummyActor(object):
            def __init__(self, aid, x, y):
                self.id = aid
                self.is_alive = True
                self._x = x
                self._y = y
            def get_location(self):
                return FakeLocation(self._x, self._y)
            def get_transform(self):
                class DummyForward:
                    def __init__(self):
                        self.x, self.y, self.z = 1.0, 0.0, 0.0
                class DummyTf:
                    def __init__(self, loc):
                        self.location = loc
                    def get_forward_vector(self):
                        return DummyForward()
                return DummyTf(self.get_location())

        class DummyWorld(object):
            def __init__(self, actors):
                self._actors = actors
            def get_actors(self):
                class ActorFilter:
                    def __init__(self, acts):
                        self.acts = acts
                    def filter(self, pattern):
                        return self.acts
                return ActorFilter(self._actors)

        ego = DummyActor(1, 0.0, 0.0)
        # Adjacent lane vehicle at y=2.8m (adjacent driving lane, not in ego lane)
        adjacent_vehicle = DummyActor(10, 15.0, 2.8)
        # Ego platoon
        controller = MissionPlatoonController.__new__(MissionPlatoonController)
        controller.platoon = [ego]
        controller.world = DummyWorld([ego, adjacent_vehicle])

        # Adjacent vehicle should NOT be detected as in-lane obstacle
        best, clearance = controller._closest_forward_obstacle(ego)
        self.assertIsNone(best)
        self.assertEqual(clearance, 80.0)

        # In-lane vehicle at y=0.3m (directly blocking our lane)
        in_lane_vehicle = DummyActor(20, 15.0, 0.3)
        controller.world = DummyWorld([ego, in_lane_vehicle])
        best, clearance = controller._closest_forward_obstacle(ego)
        self.assertIsNotNone(best)
        self.assertEqual(best.id, 20)
        self.assertLess(clearance, 15.0)


if __name__ == "__main__":
    unittest.main()

