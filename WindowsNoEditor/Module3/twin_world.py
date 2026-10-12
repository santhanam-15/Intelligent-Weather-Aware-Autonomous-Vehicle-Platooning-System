# WindowsNoEditor/Module3/twin_world.py
"""World B — Isolated CPU-Side Virtual Prediction Engine.

Does NOT mutate the CARLA simulation, spawn actors, or advance CARLA ticks.
Projects discrete vehicle trajectories forward in time over a configurable horizon (default 2.0s @ 0.1s dt).
Uses bounded kinematic models accounting for candidate tactical actions, acceleration/braking limits,
and obstacles under uncertainty.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple
from Module3.config import (
    PREDICTION_HORIZON_SEC,
    PREDICTION_DT_SEC,
    ACCEL_RATE_MPS2,
    DECEL_RATE_MPS2,
    CRUISE_RATE_MPS2,
)
from Module3.snapshot import Vector2D, VehicleSnapshot, ObstacleSnapshot, WorldSnapshot


class PredictedState:
    """State of an entity at a discrete future timestep."""
    __slots__ = ("time_offset", "position", "speed", "heading_rad")

    def __init__(self, time_offset: float, position: Vector2D, speed: float, heading_rad: float):
        self.time_offset = time_offset
        self.position = position
        self.speed = speed
        self.heading_rad = heading_rad

    def __repr__(self) -> str:
        return f"t=+{self.time_offset:.1f}s: pos={self.position}, spd={self.speed:.1f}m/s"


class TwinWorld:
    """CPU-side virtual forward projection engine."""

    def __init__(
        self,
        horizon_sec: float = PREDICTION_HORIZON_SEC,
        dt_sec: float = PREDICTION_DT_SEC,
    ):
        self.horizon_sec = max(0.5, float(horizon_sec))
        self.dt_sec = max(0.02, float(dt_sec))
        self.steps = int(round(self.horizon_sec / self.dt_sec))

    def _action_acceleration(self, action: str, current_speed: float, max_speed: float = 12.0) -> float:
        """Map candidate tactical action to forward acceleration target."""
        if action == "ACCELERATE":
            return ACCEL_RATE_MPS2 if current_speed < max_speed else 0.0
        elif action == "DECELERATE":
            return DECEL_RATE_MPS2
        elif action == "CRUISE":
            return CRUISE_RATE_MPS2
        elif action in ("CHANGE_LEFT", "CHANGE_RIGHT"):
            # Speed held steady during lateral transition
            return 0.0
        return 0.0

    def predict_platoon(
        self,
        platoon_snapshots: Tuple[VehicleSnapshot, ...],
        candidate_actions: List[str],
    ) -> Dict[int, List[PredictedState]]:
        """Project each platoon vehicle forward in time given candidate tactical actions."""
        trajectories: Dict[int, List[PredictedState]] = {}

        for idx, veh in enumerate(platoon_snapshots):
            if not veh.is_alive:
                trajectories[veh.id] = []
                continue

            action = candidate_actions[idx] if idx < len(candidate_actions) else "CRUISE"
            traj: List[PredictedState] = []
            
            curr_pos = Vector2D(veh.position.x, veh.position.y)
            curr_spd = veh.speed
            curr_heading = veh.heading_rad

            # Approximate lateral displacement rate for lane change actions (e.g. 1.2 m/s lateral)
            lat_rate = 0.0
            if action == "CHANGE_LEFT":
                lat_rate = 1.0  # lateral left offset
            elif action == "CHANGE_RIGHT":
                lat_rate = -1.0 # lateral right offset

            for step in range(1, self.steps + 1):
                t_offset = step * self.dt_sec
                accel = self._action_acceleration(action, curr_spd)
                
                # Kinematic update
                new_spd = max(0.0, curr_spd + accel * self.dt_sec)
                avg_spd = 0.5 * (curr_spd + new_spd)
                ds = avg_spd * self.dt_sec

                # Forward displacement
                dx = ds * math.cos(curr_heading)
                dy = ds * math.sin(curr_heading)

                # Lateral shift if lane-changing
                if lat_rate != 0.0:
                    d_lat = lat_rate * self.dt_sec
                    # Perpendicular vector (-sin, cos)
                    dx += -d_lat * math.sin(curr_heading)
                    dy += d_lat * math.cos(curr_heading)

                curr_pos = Vector2D(curr_pos.x + dx, curr_pos.y + dy)
                curr_spd = new_spd
                traj.append(PredictedState(t_offset, curr_pos, curr_spd, curr_heading))

            trajectories[veh.id] = traj

        return trajectories

    def predict_obstacles(
        self,
        obstacle_snapshots: Tuple[ObstacleSnapshot, ...],
    ) -> Dict[int, List[PredictedState]]:
        """Project external obstacles forward using constant velocity assumptions with bounded damping."""
        trajectories: Dict[int, List[PredictedState]] = {}

        for obs in obstacle_snapshots:
            traj: List[PredictedState] = []
            curr_pos = Vector2D(obs.position.x, obs.position.y)
            spd = obs.speed
            heading = obs.heading_rad

            for step in range(1, self.steps + 1):
                t_offset = step * self.dt_sec
                ds = spd * self.dt_sec
                dx = ds * math.cos(heading)
                dy = ds * math.sin(heading)
                curr_pos = Vector2D(curr_pos.x + dx, curr_pos.y + dy)
                traj.append(PredictedState(t_offset, curr_pos, spd, heading))

            trajectories[obs.id] = traj

        return trajectories

    def simulate_future(
        self,
        snapshot: WorldSnapshot,
        candidate_actions: List[str],
    ) -> Tuple[Dict[int, List[PredictedState]], Dict[int, List[PredictedState]]]:
        """Runs isolated forward prediction for all vehicles and obstacles without touching CARLA."""
        if not snapshot.is_valid or not snapshot.platoon:
            return {}, {}

        platoon_trajs = self.predict_platoon(snapshot.platoon, candidate_actions)
        obstacle_trajs = self.predict_obstacles(snapshot.obstacles)
        return platoon_trajs, obstacle_trajs

