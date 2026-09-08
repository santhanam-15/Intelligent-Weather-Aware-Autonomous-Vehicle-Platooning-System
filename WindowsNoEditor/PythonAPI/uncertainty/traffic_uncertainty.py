from __future__ import annotations

"""Traffic jam generation module for CARLA 0.9.9 simulator.

This module provides the TrafficJamManager class to simulate localized traffic
congestion (jams) by spawning NPC vehicles with reduced target speeds and tight
following distances via CARLA's Traffic Manager.
"""

from enum import Enum
import logging
import math
import random
from typing import Any, Dict, List, Optional, Union

import carla

logger = logging.getLogger(__name__)


class JamSeverity(Enum):
    """Severity levels for traffic jams."""

    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"


# Mapping of JamSeverity to CARLA TrafficManager speed percentage reduction.
# tm.set_percentage_speed_difference(vehicle, pct):
#   LIGHT:  60 -> 40% of normal speed
#   MEDIUM: 80 -> 20% of normal speed
#   HEAVY:  95 -> 5% of normal speed
SEVERITY_SPEED_PERCENTAGES: Dict[JamSeverity, float] = {
    JamSeverity.LIGHT: 60.0,
    JamSeverity.MEDIUM: 80.0,
    JamSeverity.HEAVY: 95.0,
}


class TrafficJamManager:
    """Manages generation, monitoring, and cleanup of traffic jams in CARLA 0.9.9.

    Attributes:
        _seed (Optional[int]): Random seed for reproducible generation.
        _rng (random.Random): Random number generator instance.
        _active_jams (List[Dict[str, Any]]): List of active jam metadata dicts.
        _spawned_actors (List[carla.Actor]): List of spawned CARLA actors for cleanup.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize the TrafficJamManager.

        Args:
            seed: Optional integer seed for reproducible random behavior.
        """
        self._seed: Optional[int] = seed
        self._rng: random.Random = random.Random(seed)
        self._active_jams: List[Dict[str, Any]] = []
        self._spawned_actors: List[carla.Actor] = []

    def trigger_jam(
        self,
        world: carla.World,
        traffic_manager: carla.TrafficManager,
        location: Optional[Union[carla.Location, carla.Transform]] = None,
        severity: JamSeverity = JamSeverity.MEDIUM,
        duration: float = 30.0,
        num_vehicles: int = 5,
        start_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Spawn NPC vehicles near a location to simulate a traffic jam.

        NPC vehicles are set to autopilot via Traffic Manager with significantly
        reduced speed and small following distance.

        Args:
            world: CARLA World instance.
            traffic_manager: CARLA TrafficManager instance.
            location: Reference location or transform near which to spawn vehicles.
                If None, a random spawn point from the map is selected.
            severity: JamSeverity level (LIGHT, MEDIUM, HEAVY). Defaults to MEDIUM.
            duration: Duration of the traffic jam in seconds. Defaults to 30.0.
            num_vehicles: Number of vehicles to spawn. Defaults to 5.
            start_time: Optional simulation start time in seconds. If None,
                retrieved from world snapshot elapsed seconds (or 0.0).

        Returns:
            Dictionary containing jam metadata ('start_time', 'duration',
            'severity', 'vehicles', 'location').
        """
        if isinstance(severity, str):
            try:
                severity = JamSeverity(severity.lower())
            except ValueError:
                try:
                    severity = JamSeverity[severity.upper()]
                except KeyError:
                    logger.warning(
                        "Unrecognized severity '%s', defaulting to JamSeverity.MEDIUM",
                        severity,
                    )
                    severity = JamSeverity.MEDIUM

        if location is None:
            center_transform = self._get_random_spawn_location(world)
            center_location = center_transform.location
            target_reference = center_transform
        else:
            if isinstance(location, carla.Transform):
                center_location = location.location
            elif hasattr(location, "location"):
                center_location = location.location
            else:
                center_location = location
            target_reference = location

        if start_time is None:
            try:
                start_time = float(world.get_snapshot().timestamp.elapsed_seconds)
            except Exception:
                start_time = 0.0

        spawn_points = self._get_nearby_spawn_points(
            world=world,
            location=target_reference,
            num=num_vehicles,
            radius=50.0,
        )

        blueprints = list(world.get_blueprint_library().filter("vehicle.*"))
        if not blueprints:
            logger.error("No vehicle blueprints found in world blueprint library.")
            jam_info: Dict[str, Any] = {
                "start_time": float(start_time),
                "duration": float(duration),
                "severity": severity,
                "vehicles": [],
                "location": center_location,
            }
            self._active_jams.append(jam_info)
            return jam_info

        speed_pct = SEVERITY_SPEED_PERCENTAGES.get(severity, 80.0)
        tm_port = traffic_manager.get_port()
        spawned_vehicles: List[carla.Vehicle] = []

        for transform in spawn_points:
            bp = self._rng.choice(blueprints)
            if bp.has_attribute("color"):
                recommended = bp.get_attribute("color").recommended_values
                if recommended:
                    bp.set_attribute("color", self._rng.choice(recommended))

            vehicle = world.try_spawn_actor(bp, transform)
            if vehicle is not None:
                try:
                    vehicle.set_autopilot(True, tm_port)
                    if hasattr(traffic_manager, "vehicle_percentage_speed_difference"):
                        traffic_manager.vehicle_percentage_speed_difference(vehicle, float(speed_pct))
                    elif hasattr(traffic_manager, "set_percentage_speed_difference"):
                        traffic_manager.set_percentage_speed_difference(vehicle, float(speed_pct))
                    traffic_manager.distance_to_leading_vehicle(vehicle, 1.0)
                except Exception as e:
                    logger.warning(
                        "Error configuring Traffic Manager for vehicle %s: %s",
                        getattr(vehicle, "id", None),
                        e,
                    )
                spawned_vehicles.append(vehicle)
                self._spawned_actors.append(vehicle)

        jam_info = {
            "start_time": float(start_time),
            "duration": float(duration),
            "severity": severity,
            "vehicles": spawned_vehicles,
            "location": center_location,
        }
        self._active_jams.append(jam_info)
        logger.info(
            "Triggered %s traffic jam with %d vehicles at %s (duration=%.1fs)",
            severity.value if hasattr(severity, "value") else severity,
            len(spawned_vehicles),
            center_location,
            duration,
        )
        return jam_info

    def maybe_trigger_jam(
        self,
        world: carla.World,
        traffic_manager: carla.TrafficManager,
        current_time: float,
        probability: float = 0.1,
        severity: Optional[JamSeverity] = None,
        duration: Optional[float] = None,
        num_vehicles: Optional[int] = None,
        location: Optional[Union[carla.Location, carla.Transform]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Probabilistically trigger a traffic jam.

        Rolls against the given probability. If triggered, samples unspecified
        parameters randomly using the configured seed.

        Args:
            world: CARLA World instance.
            traffic_manager: CARLA TrafficManager instance.
            current_time: Current simulation time in seconds.
            probability: Probability in range [0.0, 1.0] of triggering a jam.
            severity: Optional JamSeverity. If None, chosen randomly.
            duration: Optional duration in seconds. If None, chosen randomly (20.0 - 60.0s).
            num_vehicles: Optional vehicle count. If None, chosen randomly (3 - 8 vehicles).
            location: Optional spawn reference location. If None, picked randomly.

        Returns:
            Jam info dict if a jam was triggered, otherwise None.
        """
        if self._rng.random() >= probability:
            return None

        if severity is None:
            severity = self._rng.choice(list(JamSeverity))

        if duration is None:
            duration = round(self._rng.uniform(20.0, 60.0), 1)

        if num_vehicles is None:
            num_vehicles = self._rng.randint(3, 8)

        return self.trigger_jam(
            world=world,
            traffic_manager=traffic_manager,
            location=location,
            severity=severity,
            duration=duration,
            num_vehicles=num_vehicles,
            start_time=current_time,
        )

    def update(self, current_time: float) -> List[Dict[str, Any]]:
        """Check active jams and clean up expired ones.

        Removes jams where current_time > start_time + duration and destroys
        their spawned vehicles.

        Args:
            current_time: Current simulation time in seconds.

        Returns:
            List of expired jam info dictionaries that were removed.
        """
        expired_jams: List[Dict[str, Any]] = []
        remaining_jams: List[Dict[str, Any]] = []

        for jam in self._active_jams:
            start_time = jam.get("start_time", 0.0)
            duration = jam.get("duration", 0.0)
            if current_time > (start_time + duration):
                expired_jams.append(jam)
            else:
                remaining_jams.append(jam)

        for jam in expired_jams:
            for vehicle in jam.get("vehicles", []):
                try:
                    if vehicle.is_alive:
                        vehicle.destroy()
                except Exception as e:
                    logger.debug(
                        "Vehicle %s already destroyed or error during removal: %s",
                        getattr(vehicle, "id", None),
                        e,
                    )
                if vehicle in self._spawned_actors:
                    self._spawned_actors.remove(vehicle)

        self._active_jams = remaining_jams
        return expired_jams

    def cleanup(self) -> None:
        """Destroy all spawned actors and clear active jams."""
        for actor in self._spawned_actors:
            try:
                if actor.is_alive:
                    actor.destroy()
            except Exception as e:
                logger.debug(
                    "Actor %s already destroyed or error during cleanup: %s",
                    getattr(actor, "id", None),
                    e,
                )
        self._spawned_actors.clear()
        self._active_jams.clear()

    def get_active_jams(self) -> List[Dict[str, Any]]:
        """Return a copy of active jams info.

        Returns:
            A list of dictionary copies representing active jams.
        """
        return [dict(jam) for jam in self._active_jams]

    def _get_random_spawn_location(self, world: carla.World) -> carla.Transform:
        """Get a random spawn point from the world map.

        Args:
            world: CARLA World instance.

        Returns:
            A randomly chosen carla.Transform spawn point.

        Raises:
            RuntimeError: If no spawn points are available in the world map.
        """
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            logger.warning("No spawn points found in world map.")
            raise RuntimeError("No spawn points available in world map.")
        return self._rng.choice(spawn_points)

    def _get_nearby_spawn_points(
        self,
        world: carla.World,
        location: Union[carla.Location, carla.Transform],
        num: int,
        radius: float = 50.0,
    ) -> List[carla.Transform]:
        """Get spawn points within radius of location. If not enough, use closest ones available.

        Args:
            world: CARLA World instance.
            location: Reference location or transform to search around.
            num: Maximum number of spawn points to return.
            radius: Search radius in meters.

        Returns:
            List of up to `num` carla.Transform spawn points.
        """
        if num <= 0:
            return []

        if isinstance(location, carla.Transform):
            target_loc = location.location
        elif hasattr(location, "location"):
            target_loc = location.location
        else:
            target_loc = location

        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            logger.warning("No spawn points found in world map.")
            return []

        points_with_dist = []
        for sp in spawn_points:
            dx = sp.location.x - target_loc.x
            dy = sp.location.y - target_loc.y
            dz = sp.location.z - target_loc.z
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            points_with_dist.append((dist, sp))

        # Sort strictly by distance to avoid comparing carla.Transform instances
        points_with_dist.sort(key=lambda item: item[0])

        within_radius = [sp for dist, sp in points_with_dist if dist <= radius]
        if len(within_radius) >= num:
            return within_radius[:num]

        return [sp for dist, sp in points_with_dist[:num]]
