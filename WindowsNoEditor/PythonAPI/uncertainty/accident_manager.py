from __future__ import annotations

"""Accidents and road obstacles module for CARLA 0.9.9 simulator.

This module provides the AccidentManager class and related data structures to simulate
realistic road incidents such as stopped vehicles, breakdowns, multi-vehicle pileups,
partial or full lane blockages, and road debris in CARLA 0.9.9.
"""

from enum import Enum
import logging
import math
import random
from typing import List, Optional, Union

import carla

logger = logging.getLogger(__name__)


class AccidentType(Enum):
    """Enumeration of accident and road obstacle incident types."""

    STOPPED_VEHICLE = "stopped_vehicle"
    BREAKDOWN = "breakdown"
    MULTI_VEHICLE_ACCIDENT = "multi_vehicle_accident"
    PARTIAL_BLOCK = "partial_block"
    FULL_BLOCK = "full_block"
    DEBRIS = "debris"


class AccidentEvent:
    """Represents an active accident or road obstacle incident.

    Attributes:
        accident_type (AccidentType): Type of incident.
        location (carla.Transform): Reference transform of the incident.
        vehicles (List[carla.Actor]): List of spawned CARLA actors for this incident.
        start_time (float): Simulation start timestamp in seconds.
        duration (float): Lifetime of the incident in seconds.
        severity (str): Severity level ('low', 'medium', 'high').
    """

    def __init__(
        self,
        accident_type: AccidentType,
        location: carla.Transform,
        vehicles: Optional[List[carla.Actor]] = None,
        start_time: float = 0.0,
        duration: float = 60.0,
        severity: str = "medium",
    ) -> None:
        """Initialize an AccidentEvent instance.

        Args:
            accident_type: Type of incident (AccidentType).
            location: carla.Transform representing the incident origin/location.
            vehicles: Optional list of spawned carla.Actor instances.
            start_time: Simulation timestamp (seconds) when incident began.
            duration: Active duration in seconds.
            severity: Severity string ('low', 'medium', 'high').
        """
        self.accident_type: AccidentType = accident_type
        self.location: carla.Transform = location
        self.vehicles: List[carla.Actor] = vehicles if vehicles is not None else []
        self.start_time: float = float(start_time)
        self.duration: float = float(duration)
        self.severity: str = severity

    def is_expired(self, current_time: float) -> bool:
        """Check if the event duration has elapsed.

        Args:
            current_time: Current simulation time in seconds.

        Returns:
            True if current_time >= start_time + duration, False otherwise.
        """
        return current_time >= (self.start_time + self.duration)

    def remaining_time(self, current_time: float) -> float:
        """Get the remaining active duration in seconds.

        Args:
            current_time: Current simulation time in seconds.

        Returns:
            Remaining seconds (0.0 if already expired).
        """
        return max(0.0, (self.start_time + self.duration) - current_time)

    def __repr__(self) -> str:
        """Return developer-friendly string representation of the incident."""
        return (
            f"AccidentEvent(type={self.accident_type.name}, "
            f"severity={self.severity!r}, "
            f"vehicles={len(self.vehicles)}, "
            f"start={self.start_time:.1f}s, duration={self.duration:.1f}s)"
        )


class AccidentManager:
    """Manages accidents, disabled vehicles, and road obstacles in CARLA 0.9.9.

    Spawns static/stopped vehicles with brakes applied and hazard blinkers active
    to model unpredictable real-world blockages without AI/teleportation.

    Attributes:
        _seed (Optional[int]): Random seed used for reproducible sampling.
        _rng (random.Random): Random number generator instance.
        _active_events (List[AccidentEvent]): List of currently active accident events.
        _spawned_actors (List[carla.Actor]): List of all spawned actors managed by this module.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize the AccidentManager.

        Args:
            seed: Optional integer seed for reproducibility.
        """
        self._seed: Optional[int] = seed
        self._rng: random.Random = random.Random(seed)
        self._active_events: List[AccidentEvent] = []
        self._spawned_actors: List[carla.Actor] = []

    def spawn_stopped_vehicle(
        self,
        world: carla.World,
        location: Optional[Union[carla.Location, carla.Transform]] = None,
        duration: float = 60.0,
        start_time: Optional[float] = None,
        severity: str = "low",
        accident_type: AccidentType = AccidentType.STOPPED_VEHICLE,
    ) -> Optional[AccidentEvent]:
        """Spawn a stopped vehicle at location (random spawn point if None).

        DO NOT set autopilot — vehicle just sits there (apply brake via
        VehicleControl with brake=1.0, hand_brake=True).
        Enable hazard lights if possible (try vehicle.set_light_state).

        Args:
            world: CARLA World instance.
            location: Target transform or location. If None, picks a random spawn point.
            duration: Duration in seconds for which the vehicle stays. Defaults to 60.0.
            start_time: Optional simulation timestamp. If None, fetched from world snapshot.
            severity: Severity level ('low', 'medium', 'high'). Defaults to 'low'.
            accident_type: AccidentType enum member. Defaults to STOPPED_VEHICLE.

        Returns:
            AccidentEvent if vehicle was spawned successfully, otherwise None.
        """
        transform = self._resolve_target_transform(world, location)
        if transform is None:
            logger.warning("Cannot spawn stopped vehicle: no valid transform found.")
            return None

        vehicle = self._spawn_single_vehicle(world, transform)
        if vehicle is None:
            return None

        if start_time is None:
            start_time = self._get_current_time(world)

        self._spawned_actors.append(vehicle)
        event = AccidentEvent(
            accident_type=accident_type,
            location=transform,
            vehicles=[vehicle],
            start_time=start_time,
            duration=duration,
            severity=severity,
        )
        self._active_events.append(event)
        logger.info(
            "Spawned %s (id=%d) at (%.1f, %.1f, %.1f) [duration=%.1fs, severity=%s]",
            accident_type.name,
            vehicle.id,
            transform.location.x,
            transform.location.y,
            transform.location.z,
            duration,
            severity,
        )
        return event

    def spawn_accident(
        self,
        world: carla.World,
        location: Optional[Union[carla.Location, carla.Transform]] = None,
        num_vehicles: int = 2,
        duration: float = 60.0,
        start_time: Optional[float] = None,
    ) -> Optional[AccidentEvent]:
        """Spawn multiple stopped vehicles close together (slight offset in location).

        Applies full braking to all vehicles. Autopilot is NOT set.
        Hazard lights are enabled where supported.

        Args:
            world: CARLA World instance.
            location: Reference transform or location. If None, picks a random spawn point.
            num_vehicles: Number of vehicles involved in the collision (default 2).
            duration: Duration in seconds. Defaults to 60.0.
            start_time: Optional simulation timestamp. If None, fetched from world snapshot.

        Returns:
            AccidentEvent if at least one vehicle spawned, otherwise None.
        """
        transform = self._resolve_target_transform(world, location)
        if transform is None:
            logger.warning("Cannot spawn accident: no valid transform found.")
            return None

        if start_time is None:
            start_time = self._get_current_time(world)

        # Offsets for multi-vehicle collision: (longitudinal_offset, lateral_offset, yaw_offset)
        preset_offsets = [
            (0.0, 0.0, 0.0),
            (5.8, 0.7, 25.0),
            (-5.8, -0.7, -15.0),
            (11.5, 1.2, 10.0),
            (-11.5, -1.0, -20.0),
        ]

        spawned_vehicles: List[carla.Actor] = []
        for i in range(num_vehicles):
            if i < len(preset_offsets):
                ox, oy, yaw = preset_offsets[i]
            else:
                sign = 1.0 if (i % 2 == 1) else -1.0
                ox = sign * (6.0 * ((i // 2) + 1))
                oy = 0.5 * (1.0 if i % 2 == 1 else -1.0)
                yaw = self._rng.uniform(-20.0, 20.0)

            v_transform = self._offset_location(
                transform, offset_x=ox, offset_y=oy, yaw_offset=yaw
            )
            vehicle = self._spawn_single_vehicle(world, v_transform)
            if vehicle is not None:
                spawned_vehicles.append(vehicle)
                self._spawned_actors.append(vehicle)

        if not spawned_vehicles:
            logger.warning("Failed to spawn any vehicles for multi-vehicle accident.")
            return None

        severity = "high" if len(spawned_vehicles) >= 3 else "medium"
        event = AccidentEvent(
            accident_type=AccidentType.MULTI_VEHICLE_ACCIDENT,
            location=transform,
            vehicles=spawned_vehicles,
            start_time=start_time,
            duration=duration,
            severity=severity,
        )
        self._active_events.append(event)
        logger.info(
            "Spawned MULTI_VEHICLE_ACCIDENT with %d vehicles at (%.1f, %.1f) [duration=%.1fs, severity=%s]",
            len(spawned_vehicles),
            transform.location.x,
            transform.location.y,
            duration,
            severity,
        )
        return event

    def block_lane(
        self,
        world: carla.World,
        location: Optional[Union[carla.Location, carla.Transform]] = None,
        block_type: str = "partial",
        duration: float = 60.0,
        start_time: Optional[float] = None,
    ) -> Optional[AccidentEvent]:
        """Block a lane partially or fully using stopped vehicles.

        'partial': spawn 1 vehicle slightly offset into the lane.
        'full': spawn 2 vehicles side by side to fully block.

        Args:
            world: CARLA World instance.
            location: Target location or transform. If None, random spawn point used.
            block_type: 'partial' or 'full'.
            duration: Duration in seconds. Defaults to 60.0.
            start_time: Optional simulation timestamp. If None, fetched from world snapshot.

        Returns:
            AccidentEvent if successfully spawned, otherwise None.
        """
        transform = self._resolve_target_transform(world, location)
        if transform is None:
            logger.warning("Cannot block lane: no valid transform found.")
            return None

        if start_time is None:
            start_time = self._get_current_time(world)

        b_type = str(block_type).lower().strip()
        spawned_vehicles: List[carla.Actor] = []

        if b_type == "full":
            # Spawn 2 vehicles side by side (staggered slightly to avoid overlap on spawn)
            tf1 = self._offset_location(
                transform, offset_x=-1.5, offset_y=-1.8, yaw_offset=-10.0
            )
            tf2 = self._offset_location(
                transform, offset_x=1.5, offset_y=1.8, yaw_offset=15.0
            )

            v1 = self._spawn_single_vehicle(world, tf1)
            if v1 is not None:
                spawned_vehicles.append(v1)
                self._spawned_actors.append(v1)

            v2 = self._spawn_single_vehicle(world, tf2)
            if v2 is not None:
                spawned_vehicles.append(v2)
                self._spawned_actors.append(v2)

            accident_type = AccidentType.FULL_BLOCK
            severity = "high"
        else:
            # 'partial' block: spawn 1 vehicle offset into lane at an angle
            offset_y = self._rng.choice([-1.4, 1.4])
            yaw_offset = self._rng.uniform(15.0, 30.0) * (1.0 if offset_y > 0 else -1.0)
            tf = self._offset_location(
                transform, offset_x=0.0, offset_y=offset_y, yaw_offset=yaw_offset
            )
            v = self._spawn_single_vehicle(world, tf)
            if v is not None:
                spawned_vehicles.append(v)
                self._spawned_actors.append(v)

            accident_type = AccidentType.PARTIAL_BLOCK
            severity = "medium"

        if not spawned_vehicles:
            logger.warning("Failed to spawn vehicle(s) for %s lane block.", b_type)
            return None

        event = AccidentEvent(
            accident_type=accident_type,
            location=transform,
            vehicles=spawned_vehicles,
            start_time=start_time,
            duration=duration,
            severity=severity,
        )
        self._active_events.append(event)
        logger.info(
            "Spawned %s lane block with %d vehicle(s) at (%.1f, %.1f) [duration=%.1fs, severity=%s]",
            b_type.upper(),
            len(spawned_vehicles),
            transform.location.x,
            transform.location.y,
            duration,
            severity,
        )
        return event

    def spawn_breakdown(
        self,
        world: carla.World,
        location: Optional[Union[carla.Location, carla.Transform]] = None,
        duration: float = 60.0,
        start_time: Optional[float] = None,
    ) -> Optional[AccidentEvent]:
        """Spawn a single broken-down vehicle with hazard lights active.

        Args:
            world: CARLA World instance.
            location: Target location or transform. If None, random spawn point used.
            duration: Duration in seconds. Defaults to 60.0.
            start_time: Optional simulation timestamp.

        Returns:
            AccidentEvent if vehicle was spawned, otherwise None.
        """
        return self.spawn_stopped_vehicle(
            world=world,
            location=location,
            duration=duration,
            start_time=start_time,
            severity="low",
            accident_type=AccidentType.BREAKDOWN,
        )

    def spawn_debris(
        self,
        world: carla.World,
        location: Optional[Union[carla.Location, carla.Transform]] = None,
        duration: float = 60.0,
        start_time: Optional[float] = None,
    ) -> Optional[AccidentEvent]:
        """Spawn road debris or obstacles (static props or disabled small vehicle).

        Args:
            world: CARLA World instance.
            location: Optional target location. If None, random spawn point used.
            duration: Duration in seconds. Defaults to 60.0.
            start_time: Optional simulation timestamp.

        Returns:
            AccidentEvent if spawned, otherwise None.
        """
        transform = self._resolve_target_transform(world, location)
        if transform is None:
            logger.warning("Cannot spawn debris: no valid transform found.")
            return None

        if start_time is None:
            start_time = self._get_current_time(world)

        # Attempt to find static prop blueprints (e.g. box, barrel, cone)
        prop_bps = world.get_blueprint_library().filter("static.prop.*")
        spawned_props: List[carla.Actor] = []

        if prop_bps:
            num_props = self._rng.randint(2, 4)
            for _ in range(num_props):
                bp = self._rng.choice(prop_bps)
                ox = self._rng.uniform(-2.0, 2.0)
                oy = self._rng.uniform(-1.5, 1.5)
                prop_tf = self._offset_location(
                    transform, offset_x=ox, offset_y=oy, offset_z=0.2
                )
                prop_actor = world.try_spawn_actor(bp, prop_tf)
                if prop_actor is not None:
                    spawned_props.append(prop_actor)
                    self._spawned_actors.append(prop_actor)

        # Fallback to disabled small vehicle if static props unavailable or spawn failed
        if not spawned_props:
            veh = self._spawn_single_vehicle(world, transform)
            if veh is not None:
                spawned_props.append(veh)
                self._spawned_actors.append(veh)

        if not spawned_props:
            logger.warning("Failed to spawn debris props or fallback vehicle.")
            return None

        event = AccidentEvent(
            accident_type=AccidentType.DEBRIS,
            location=transform,
            vehicles=spawned_props,
            start_time=start_time,
            duration=duration,
            severity="low",
        )
        self._active_events.append(event)
        logger.info(
            "Spawned DEBRIS (%d object(s)) at (%.1f, %.1f) [duration=%.1fs]",
            len(spawned_props),
            transform.location.x,
            transform.location.y,
            duration,
        )
        return event

    def spawn_random_incident(
        self,
        world: carla.World,
        current_time: float,
        probability: float = 0.1,
        duration: Optional[float] = None,
        location: Optional[Union[carla.Location, carla.Transform]] = None,
    ) -> Optional[AccidentEvent]:
        """Roll probability, pick a random incident type, and spawn it.

        Args:
            world: CARLA World instance.
            current_time: Current simulation timestamp in seconds.
            probability: Probability in [0.0, 1.0] of spawning an incident. Defaults to 0.1.
            duration: Optional duration in seconds. If None, samples between 30.0 and 120.0s.
            location: Optional location to spawn near or at. If None, chosen randomly.

        Returns:
            AccidentEvent if an incident was spawned, otherwise None.
        """
        if self._rng.random() >= probability:
            return None

        if duration is None:
            duration = round(self._rng.uniform(30.0, 120.0), 1)

        # Randomly choose one of the available incident types
        incident_types = list(AccidentType)
        chosen_type = self._rng.choice(incident_types)

        if chosen_type == AccidentType.STOPPED_VEHICLE:
            return self.spawn_stopped_vehicle(
                world=world,
                location=location,
                duration=duration,
                start_time=current_time,
            )
        elif chosen_type == AccidentType.BREAKDOWN:
            return self.spawn_breakdown(
                world=world,
                location=location,
                duration=duration,
                start_time=current_time,
            )
        elif chosen_type == AccidentType.MULTI_VEHICLE_ACCIDENT:
            num_v = self._rng.choice([2, 3])
            return self.spawn_accident(
                world=world,
                location=location,
                num_vehicles=num_v,
                duration=duration,
                start_time=current_time,
            )
        elif chosen_type == AccidentType.PARTIAL_BLOCK:
            return self.block_lane(
                world=world,
                location=location,
                block_type="partial",
                duration=duration,
                start_time=current_time,
            )
        elif chosen_type == AccidentType.FULL_BLOCK:
            return self.block_lane(
                world=world,
                location=location,
                block_type="full",
                duration=duration,
                start_time=current_time,
            )
        elif chosen_type == AccidentType.DEBRIS:
            return self.spawn_debris(
                world=world,
                location=location,
                duration=duration,
                start_time=current_time,
            )
        else:
            return self.spawn_stopped_vehicle(
                world=world,
                location=location,
                duration=duration,
                start_time=current_time,
            )

    def update(self, current_time: float) -> List[AccidentEvent]:
        """Check active events, remove expired ones, and destroy their actors.

        Args:
            current_time: Current simulation time in seconds.

        Returns:
            List of expired AccidentEvent objects that were removed and cleaned up.
        """
        expired_events: List[AccidentEvent] = []
        remaining_events: List[AccidentEvent] = []

        for event in self._active_events:
            if event.is_expired(current_time):
                expired_events.append(event)
            else:
                remaining_events.append(event)

        for event in expired_events:
            for actor in event.vehicles:
                try:
                    if actor is not None and getattr(actor, "is_alive", True):
                        actor.destroy()
                except Exception as exc:
                    logger.debug(
                        "Actor %s already destroyed or error during removal: %s",
                        getattr(actor, "id", None),
                        exc,
                    )
                if actor in self._spawned_actors:
                    self._spawned_actors.remove(actor)

            logger.info(
                "Removed expired incident %s (type=%s, duration=%.1fs)",
                event,
                event.accident_type.name,
                event.duration,
            )

        self._active_events = remaining_events
        return expired_events

    def cleanup(self) -> None:
        """Destroy all spawned actors and clear all active accident events."""
        for actor in self._spawned_actors:
            try:
                if actor is not None and getattr(actor, "is_alive", True):
                    actor.destroy()
            except Exception as exc:
                logger.debug(
                    "Actor %s already destroyed or error during cleanup: %s",
                    getattr(actor, "id", None),
                    exc,
                )
        self._spawned_actors.clear()
        self._active_events.clear()
        logger.info("Cleaned up all accident actors and cleared active events.")

    def get_active_events(self) -> List[AccidentEvent]:
        """Return a list of currently active accident events.

        Returns:
            List of active AccidentEvent instances.
        """
        return list(self._active_events)

    def _get_spawn_location(
        self,
        world: carla.World,
        near_location: Optional[Union[carla.Location, carla.Transform]] = None,
    ) -> Optional[carla.Transform]:
        """Get a spawn point from the map, optionally near a given location.

        Args:
            world: CARLA World instance.
            near_location: Optional location or transform to find a spawn point near.

        Returns:
            A carla.Transform spawn point, or None if no spawn points exist.
        """
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            logger.warning("No spawn points found on current CARLA map.")
            return None

        if near_location is None:
            return self._rng.choice(spawn_points)

        if isinstance(near_location, carla.Transform):
            target_loc = near_location.location
        elif hasattr(near_location, "location"):
            target_loc = near_location.location
        else:
            target_loc = near_location

        points_with_dist = []
        for sp in spawn_points:
            dx = sp.location.x - target_loc.x
            dy = sp.location.y - target_loc.y
            dz = sp.location.z - target_loc.z
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            points_with_dist.append((dist, sp))

        points_with_dist.sort(key=lambda item: item[0])
        # Sample from top 3 closest spawn points to add slight variation
        candidates = [sp for _, sp in points_with_dist[: min(3, len(points_with_dist))]]
        return self._rng.choice(candidates)

    def _offset_location(
        self,
        transform: carla.Transform,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        offset_z: float = 0.0,
        yaw_offset: float = 0.0,
        local_coordinates: bool = True,
    ) -> carla.Transform:
        """Return a new Transform with a small positional and optional angular offset.

        When local_coordinates is True (default), offset_x is longitudinal (forward
        along the vehicle heading) and offset_y is lateral (perpendicular/right),
        accounting for vehicle orientation.

        Args:
            transform: Base carla.Transform.
            offset_x: Forward (longitudinal) offset in meters (default 0.0).
            offset_y: Rightward (lateral) offset in meters (default 0.0).
            offset_z: Vertical (Z) offset in meters (default 0.0).
            yaw_offset: Additional yaw rotation in degrees (default 0.0).
            local_coordinates: Whether offsets are relative to transform heading.

        Returns:
            A new carla.Transform with the offset applied.
        """
        if local_coordinates:
            yaw_rad = math.radians(transform.rotation.yaw)
            dx = offset_x * math.cos(yaw_rad) - offset_y * math.sin(yaw_rad)
            dy = offset_x * math.sin(yaw_rad) + offset_y * math.cos(yaw_rad)
        else:
            dx = offset_x
            dy = offset_y

        new_loc = carla.Location(
            x=transform.location.x + dx,
            y=transform.location.y + dy,
            z=transform.location.z + offset_z,
        )
        new_rot = carla.Rotation(
            pitch=transform.rotation.pitch,
            yaw=transform.rotation.yaw + yaw_offset,
            roll=transform.rotation.roll,
        )
        return carla.Transform(new_loc, new_rot)

    def _get_vehicle_blueprint(
        self, world: carla.World
    ) -> Optional[carla.ActorBlueprint]:
        """Select a random sedan or small passenger car blueprint from the library.

        Filters for 4-wheeled vehicles, prioritizing passenger cars / sedans
        and excluding heavy or specialty vehicles (trucks, vans, bikes).

        Args:
            world: CARLA World instance.

        Returns:
            A carla.ActorBlueprint, or None if no blueprints are available.
        """
        bp_library = world.get_blueprint_library()
        vehicles = bp_library.filter("vehicle.*")
        if not vehicles:
            logger.warning("No vehicle blueprints found in blueprint library.")
            return None

        # Filter 4-wheeled vehicles
        four_wheelers = [
            bp
            for bp in vehicles
            if bp.has_attribute("number_of_wheels")
            and int(bp.get_attribute("number_of_wheels")) == 4
        ]
        if not four_wheelers:
            four_wheelers = list(vehicles)

        # Exclude heavy, oversized, emergency vehicles, or microcars
        heavy_keywords = (
            "carlacola",
            "cybertruck",
            "t2",
            "sprinter",
            "ambulance",
            "firetruck",
            "truck",
            "van",
            "bus",
            "box",
            "garbage",
            "isetta",
        )
        small_sedan_bps = [
            bp
            for bp in four_wheelers
            if not any(k in bp.id.lower() for k in heavy_keywords)
        ]

        selected_pool = small_sedan_bps if small_sedan_bps else four_wheelers
        bp = self._rng.choice(selected_pool)

        # Set random recommended color if available
        if bp.has_attribute("color"):
            colors = bp.get_attribute("color").recommended_values
            if colors:
                bp.set_attribute("color", self._rng.choice(colors))

        return bp

    def _spawn_single_vehicle(
        self, world: carla.World, transform: carla.Transform
    ) -> Optional[carla.Vehicle]:
        """Helper to spawn a stopped vehicle at a given transform.

        Applies full braking (VehicleControl(brake=1.0, hand_brake=True)) and
        attempts to activate hazard lights upon successful spawn. Autopilot is NOT set.

        Args:
            world: CARLA World instance.
            transform: Target carla.Transform for spawning.

        Returns:
            The spawned carla.Vehicle, or None if spawning failed.
        """
        bp = self._get_vehicle_blueprint(world)
        if bp is None:
            return None

        # Try spawning at transform; if collision fails, retry with small Z lift
        vehicle = world.try_spawn_actor(bp, transform)
        if vehicle is None:
            elevated_tf = carla.Transform(
                carla.Location(
                    x=transform.location.x,
                    y=transform.location.y,
                    z=transform.location.z + 0.3,
                ),
                transform.rotation,
            )
            vehicle = world.try_spawn_actor(bp, elevated_tf)

        if vehicle is None:
            logger.warning(
                "Failed to spawn vehicle at (%.1f, %.1f, %.1f)",
                transform.location.x,
                transform.location.y,
                transform.location.z,
            )
            return None

        # DO NOT set autopilot — vehicle just sits there (apply brake & handbrake)
        vehicle.apply_control(
            carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=True)
        )

        # Enable hazard lights if possible
        self._set_hazard_lights(vehicle)

        return vehicle

    def _set_hazard_lights(self, vehicle: carla.Vehicle) -> bool:
        """Enable hazard blinker lights on the vehicle if supported.

        Args:
            vehicle: CARLA vehicle actor.

        Returns:
            True if hazard lights were successfully applied, False otherwise.
        """
        try:
            if hasattr(carla, "VehicleLightState"):
                hazard_mask = (
                    carla.VehicleLightState.RightBlinker
                    | carla.VehicleLightState.LeftBlinker
                )
                vehicle.set_light_state(carla.VehicleLightState(hazard_mask))
                return True
        except Exception as exc:
            logger.debug(
                "Could not set hazard lights on vehicle %s: %s",
                getattr(vehicle, "id", None),
                exc,
            )
        return False

    def _resolve_target_transform(
        self,
        world: carla.World,
        location: Optional[Union[carla.Location, carla.Transform]] = None,
    ) -> Optional[carla.Transform]:
        """Convert an optional location/transform into a valid spawn Transform.

        Args:
            world: CARLA World instance.
            location: Optional target Location or Transform.

        Returns:
            A carla.Transform or None if resolution fails.
        """
        if location is None:
            return self._get_spawn_location(world)

        if isinstance(location, carla.Transform):
            return carla.Transform(
                carla.Location(
                    x=location.location.x,
                    y=location.location.y,
                    z=location.location.z + 0.1,
                ),
                carla.Rotation(
                    pitch=location.rotation.pitch,
                    yaw=location.rotation.yaw,
                    roll=location.rotation.roll,
                ),
            )

        if isinstance(location, carla.Location) or hasattr(location, "x"):
            waypoint = world.get_map().get_waypoint(location)
            if waypoint is not None:
                tf = waypoint.transform
                tf.location.z += 0.3
                return tf
            return carla.Transform(
                carla.Location(x=location.x, y=location.y, z=location.z + 0.3),
                carla.Rotation(),
            )

        return self._get_spawn_location(world, near_location=location)

    def _get_current_time(self, world: carla.World) -> float:
        """Retrieve elapsed simulation time in seconds from world snapshot.

        Args:
            world: CARLA World instance.

        Returns:
            Elapsed seconds as float, or 0.0 on failure.
        """
        try:
            snapshot = world.get_snapshot()
            if snapshot is not None and snapshot.timestamp is not None:
                return float(snapshot.timestamp.elapsed_seconds)
        except Exception:
            pass
        return 0.0
