from __future__ import annotations

"""Human driver behavior profiles module for CARLA 0.9.9 simulator.

This module models diverse human driving styles and unexpected behaviors
for traffic vehicles managed by CARLA's Traffic Manager. It supports predefined
behavior profiles (normal, aggressive, conservative, distracted, erratic, slow),
dynamic behavioral variations over time, and tactical event triggers (sudden braking,
acceleration, forced lane changes, cut-ins).
"""

from enum import Enum
import logging
import random
from typing import Any, Dict, List, Optional, Sequence, Union

import carla

logger = logging.getLogger(__name__)


class DriverProfile(Enum):
    """Enumeration of human driver behavior profiles."""

    NORMAL = "normal"
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    DISTRACTED = "distracted"
    ERRATIC = "erratic"
    SLOW = "slow"

    @classmethod
    def from_string(cls, name: str) -> "DriverProfile":
        """Convert a string name or value to DriverProfile.

        Args:
            name: Case-insensitive profile name (e.g. 'NORMAL', 'normal').

        Returns:
            Corresponding DriverProfile enum member.

        Raises:
            ValueError: If the name does not match any profile.
        """
        normalized = name.strip().upper()
        for member in cls:
            if member.name == normalized or member.value.upper() == normalized:
                return member
        raise ValueError(f"Unknown DriverProfile: {name}")


def _set_vehicle_speed_diff(tm: Any, vehicle: carla.Vehicle, diff: float) -> None:
    """Set speed difference percentage supporting both CARLA 0.9.9 and newer TM APIs."""
    if hasattr(tm, "vehicle_percentage_speed_difference"):
        tm.vehicle_percentage_speed_difference(vehicle, float(diff))
    elif hasattr(tm, "set_percentage_speed_difference"):
        tm.set_percentage_speed_difference(vehicle, float(diff))


# Traffic Manager settings per driver profile.
# Note: In CARLA TM, vehicle_percentage_speed_difference uses POSITIVE values
# to go SLOWER and NEGATIVE values to go FASTER.
PROFILE_PARAMS: Dict[DriverProfile, Dict[str, Any]] = {
    DriverProfile.NORMAL: {
        "speed_diff": 0.0,
        "following_distance": 2.5,
        "following_distance": 3.0,
        "auto_lane_change": True,
        "ignore_lights_pct": 0.0,
    },
    DriverProfile.AGGRESSIVE: {
        "speed_diff": -20.0,
        "following_distance": 0.5,
        "speed_diff": -15.0,
        "following_distance": 2.0,
        "auto_lane_change": True,
        "ignore_lights_pct": 20.0,
        "ignore_lights_pct": 0.0,
    },
    DriverProfile.CONSERVATIVE: {
        "speed_diff": 30.0,
        "following_distance": 5.0,
        "speed_diff": 20.0,
        "following_distance": 4.5,
        "auto_lane_change": False,
        "ignore_lights_pct": 0.0,
    },
    DriverProfile.DISTRACTED: {
        "speed_diff": 10.0,
        "following_distance": 3.0,
        "speed_diff": 5.0,
        "following_distance": 3.5,
        "auto_lane_change": True,
        "ignore_lights_pct": 10.0,
        "ignore_lights_pct": 0.0,
    },
    DriverProfile.ERRATIC: {
        "speed_diff": -10.0,
        "following_distance": 1.0,
        "following_distance": 2.5,
        "auto_lane_change": True,
        "ignore_lights_pct": 30.0,
        "ignore_lights_pct": 0.0,
    },
    DriverProfile.SLOW: {
        "speed_diff": 50.0,
        "speed_diff": 30.0,
        "following_distance": 4.0,
        "auto_lane_change": False,
        "ignore_lights_pct": 0.0,
    },
}

# Default probability distribution for random profile assignments.
DEFAULT_PROFILE_WEIGHTS: Dict[DriverProfile, float] = {
    DriverProfile.NORMAL: 0.40,
    DriverProfile.AGGRESSIVE: 0.15,
    DriverProfile.CONSERVATIVE: 0.15,
    DriverProfile.DISTRACTED: 0.10,
    DriverProfile.ERRATIC: 0.10,
    DriverProfile.SLOW: 0.10,
}


class HumanBehaviorManager:
    """Manages human driver behavior profiles for CARLA vehicles.

    Configures Traffic Manager parameters based on behavioral profiles,
    handles stochastic dynamic updates (distracted speed drift, erratic actions),
    and triggers tactical maneuvers (sudden braking, sudden acceleration, lane changes, cut-ins).

    Attributes:
        _seed (Optional[int]): Random seed for reproducible behavior.
        _rng (random.Random): Random number generator instance.
        _vehicle_profiles (Dict[int, DriverProfile]): Mapping of vehicle.id to DriverProfile.
        _distracted_state (Dict[int, Dict[str, Any]]): State tracker for DISTRACTED and ERRATIC vehicles.
        _vehicles (Dict[int, carla.Vehicle]): Mapping of vehicle.id to vehicle actors.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize the HumanBehaviorManager.

        Args:
            seed: Optional integer seed for reproducibility.
        """
        self._seed: Optional[int] = seed
        self._rng: random.Random = random.Random(seed)
        self._vehicle_profiles: Dict[int, DriverProfile] = {}
        self._distracted_state: Dict[int, Dict[str, Any]] = {}
        self._vehicles: Dict[int, carla.Vehicle] = {}

        logger.info("HumanBehaviorManager initialized (seed=%s)", seed)

    def assign_profile(
        self,
        vehicle: carla.Vehicle,
        traffic_manager: Any,
        profile: DriverProfile,
    ) -> None:
        """Apply Traffic Manager settings for a driver profile to a vehicle.

        Configures:
        - set_percentage_speed_difference(vehicle, speed_diff)
        - distance_to_leading_vehicle(vehicle, following_distance)
        - auto_lane_change(vehicle, auto_lane_change)
        - ignore_lights_percentage(vehicle, ignore_lights_pct)

        Args:
            vehicle: CARLA vehicle actor to configure.
            traffic_manager: CARLA Traffic Manager instance.
            profile: DriverProfile enum member to apply.

        Raises:
            KeyError: If profile is not recognized in PROFILE_PARAMS.
        """
        if isinstance(profile, str):
            profile = DriverProfile.from_string(profile)

        if profile not in PROFILE_PARAMS:
            raise KeyError(f"Unknown driver profile: {profile}")

        params = PROFILE_PARAMS[profile]

        _set_vehicle_speed_diff(
            traffic_manager, vehicle, float(params["speed_diff"])
        )
        traffic_manager.distance_to_leading_vehicle(
            vehicle, float(params["following_distance"])
        )
        traffic_manager.auto_lane_change(
            vehicle, bool(params["auto_lane_change"])
        )
        traffic_manager.ignore_lights_percentage(
            vehicle, float(params["ignore_lights_pct"])
        )

        self._vehicle_profiles[vehicle.id] = profile
        self._vehicles[vehicle.id] = vehicle

        if profile in (DriverProfile.DISTRACTED, DriverProfile.ERRATIC):
            self._distracted_state[vehicle.id] = {
                "vehicle": vehicle,
                "profile": profile,
                "last_update_time": None,
                "interval": self._rng.uniform(2.0, 4.0),
                "base_speed_diff": float(params["speed_diff"]),
                "current_speed_diff": float(params["speed_diff"]),
            }
        else:
            self._distracted_state.pop(vehicle.id, None)

        logger.info(
            "Vehicle %d assigned profile %s (speed_diff=%.1f%%, dist=%.1fm, auto_lc=%s, ignore_lights=%.1f%%)",
            vehicle.id,
            profile.name,
            params["speed_diff"],
            params["following_distance"],
            params["auto_lane_change"],
            params["ignore_lights_pct"],
        )

    def assign_random_profiles(
        self,
        vehicles: Sequence[carla.Vehicle],
        traffic_manager: Any,
        seed: Optional[int] = None,
        weights: Optional[Union[Dict[Any, float], Sequence[float]]] = None,
    ) -> Dict[int, DriverProfile]:
        """Assign random profiles to all given vehicles based on weights.

        If weights is None, default distribution is used:
        NORMAL=40%, AGGRESSIVE=15%, CONSERVATIVE=15%, DISTRACTED=10%, ERRATIC=10%, SLOW=10%.

        Args:
            vehicles: Sequence of CARLA vehicle actors.
            traffic_manager: CARLA Traffic Manager instance.
            seed: Optional integer seed to override generator for this assignment.
            weights: Optional distribution weights. Can be a dict mapping DriverProfile
                (or string name) to float, or a sequence of floats matching the order
                of DriverProfile members.

        Returns:
            Dict mapping vehicle.id to assigned DriverProfile.
        """
        rng = random.Random(seed) if seed is not None else self._rng

        profiles: List[DriverProfile] = []
        weight_values: List[float] = []

        if weights is None:
            profiles = list(DEFAULT_PROFILE_WEIGHTS.keys())
            weight_values = list(DEFAULT_PROFILE_WEIGHTS.values())
        elif isinstance(weights, dict):
            for k, v in weights.items():
                p = DriverProfile.from_string(k) if isinstance(k, str) else k
                profiles.append(p)
                weight_values.append(float(v))
        elif isinstance(weights, (list, tuple)):
            all_profiles = list(DriverProfile)
            if len(weights) != len(all_profiles):
                raise ValueError(
                    f"Expected {len(all_profiles)} weights, got {len(weights)}"
                )
            profiles = all_profiles
            weight_values = [float(w) for w in weights]
        else:
            raise TypeError(f"Unsupported weights type: {type(weights)}")

        assigned: Dict[int, DriverProfile] = {}
        for vehicle in vehicles:
            chosen_profile = rng.choices(profiles, weights=weight_values, k=1)[0]
            self.assign_profile(vehicle, traffic_manager, chosen_profile)
            assigned[vehicle.id] = chosen_profile

        logger.info(
            "Assigned random profiles to %d vehicles (seed=%s)",
            len(vehicles),
            seed if seed is not None else self._seed,
        )
        return assigned

    def trigger_sudden_brake(self, vehicle: carla.Vehicle) -> None:
        """Apply an emergency brake to the vehicle.

        Applies full braking control (brake=1.0, hand_brake=False).

        Args:
            vehicle: CARLA vehicle actor to brake.
        """
        vehicle.apply_control(carla.VehicleControl(brake=1.0, hand_brake=False))
        logger.info("Triggered sudden brake on vehicle %d", vehicle.id)

    def trigger_sudden_acceleration(self, vehicle: carla.Vehicle) -> None:
        """Apply sudden maximum throttle to the vehicle.

        Applies full throttle control (throttle=1.0).

        Args:
            vehicle: CARLA vehicle actor to accelerate.
        """
        vehicle.apply_control(carla.VehicleControl(throttle=1.0))
        logger.info("Triggered sudden acceleration on vehicle %d", vehicle.id)

    def trigger_lane_change(
        self,
        vehicle: carla.Vehicle,
        traffic_manager: Any,
        direction: Optional[bool] = None,
    ) -> bool:
        """Force a lane change on the vehicle via Traffic Manager.

        If direction is None, picks random True (right) or False (left).

        Args:
            vehicle: CARLA vehicle actor to command.
            traffic_manager: CARLA Traffic Manager instance.
            direction: Optional boolean direction. True for right, False for left.
                If None, randomly selected.

        Returns:
            The direction (bool) executed.
        """
        if direction is None:
            direction = self._rng.choice([True, False])
        traffic_manager.force_lane_change(vehicle, direction)
        logger.info(
            "Triggered lane change on vehicle %d (direction=%s)",
            vehicle.id,
            "right" if direction else "left",
        )
        return direction

    def trigger_cut_in(
        self,
        vehicle: carla.Vehicle,
        traffic_manager: Any,
    ) -> bool:
        """Force an abrupt cut-in lane change towards an adjacent lane.

        Forces a lane change with a random direction.

        Args:
            vehicle: CARLA vehicle actor to command.
            traffic_manager: CARLA Traffic Manager instance.

        Returns:
            The direction (bool) executed.
        """
        direction = self._rng.choice([True, False])
        traffic_manager.force_lane_change(vehicle, direction)
        logger.info(
            "Triggered cut-in on vehicle %d (direction=%s)",
            vehicle.id,
            "right" if direction else "left",
        )
        return direction

    def update_distracted_vehicles(
        self,
        traffic_manager: Any,
        current_time: float,
    ) -> None:
        """Update behavioral dynamics for DISTRACTED and ERRATIC vehicles.

        For DISTRACTED vehicles:
            Randomly varies their speed difference (±15%) every few seconds.

        For ERRATIC vehicles:
            Randomly triggers sudden speed changes or lane changes with low probability.

        Args:
            traffic_manager: CARLA Traffic Manager instance.
            current_time: Current simulation elapsed time in seconds.
        """
        stale_vehicle_ids: List[int] = []

        for veh_id, state in list(self._distracted_state.items()):
            vehicle = state.get("vehicle")
            if vehicle is None:
                continue

            try:
                if hasattr(vehicle, "is_alive") and not vehicle.is_alive:
                    stale_vehicle_ids.append(veh_id)
                    continue
            except Exception:
                stale_vehicle_ids.append(veh_id)
                continue

            profile = state.get("profile")
            last_update = state.get("last_update_time")
            interval = state.get("interval", 3.0)

            if last_update is None:
                state["last_update_time"] = current_time
                continue

            if (current_time - last_update) < interval:
                continue

            state["last_update_time"] = current_time
            state["interval"] = self._rng.uniform(2.0, 4.0)

            try:
                if profile == DriverProfile.DISTRACTED:
                    base_diff = state.get(
                        "base_speed_diff",
                        PROFILE_PARAMS[DriverProfile.DISTRACTED]["speed_diff"],
                    )
                    delta = self._rng.uniform(-15.0, 15.0)
                    new_speed_diff = base_diff + delta
                    _set_vehicle_speed_diff(
                        traffic_manager, vehicle, float(new_speed_diff)
                    )
                    state["current_speed_diff"] = new_speed_diff
                    logger.debug(
                        "DISTRACTED vehicle %d: varied speed_diff to %.1f%% (delta %+.1f%%)",
                        veh_id,
                        new_speed_diff,
                        delta,
                    )

                elif profile == DriverProfile.ERRATIC:
                    if self._rng.random() < 0.30:
                        erratic_action = self._rng.choice(["lane_change", "speed_change"])
                        if erratic_action == "lane_change":
                            self.trigger_lane_change(vehicle, traffic_manager)
                        else:
                            sub_choice = self._rng.choice(["speed_diff", "brake", "accel"])
                            if sub_choice == "speed_diff":
                                new_speed_diff = self._rng.uniform(-35.0, 35.0)
                                _set_vehicle_speed_diff(
                                    traffic_manager, vehicle, float(new_speed_diff)
                                )
                                state["current_speed_diff"] = new_speed_diff
                                logger.debug(
                                    "ERRATIC vehicle %d: sudden speed_diff set to %.1f%%",
                                    veh_id,
                                    new_speed_diff,
                                )
                            elif sub_choice == "brake":
                                self.trigger_sudden_brake(vehicle)
                            else:
                                self.trigger_sudden_acceleration(vehicle)

            except Exception as exc:
                logger.warning(
                    "Error updating distracted/erratic vehicle %d: %s", veh_id, exc
                )
                stale_vehicle_ids.append(veh_id)

        for dead_id in stale_vehicle_ids:
            self.remove_vehicle(dead_id)

    def get_profile(
        self, vehicle: Union[carla.Vehicle, int]
    ) -> Optional[DriverProfile]:
        """Get the assigned DriverProfile for a vehicle.

        Args:
            vehicle: CARLA vehicle actor or integer vehicle ID.

        Returns:
            DriverProfile enum member if assigned, otherwise None.
        """
        veh_id = vehicle.id if hasattr(vehicle, "id") else int(vehicle)
        return self._vehicle_profiles.get(veh_id)

    def get_all_profiles(self) -> Dict[int, DriverProfile]:
        """Get a copy of all assigned vehicle profiles.

        Returns:
            Dictionary mapping vehicle.id (int) to DriverProfile.
        """
        return dict(self._vehicle_profiles)

    def get_distracted_state(
        self, vehicle: Union[carla.Vehicle, int]
    ) -> Optional[Dict[str, Any]]:
        """Get the distraction state dictionary for a vehicle.

        Args:
            vehicle: CARLA vehicle actor or integer vehicle ID.

        Returns:
            Distraction state dict if vehicle is tracked as DISTRACTED or ERRATIC, else None.
        """
        veh_id = vehicle.id if hasattr(vehicle, "id") else int(vehicle)
        return self._distracted_state.get(veh_id)

    def remove_vehicle(self, vehicle: Union[carla.Vehicle, int]) -> None:
        """Remove a vehicle from tracking.

        Args:
            vehicle: CARLA vehicle actor or integer vehicle ID.
        """
        veh_id = vehicle.id if hasattr(vehicle, "id") else int(vehicle)
        self._vehicle_profiles.pop(veh_id, None)
        self._distracted_state.pop(veh_id, None)
        self._vehicles.pop(veh_id, None)

    def reset(self) -> None:
        """Reset all tracked vehicle profiles and distraction states."""
        self._vehicle_profiles.clear()
        self._distracted_state.clear()
        self._vehicles.clear()
        logger.info("HumanBehaviorManager reset")
