from __future__ import annotations

"""Friction uncertainty module for CARLA 0.9.9 simulator.

This module provides fine-grained control over vehicle tire-road friction
coefficients (mu) to simulate varied road surface conditions such as dry,
wet, slippery, or icy pavement under CARLA 0.9.9 physics.
"""

from enum import Enum
import random
from typing import Dict, Optional, Tuple

import carla


class FrictionCondition(Enum):
    """Enumeration of road friction conditions."""

    DRY = "dry"
    WET = "wet"
    SLIPPERY = "slippery"
    VERY_SLIPPERY = "very_slippery"


FRICTION_VALUES: Dict[FrictionCondition, float] = {
    FrictionCondition.DRY: 1.0,
    FrictionCondition.WET: 0.6,
    FrictionCondition.SLIPPERY: 0.4,
    FrictionCondition.VERY_SLIPPERY: 0.25,
}


class FrictionManager:
    """Manages road friction uncertainty for CARLA vehicles.

    Adjusts wheel tire friction coefficients via CARLA's VehiclePhysicsControl
    to simulate realistic surface adhesion variations.

    Attributes:
        _default_friction (float): Base CARLA tire friction coefficient (default 3.5).
        _current_mu (Dict[int, float]): Mapping of vehicle IDs to current friction multipliers (mu).
        _seed (Optional[int]): Random seed used for reproducible friction sampling.
        _rng (random.Random): Random number generator instance.
    """

    _default_friction: float = 3.5

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize the FrictionManager.

        Args:
            seed: Optional integer seed for reproducible random friction generation.
        """
        self._seed: Optional[int] = seed
        self._rng: random.Random = random.Random(seed)
        self._current_mu: Dict[int, float] = {}

    def set_default_friction(self, value: float) -> None:
        """Set the baseline tire friction coefficient.

        Args:
            value: Base tire friction value (CARLA default is typically 3.5).
        """
        if value <= 0:
            raise ValueError(f"Default friction must be positive, got {value}")
        self._default_friction = float(value)

    def get_default_friction(self) -> float:
        """Get the baseline tire friction coefficient.

        Returns:
            The current base tire friction value.
        """
        return self._default_friction

    def set_friction(self, vehicle: carla.Vehicle, mu: float) -> None:
        """Set tire friction for all wheels of a vehicle scaled by mu.

        Fetches the vehicle's physics control, updates the tire_friction attribute
        on all wheels to (mu * _default_friction), and applies the updated physics.

        Args:
            vehicle: CARLA vehicle actor whose tire friction is to be modified.
            mu: Friction multiplier relative to default friction (e.g. 1.0 for normal, 0.4 for slippery).

        Raises:
            ValueError: If mu is negative.
        """
        if mu < 0:
            raise ValueError(f"Friction multiplier mu must be non-negative, got {mu}")

        physics = vehicle.get_physics_control()
        wheels = physics.wheels
        for wheel in wheels:
            wheel.tire_friction = float(mu * self._default_friction)
        physics.wheels = wheels
        vehicle.apply_physics_control(physics)

        self._current_mu[vehicle.id] = float(mu)

    def set_friction_for_condition(
        self, vehicle: carla.Vehicle, condition: FrictionCondition
    ) -> None:
        """Set vehicle tire friction according to a predefined road condition.

        Args:
            vehicle: CARLA vehicle actor to modify.
            condition: FrictionCondition enum member (DRY, WET, SLIPPERY, VERY_SLIPPERY).

        Raises:
            KeyError: If the condition is not recognized in FRICTION_VALUES.
        """
        if condition not in FRICTION_VALUES:
            raise KeyError(f"Unknown friction condition: {condition}")

        mu = FRICTION_VALUES[condition]
        self.set_friction(vehicle, mu)

    def randomize_friction(
        self,
        vehicle: carla.Vehicle,
        seed: Optional[int] = None,
        mu_range: Tuple[float, float] = (0.25, 1.0),
    ) -> float:
        """Randomly sample and apply a friction multiplier within a given range.

        Args:
            vehicle: CARLA vehicle actor to modify.
            seed: Optional seed to override generator for this operation.
            mu_range: Tuple of (min_mu, max_mu) defining the sampling interval.

        Returns:
            The sampled friction multiplier (mu) that was applied to the vehicle.

        Raises:
            ValueError: If mu_range is invalid (min > max or min < 0).
        """
        min_mu, max_mu = mu_range
        if min_mu < 0 or max_mu < min_mu:
            raise ValueError(f"Invalid mu_range: {mu_range}. Must satisfy 0 <= min <= max.")

        rng = random.Random(seed) if seed is not None else self._rng
        sampled_mu = rng.uniform(min_mu, max_mu)
        self.set_friction(vehicle, sampled_mu)
        return sampled_mu

    def get_friction(self, vehicle: carla.Vehicle) -> Optional[float]:
        """Get the current friction multiplier (mu) for a vehicle.

        Args:
            vehicle: CARLA vehicle actor to query.

        Returns:
            The current mu multiplier if managed by this instance, otherwise None.
        """
        return self._current_mu.get(vehicle.id)

