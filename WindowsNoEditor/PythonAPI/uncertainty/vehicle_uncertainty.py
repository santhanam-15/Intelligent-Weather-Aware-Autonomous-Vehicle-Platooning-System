from __future__ import annotations

"""Actuator and vehicle control uncertainty module for CARLA 0.9.9 simulator.

This module provides stochastic noise injection and transport delay simulation
for vehicle actuator commands (throttle, steering, braking) under CARLA 0.9.9.
"""

import collections
import random
from typing import Any, Dict, Optional

import carla


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a numeric value between a minimum and maximum threshold.

    Args:
        value (float): The numeric value to clamp.
        min_val (float): Lower bound.
        max_val (float): Upper bound.

    Returns:
        float: Clamped value within [min_val, max_val].
    """
    return max(min_val, min(value, max_val))


class ActuatorUncertainty:
    """Simulates actuator noise and latency uncertainties for CARLA vehicles.

    Injects zero-mean Gaussian perturbations into throttle, steer, and brake
    control inputs, and simulates transport and mechanical delay via a FIFO tick queue.

    Attributes:
        _seed (Optional[int]): Seed used for reproducible noise generation.
        _rng (random.Random): Dedicated random number generator instance.
        _throttle_noise (float): Standard deviation of Gaussian noise on throttle.
        _steer_noise (float): Standard deviation of Gaussian noise on steer.
        _brake_noise (float): Standard deviation of Gaussian noise on brake.
        _delay_ticks (int): Latency in simulation ticks before controls take effect.
        _delay_queue (collections.deque): FIFO buffer storing delayed control commands.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        throttle_noise: float = 0.05,
        steer_noise: float = 0.02,
        brake_noise: float = 0.05,
        delay_ticks: int = 0,
    ) -> None:
        """Initialize actuator uncertainty settings.

        Args:
            seed (Optional[int]): Random seed for reproducible noise generation.
            throttle_noise (float): Standard deviation of Gaussian throttle noise.
            steer_noise (float): Standard deviation of Gaussian steering noise.
            brake_noise (float): Standard deviation of Gaussian brake noise.
            delay_ticks (int): Simulation tick delay for actuator response.
        """
        self._seed: Optional[int] = seed
        self._rng: random.Random = random.Random(seed)
        self._throttle_noise: float = throttle_noise
        self._steer_noise: float = steer_noise
        self._brake_noise: float = brake_noise
        self._delay_ticks: int = delay_ticks
        self._delay_queue: collections.deque = collections.deque(
            maxlen=max(1, delay_ticks + 1)
        )

    def apply_uncertainty(self, control: carla.VehicleControl) -> carla.VehicleControl:
        """Apply Gaussian noise and latency uncertainty to a vehicle control command.

        Creates a new carla.VehicleControl instance without mutating the input control.
        Adds zero-mean Gaussian noise to throttle, steer, and brake using the internal
        random generator, clamps each value to its valid range, and queues the command
        if delay_ticks > 0 to simulate actuator latency. Preserves discrete properties
        such as hand_brake, reverse, manual_gear_shift, and gear.

        Args:
            control (carla.VehicleControl): Original nominal control input.

        Returns:
            carla.VehicleControl: A new vehicle control object with uncertainty applied.
        """
        noisy_throttle = control.throttle
        if self._throttle_noise > 0.0:
            noisy_throttle += self._rng.gauss(0.0, self._throttle_noise)
        noisy_throttle = clamp(noisy_throttle, 0.0, 1.0)

        noisy_steer = control.steer
        if self._steer_noise > 0.0:
            noisy_steer += self._rng.gauss(0.0, self._steer_noise)
        noisy_steer = clamp(noisy_steer, -1.0, 1.0)

        noisy_brake = control.brake
        if self._brake_noise > 0.0:
            noisy_brake += self._rng.gauss(0.0, self._brake_noise)
        noisy_brake = clamp(noisy_brake, 0.0, 1.0)

        noisy_control = carla.VehicleControl(
            throttle=noisy_throttle,
            steer=noisy_steer,
            brake=noisy_brake,
            hand_brake=control.hand_brake,
            reverse=control.reverse,
            manual_gear_shift=control.manual_gear_shift,
            gear=control.gear,
        )

        if self._delay_ticks > 0:
            self._delay_queue.append(noisy_control)
            return self._delay_queue[0]

        return noisy_control

    def apply_to_vehicle(
        self, vehicle: Any, control: carla.VehicleControl
    ) -> None:
        """Apply the uncertain control command to a CARLA vehicle actor.

        Args:
            vehicle (carla.Vehicle): CARLA vehicle actor to which the control is applied.
            control (carla.VehicleControl): Nominal control input to perturb and apply.
        """
        vehicle.apply_control(self.apply_uncertainty(control))

    def get_config(self) -> Dict[str, Any]:
        """Get the current noise levels and delay settings.

        Returns:
            Dict[str, Any]: Dictionary containing current noise and delay configuration.
        """
        return {
            "throttle_noise": self._throttle_noise,
            "steer_noise": self._steer_noise,
            "brake_noise": self._brake_noise,
            "delay_ticks": self._delay_ticks,
        }

    def set_noise(
        self,
        throttle_noise: Optional[float] = None,
        steer_noise: Optional[float] = None,
        brake_noise: Optional[float] = None,
    ) -> None:
        """Update noise levels for actuator inputs.

        Only non-None arguments are updated.

        Args:
            throttle_noise (Optional[float]): New throttle Gaussian noise std dev.
            steer_noise (Optional[float]): New steer Gaussian noise std dev.
            brake_noise (Optional[float]): New brake Gaussian noise std dev.
        """
        if throttle_noise is not None:
            self._throttle_noise = float(throttle_noise)
        if steer_noise is not None:
            self._steer_noise = float(steer_noise)
        if brake_noise is not None:
            self._brake_noise = float(brake_noise)

    def set_delay(self, delay_ticks: int) -> None:
        """Update actuator delay ticks and recreate the delay queue.

        Args:
            delay_ticks (int): Simulation tick delay for actuator response.
        """
        self._delay_ticks = int(delay_ticks)
        self._delay_queue = collections.deque(
            maxlen=max(1, self._delay_ticks + 1)
        )

    def reset(self) -> None:
        """Reset internal state, clearing the actuator delay queue."""
        self._delay_queue.clear()
