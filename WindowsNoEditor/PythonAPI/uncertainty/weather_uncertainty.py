from __future__ import annotations

"""Weather uncertainty module for CARLA 0.9.9 simulator.

This module provides fine-grained control and reproducible randomization
of CARLA 0.9.9 weather parameters (cloudiness, precipitation, fog, wind, wetness).
Note that in CARLA 0.9.9, weather parameters do NOT automatically alter physical
road-tire friction coefficients; friction uncertainty is decoupled and managed
separately via FrictionManager.
"""

from enum import Enum
import random
from typing import Dict, Optional, Union

import carla


class WeatherPreset(Enum):
    """Enumeration of standard weather presets."""

    CLEAR = "clear"
    CLOUDY = "cloudy"
    LIGHT_RAIN = "light_rain"
    MODERATE_RAIN = "moderate_rain"
    HEAVY_RAIN = "heavy_rain"
    FOG = "fog"
    HEAVY_FOG = "heavy_fog"
    RAIN_FOG = "rain_fog"


WEATHER_PARAMS: Dict[WeatherPreset, Dict[str, float]] = {
    WeatherPreset.CLEAR: {
        "cloudiness": 10.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wind_intensity": 5.0,
        "fog_density": 0.0,
        "fog_distance": 0.0,
        "wetness": 0.0,
    },
    WeatherPreset.CLOUDY: {
        "cloudiness": 60.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wind_intensity": 20.0,
        "fog_density": 0.0,
        "fog_distance": 0.0,
        "wetness": 0.0,
    },
    WeatherPreset.LIGHT_RAIN: {
        "cloudiness": 50.0,
        "precipitation": 20.0,
        "precipitation_deposits": 20.0,
        "wind_intensity": 30.0,
        "fog_density": 5.0,
        "fog_distance": 0.0,
        "wetness": 30.0,
    },
    WeatherPreset.MODERATE_RAIN: {
        "cloudiness": 70.0,
        "precipitation": 50.0,
        "precipitation_deposits": 50.0,
        "wind_intensity": 50.0,
        "fog_density": 10.0,
        "fog_distance": 0.0,
        "wetness": 60.0,
    },
    WeatherPreset.HEAVY_RAIN: {
        "cloudiness": 90.0,
        "precipitation": 80.0,
        "precipitation_deposits": 80.0,
        "wind_intensity": 80.0,
        "fog_density": 20.0,
        "fog_distance": 0.0,
        "wetness": 90.0,
    },
    WeatherPreset.FOG: {
        "cloudiness": 40.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wind_intensity": 10.0,
        "fog_density": 50.0,
        "fog_distance": 10.0,
        "wetness": 0.0,
    },
    WeatherPreset.HEAVY_FOG: {
        "cloudiness": 50.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wind_intensity": 5.0,
        "fog_density": 90.0,
        "fog_distance": 5.0,
        "wetness": 10.0,
    },
    WeatherPreset.RAIN_FOG: {
        "cloudiness": 80.0,
        "precipitation": 60.0,
        "precipitation_deposits": 60.0,
        "wind_intensity": 60.0,
        "fog_density": 60.0,
        "fog_distance": 10.0,
        "wetness": 70.0,
    },
}


class WeatherManager:
    """Manages weather uncertainty and environmental transitions in CARLA 0.9.9.

    Attributes:
        _seed (Optional[int]): Random seed used for reproducible weather selection.
        _rng (random.Random): Random number generator instance.
        _current_preset (Optional[WeatherPreset]): The most recently applied preset.
        _transition_start (Optional[carla.WeatherParameters]): Baseline weather for lerp transitions.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize the WeatherManager.

        Args:
            seed: Optional integer seed for reproducible random weather selection.
        """
        self._seed: Optional[int] = seed
        self._rng: random.Random = random.Random(seed)
        self._current_preset: Optional[WeatherPreset] = None
        self._transition_start: Optional[carla.WeatherParameters] = None
        self._transition_target: Optional[WeatherPreset] = None

    @staticmethod
    def _resolve_preset(preset: Union[WeatherPreset, str]) -> WeatherPreset:
        """Resolve a WeatherPreset enum or string into a WeatherPreset member.

        Args:
            preset: WeatherPreset enum member or preset name string.

        Returns:
            The resolved WeatherPreset enum member.

        Raises:
            ValueError: If preset string does not match any WeatherPreset.
            TypeError: If preset is neither WeatherPreset nor str.
        """
        if isinstance(preset, WeatherPreset):
            return preset
        if isinstance(preset, str):
            clean_name = preset.strip().upper()
            try:
                return WeatherPreset[clean_name]
            except KeyError:
                pass
            clean_val = preset.strip().lower()
            for p in WeatherPreset:
                if p.value == clean_val:
                    return p
            valid = [p.name for p in WeatherPreset]
            raise ValueError(f"Unknown weather preset '{preset}'. Valid presets: {valid}")
        raise TypeError(f"Preset must be WeatherPreset or str, got {type(preset).__name__}")

    def set_seed(self, seed: Optional[int]) -> None:
        """Set or update the random seed.

        Args:
            seed: Integer seed for reproducible random weather selection.
        """
        self._seed = seed
        self._rng = random.Random(seed)

    @property
    def seed(self) -> Optional[int]:
        """Return the random seed."""
        return self._seed

    @property
    def current_preset(self) -> Optional[WeatherPreset]:
        """Return the currently applied WeatherPreset, or None if custom/in-transition."""
        return self._current_preset

    def set_weather(
        self,
        world: carla.World,
        preset: Union[WeatherPreset, str],
    ) -> carla.WeatherParameters:
        """Apply a predefined weather preset to the CARLA world.

        Args:
            world: CARLA world instance.
            preset: WeatherPreset enum member or preset name string.

        Returns:
            The applied carla.WeatherParameters instance.
        """
        resolved = self._resolve_preset(preset)
        params = WEATHER_PARAMS[resolved]

        try:
            current_weather = world.get_weather()
            sun_azimuth = current_weather.sun_azimuth_angle
            sun_altitude = current_weather.sun_altitude_angle
        except Exception:
            sun_azimuth = 0.0
            sun_altitude = 75.0

        weather_params = carla.WeatherParameters(
            cloudiness=params["cloudiness"],
            precipitation=params["precipitation"],
            precipitation_deposits=params["precipitation_deposits"],
            wind_intensity=params["wind_intensity"],
            fog_density=params["fog_density"],
            fog_distance=params["fog_distance"],
            wetness=params["wetness"],
            sun_azimuth_angle=sun_azimuth,
            sun_altitude_angle=sun_altitude,
        )

        world.set_weather(weather_params)
        self._current_preset = resolved
        self._transition_start = None
        self._transition_target = None
        return weather_params

    def set_custom_weather(
        self,
        world: carla.World,
        **kwargs: float,
    ) -> carla.WeatherParameters:
        """Apply custom weather parameters to the CARLA world.

        Args:
            world: CARLA world instance.
            **kwargs: Weather parameter overrides (e.g. cloudiness, precipitation,
                precipitation_deposits, wind_intensity, fog_density, fog_distance,
                wetness, sun_azimuth_angle, sun_altitude_angle).

        Returns:
            The applied carla.WeatherParameters instance.
        """
        try:
            current = world.get_weather()
        except Exception:
            current = carla.WeatherParameters()

        cloudiness = float(kwargs.get("cloudiness", current.cloudiness))
        precipitation = float(kwargs.get("precipitation", current.precipitation))
        precipitation_deposits = float(
            kwargs.get("precipitation_deposits", current.precipitation_deposits)
        )
        wind_intensity = float(kwargs.get("wind_intensity", current.wind_intensity))
        fog_density = float(kwargs.get("fog_density", current.fog_density))
        fog_distance = float(kwargs.get("fog_distance", current.fog_distance))
        wetness = float(kwargs.get("wetness", current.wetness))
        sun_azimuth_angle = float(kwargs.get("sun_azimuth_angle", current.sun_azimuth_angle))
        sun_altitude_angle = float(kwargs.get("sun_altitude_angle", current.sun_altitude_angle))

        cloudiness = max(0.0, min(100.0, cloudiness))
        precipitation = max(0.0, min(100.0, precipitation))
        precipitation_deposits = max(0.0, min(100.0, precipitation_deposits))
        wind_intensity = max(0.0, min(100.0, wind_intensity))
        fog_density = max(0.0, min(100.0, fog_density))
        fog_distance = max(0.0, fog_distance)
        wetness = max(0.0, min(100.0, wetness))

        weather_params = carla.WeatherParameters(
            cloudiness=cloudiness,
            precipitation=precipitation,
            precipitation_deposits=precipitation_deposits,
            wind_intensity=wind_intensity,
            fog_density=fog_density,
            fog_distance=fog_distance,
            wetness=wetness,
            sun_azimuth_angle=sun_azimuth_angle,
            sun_altitude_angle=sun_altitude_angle,
        )

        world.set_weather(weather_params)
        self._current_preset = None
        self._transition_start = None
        self._transition_target = None
        return weather_params

    def random_weather(
        self,
        world: carla.World,
        seed: Optional[int] = None,
    ) -> WeatherPreset:
        """Pick and apply a random weather preset using a seeded RNG.

        Args:
            world: CARLA world instance.
            seed: Optional seed to override generator for this call.

        Returns:
            The selected WeatherPreset that was applied.
        """
        rng = random.Random(seed) if seed is not None else self._rng
        presets = list(WeatherPreset)
        chosen = rng.choice(presets)
        self.set_weather(world, chosen)
        return chosen

    def transition_weather(
        self,
        world: carla.World,
        target_preset: Union[WeatherPreset, str],
        progress: float,
    ) -> carla.WeatherParameters:
        """Interpolate from current weather to a target preset.

        Linearly interpolates (lerps) atmospheric parameters between the transition
        start weather and the target preset based on progress in [0.0, 1.0].

        Args:
            world: CARLA world instance.
            target_preset: Target WeatherPreset or preset name string.
            progress: Float value between 0.0 (initial state) and 1.0 (target reached).

        Returns:
            The interpolated carla.WeatherParameters applied to the world.
        """
        target = self._resolve_preset(target_preset)
        t = max(0.0, min(1.0, float(progress)))

        if (
            self._transition_start is None
            or getattr(self, "_transition_target", None) != target
            or t <= 0.0
        ):
            self._transition_start = world.get_weather()
            self._transition_target = target

        start = self._transition_start
        target_params = WEATHER_PARAMS[target]

        def _lerp(start_val: float, end_val: float, factor: float) -> float:
            val = start_val + (end_val - start_val) * factor
            return max(0.0, min(100.0, val))

        cloudiness = _lerp(start.cloudiness, target_params["cloudiness"], t)
        precipitation = _lerp(start.precipitation, target_params["precipitation"], t)
        precipitation_deposits = _lerp(
            start.precipitation_deposits, target_params["precipitation_deposits"], t
        )
        wind_intensity = _lerp(start.wind_intensity, target_params["wind_intensity"], t)
        fog_density = _lerp(start.fog_density, target_params["fog_density"], t)
        fog_distance = max(
            0.0, start.fog_distance + (target_params["fog_distance"] - start.fog_distance) * t
        )
        wetness = _lerp(start.wetness, target_params["wetness"], t)

        weather_params = carla.WeatherParameters(
            cloudiness=cloudiness,
            precipitation=precipitation,
            precipitation_deposits=precipitation_deposits,
            wind_intensity=wind_intensity,
            fog_density=fog_density,
            fog_distance=fog_distance,
            wetness=wetness,
            sun_azimuth_angle=start.sun_azimuth_angle,
            sun_altitude_angle=start.sun_altitude_angle,
        )

        world.set_weather(weather_params)

        if t >= 1.0:
            self._current_preset = target
            self._transition_start = None
            self._transition_target = None
        else:
            self._current_preset = None

        return weather_params

    def get_weather_info(self, world: carla.World) -> Dict[str, float]:
        """Retrieve current weather status information from the world.

        Args:
            world: CARLA world instance.

        Returns:
            Dictionary containing:
                - rain_intensity (float): Current precipitation level (0-100).
                - fog_intensity (float): Current fog density level (0-100).
                - cloudiness (float): Current cloudiness level (0-100).
                - wetness (float): Current road surface wetness level (0-100).
                - wind_intensity (float): Current wind intensity level (0-100).
                - visibility (float): Visibility computed as max(0.0, 100.0 - fog_density).
        """
        current = world.get_weather()
        fog = float(current.fog_density)
        visibility = float(max(0.0, 100.0 - fog))
        return {
            "rain_intensity": float(current.precipitation),
            "fog_intensity": fog,
            "cloudiness": float(current.cloudiness),
            "wetness": float(current.wetness),
            "wind_intensity": float(current.wind_intensity),
            "visibility": visibility,
        }
