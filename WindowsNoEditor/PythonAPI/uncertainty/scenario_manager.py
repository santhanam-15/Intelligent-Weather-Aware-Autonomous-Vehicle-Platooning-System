"""
Scenario Manager for CARLA 0.9.9

Combines all uncertainty modules into predefined scenario levels:
LOW, MEDIUM, HIGH, EXTREME.

Each level configures weather, friction, communication, traffic,
human behavior, and event probabilities to create a cohesive
uncertainty scenario.

Compatible with CARLA 0.9.9.
"""

import logging
from enum import Enum

logger = logging.getLogger(__name__)


class ScenarioLevel(Enum):
    """Predefined scenario difficulty levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


# Scenario presets: each level defines parameters for all sub-modules
SCENARIO_PRESETS = {
    ScenarioLevel.LOW: {
        "description": "Mild weather variation, normal traffic, minimal uncertainty",
        "weather": {
            "enabled": True,
            "preset": "CLOUDY",
        },
        "friction": {
            "enabled": True,
            "condition": "DRY",
        },
        "vehicle_uncertainty": {
            "enabled": True,
            "throttle_noise": 0.02,
            "steer_noise": 0.01,
            "brake_noise": 0.02,
            "delay_ticks": 0,
        },
        "communication": {
            "enabled": True,
            "preset": "NORMAL",
        },
        "traffic_jam": {
            "enabled": False,
            "probability": 0.0,
        },
        "accident": {
            "enabled": False,
            "probability": 0.0,
        },
        "human_behavior": {
            "enabled": True,
            "profile_weights": {
                "NORMAL": 0.70,
                "AGGRESSIVE": 0.05,
                "CONSERVATIVE": 0.15,
                "DISTRACTED": 0.05,
                "ERRATIC": 0.0,
                "SLOW": 0.05,
            },
        },
        "events": {
            "enabled": True,
            "sudden_braking": {"probability": 0.01, "cooldown": 60.0,
                               "duration": 2.0, "severity": "low"},
            "sudden_acceleration": {"probability": 0.01, "cooldown": 60.0,
                                    "duration": 3.0, "severity": "low"},
        },
    },

    ScenarioLevel.MEDIUM: {
        "description": "Rain, friction variation, comm delay, behavior variation",
        "weather": {
            "enabled": True,
            "preset": "MODERATE_RAIN",
        },
        "friction": {
            "enabled": True,
            "condition": "WET",
        },
        "vehicle_uncertainty": {
            "enabled": True,
            "throttle_noise": 0.05,
            "steer_noise": 0.02,
            "brake_noise": 0.05,
            "delay_ticks": 1,
        },
        "communication": {
            "enabled": True,
            "preset": "MEDIUM",
        },
        "traffic_jam": {
            "enabled": True,
            "probability": 0.05,
            "severity": "LIGHT",
            "duration": 20.0,
            "num_vehicles": 3,
        },
        "accident": {
            "enabled": True,
            "probability": 0.02,
            "duration": 30.0,
        },
        "human_behavior": {
            "enabled": True,
            "profile_weights": {
                "NORMAL": 0.40,
                "AGGRESSIVE": 0.15,
                "CONSERVATIVE": 0.15,
                "DISTRACTED": 0.10,
                "ERRATIC": 0.10,
                "SLOW": 0.10,
            },
        },
        "events": {
            "enabled": True,
            "sudden_braking": {"probability": 0.05, "cooldown": 30.0,
                               "duration": 3.0, "severity": "medium"},
            "sudden_acceleration": {"probability": 0.03, "cooldown": 30.0,
                                    "duration": 5.0, "severity": "medium"},
            "lane_cut_in": {"probability": 0.03, "cooldown": 25.0,
                            "duration": 5.0, "severity": "medium"},
            "comm_failure": {"probability": 0.02, "cooldown": 45.0,
                             "duration": 8.0, "severity": "medium"},
        },
    },

    ScenarioLevel.HIGH: {
        "description": "Heavy rain, low friction, traffic jams, aggressive "
                       "drivers, communication problems",
        "weather": {
            "enabled": True,
            "preset": "HEAVY_RAIN",
        },
        "friction": {
            "enabled": True,
            "condition": "SLIPPERY",
        },
        "vehicle_uncertainty": {
            "enabled": True,
            "throttle_noise": 0.08,
            "steer_noise": 0.04,
            "brake_noise": 0.08,
            "delay_ticks": 2,
        },
        "communication": {
            "enabled": True,
            "preset": "BAD",
        },
        "traffic_jam": {
            "enabled": True,
            "probability": 0.15,
            "severity": "MEDIUM",
            "duration": 30.0,
            "num_vehicles": 5,
        },
        "accident": {
            "enabled": True,
            "probability": 0.05,
            "duration": 45.0,
        },
        "human_behavior": {
            "enabled": True,
            "profile_weights": {
                "NORMAL": 0.20,
                "AGGRESSIVE": 0.25,
                "CONSERVATIVE": 0.10,
                "DISTRACTED": 0.15,
                "ERRATIC": 0.20,
                "SLOW": 0.10,
            },
        },
        "events": {
            "enabled": True,
            "sudden_braking": {"probability": 0.10, "cooldown": 20.0,
                               "duration": 4.0, "severity": "high"},
            "sudden_acceleration": {"probability": 0.05, "cooldown": 25.0,
                                    "duration": 5.0, "severity": "high"},
            "lane_cut_in": {"probability": 0.08, "cooldown": 15.0,
                            "duration": 5.0, "severity": "high"},
            "traffic_jam": {"probability": 0.03, "cooldown": 45.0,
                            "duration": 30.0, "severity": "high"},
            "accident": {"probability": 0.02, "cooldown": 90.0,
                         "duration": 45.0, "severity": "high"},
            "comm_failure": {"probability": 0.05, "cooldown": 30.0,
                             "duration": 12.0, "severity": "high"},
            "sudden_rain": {"probability": 0.03, "cooldown": 45.0,
                            "duration": 25.0, "severity": "high"},
            "friction_reduction": {"probability": 0.03, "cooldown": 45.0,
                                   "duration": 15.0, "severity": "high"},
        },
    },

    ScenarioLevel.EXTREME: {
        "description": "Heavy rain + fog, very low friction, traffic jams, "
                       "accidents, erratic drivers, comm outage, sudden events",
        "weather": {
            "enabled": True,
            "preset": "RAIN_FOG",
        },
        "friction": {
            "enabled": True,
            "condition": "VERY_SLIPPERY",
        },
        "vehicle_uncertainty": {
            "enabled": True,
            "throttle_noise": 0.12,
            "steer_noise": 0.06,
            "brake_noise": 0.12,
            "delay_ticks": 3,
        },
        "communication": {
            "enabled": True,
            "preset": "BAD",
            "outage_probability": 0.10,
            "outage_duration_s": 10.0,
        },
        "traffic_jam": {
            "enabled": True,
            "probability": 0.25,
            "severity": "HEAVY",
            "duration": 45.0,
            "num_vehicles": 8,
        },
        "accident": {
            "enabled": True,
            "probability": 0.10,
            "duration": 60.0,
        },
        "human_behavior": {
            "enabled": True,
            "profile_weights": {
                "NORMAL": 0.10,
                "AGGRESSIVE": 0.30,
                "CONSERVATIVE": 0.05,
                "DISTRACTED": 0.15,
                "ERRATIC": 0.30,
                "SLOW": 0.10,
            },
        },
        "events": {
            "enabled": True,
            "sudden_braking": {"probability": 0.15, "cooldown": 10.0,
                               "duration": 5.0, "severity": "high"},
            "sudden_acceleration": {"probability": 0.08, "cooldown": 15.0,
                                    "duration": 5.0, "severity": "high"},
            "lane_cut_in": {"probability": 0.12, "cooldown": 10.0,
                            "duration": 5.0, "severity": "high"},
            "traffic_jam": {"probability": 0.05, "cooldown": 30.0,
                            "duration": 40.0, "severity": "high"},
            "accident": {"probability": 0.04, "cooldown": 60.0,
                         "duration": 60.0, "severity": "high"},
            "stopped_vehicle": {"probability": 0.05, "cooldown": 30.0,
                                "duration": 30.0, "severity": "high"},
            "comm_failure": {"probability": 0.08, "cooldown": 20.0,
                             "duration": 15.0, "severity": "high"},
            "sudden_rain": {"probability": 0.05, "cooldown": 30.0,
                            "duration": 30.0, "severity": "high"},
            "sudden_fog": {"probability": 0.05, "cooldown": 30.0,
                           "duration": 30.0, "severity": "high"},
            "friction_reduction": {"probability": 0.05, "cooldown": 30.0,
                                   "duration": 20.0, "severity": "high"},
        },
    },
}


class ScenarioManager:
    """
    Manages predefined uncertainty scenarios for CARLA 0.9.9.

    Provides LOW, MEDIUM, HIGH, EXTREME difficulty levels that
    configure all uncertainty sub-modules with appropriate parameters.

    This manager does NOT control AI models, reward functions, or
    training. It only configures the CARLA environment uncertainty.

    Usage:
        manager = ScenarioManager(seed=42)
        config = manager.configure(ScenarioLevel.HIGH)
        # Use the returned config dict to set up sub-modules
    """

    def __init__(self, seed=None):
        """
        Initialize the scenario manager.

        Args:
            seed (int, optional): Random seed for reproducibility.
        """
        self._seed = seed
        self._current_level = None
        self._current_config = None
        logger.info("ScenarioManager initialized (seed=%s)", seed)

    def configure(self, level):
        """
        Get the configuration for a scenario level.

        Args:
            level (ScenarioLevel or str): The scenario level. Can be a
                ScenarioLevel enum or a string ('low', 'medium', 'high',
                'extreme').

        Returns:
            dict: Configuration dictionary for all sub-modules.
        """
        if isinstance(level, str):
            level = ScenarioLevel(level.lower())

        self._current_level = level
        self._current_config = SCENARIO_PRESETS[level].copy()

        # Inject global seed
        self._current_config["global"] = {
            "seed": self._seed,
            "scenario_level": level.value,
        }

        logger.info("Scenario configured: %s - %s",
                     level.value,
                     SCENARIO_PRESETS[level]["description"])

        return self._current_config

    def get_level(self):
        """
        Get the current scenario level.

        Returns:
            ScenarioLevel or None: Current level, or None if not configured.
        """
        return self._current_level

    def get_config(self):
        """
        Get the current configuration.

        Returns:
            dict or None: Current config, or None if not configured.
        """
        return self._current_config

    def get_description(self, level=None):
        """
        Get a human-readable description of a scenario level.

        Args:
            level (ScenarioLevel, optional): Level to describe.
                Defaults to current level.

        Returns:
            str: Description of the scenario.
        """
        if level is None:
            level = self._current_level
        if level is None:
            return "No scenario configured"
        return SCENARIO_PRESETS[level]["description"]

    def get_available_levels(self):
        """
        Get all available scenario levels.

        Returns:
            list: List of ScenarioLevel enum values.
        """
        return list(ScenarioLevel)

    def get_module_config(self, module_name):
        """
        Get the configuration for a specific module.

        Args:
            module_name (str): Name of the module (e.g., 'weather',
                'friction', 'communication').

        Returns:
            dict or None: Module configuration, or None if not found.
        """
        if self._current_config is None:
            return None
        return self._current_config.get(module_name)

    def customize(self, overrides):
        """
        Apply custom overrides to the current configuration.

        Args:
            overrides (dict): Dictionary of overrides to apply.
                Keys are module names, values are dicts of parameters.
                Example: {"friction": {"condition": "SLIPPERY"}}

        Returns:
            dict: Updated configuration.
        """
        if self._current_config is None:
            raise RuntimeError("No scenario configured. Call configure() first.")

        for module_name, params in overrides.items():
            if module_name in self._current_config:
                self._current_config[module_name].update(params)
            else:
                self._current_config[module_name] = params

        logger.info("Scenario customized with overrides: %s",
                     list(overrides.keys()))
        return self._current_config

    def __repr__(self):
        if self._current_level:
            return (f"ScenarioManager(level={self._current_level.value}, "
                    f"seed={self._seed})")
        return f"ScenarioManager(unconfigured, seed={self._seed})"

