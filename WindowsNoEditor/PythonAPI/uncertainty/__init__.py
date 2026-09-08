"""CARLA 0.9.9 Uncertainty Toolkit.

This package provides modules for introducing realistic physical,
environmental, and behavioral uncertainties into the CARLA 0.9.9 simulator.

Modules:
    weather_uncertainty     - Weather presets and transitions
    friction_uncertainty    - Road/tire friction control
    vehicle_uncertainty     - Actuator noise and delay
    communication_uncertainty - V2V communication simulation
    traffic_uncertainty     - Traffic jam generation
    accident_manager        - Accidents and road obstacles
    human_behavior          - Human driver behavior profiles
    event_manager           - Sudden event triggers
    scenario_manager        - Predefined scenario levels
    uncertainty_manager     - Unified top-level API
"""

import glob
import os
import sys

# Set up sys.path to find the CARLA egg
try:
    sys.path.append(
        glob.glob(
            os.path.join(
                os.path.dirname(__file__),
                '..',
                'carla',
                'dist',
                'carla-*%d.%d-%s.egg'
                % (
                    sys.version_info.major,
                    sys.version_info.minor,
                    'win-amd64' if os.name == 'nt' else 'linux-x86_64',
                ),
            )
        )[0]
    )
except IndexError:
    pass

import carla  # noqa: E402

__version__ = '1.0.0'

# --- Import and expose all main classes ---

from .weather_uncertainty import WeatherPreset, WeatherManager, WEATHER_PARAMS
from .friction_uncertainty import FrictionCondition, FrictionManager, FRICTION_VALUES
from .vehicle_uncertainty import ActuatorUncertainty
from .communication_uncertainty import V2VChannel, CommPreset, Message
from .traffic_uncertainty import TrafficJamManager, JamSeverity
from .accident_manager import AccidentManager, AccidentType
from .human_behavior import HumanBehaviorManager, DriverProfile
from .event_manager import EventManager, EventType, Event
from .scenario_manager import ScenarioManager, ScenarioLevel
from .uncertainty_manager import UncertaintyManager

__all__ = [
    '__version__',
    # Weather
    'WeatherPreset', 'WeatherManager', 'WEATHER_PARAMS',
    # Friction
    'FrictionCondition', 'FrictionManager', 'FRICTION_VALUES',
    # Vehicle / Actuator
    'ActuatorUncertainty',
    # Communication
    'V2VChannel', 'CommPreset', 'Message',
    # Traffic
    'TrafficJamManager', 'JamSeverity',
    # Accidents
    'AccidentManager', 'AccidentType',
    # Human Behavior
    'HumanBehaviorManager', 'DriverProfile',
    # Events
    'EventManager', 'EventType', 'Event',
    # Scenario
    'ScenarioManager', 'ScenarioLevel',
    # Unified Manager
    'UncertaintyManager',
]
