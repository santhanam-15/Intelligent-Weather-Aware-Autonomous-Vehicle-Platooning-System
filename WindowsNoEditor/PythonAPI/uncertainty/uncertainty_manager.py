from __future__ import annotations

"""
Uncertainty Manager for CARLA 0.9.9

Top-level façade that unifies all uncertainty modules into a single
interface. Loads configuration from JSON, sets up sub-modules, and
provides a tick() method for simulation-loop integration.

Compatible with CARLA 0.9.9.
"""

import json
import logging
import os

import carla

from .weather_uncertainty import WeatherManager, WeatherPreset
from .friction_uncertainty import FrictionManager, FrictionCondition
from .vehicle_uncertainty import ActuatorUncertainty
from .communication_uncertainty import V2VChannel, CommPreset
from .traffic_uncertainty import TrafficJamManager, JamSeverity
from .accident_manager import AccidentManager
from .human_behavior import HumanBehaviorManager, DriverProfile
from .event_manager import EventManager, EventType
from .scenario_manager import ScenarioManager, ScenarioLevel

logger = logging.getLogger(__name__)


class UncertaintyManager:
    """
    Unified uncertainty manager for CARLA 0.9.9.

    Combines all uncertainty sub-modules into a single interface.
    Supports configuration from JSON files and scenario presets.

    Data flow:
        CARLA → UncertaintyManager → Modified/Uncertain Environment → CARLA

    Usage:
        manager = UncertaintyManager(seed=42)
        manager.load_config("config/uncertainty_config.json")
        manager.setup(world, ego_vehicle, npc_vehicles, traffic_manager)

        # In simulation loop:
        manager.tick(delta_seconds, current_time)

        # When done:
        manager.cleanup()
    """

    def __init__(self, seed=None):
        """
        Initialize the uncertainty manager.

        Args:
            seed (int, optional): Global random seed for reproducibility.
        """
        self._seed = seed
        self._config = {}

        # Sub-managers
        self.weather = WeatherManager(seed=seed)
        self.friction = FrictionManager(seed=seed)
        self.actuator = ActuatorUncertainty(seed=seed)
        self.communication = V2VChannel(seed=seed)
        self.traffic = TrafficJamManager(seed=seed)
        self.accident = AccidentManager(seed=seed)
        self.behavior = HumanBehaviorManager(seed=seed)
        self.events = EventManager(seed=seed)
        self.scenario = ScenarioManager(seed=seed)

        # Module enable flags
        self._enabled = {
            "weather": True,
            "friction": True,
            "vehicle_uncertainty": True,
            "communication": True,
            "traffic_jam": True,
            "accident": True,
            "human_behavior": True,
            "events": True,
        }

        # References set during setup()
        self._world = None
        self._ego_vehicle = None
        self._npc_vehicles = []
        self._traffic_manager = None
        self._is_setup = False

        logger.info("UncertaintyManager initialized (seed=%s)", seed)

    def load_config(self, config_path):
        """
        Load configuration from a JSON file.

        Args:
            config_path (str): Path to the uncertainty_config.json file.
        """
        if not os.path.exists(config_path):
            logger.warning("Config file not found: %s", config_path)
            return

        with open(config_path, 'r') as f:
            self._config = json.load(f)

        # Apply global settings
        global_cfg = self._config.get("global", {})
        if "seed" in global_cfg and self._seed is None:
            self._seed = global_cfg["seed"]
            self._reinit_submodules()

        # Apply enable/disable flags
        for module_name in self._enabled:
            module_cfg = self._config.get(module_name, {})
            if "enabled" in module_cfg:
                self._enabled[module_name] = module_cfg["enabled"]

        # Configure vehicle uncertainty noise
        vu_cfg = self._config.get("vehicle_uncertainty", {})
        if vu_cfg.get("enabled", True):
            self.actuator.set_noise(
                throttle_noise=vu_cfg.get("throttle_noise"),
                steer_noise=vu_cfg.get("steer_noise"),
                brake_noise=vu_cfg.get("brake_noise"),
            )
            if "delay_ticks" in vu_cfg:
                self.actuator.set_delay(vu_cfg["delay_ticks"])

        # Configure communication
        comm_cfg = self._config.get("communication", {})
        if comm_cfg.get("enabled", True):
            preset_str = comm_cfg.get("preset")
            if preset_str:
                self.communication.configure_from_preset(
                    CommPreset[preset_str])

        # Configure events
        events_cfg = self._config.get("events", {})
        if events_cfg.get("enabled", True):
            event_items = {k: v for k, v in events_cfg.items()
                          if k != "enabled" and isinstance(v, dict)}
            self.events.configure_all_from_dict(event_items)

        logger.info("Configuration loaded from: %s", config_path)

    def load_scenario(self, level):
        """
        Load a predefined scenario level.

        Args:
            level (ScenarioLevel or str): Scenario level
                ('low', 'medium', 'high', 'extreme').
        """
        self._config = self.scenario.configure(level)

        # Apply enable/disable and module configs from scenario
        for module_name in self._enabled:
            module_cfg = self._config.get(module_name, {})
            if "enabled" in module_cfg:
                self._enabled[module_name] = module_cfg["enabled"]

        # Configure vehicle uncertainty
        vu_cfg = self._config.get("vehicle_uncertainty", {})
        if vu_cfg.get("enabled", True):
            self.actuator.set_noise(
                throttle_noise=vu_cfg.get("throttle_noise"),
                steer_noise=vu_cfg.get("steer_noise"),
                brake_noise=vu_cfg.get("brake_noise"),
            )
            if "delay_ticks" in vu_cfg:
                self.actuator.set_delay(vu_cfg["delay_ticks"])

        # Configure communication
        comm_cfg = self._config.get("communication", {})
        if comm_cfg.get("enabled", True):
            preset_str = comm_cfg.get("preset")
            if preset_str:
                self.communication.configure_from_preset(
                    CommPreset[preset_str])

        # Configure events
        events_cfg = self._config.get("events", {})
        if events_cfg.get("enabled", True):
            event_items = {k: v for k, v in events_cfg.items()
                          if k != "enabled" and isinstance(v, dict)}
            self.events.configure_all_from_dict(event_items)

        logger.info("Scenario loaded: %s", level)

    def setup(self, world, ego_vehicle=None, npc_vehicles=None,
              traffic_manager=None):
        """
        Set up the uncertainty manager with CARLA references.

        Args:
            world: CARLA world object.
            ego_vehicle: The ego (player) vehicle actor.
            npc_vehicles (list): List of NPC vehicle actors.
            traffic_manager: CARLA traffic manager instance.
        """
        self._world = world
        self._ego_vehicle = ego_vehicle
        self._npc_vehicles = npc_vehicles or []
        self._traffic_manager = traffic_manager
        self._is_setup = True

        # Apply weather
        if self._enabled["weather"]:
            weather_cfg = self._config.get("weather", {})
            preset_str = weather_cfg.get("preset", "CLEAR")
            try:
                preset = WeatherPreset[preset_str]
                self.weather.set_weather(world, preset)
                logger.info("Weather set to: %s", preset_str)
            except KeyError:
                logger.warning("Unknown weather preset: %s", preset_str)

        # Apply friction to ego vehicle
        if self._enabled["friction"] and ego_vehicle:
            friction_cfg = self._config.get("friction", {})
            condition_str = friction_cfg.get("condition")
            mu_override = friction_cfg.get("mu_override")
            if mu_override is not None:
                self.friction.set_friction(ego_vehicle, mu_override)
            elif condition_str:
                try:
                    condition = FrictionCondition[condition_str]
                    self.friction.set_friction_for_condition(
                        ego_vehicle, condition)
                except KeyError:
                    logger.warning("Unknown friction condition: %s",
                                   condition_str)

        # Apply human behavior profiles to NPCs
        if (self._enabled["human_behavior"] and npc_vehicles
                and traffic_manager):
            behavior_cfg = self._config.get("human_behavior", {})
            weights_dict = behavior_cfg.get("profile_weights")
            if weights_dict:
                # Convert string keys to DriverProfile enums
                weights = {}
                for key, val in weights_dict.items():
                    try:
                        weights[DriverProfile[key]] = val
                    except KeyError:
                        pass
                self.behavior.assign_random_profiles(
                    npc_vehicles, traffic_manager, weights=weights)

        logger.info("UncertaintyManager setup complete with %d NPCs",
                     len(self._npc_vehicles))

    def tick(self, delta_seconds, current_time):
        """
        Update all uncertainty modules for one simulation step.

        Call this once per simulation tick.

        Args:
            delta_seconds (float): Time elapsed since last tick.
            current_time (float): Current simulation time in seconds.

        Returns:
            dict: Summary of events and state changes this tick.
        """
        if not self._is_setup:
            logger.warning("tick() called before setup()")
            return {}

        result = {
            "triggered_events": [],
            "active_events": [],
            "weather_info": None,
            "communication_stats": None,
        }

        # Update event manager
        if self._enabled["events"]:
            triggered = self.events.update(current_time)
            result["triggered_events"] = triggered

            # Execute triggered events
            for event in triggered:
                self._execute_event(event, current_time)

        # Update traffic jams (expire old ones)
        if self._enabled["traffic_jam"]:
            self.traffic.update(current_time)

            # Maybe trigger new jam
            jam_cfg = self._config.get("traffic_jam", {})
            if jam_cfg.get("enabled", True) and self._traffic_manager:
                severity_str = jam_cfg.get("severity", "MEDIUM")
                try:
                    severity = JamSeverity[severity_str]
                except KeyError:
                    severity = JamSeverity.MEDIUM

                self.traffic.maybe_trigger_jam(
                    self._world,
                    self._traffic_manager,
                    current_time,
                    probability=jam_cfg.get("probability", 0.1),
                    severity=severity,
                    duration=jam_cfg.get("duration"),
                    num_vehicles=jam_cfg.get("num_vehicles"),
                )

        # Update accidents (expire old ones)
        if self._enabled["accident"]:
            self.accident.update(current_time)

        # Update communication (check outages)
        if self._enabled["communication"]:
            self.communication.update(current_time)
            result["communication_stats"] = self.communication.get_stats()

        # Update distracted/erratic driver behaviors
        if (self._enabled["human_behavior"]
                and self._traffic_manager):
            self.behavior.update_distracted_vehicles(
                self._traffic_manager, current_time)

        # Get weather info
        if self._enabled["weather"]:
            result["weather_info"] = self.weather.get_weather_info(
                self._world)

        # Get active events
        result["active_events"] = self.events.get_active_events()

        return result

    def apply_control_with_uncertainty(self, vehicle, control):
        """
        Apply a vehicle control with actuator uncertainty.

        Args:
            vehicle: CARLA vehicle actor.
            control: carla.VehicleControl to apply.

        Returns:
            carla.VehicleControl: The actual control applied (with noise).
        """
        if self._enabled["vehicle_uncertainty"]:
            uncertain_control = self.actuator.apply_uncertainty(control)
            vehicle.apply_control(uncertain_control)
            return uncertain_control
        else:
            vehicle.apply_control(control)
            return control

    def send_v2v_message(self, sender_id, receiver_id, data, current_time):
        """
        Send a V2V message through the uncertain communication channel.

        Args:
            sender_id (int): Sender vehicle ID.
            receiver_id (int): Receiver vehicle ID.
            data (dict): Message data.
            current_time (float): Current simulation time.

        Returns:
            bool: True if message was sent (not lost), False otherwise.
        """
        if self._enabled["communication"]:
            return self.communication.send_message(
                sender_id, receiver_id, data, current_time)
        return True

    def receive_v2v_messages(self, receiver_id, current_time):
        """
        Receive V2V messages for a vehicle.

        Args:
            receiver_id (int): Receiver vehicle ID.
            current_time (float): Current simulation time.

        Returns:
            list: List of received Message objects.
        """
        if self._enabled["communication"]:
            return self.communication.receive_messages(
                receiver_id, current_time)
        return []

    def _execute_event(self, event, current_time):
        """
        Execute a triggered event by calling the appropriate sub-module.

        Args:
            event: Event object from the EventManager.
            current_time (float): Current simulation time.
        """
        et = event.event_type

        if et == EventType.SUDDEN_BRAKING and self._npc_vehicles:
            import random
            rng = random.Random(self._seed)
            target = rng.choice(self._npc_vehicles)
            self.behavior.trigger_sudden_brake(target)
            logger.info("Executed SUDDEN_BRAKING on vehicle %d", target.id)

        elif et == EventType.SUDDEN_ACCELERATION and self._npc_vehicles:
            import random
            rng = random.Random(self._seed)
            target = rng.choice(self._npc_vehicles)
            self.behavior.trigger_sudden_acceleration(target)
            logger.info("Executed SUDDEN_ACCELERATION on vehicle %d",
                        target.id)

        elif et == EventType.LANE_CUT_IN and (self._npc_vehicles
                                               and self._traffic_manager):
            import random
            rng = random.Random(self._seed)
            target = rng.choice(self._npc_vehicles)
            self.behavior.trigger_cut_in(target, self._traffic_manager)
            logger.info("Executed LANE_CUT_IN on vehicle %d", target.id)

        elif et == EventType.TRAFFIC_JAM and self._traffic_manager:
            self.traffic.trigger_jam(
                self._world, self._traffic_manager,
                duration=event.duration)
            logger.info("Executed TRAFFIC_JAM event")

        elif et == EventType.ACCIDENT:
            self.accident.spawn_random_incident(
                self._world, current_time, probability=1.0,
                duration=event.duration)
            logger.info("Executed ACCIDENT event")

        elif et == EventType.STOPPED_VEHICLE:
            self.accident.spawn_stopped_vehicle(
                self._world, duration=event.duration)
            logger.info("Executed STOPPED_VEHICLE event")

        elif et == EventType.COMM_FAILURE:
            self.communication._outage_active = True
            self.communication._outage_end_time = (
                current_time + event.duration)
            logger.info("Executed COMM_FAILURE event (dur=%.1f)",
                        event.duration)

        elif et == EventType.SUDDEN_RAIN:
            self.weather.set_weather(self._world, WeatherPreset.HEAVY_RAIN)
            logger.info("Executed SUDDEN_RAIN event")

        elif et == EventType.SUDDEN_FOG:
            self.weather.set_weather(self._world, WeatherPreset.HEAVY_FOG)
            logger.info("Executed SUDDEN_FOG event")

        elif et == EventType.FRICTION_REDUCTION and self._ego_vehicle:
            self.friction.set_friction_for_condition(
                self._ego_vehicle, FrictionCondition.SLIPPERY)
            logger.info("Executed FRICTION_REDUCTION event")

    def get_state(self):
        """
        Get a summary of the current uncertainty state.

        Returns:
            dict: Summary of all active uncertainties.
        """
        state = {
            "enabled_modules": {k: v for k, v in self._enabled.items()},
            "scenario_level": (self.scenario.get_level().value
                               if self.scenario.get_level() else None),
            "event_stats": self.events.get_stats(),
        }

        if self._is_setup:
            if self._enabled["weather"]:
                state["weather"] = self.weather.get_weather_info(
                    self._world)
            if self._enabled["friction"] and self._ego_vehicle:
                state["friction_mu"] = self.friction.get_friction(
                    self._ego_vehicle)
            if self._enabled["communication"]:
                state["communication"] = self.communication.get_stats()
            if self._enabled["vehicle_uncertainty"]:
                state["actuator"] = self.actuator.get_config()
            if self._enabled["human_behavior"]:
                state["behavior_profiles"] = self.behavior.get_all_profiles()
            if self._enabled["traffic_jam"]:
                state["active_jams"] = len(
                    self.traffic.get_active_jams())
            if self._enabled["accident"]:
                state["active_accidents"] = len(
                    self.accident.get_active_events())

        return state

    def enable_module(self, module_name, enabled=True):
        """
        Enable or disable a specific uncertainty module.

        Args:
            module_name (str): Module name (e.g., 'weather', 'friction').
            enabled (bool): True to enable, False to disable.
        """
        if module_name in self._enabled:
            self._enabled[module_name] = enabled
            logger.info("Module '%s' %s",
                        module_name, "enabled" if enabled else "disabled")
        else:
            logger.warning("Unknown module: %s", module_name)

    def cleanup(self):
        """
        Clean up all spawned actors and reset state.

        Call this when the simulation ends.
        """
        self.traffic.cleanup()
        self.accident.cleanup()
        self.events.reset()
        self._is_setup = False
        logger.info("UncertaintyManager cleanup complete")

    def _reinit_submodules(self):
        """Reinitialize sub-modules with the updated seed."""
        self.weather = WeatherManager(seed=self._seed)
        self.friction = FrictionManager(seed=self._seed)
        self.actuator = ActuatorUncertainty(seed=self._seed)
        self.communication = V2VChannel(seed=self._seed)
        self.traffic = TrafficJamManager(seed=self._seed)
        self.accident = AccidentManager(seed=self._seed)
        self.behavior = HumanBehaviorManager(seed=self._seed)
        self.events = EventManager(seed=self._seed)

    def __repr__(self):
        return (f"UncertaintyManager(seed={self._seed}, "
                f"setup={self._is_setup}, "
                f"level={self.scenario.get_level()})")

