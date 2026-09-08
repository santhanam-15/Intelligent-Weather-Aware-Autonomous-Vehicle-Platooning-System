"""
Sudden Event Manager for CARLA 0.9.9

Triggers unexpected events such as sudden braking, lane cut-ins,
traffic jams, accidents, communication failures, weather changes,
and friction reductions with configurable probability, cooldown,
duration, and severity.

Compatible with CARLA 0.9.9.
"""

import random
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of sudden events that can be triggered."""
    NO_EVENT = "no_event"
    SUDDEN_BRAKING = "sudden_braking"
    SUDDEN_ACCELERATION = "sudden_acceleration"
    LANE_CUT_IN = "lane_cut_in"
    TRAFFIC_JAM = "traffic_jam"
    ACCIDENT = "accident"
    STOPPED_VEHICLE = "stopped_vehicle"
    COMM_FAILURE = "comm_failure"
    SUDDEN_RAIN = "sudden_rain"
    SUDDEN_FOG = "sudden_fog"
    FRICTION_REDUCTION = "friction_reduction"


class Event:
    """Represents a triggered event with its metadata."""

    def __init__(self, event_type, trigger_time, duration=10.0,
                 severity="medium", data=None):
        """
        Args:
            event_type (EventType): The type of event.
            trigger_time (float): Simulation time when the event was triggered.
            duration (float): How long the event lasts in seconds.
            severity (str): 'low', 'medium', or 'high'.
            data (dict): Additional event-specific data.
        """
        self.event_type = event_type
        self.trigger_time = trigger_time
        self.duration = duration
        self.severity = severity
        self.data = data or {}
        self.is_active = True

    @property
    def end_time(self):
        """Time when this event expires."""
        return self.trigger_time + self.duration

    def __repr__(self):
        return (f"Event({self.event_type.value}, t={self.trigger_time:.1f}, "
                f"dur={self.duration:.1f}, sev={self.severity})")


class EventConfig:
    """Configuration for a single event type."""

    def __init__(self, event_type, probability=0.05, cooldown=30.0,
                 duration=10.0, severity="medium", enabled=True):
        """
        Args:
            event_type (EventType): The event type this config applies to.
            probability (float): Probability of triggering per update call (0-1).
            cooldown (float): Minimum seconds between triggers of this type.
            duration (float): Default duration of the event in seconds.
            severity (str): Default severity: 'low', 'medium', or 'high'.
            enabled (bool): Whether this event type is enabled.
        """
        self.event_type = event_type
        self.probability = probability
        self.cooldown = cooldown
        self.duration = duration
        self.severity = severity
        self.enabled = enabled


class EventManager:
    """
    Manages sudden/unexpected events in the CARLA simulation.

    Provides a probabilistic event trigger system with cooldowns,
    configurable severity, and duration tracking.

    Usage:
        manager = EventManager(seed=42)
        manager.register_event(EventType.SUDDEN_BRAKING, probability=0.1)
        manager.register_event(EventType.TRAFFIC_JAM, probability=0.05)

        # In simulation loop:
        triggered = manager.update(current_time)
        for event in triggered:
            # Handle event based on event.event_type
            pass
    """

    def __init__(self, seed=None):
        """
        Initialize the event manager.

        Args:
            seed (int, optional): Random seed for reproducibility.
        """
        self._rng = random.Random(seed)
        self._seed = seed
        self._event_configs = {}  # EventType -> EventConfig
        self._last_trigger_time = {}  # EventType -> float
        self._active_events = []  # List of active Event objects
        self._event_history = []  # All triggered events
        self._total_updates = 0

        logger.info("EventManager initialized (seed=%s)", seed)

    def register_event(self, event_type, probability=0.05, cooldown=30.0,
                       duration=10.0, severity="medium", enabled=True):
        """
        Register an event type with its configuration.

        Args:
            event_type (EventType): The event type to register.
            probability (float): Probability of triggering per update (0-1).
            cooldown (float): Minimum seconds between triggers.
            duration (float): Default event duration in seconds.
            severity (str): Default severity: 'low', 'medium', 'high'.
            enabled (bool): Whether this event is enabled.
        """
        config = EventConfig(
            event_type=event_type,
            probability=probability,
            cooldown=cooldown,
            duration=duration,
            severity=severity,
            enabled=enabled
        )
        self._event_configs[event_type] = config
        self._last_trigger_time[event_type] = -999999.0
        logger.debug("Registered event: %s (prob=%.3f, cd=%.1f)",
                      event_type.value, probability, cooldown)

    def unregister_event(self, event_type):
        """
        Remove an event type from the manager.

        Args:
            event_type (EventType): The event type to remove.
        """
        self._event_configs.pop(event_type, None)
        self._last_trigger_time.pop(event_type, None)

    def enable_event(self, event_type, enabled=True):
        """
        Enable or disable an event type.

        Args:
            event_type (EventType): The event type.
            enabled (bool): True to enable, False to disable.
        """
        if event_type in self._event_configs:
            self._event_configs[event_type].enabled = enabled

    def update(self, current_time):
        """
        Check all registered events and trigger any that pass
        their probability roll and cooldown check.

        Args:
            current_time (float): Current simulation time in seconds.

        Returns:
            list: List of newly triggered Event objects. Empty list if
                  no events were triggered.
        """
        self._total_updates += 1
        triggered_events = []

        # Expire old active events
        still_active = []
        for event in self._active_events:
            if current_time >= event.end_time:
                event.is_active = False
                logger.info("Event expired: %s", event)
            else:
                still_active.append(event)
        self._active_events = still_active

        # Check each registered event type
        for event_type, config in self._event_configs.items():
            if not config.enabled:
                continue

            # Check cooldown
            time_since_last = current_time - self._last_trigger_time[event_type]
            if time_since_last < config.cooldown:
                continue

            # Roll probability
            if self._rng.random() < config.probability:
                event = Event(
                    event_type=event_type,
                    trigger_time=current_time,
                    duration=config.duration,
                    severity=config.severity
                )
                self._active_events.append(event)
                self._event_history.append(event)
                self._last_trigger_time[event_type] = current_time
                triggered_events.append(event)
                logger.info("Event triggered: %s at t=%.1f",
                            event_type.value, current_time)

        return triggered_events

    def force_trigger(self, event_type, current_time, duration=None,
                      severity=None):
        """
        Force-trigger a specific event regardless of probability/cooldown.

        Args:
            event_type (EventType): The event to trigger.
            current_time (float): Current simulation time.
            duration (float, optional): Override duration.
            severity (str, optional): Override severity.

        Returns:
            Event: The triggered event.
        """
        config = self._event_configs.get(event_type)
        dur = duration if duration is not None else (
            config.duration if config else 10.0)
        sev = severity if severity is not None else (
            config.severity if config else "medium")

        event = Event(
            event_type=event_type,
            trigger_time=current_time,
            duration=dur,
            severity=sev
        )
        self._active_events.append(event)
        self._event_history.append(event)
        self._last_trigger_time[event_type] = current_time
        logger.info("Force triggered: %s at t=%.1f", event_type.value,
                    current_time)
        return event

    def get_active_events(self):
        """
        Get all currently active events.

        Returns:
            list: List of active Event objects.
        """
        return list(self._active_events)

    def get_active_event_types(self):
        """
        Get the set of currently active event types.

        Returns:
            set: Set of EventType values.
        """
        return {e.event_type for e in self._active_events}

    def is_event_active(self, event_type):
        """
        Check if a specific event type is currently active.

        Args:
            event_type (EventType): The event type to check.

        Returns:
            bool: True if the event is currently active.
        """
        return any(e.event_type == event_type for e in self._active_events)

    def get_history(self):
        """
        Get the full history of triggered events.

        Returns:
            list: All Event objects ever triggered.
        """
        return list(self._event_history)

    def get_stats(self):
        """
        Get event manager statistics.

        Returns:
            dict: Stats including total updates, events triggered, active count.
        """
        return {
            "total_updates": self._total_updates,
            "total_events_triggered": len(self._event_history),
            "active_events": len(self._active_events),
            "registered_event_types": len(self._event_configs),
            "active_event_types": [e.event_type.value
                                   for e in self._active_events]
        }

    def reset(self):
        """Reset the event manager, clearing all active events and history."""
        self._active_events.clear()
        self._event_history.clear()
        for et in self._last_trigger_time:
            self._last_trigger_time[et] = -999999.0
        self._total_updates = 0
        logger.info("EventManager reset")

    def configure_all_from_dict(self, config_dict):
        """
        Configure events from a dictionary.

        Args:
            config_dict (dict): Mapping of event type string to config params.
                Example: {
                    "sudden_braking": {"probability": 0.1, "cooldown": 20},
                    "traffic_jam": {"probability": 0.05, "enabled": False}
                }
        """
        type_map = {et.value: et for et in EventType}
        for type_str, params in config_dict.items():
            event_type = type_map.get(type_str)
            if event_type and event_type != EventType.NO_EVENT:
                self.register_event(event_type, **params)

