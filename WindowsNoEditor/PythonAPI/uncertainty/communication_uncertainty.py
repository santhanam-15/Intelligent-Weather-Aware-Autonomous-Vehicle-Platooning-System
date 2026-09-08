"""Vehicle-to-Vehicle (V2V) communication uncertainty simulation module.

This module provides a pure-Python simulation of V2V communication channels with
configurable latency, jitter, packet drop, Gaussian sensor/state noise, and
communication outages. It does not interact with CARLA physics directly, making
it lightweight, modular, and fast.

Key features:
    - CommPreset configurations (NORMAL, MEDIUM, BAD).
    - Latency modeling with random jitter (±20% of base delay).
    - Probabilistic packet loss.
    - Temporary communication outages with configurable duration.
    - Gaussian noise addition to numerical fields in payload dicts.
    - Complete reproducibility via random seeds.
"""

import copy
from dataclasses import dataclass
from enum import Enum
import random
from typing import Any, Dict, List, Optional, Union

import carla


class CommPreset(Enum):
    """Presets representing different V2V communication channel condition levels."""

    NORMAL = "normal"
    MEDIUM = "medium"
    BAD = "bad"


class _PresetDict(dict):
    """Dictionary supporting lookup by CommPreset enum or string key."""

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            try:
                return super().__getitem__(CommPreset[key.upper()])
            except (KeyError, ValueError):
                try:
                    return super().__getitem__(CommPreset(key.lower()))
                except (KeyError, ValueError):
                    pass
        return super().__getitem__(key)

    def __contains__(self, key: Any) -> bool:
        if isinstance(key, str):
            try:
                return super().__contains__(CommPreset[key.upper()])
            except (KeyError, ValueError):
                try:
                    return super().__contains__(CommPreset(key.lower()))
                except (KeyError, ValueError):
                    pass
        return super().__contains__(key)

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


COMM_PRESETS: Dict[CommPreset, Dict[str, float]] = _PresetDict({
    CommPreset.NORMAL: {
        "delay_ms": 0,
        "packet_loss_rate": 0.0,
        "noise_std": 0.0,
        "outage_probability": 0.0,
    },
    CommPreset.MEDIUM: {
        "delay_ms": 100,
        "packet_loss_rate": 0.05,
        "noise_std": 0.01,
        "outage_probability": 0.01,
    },
    CommPreset.BAD: {
        "delay_ms": 250,
        "packet_loss_rate": 0.15,
        "noise_std": 0.05,
        "outage_probability": 0.05,
    },
})


@dataclass
class Message:
    """Represents a V2V communication packet transmitted across the channel.

    Attributes:
        sender_id: ID of the transmitting vehicle/agent.
        receiver_id: ID of the destination vehicle/agent.
        data: Payload dictionary carried by the message.
        send_time: Simulation timestamp (seconds) when the message was sent.
        receive_time: Simulation timestamp (seconds) when the message is delivered.
        is_corrupted: Flag indicating if noise/corruption was applied to the payload.
    """

    sender_id: int
    receiver_id: int
    data: dict
    send_time: float
    receive_time: float
    is_corrupted: bool = False


def _add_noise_to_data(data: dict, noise_std: float, rng: random.Random) -> dict:
    """Recursively add Gaussian noise to numeric values in a dictionary.

    Non-numeric values (such as strings, booleans, None) are left unchanged.
    Nested dictionaries, lists, and tuples containing numbers are recursively processed.

    Args:
        data: Input dictionary containing message payload.
        noise_std: Standard deviation of the zero-mean Gaussian noise.
        rng: random.Random instance for deterministic noise generation.

    Returns:
        dict: A new dictionary with Gaussian noise added to numeric values.
    """
    if noise_std <= 0.0:
        return copy.deepcopy(data)

    def _process_value(val: Any) -> Any:
        # In Python, bool is a subclass of int (isinstance(True, int) == True).
        # We must explicitly check for bool first to avoid corrupting boolean flags.
        if isinstance(val, bool):
            return val
        elif isinstance(val, (int, float)):
            return val + rng.gauss(0.0, noise_std)
        elif isinstance(val, dict):
            return {k: _process_value(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [_process_value(elem) for elem in val]
        elif isinstance(val, tuple):
            return tuple(_process_value(elem) for elem in val)
        else:
            return copy.deepcopy(val)

    return {k: _process_value(v) for k, v in data.items()}


class V2VChannel:
    """Simulates a Vehicle-to-Vehicle (V2V) wireless communication channel.

    Models transmission delay with jitter, packet loss, communication outages,
    and payload noise corruption.

    Attributes:
        _rng: Instance of random.Random for reproducible stochastic behavior.
        _message_buffer: List of Message objects currently in transit.
        _outage_active: Boolean flag indicating if an outage is currently active.
        _outage_end_time: Simulation timestamp when the current outage ends.
        _messages_sent: Counter for total transmission attempts.
        _messages_lost: Counter for messages lost due to outage or packet drop.
        _messages_delivered: Counter for messages delivered to receivers.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        delay_ms: float = 0,
        packet_loss_rate: float = 0.0,
        noise_std: float = 0.0,
        outage_probability: float = 0.0,
        outage_duration_s: float = 5.0,
    ) -> None:
        """Initialize the V2VChannel with given uncertainty parameters.

        Args:
            seed: Optional integer seed for random number generator.
            delay_ms: Base transmission delay in milliseconds (default 0).
            packet_loss_rate: Probability of packet drop [0.0, 1.0] (default 0.0).
            noise_std: Standard deviation of Gaussian noise added to numeric data (default 0.0).
            outage_probability: Probability of entering an outage on update() [0.0, 1.0] (default 0.0).
            outage_duration_s: Duration of an outage in seconds (default 5.0).
        """
        self._seed = seed
        self._rng = random.Random(seed)
        self._delay_ms = float(delay_ms)
        self._packet_loss_rate = float(packet_loss_rate)
        self._noise_std = float(noise_std)
        self._outage_probability = float(outage_probability)
        self._outage_duration_s = float(outage_duration_s)

        self._message_buffer: List[Message] = []
        self._outage_active: bool = False
        self._outage_end_time: float = 0.0

        self._messages_sent: int = 0
        self._messages_lost: int = 0
        self._messages_delivered: int = 0

    @staticmethod
    def _add_noise_to_data(data: dict, noise_std: float, rng: random.Random) -> dict:
        """Add Gaussian noise to numeric values in a dict recursively.

        Static helper exposing module-level _add_noise_to_data.
        """
        return _add_noise_to_data(data, noise_std, rng)

    def configure_from_preset(self, preset: Union[CommPreset, str]) -> None:
        """Load communication channel parameters from a preset.

        Args:
            preset: CommPreset enum or string ('NORMAL', 'MEDIUM', 'BAD').

        Raises:
            ValueError: If the preset is unrecognized or invalid.
        """
        if isinstance(preset, str):
            try:
                preset_enum = CommPreset[preset.upper()]
            except KeyError:
                try:
                    preset_enum = CommPreset(preset.lower())
                except ValueError:
                    valid_names = [p.name for p in CommPreset]
                    raise ValueError(
                        f"Unknown preset '{preset}'. Expected one of: {valid_names}"
                    )
        elif isinstance(preset, CommPreset):
            preset_enum = preset
        else:
            raise ValueError(
                f"Invalid preset type {type(preset)}. Expected CommPreset or str."
            )

        config = COMM_PRESETS[preset_enum]
        self._delay_ms = float(config["delay_ms"])
        self._packet_loss_rate = float(config["packet_loss_rate"])
        self._noise_std = float(config["noise_std"])
        self._outage_probability = float(config["outage_probability"])

    def send_message(
        self,
        sender_id: int,
        receiver_id: int,
        data: dict,
        current_time: float,
    ) -> bool:
        """Attempt to transmit a message through the V2V channel.

        Checks for active outage and rolls for packet loss. If dropped,
        increments the lost counter and returns False.
        Otherwise, calculates latency with jitter (±20%), applies Gaussian noise
        to numeric values in the payload if noise_std > 0, enqueues the message
        into the transit buffer, and returns True.

        Args:
            sender_id: ID of the transmitting vehicle/agent.
            receiver_id: ID of the target vehicle/agent.
            data: Payload dictionary carried by the message.
            current_time: Current simulation timestamp in seconds.

        Returns:
            bool: True if message was successfully enqueued into the channel buffer,
                  False if dropped due to outage or packet loss.
        """
        self._messages_sent += 1

        # Check if channel outage is active -> drop message
        if self._outage_active:
            self._messages_lost += 1
            return False

        # Roll for packet loss -> drop message
        if self._packet_loss_rate > 0.0 and self._rng.random() < self._packet_loss_rate:
            self._messages_lost += 1
            return False

        # Calculate delay = self._delay_ms / 1000.0 + random jitter (±20% of delay)
        base_delay = self._delay_ms / 1000.0
        if base_delay > 0.0:
            jitter = self._rng.uniform(-0.2 * base_delay, 0.2 * base_delay)
            delay = max(0.0, base_delay + jitter)
        else:
            delay = 0.0

        receive_time = current_time + delay

        # Add noise to numeric values in data dict if noise_std > 0
        if self._noise_std > 0.0:
            payload = _add_noise_to_data(data, self._noise_std, self._rng)
            is_corrupted = True
        else:
            payload = copy.deepcopy(data)
            is_corrupted = False

        # Create message and add to transit buffer
        message = Message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            data=payload,
            send_time=current_time,
            receive_time=receive_time,
            is_corrupted=is_corrupted,
        )
        self._message_buffer.append(message)
        return True

    def receive_messages(
        self,
        receiver_id: int,
        current_time: float,
    ) -> List[Message]:
        """Return and remove all messages from the buffer ready for the receiver.

        Filters buffer for messages where receiver_id matches AND receive_time <= current_time.
        Matching messages are removed from transit buffer, sorted by receive_time,
        and returned.

        Args:
            receiver_id: ID of the receiving vehicle/agent.
            current_time: Current simulation timestamp in seconds.

        Returns:
            List[Message]: Delivered messages ready at current_time.
        """
        delivered: List[Message] = []
        remaining: List[Message] = []

        for msg in self._message_buffer:
            if msg.receiver_id == receiver_id and msg.receive_time <= current_time:
                delivered.append(msg)
            else:
                remaining.append(msg)

        # Sort delivered messages chronologically by receive timestamp
        delivered.sort(key=lambda m: m.receive_time)

        self._message_buffer = remaining
        self._messages_delivered += len(delivered)
        return delivered

    def update(self, current_time: float) -> None:
        """Update the channel state for the current simulation time step.

        If in an outage and current_time >= outage_end_time, deactivates outage.
        If not in an outage, rolls against outage_probability to potentially
        trigger a new outage with duration outage_duration_s.

        Args:
            current_time: Current simulation timestamp in seconds.
        """
        if self._outage_active:
            if current_time >= self._outage_end_time:
                self._outage_active = False
        else:
            if (
                self._outage_probability > 0.0
                and self._rng.random() < self._outage_probability
            ):
                self._outage_active = True
                self._outage_end_time = current_time + self._outage_duration_s

    def is_outage_active(self) -> bool:
        """Check whether a communication outage is currently active.

        Returns:
            bool: True if outage is active, False otherwise.
        """
        return self._outage_active

    def get_stats(self) -> Dict[str, Any]:
        """Return transmission statistics and current outage status.

        Returns:
            dict: Dictionary with:
                - 'messages_sent': Total messages attempted to send.
                - 'messages_lost': Messages dropped by outage or packet loss.
                - 'messages_delivered': Messages received and removed by receivers.
                - 'current_outage': Boolean indicating if outage is currently active.
        """
        return {
            "messages_sent": self._messages_sent,
            "messages_lost": self._messages_lost,
            "messages_delivered": self._messages_delivered,
            "current_outage": self._outage_active,
        }

    def reset(self) -> None:
        """Reset the channel state, clearing message buffer and resetting statistics."""
        self._message_buffer.clear()
        self._outage_active = False
        self._outage_end_time = 0.0
        self._messages_sent = 0
        self._messages_lost = 0
        self._messages_delivered = 0

    def set_seed(self, seed: Optional[int]) -> None:
        """Set random seed for reproducibility.

        Args:
            seed: Integer random seed or None.
        """
        self._seed = seed
        self._rng = random.Random(seed)

    @property
    def delay_ms(self) -> float:
        """Get the base transmission delay in milliseconds."""
        return self._delay_ms

    @delay_ms.setter
    def delay_ms(self, value: float) -> None:
        """Set the base transmission delay in milliseconds."""
        self._delay_ms = float(value)

    @property
    def packet_loss_rate(self) -> float:
        """Get the packet loss rate."""
        return self._packet_loss_rate

    @packet_loss_rate.setter
    def packet_loss_rate(self, value: float) -> None:
        """Set the packet loss rate."""
        self._packet_loss_rate = float(value)

    @property
    def noise_std(self) -> float:
        """Get standard deviation of Gaussian payload noise."""
        return self._noise_std

    @noise_std.setter
    def noise_std(self, value: float) -> None:
        """Set standard deviation of Gaussian payload noise."""
        self._noise_std = float(value)

    @property
    def outage_probability(self) -> float:
        """Get the outage trigger probability per update step."""
        return self._outage_probability

    @outage_probability.setter
    def outage_probability(self, value: float) -> None:
        """Set the outage trigger probability per update step."""
        self._outage_probability = float(value)

    @property
    def outage_duration_s(self) -> float:
        """Get the duration of communication outages in seconds."""
        return self._outage_duration_s

    @outage_duration_s.setter
    def outage_duration_s(self, value: float) -> None:
        """Set the duration of communication outages in seconds."""
        self._outage_duration_s = float(value)

    @property
    def message_buffer(self) -> List[Message]:
        """Get the list of messages currently in transit."""
        return self._message_buffer
