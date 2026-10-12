# WindowsNoEditor/Module2/logger.py
"""Dedicated thread-safe, non-blocking logger for live MAPPO tactical control telemetry."""

from __future__ import annotations

import datetime
import os
import queue
import sys
import threading
import time
from typing import Any, Dict, List, Optional

from Module2.config import DECISION_LOG_FILE


class MAPPODecisionLogger:
    """Non-blocking queue-based logger streaming MAPPO telemetry to disk for the dedicated console."""

    _instance: Optional["MAPPODecisionLogger"] = None
    _lock = threading.Lock()

    def __init__(self, log_path: str = DECISION_LOG_FILE):
        self.log_path = log_path
        self.log_dir = os.path.dirname(os.path.abspath(log_path))
        if not os.path.exists(self.log_dir):
            try:
                os.makedirs(self.log_dir, exist_ok=True)
            except Exception:
                pass

        self._queue: queue.Queue = queue.Queue(maxsize=10000)
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._worker_thread.start()
        self._write_banner()

    @classmethod
    def get_instance(cls) -> "MAPPODecisionLogger":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _write_banner(self) -> None:
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        banner = (
            "\n" + "=" * 80 + "\n"
            f"  CARLA 0.9.9 - MAPPO TACTICAL CONTROL & PLATOON DECISION STREAM\n"
            f"  Shared Actor-Critic Architecture  |  Time: {now_str}\n"
            f"  Observing: Tactical Actions, Speed-Gap Regulation, Waypoints, Role Failover\n"
            + "=" * 80 + "\n"
        )
        self._enqueue(banner)

    def _enqueue(self, text: str) -> None:
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            pass

    def _writer_loop(self) -> None:
        while not self._stop_event.is_set():
            lines_to_write = []
            try:
                item = self._queue.get(timeout=0.05)
                lines_to_write.append(item)
                while len(lines_to_write) < 100:
                    try:
                        lines_to_write.append(self._queue.get_nowait())
                    except queue.Empty:
                        break
            except queue.Empty:
                continue

            if lines_to_write:
                try:
                    with open(self.log_path, "a", encoding="utf-8") as f:
                        for line in lines_to_write:
                            f.write(line if line.endswith("\n") else line + "\n")
                        f.flush()
                except Exception:
                    pass

    @staticmethod
    def _format_time(sim_time_sec: float) -> str:
        mins = int(sim_time_sec) // 60
        secs = sim_time_sec % 60.0
        return f"{mins:02d}:{secs:06.3f}"

    def log_startup(
        self,
        pid: int,
        channel: str = "MAPPO Tactical Control Stream",
        mode: str = "EVALUATE",
        leader_id: int = 0,
        status: str = "SUCCESS",
    ) -> None:
        """Stream formal connection notification and startup summary."""
        msg = (
            f"[MAPPO CONSOLE] Dedicated MAPPO logging window connected.\n"
            f"[MAPPO STARTUP SUMMARY]\n"
            f"  - Active Process PID:      {pid}\n"
            f"  - Log Channel:             {channel}\n"
            f"  - Inference Mode:          {mode.upper()}\n"
            f"  - Authoritative Leader ID: {leader_id}\n"
            f"  - Logging Connection:      {status}\n"
        )
        self._enqueue(msg)

    def log_initialization(
        self,
        ckpt_path: str,
        ckpt_hash: str,
        obs_dim: int,
        action_dim: int,
        update_step: int,
        is_deterministic: bool,
        actor_loaded: bool,
        critic_loaded: bool,
    ) -> None:
        """Log checkpoint restoration and network initialization parameters."""
        sep = "=" * 80
        msg = (
            f"\n{sep}\n"
            f"  MAPPO CONFIGURATION & CHECKPOINT RESTORATION\n"
            f"{sep}\n"
            f"  CHECKPOINT PATH:           {ckpt_path}\n"
            f"  CHECKPOINT SHA-256:        {ckpt_hash}\n"
            f"  OBSERVATION DIM:           {obs_dim}\n"
            f"  ACTION DIM:                {action_dim}\n"
            f"  UPDATE STEP:               {update_step}\n"
            f"  DETERMINISTIC EVALUATION:  {is_deterministic}\n"
            f"  ACTOR NETWORK LOADED:      {actor_loaded}\n"
            f"  CRITIC NETWORK LOADED:     {critic_loaded}\n"
            f"{sep}\n"
        )
        self._enqueue(msg)

    def log_tactical_cycle(
        self,
        sim_time_sec: float,
        step: int,
        mode: str,
        vehicle_records: List[Dict[str, Any]],
        training_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Stream structured tactical record for all platoon vehicles distinguishing proposed, shield, and executed actions."""
        t_str = self._format_time(sim_time_sec)
        header = f"[MAPPO STEP {step:05d}] {t_str} | Mode={mode.upper()}"
        self._enqueue(header)

        for rec in vehicle_records:
            v_id = rec.get("id", 0)
            role = rec.get("role", "Unknown")
            action = rec.get("action", rec.get("executed_action", "CRUISE"))
            prop_action = rec.get("proposed_action", action)
            shield_action = rec.get("shield_action", action)
            shield_mode = rec.get("shield_mode", "SHADOW")
            reason = rec.get("reason", rec.get("action_reason", "NOMINAL"))
            spd_kmh = rec.get("speed_mps", 0.0) * 3.6
            tgt_kmh = rec.get("target_mps", 0.0) * 3.6
            gap = rec.get("gap_m", 0.0)
            desired = rec.get("desired_gap_m", 0.0)
            cursor = rec.get("cursor", 0)
            total_wps = rec.get("total_wps", 1)
            pct = (cursor / max(1, total_wps)) * 100.0

            gap_str = f"Gap={gap:5.1f}m (Des={desired:5.1f}m)" if "Leader" not in role else "Leading Platoon     "
            
            line = (
                f"  -> Actor={v_id:<3} ({role:<16}) | PROPOSED={prop_action:<11} | "
                f"SHIELD={shield_action:<11} ({shield_mode}) | EXECUTED={action:<11} | "
                f"REASON={reason:<20} | Spd={spd_kmh:4.1f}->{tgt_kmh:4.1f} km/h | {gap_str} | "
                f"WP={cursor:>3}/{total_wps:<3} ({pct:4.1f}%)"
            )
            self._enqueue(line)

        if training_metrics and mode.upper() == "TRAIN":
            loss_a = training_metrics.get("loss_actor", 0.0)
            loss_c = training_metrics.get("loss_critic", 0.0)
            entropy = training_metrics.get("entropy", 0.0)
            up_step = training_metrics.get("update_step", 0)
            m_line = (
                f"  [TRAINING UPDATE #{up_step}] Loss_Actor={loss_a:.4f} | "
                f"Loss_Critic={loss_c:.4f} | Entropy={entropy:.4f}"
            )
            self._enqueue(m_line)

    def log_role_failover(
        self,
        sim_time_sec: float,
        old_leader_id: int,
        promoted_leader_id: int,
        reason: str,
    ) -> None:
        """Stream critical failover notification."""
        t_str = self._format_time(sim_time_sec)
        msg = (
            f"[MAPPO FAILOVER] {t_str} | Old Leader #{old_leader_id} DISABLED -> "
            f"Promoted Leader #{promoted_leader_id} | Reason={reason}"
        )
        self._enqueue(msg)

    def log_overtake_event(
        self,
        sim_time_sec: float,
        vehicle_id: int,
        role: str,
        event_tag: str,
        details: str = "",
    ) -> None:
        """Log structured overtaking event for the dedicated MAPPO terminal."""
        t_str = self._format_time(sim_time_sec)
        detail_part = f" | {details}" if details else ""
        msg = f"[OVERTAKE] {t_str} | Veh={vehicle_id:<3} ({role:<14}) | {event_tag}{detail_part}"
        self._enqueue(msg)

    def log_lane_change_event(
        self,
        sim_time_sec: float,
        vehicle_id: int,
        role: str,
        event_tag: str,
        details: str = "",
    ) -> None:
        """Log structured lane change event for the dedicated MAPPO terminal."""
        t_str = self._format_time(sim_time_sec)
        detail_part = f" | {details}" if details else ""
        msg = f"[LANE_CHANGE] {t_str} | Veh={vehicle_id:<3} ({role:<14}) | {event_tag}{detail_part}"
        self._enqueue(msg)

    def log_platoon_event(
        self,
        sim_time_sec: float,
        vehicle_id: int,
        role: str,
        event_tag: str,
        details: str = "",
    ) -> None:
        """Log structured platoon rejoining and follower status event."""
        t_str = self._format_time(sim_time_sec)
        detail_part = f" | {details}" if details else ""
        msg = f"[PLATOON] {t_str} | Veh={vehicle_id:<3} ({role:<14}) | {event_tag}{detail_part}"
        self._enqueue(msg)

    def log_signal_event(
        self,
        sim_time_sec: float,
        vehicle_id: int,
        role: str,
        event_tag: str,
        details: str = "",
    ) -> None:
        """Log structured signal and intersection compliance event."""
        t_str = self._format_time(sim_time_sec)
        detail_part = f" | {details}" if details else ""
        msg = f"[SIGNAL] {t_str} | Veh={vehicle_id:<3} ({role:<14}) | {event_tag}{detail_part}"
        self._enqueue(msg)

    def close(self) -> None:
        self._stop_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=0.5)


