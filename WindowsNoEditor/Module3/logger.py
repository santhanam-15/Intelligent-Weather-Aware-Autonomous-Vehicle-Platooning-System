# WindowsNoEditor/Module3/logger.py
"""Dedicated thread-safe, non-blocking live decision logger for Module 3 Twin-World Safety Projection.

Formats structured records for real-time display in the dedicated decision terminal
and persists to a rotating log file.
"""

from __future__ import annotations

import datetime
import os
import queue
import sys
import threading
import time
from typing import Any, Dict, List, Optional

from Module3.config import MODULE3_LOG_FILE, MODULE3_LOG_HEARTBEAT_SEC, MODULE3_MODE


class Module3DecisionLogger:
    """Non-blocking structured decision logger streaming to disk and console monitor."""

    _instance: Optional["Module3DecisionLogger"] = None
    _lock = threading.Lock()

    def __init__(self, log_path: str = MODULE3_LOG_FILE, heartbeat_sec: float = MODULE3_LOG_HEARTBEAT_SEC):
        self.log_path = log_path
        self.heartbeat_sec = float(heartbeat_sec)
        self.log_dir = os.path.dirname(os.path.abspath(log_path))
        if not os.path.exists(self.log_dir):
            try:
                os.makedirs(self.log_dir, exist_ok=True)
            except Exception:
                pass

        self._queue: queue.Queue = queue.Queue(maxsize=10000)
        self._stop_event = threading.Event()
        self._last_normal_log_time: Dict[int, float] = {}
        self._last_state: Dict[int, str] = {}

        # Worker thread for asynchronous file writing
        self._worker_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._worker_thread.start()

        # Emit startup banner
        self._write_banner()

    @classmethod
    def get_instance(cls) -> "Module3DecisionLogger":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _write_banner(self) -> None:
        mode_str = MODULE3_MODE.upper()
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        banner = (
            "\n" + "=" * 80 + "\n"
            f"  MODULE 3 TWIN-WORLD PREDICTIVE SAFETY PROJECTION SYSTEM\n"
            f"  Operating Mode: {mode_str}  |  CARLA 0.9.9  |  Time: {now_str}\n"
            f"  Dynamic Rear-Impact Evasion  &  Multi-Vehicle Platoon Rejoining\n"
            + "=" * 80 + "\n"
        )
        self._enqueue(banner)

    def _enqueue(self, text: str) -> None:
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            pass  # Avoid blocking the simulation control thread under high load

    def _writer_loop(self) -> None:
        while not self._stop_event.is_set():
            lines_to_write = []
            try:
                # Batch read up to 100 items or wait 50ms
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
        channel: str = "Module 3 Twin-World Safety Channel",
        mode: str = "SHADOW",
        leader_id: int = 0,
        status: str = "SUCCESS",
    ) -> None:
        """Stream formal connection notification and startup summary."""
        msg = (
            f"[MODULE 3 CONSOLE] Dedicated Module 3 logging window connected.\n"
            f"[MODULE 3 STARTUP SUMMARY]\n"
            f"  - Active Process PID:      {pid}\n"
            f"  - Log Channel:             {channel}\n"
            f"  - Operating Mode:          {mode.upper()}\n"
            f"  - Authoritative Leader ID: {leader_id}\n"
            f"  - Logging Connection:      {status}\n"
        )
        self._enqueue(msg)

    def log_initialization(
        self,
        mode: str,
        enabled: bool,
        horizon_sec: float,
        dt_sec: float,
        budget_ms: float,
    ) -> None:
        """Log Module 3 Twin-World initialization parameters."""
        sep = "=" * 80
        msg = (
            f"\n{sep}\n"
            f"  MODULE 3 TWIN-WORLD CONFIGURATION & PARAMETERS\n"
            f"{sep}\n"
            f"  OPERATING MODE:            {mode.upper()}\n"
            f"  SHIELD ENABLED:            {enabled}\n"
            f"  PREDICTION HORIZON:        {horizon_sec:.1f}s\n"
            f"  DISCRETE STEP (dt):        {dt_sec:.2f}s\n"
            f"  EXECUTION BUDGET:          {budget_ms:.1f}ms\n"
            f"{sep}\n"
        )
        self._enqueue(msg)

    def log_risk_assessment(
        self,
        sim_time_sec: float,
        actor_id: int,
        state: str,
        risk: str,
        rear_ttc_s: float,
        gap_m: float,
        closing_mps: float,
    ) -> None:
        """[MODULE3] Time | Actor | State | Risk | Rear TTC | Gap | Closing Speed"""
        now = time.time()
        last_t = self._last_normal_log_time.get(actor_id, 0.0)

        # Rate-limit identical NORMAL records to avoid log spamming
        if risk == "NORMAL" and state == "NORMAL":
            if now - last_t < self.heartbeat_sec:
                return
            self._last_normal_log_time[actor_id] = now

        ttc_str = f"{rear_ttc_s:.2f}s" if (rear_ttc_s < 99.0 and rear_ttc_s > 0.0) else "inf"
        t_str = self._format_time(sim_time_sec)
        msg = f"[MODULE3] {t_str} | Actor={actor_id:<3} | State={state:<20} | Risk={risk:<10} | RearTTC={ttc_str:<7} | Gap={gap_m:5.1f}m | Closing={closing_mps:5.1f}m/s"
        self._enqueue(msg)

    def log_candidate_validation(
        self,
        sim_time_sec: float,
        actor_id: int,
        candidate_action: str,
        target_lane: Optional[int],
        is_valid: bool,
        rejection_reason: str = "",
    ) -> None:
        """[MODULE3] Time | Actor | Candidate | Target Lane | Validation | Rejection Reason"""
        t_str = self._format_time(sim_time_sec)
        val_str = "SAFE" if is_valid else "REJECTED"
        lane_str = str(target_lane) if target_lane is not None else "N/A"
        reason_str = f" | Reason={rejection_reason}" if rejection_reason else ""
        msg = f"[MODULE3] {t_str} | Actor={actor_id:<3} | Candidate={candidate_action:<12} | TargetLane={lane_str:<3} | Validation={val_str:<8}{reason_str}"
        self._enqueue(msg)

    def log_state_transition(
        self,
        sim_time_sec: float,
        actor_id: int,
        prev_state: str,
        new_state: str,
        reason: str,
    ) -> None:
        """[MODULE3] Time | Actor | State Transition | Previous State -> New State | Reason"""
        t_str = self._format_time(sim_time_sec)
        msg = f"[MODULE3] {t_str} | Actor={actor_id:<3} | State Transition | {prev_state} -> {new_state} | Reason={reason}"
        self._enqueue(msg)

    def log_action_arbitration(
        self,
        sim_time_sec: float,
        actor_id: int,
        mappo_action: str,
        proposed_safety_action: str,
        final_action: str,
        emergency_brake: bool = False,
    ) -> None:
        """[MODULE3] Time | Actor | MAPPO Action | Proposed Safety Action | Final Action"""
        t_str = self._format_time(sim_time_sec)
        brk_str = " [EMERGENCY_BRAKE]" if emergency_brake else ""
        msg = f"[MODULE3] {t_str} | Actor={actor_id:<3} | MAPPO={mappo_action:<11} | Proposed={proposed_safety_action:<12} | Final={final_action:<12}{brk_str}"
        self._enqueue(msg)

    def log_rejoin_status(
        self,
        sim_time_sec: float,
        actor_id: int,
        status: str,
        safe_gap_m: float,
        decision: str,
    ) -> None:
        """[MODULE3] Time | Actor | Rejoin Status | Safe Gap | Rejoin Decision"""
        t_str = self._format_time(sim_time_sec)
        msg = f"[MODULE3] {t_str} | Actor={actor_id:<3} | Rejoin Status={status:<15} | SafeGap={safe_gap_m:5.1f}m | Decision={decision}"
        self._enqueue(msg)

    def log_emergency(
        self,
        sim_time_sec: float,
        actor_id: int,
        reason: str,
        selected_response: str,
    ) -> None:
        """[MODULE3] Time | Actor | Emergency | Reason | Selected Response"""
        t_str = self._format_time(sim_time_sec)
        msg = f"[MODULE3] {t_str} | Actor={actor_id:<3} | Emergency | Reason={reason} | Response={selected_response}"
        self._enqueue(msg)

    def log_pipeline_trace(
        self,
        decision_id: str,
        timestamp: float,
        frame: int,
        snapshot_valid: bool,
        n_platoon: int,
        n_obstacles: int,
        pred_duration_ms: float,
        eval_duration_ms: float,
        total_duration_ms: float,
        budget_exceeded: bool,
        mode: str,
        proposed_actions: List[str],
        selected_actions: List[str],
        emergency_flags: List[bool],
        delivered_actions: List[str],
        lateral_offsets: Dict[int, float],
    ) -> None:
        """[MODULE3 TRACE] SNAPSHOT -> PREDICTION -> RISK EVALUATION -> SHIELD -> MODULE 2 COMMAND"""
        t_str = self._format_time(timestamp)
        over_str = " [BUDGET_OVERRUN]" if budget_exceeded else ""
        trace_line = (
            f"[MODULE3 TRACE] {t_str} | ID={decision_id} | Mode={mode} | "
            f"SNAPSHOT[valid={snapshot_valid}, plt={n_platoon}, obs={n_obstacles}] -> "
            f"PREDICT[{pred_duration_ms:.1f}ms] -> EVAL[{eval_duration_ms:.1f}ms] -> "
            f"TOTAL[{total_duration_ms:.1f}ms{over_str}] -> "
            f"SHIELD[props={proposed_actions} -> sel={selected_actions}] -> "
            f"MODULE2[cmds={delivered_actions}, emerg={emergency_flags}]"
        )
        self._enqueue(trace_line)

    def log_overtake_decision(
        self,
        sim_time_sec: float,
        actor_id: int,
        role: str,
        event_tag: str,
        target_lane: str = "NONE",
        gap_front: float = 0.0,
        gap_rear: float = 0.0,
        decision: str = "APPROVED",
        reason: str = "CLEAR",
    ) -> None:
        """[OVERTAKE] Time | Actor | Event | Target Lane | Gaps | Decision | Reason"""
        t_str = self._format_time(sim_time_sec)
        msg = (
            f"[OVERTAKE] {t_str} | Actor={actor_id:<3} ({role:<14}) | {event_tag} | "
            f"TargetLane={target_lane} | Gaps[F={gap_front:4.1f}m, R={gap_rear:4.1f}m] | "
            f"Decision={decision} | Reason={reason}"
        )
        self._enqueue(msg)

    def close(self) -> None:
        self._stop_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=0.5)

