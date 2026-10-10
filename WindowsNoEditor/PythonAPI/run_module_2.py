#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified Uncertain Platoon Simulation - Module 2 MAPPO Integration
Optimized Decoupled Architecture (CARLA Simulation Thread + Main UI Display Loop)
"""

from __future__ import annotations

import argparse
import ctypes
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import carla
import numpy as np
import pygame
from pygame.locals import K_c, K_ESCAPE, K_p, K_q, K_TAB, KEYDOWN, QUIT

try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

# Path configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CARLA_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, '..'))

try:
    import glob
    sys.path.append(glob.glob(os.path.join(
        SCRIPT_DIR, 'carla', 'dist',
        'carla-*%d.%d-%s.egg' % (
            sys.version_info.major, sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'
        )
    ))[0])
except IndexError:
    pass

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from autopilot_platoon_window import (
    setup_logging, wait_for_server, maybe_launch_carla,
    configure_world_timing, restore_world_timing, advance_world,
    cleanup_existing_actors, spawn_platoon, CameraManager,
    stop_launched_carla,
    MiniHUD, PlatoonSafetyMonitor, brake_platoon, platoon_rows, traffic_summary, walker_summary,
    WINDOW_WIDTH, WINDOW_HEIGHT, TARGET_FPS, DEFAULT_LOG_FILE, TM_PORT
)

from real_world_simulation import (
    spawn_npc_traffic, spawn_pedestrians, UncertaintyTimelineDirector
)
from uncertainty.uncertainty_manager import UncertaintyManager
from uncertainty.human_behavior import DriverProfile

if CARLA_ROOT not in sys.path:
    sys.path.insert(0, CARLA_ROOT)

from Module2.module2_manager import Module2Manager
from Module2.mission_controller import MissionPlatoonController

logger = logging.getLogger("UncertainPlatoonMain")


# ── HARDWARE METRICS MONITOR ────────────────────────────────────────────────
class HardwareMonitor:
    """Non-blocking hardware metrics sampler (CPU % via Windows API, GPU via nvidia-smi)."""

    def __init__(self):
        self.cpu_pct: float = 0.0
        self.gpu_util: Optional[int] = None
        self.gpu_mem_used: Optional[int] = None
        self.gpu_mem_total: Optional[int] = None
        self._last_cpu_time: float = time.time()
        self._last_gpu_query: float = 0.0
        self._idle: int = 0
        self._kernel: int = 0
        self._user: int = 0
        self._init_cpu()

    def _init_cpu(self):
        if os.name == "nt":
            try:
                class FILETIME(ctypes.Structure):
                    _fields_ = [('dwLowDateTime', ctypes.c_uint32), ('dwHighDateTime', ctypes.c_uint32)]
                i, k, u = FILETIME(), FILETIME(), FILETIME()
                ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(i), ctypes.byref(k), ctypes.byref(u))
                self._idle = (i.dwHighDateTime << 32) + i.dwLowDateTime
                self._kernel = (k.dwHighDateTime << 32) + k.dwLowDateTime
                self._user = (u.dwHighDateTime << 32) + u.dwLowDateTime
            except Exception:
                pass

    def sample(self):
        now = time.time()
        # Fast Windows CPU sampling every 0.5s
        if now - self._last_cpu_time >= 0.5 and os.name == "nt":
            try:
                class FILETIME(ctypes.Structure):
                    _fields_ = [('dwLowDateTime', ctypes.c_uint32), ('dwHighDateTime', ctypes.c_uint32)]
                i, k, u = FILETIME(), FILETIME(), FILETIME()
                ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(i), ctypes.byref(k), ctypes.byref(u))
                i_cur = (i.dwHighDateTime << 32) + i.dwLowDateTime
                k_cur = (k.dwHighDateTime << 32) + k.dwLowDateTime
                u_cur = (u.dwHighDateTime << 32) + u.dwLowDateTime

                i_diff = i_cur - self._idle
                k_diff = k_cur - self._kernel
                u_diff = u_cur - self._user
                total = k_diff + u_diff
                if total > 0:
                    self.cpu_pct = max(0.0, min(100.0, 100.0 * (1.0 - (i_diff / total))))
                self._idle, self._kernel, self._user = i_cur, k_cur, u_cur
                self._last_cpu_time = now
            except Exception:
                pass

        # Asynchronous non-blocking GPU query every 2.0s
        if now - self._last_gpu_query >= 2.0:
            self._last_gpu_query = now

            def _query_gpu():
                try:
                    out = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                        stderr=subprocess.DEVNULL, timeout=1.0
                    ).decode("utf-8").strip()
                    parts = [int(p.strip()) for p in out.split(",")]
                    self.gpu_util, self.gpu_mem_used, self.gpu_mem_total = parts[0], parts[1], parts[2]
                except Exception:
                    pass

            threading.Thread(target=_query_gpu, daemon=True).start()


# ── TELEMETRY SNAPSHOT FOR UI ────────────────────────────────────────────────
class TelemetrySnapshot:
    """Thread-safe state snapshot published by the CARLA simulation thread for the UI."""

    def __init__(self):
        self.platoon_rows: List[Dict[str, Any]] = []
        self.npc_stats: Tuple[int, int, int, float] = (0, 0, 0, 0.0)
        self.walker_stats: Tuple[int, int] = (0, 0)
        self.control_active: bool = True
        self.camera_name: str = "Platoon Overview"
        self.elapsed_sim: float = 0.0
        self.town_name: str = "Town01"
        self.synchronous: bool = True
        self.sim_fps: float = 30.0
        self.ghost_status: Optional[Dict[str, Any]] = None
        self.mappo_actions: Dict[str, str] = {}
        self.leader_diag: Dict[str, Any] = {}
        self.tl_label: str = "NONE"
        self.tl_color: Tuple[int, int, int] = (160, 160, 160)
        self.leader_thr: float = 0.0
        self.leader_brk: float = 0.0
        self.cpu_util: float = 0.0
        self.gpu_util: Optional[int] = None
        self.gpu_mem_used: Optional[int] = None
        self.gpu_mem_total: Optional[int] = None
        self.ckpt_step: int = 0
        self.ckpt_hash: str = "--------"
        self.platoon_state: str = "NORMAL"
        self.overrides: Dict[str, str] = {}


class SnapshotHolder:
    def __init__(self):
        self._lock = threading.Lock()
        self._snapshot = TelemetrySnapshot()

    def set(self, snap: TelemetrySnapshot) -> None:
        with self._lock:
            self._snapshot = snap

    def get(self) -> TelemetrySnapshot:
        with self._lock:
            return self._snapshot


def apply_platoon_friction(uncertainty_manager, platoon, weather_preset):
    friction_val = {
        "CLEAR": 1.0, "CLOUDY": 0.9, "LIGHT_RAIN": 0.6,
        "HEAVY_RAIN": 0.4, "HEAVY_FOG": 0.7,
    }.get(weather_preset.name.upper(), 1.0)
    for v in platoon:
        if v and v.is_alive:
            try:
                uncertainty_manager.friction.set_friction(v, friction_val)
            except Exception:
                pass


# ── UNIFIED HUD (WITH SNAPSHOT FAST-PATH) ────────────────────────────────────
class UnifiedHUD(MiniHUD):
    """Full-width dark telemetry dashboard strip rendered below the camera viewport."""
    STRIP_H = 210

    ACTION_COLORS = {
        "CRUISE": (120, 200, 255), "ACCELERATE": (80, 255, 130),
        "DECELERATE": (255, 190, 60), "CHANGE_LEFT": (220, 130, 255),
        "CHANGE_RIGHT": (220, 130, 255), "BRAKE": (255, 90, 90)
    }

    def __init__(self, width: int, height: int):
        super().__init__(width, height)
        self.font_s  = pygame.font.SysFont("consolas", 12, bold=True)
        self.font_m  = pygame.font.SysFont("consolas", 14, bold=True)
        self.font_lg = pygame.font.SysFont("consolas", 15, bold=True)

        self.top_surface = pygame.Surface((self.width, 28), pygame.SRCALPHA)
        self.top_surface.fill((8, 12, 20, 200))

        self.base_strip = pygame.Surface((self.width, self.STRIP_H))
        self.base_strip.fill((8, 10, 18))
        pygame.draw.line(self.base_strip, (60, 100, 160), (0, 0), (self.width, 0), 2)

        self.col1_x = 12
        self.col2_x = 420 if self.width >= 1200 else int(self.width * 0.32)
        self.col3_x = 650 if self.width >= 1200 else int(self.width * 0.51)
        self.col4_x = 900 if self.width >= 1200 else int(self.width * 0.70)

        c1_head = self.font_lg.render("PLATOON TELEMETRY", True, (100, 180, 255))
        self.base_strip.blit(c1_head, (self.col1_x, 8))
        pygame.draw.line(self.base_strip, (60, 100, 160), (self.col1_x, 28), (self.col1_x + 380, 28), 1)
        c1_sub = self.font_s.render(f"{'ROLE':<12}{'ID':<6}{'SPD':>7}{'GAP':>8}  STATUS", True, (130, 155, 190))
        self.base_strip.blit(c1_sub, (self.col1_x, 34))

        c2_head = self.font_lg.render("CAV CONTROL PIPELINE", True, (100, 180, 255))
        self.base_strip.blit(c2_head, (self.col2_x, 8))
        pygame.draw.line(self.base_strip, (60, 100, 160), (self.col2_x, 28), (self.col2_x + 215, 28), 1)
        c2_ap = self.font_s.render("CARLA Autopilot: OFF", True, (120, 255, 180))
        self.base_strip.blit(c2_ap, (self.col2_x, 34))

        c3_head = self.font_lg.render("MAPPO ACTIONS", True, (100, 180, 255))
        self.base_strip.blit(c3_head, (self.col3_x, 8))
        pygame.draw.line(self.base_strip, (60, 100, 160), (self.col3_x, 28), (self.col3_x + 220, 28), 1)

        pygame.draw.line(self.base_strip, (60, 100, 160), (self.col4_x, 28), (self.col4_x + 250, 28), 1)
        self.ghost_head_inactive = self.font_lg.render("GHOST DRIVER", True, (100, 180, 255))
        self.ghost_head_active   = self.font_lg.render("GHOST DRIVER", True, (255, 140, 255))

        hint = "[P] Toggle MAPPO  [TAB] Camera  [C] View  [ESC/Q] Exit  |  Live logs in terminal"
        self.hint_surf = self.font_s.render(hint, True, (130, 155, 190))
        hint_x = (self.width - self.hint_surf.get_width()) // 2
        hint_y = self.STRIP_H - 22
        self.base_strip.blit(self.hint_surf, (hint_x, hint_y))
        self.active_strip = self.base_strip.copy()

    def render(
        self, display, leader, platoon, vehicles, walkers,
        control_active, camera_name, elapsed, town_name, synchronous, fps,
        ghost_status=None, mappo_actions=None, leader_diag=None, ui_fps=0.0,
        snapshot: Optional[TelemetrySnapshot] = None
    ) -> None:
        W, H = self.width, self.height
        STRIP_Y = H - self.STRIP_H

        # If a pre-collected snapshot is provided, render with 0 CARLA RPC calls
        if snapshot is not None:
            mins, secs = int(snapshot.elapsed_sim) // 60, int(snapshot.elapsed_sim) % 60
            mode = "SYNC 30FPS" if snapshot.synchronous else "ASYNC"
            ui_txt = f"UI: {ui_fps:.0f} FPS" if ui_fps > 0 else ""
            cpu_txt = f"CPU: {snapshot.cpu_util:.0f}%"
            gpu_txt = f"GPU: {snapshot.gpu_util:.0f}% ({snapshot.gpu_mem_used}MB)" if snapshot.gpu_util is not None else "GPU: RTX 3050"
            display.blit(self.top_surface, (0, 0))
            bar_txt = f"  {snapshot.town_name}  |  {mins:02d}:{secs:02d}  |  Sim: {snapshot.sim_fps:.0f} FPS  |  {ui_txt}  |  {cpu_txt}  |  {gpu_txt}  |  Cam: {snapshot.camera_name}"
            display.blit(self.font_s.render(bar_txt, True, (160, 200, 240)), (8, 6))

            self.active_strip.blit(self.base_strip, (0, 0))
            def txt(text, x, y, color=(210, 225, 255), fnt=None):
                fnt = fnt or self.font_m
                self.active_strip.blit(fnt.render(text, True, color), (x, y))

            # Column 1
            rows = snapshot.platoon_rows
            for idx, row in enumerate(rows[:5]):
                yy = 52 + idx * 30
                gap_str = "--    " if row["gap"] is None else f"{row['gap']:5.1f}m"
                status  = "MOVING" if row["speed"] > 1.0 else "WAIT"
                color   = (90, 255, 140) if row["speed"] > 1.0 else (255, 215, 70)
                if row["gap"] is not None and row["gap"] < 3.0:
                    status, color = "CLOSE!", (255, 80, 80)
                if not row["alive"]:
                    status, color = "MISSING", (255, 50, 50)
                line = f"{row['role']:<12}#{row['id']:<5}{row['speed']:5.1f}km/h  {gap_str}  {status}"
                txt(line, self.col1_x, yy, color, self.font_s)

            # Column 2
            col2_x = self.col2_x
            leader_spd = rows[0]["speed"] if rows else 0.0
            tl_label = snapshot.tl_label
            tl_color = snapshot.tl_color
            ctrl_status = "MAPPO (ACTIVE)" if snapshot.control_active else "MAPPO (PAUSED)"
            ctrl_col = (90, 255, 140) if snapshot.control_active else (255, 180, 60)
            txt(f"Controller: {ctrl_status}", col2_x, 48, ctrl_col, self.font_s)
            txt(f"Ckpt: Step {snapshot.ckpt_step} [{snapshot.ckpt_hash}]", col2_x, 62, (180, 210, 240), self.font_s)

            leader_diag = snapshot.leader_diag
            target_spd_str = f"{leader_diag.get('target_speed', 0.0) * 3.6:4.1f} km/h" if leader_diag else "--"
            txt(f"Speed: {leader_spd:4.1f} | Target: {target_spd_str}", col2_x, 76, (210, 240, 255), self.font_s)

            safety_ovr = leader_diag.get('safety_override', '') if leader_diag else ''
            stop_reason = leader_diag.get('stop_reason', 'NONE') if leader_diag else 'NONE'
            if stop_reason and stop_reason != 'NONE':
                status_text = f"STOP: {stop_reason[:16]}"
                status_col = (255, 80, 80)
            elif safety_ovr and safety_ovr != 'NONE':
                status_text = f"Override: {safety_ovr[:16]}"
                status_col = (255, 215, 80)
            else:
                status_text = "Safety: PURE PURSUIT"
                status_col = (180, 210, 240)
            txt(status_text, col2_x, 90, status_col, self.font_s)
            txt(f"Traffic Lt: {tl_label}", col2_x, 104, tl_color, self.font_s)

            thr, brk = snapshot.leader_thr, snapshot.leader_brk
            txt(f"THR: {thr:4.2f}", col2_x, 118, (180, 220, 180), self.font_s)
            pygame.draw.rect(self.active_strip, (60, 220, 90), (col2_x + 70, 120, int(60 * thr), 7))
            txt(f"BRK: {brk:4.2f}", col2_x, 132, (220, 180, 180), self.font_s)
            pygame.draw.rect(self.active_strip, (220, 60, 60), (col2_x + 70, 134, int(60 * brk), 7))

            # Column 3
            col3_x = self.col3_x
            if snapshot.mappo_actions:
                for mi, (role, act) in enumerate(snapshot.mappo_actions.items()):
                    acolor = self.ACTION_COLORS.get(act, (200, 200, 200))
                    ovr = snapshot.overrides.get(role, "NONE")
                    if ovr and ovr != "NONE":
                        txt(f"{role:<6}: {act:<9} !{ovr[:9]}", col3_x, 38 + mi * 22, (255, 130, 130), self.font_s)
                    else:
                        txt(f"{role:<6}: {act}", col3_x, 38 + mi * 22, acolor, self.font_s)
            else:
                txt("Waiting for rollout...", col3_x, 44, (130, 140, 160), self.font_s)

            # Column 4
            col4_x = self.col4_x
            ghost_status = snapshot.ghost_status
            ghost_active = ghost_status.get("active", False) if ghost_status else False
            self.active_strip.blit(
                self.ghost_head_active if ghost_active else self.ghost_head_inactive,
                (col4_x, 8)
            )
            if ghost_status:
                if ghost_active:
                    tid = ghost_status.get("target_id")
                    disruption = ghost_status.get("disruption_name", "?")
                    subphase = ghost_status.get("subphase", "ACT")
                    prog = int(ghost_status.get("progress", 0.0) * 100)
                    rem = ghost_status.get("remaining_seconds", 0.0)
                    txt(f"POSSESSED #{tid} [{subphase}]", col4_x, 38, (255, 90, 90), self.font_s)
                    txt(f"{disruption[:20]} ({prog}%)", col4_x, 54, (255, 170, 255), self.font_s)
                    txt(f"Time: {rem:.1f}s | Plt: {snapshot.platoon_state}", col4_x, 70, (255, 210, 100), self.font_s)
                else:
                    cooldown = ghost_status.get("cooldown_remaining", 0.0)
                    txt("All → NORMAL", col4_x, 38, (90, 255, 140), self.font_s)
                    txt(f"Lurking... {cooldown:.1f}s", col4_x, 54, (160, 200, 200), self.font_s)
                    txt(f"Platoon State: {snapshot.platoon_state}", col4_x, 70, (140, 220, 255), self.font_s)

            live_npcs, moving_npcs, stopped_npcs, avg_spd = snapshot.npc_stats
            live_w, moving_w = snapshot.walker_stats
            env_x = col4_x
            txt(f"NPCs: {live_npcs}  Mv:{moving_npcs}  St:{stopped_npcs}", env_x, 100, (160, 190, 215), self.font_s)
            txt(f"Avg NPC: {avg_spd:.1f} km/h", env_x, 118, (160, 190, 215), self.font_s)
            txt(f"Walkers: {live_w} live / {moving_w} moving", env_x, 136, (160, 190, 215), self.font_s)

            display.blit(self.active_strip, (0, STRIP_Y))
            return

        # Fallback for legacy calls without snapshot
        display.blit(self.top_surface, (0, 0))
        mins, secs = int(elapsed) // 60, int(elapsed) % 60
        mode = "SYNC 30FPS" if synchronous else "ASYNC"
        ui_txt = f" (UI: {ui_fps:.0f})" if ui_fps > 0 else ""
        bar_txt = f"  {town_name}  |  {mins:02d}:{secs:02d}  |  {fps:.0f} FPS{ui_txt}  |  {mode}  |  Cam: {camera_name}"
        display.blit(self.font_s.render(bar_txt, True, (160, 200, 240)), (8, 6))

        self.active_strip.blit(self.base_strip, (0, 0))
        def txt_fb(text, x, y, color=(210, 225, 255), fnt=None):
            fnt = fnt or self.font_m
            self.active_strip.blit(fnt.render(text, True, color), (x, y))

        rows = platoon_rows(platoon) if platoon else []
        for idx, row in enumerate(rows[:5]):
            yy = 52 + idx * 30
            gap_str = "--    " if row["gap"] is None else f"{row['gap']:5.1f}m"
            status  = "MOVING" if row["speed"] > 1.0 else "WAIT"
            color   = (90, 255, 140) if row["speed"] > 1.0 else (255, 215, 70)
            if row["gap"] is not None and row["gap"] < 3.0:
                status, color = "CLOSE!", (255, 80, 80)
            if not row["alive"]:
                status, color = "MISSING", (255, 50, 50)
            line = f"{row['role']:<12}#{row['id']:<5}{row['speed']:5.1f}km/h  {gap_str}  {status}"
            txt_fb(line, self.col1_x, yy, color, self.font_s)

        col2_x = self.col2_x
        leader_spd = rows[0]["speed"] if rows else 0.0
        tl_color = (160, 160, 160)
        tl_label = "NONE"
        if leader and getattr(leader, "is_alive", False):
            try:
                tl = leader.get_traffic_light()
                if tl is not None:
                    state = tl.get_state()
                    if state == carla.TrafficLightState.Red:
                        tl_label, tl_color = "RED  ●", (255, 80, 80)
                    elif state == carla.TrafficLightState.Yellow:
                        tl_label, tl_color = "AMBER ●", (255, 210, 50)
                    elif state == carla.TrafficLightState.Green:
                        tl_label, tl_color = "GREEN ●", (70, 255, 120)
            except Exception:
                pass

        ctrl_status = "MAPPO (ACTIVE)" if control_active else "MAPPO (PAUSED)"
        ctrl_col = (90, 255, 140) if control_active else (255, 180, 60)
        txt_fb(f"Controller: {ctrl_status}", col2_x, 50, ctrl_col, self.font_s)
        target_spd_str = f"{leader_diag.get('target_speed', 0.0) * 3.6:4.1f} km/h" if leader_diag else "--"
        txt_fb(f"Speed: {leader_spd:4.1f} | Target: {target_spd_str}", col2_x, 66, (210, 240, 255), self.font_s)

        safety_ovr = leader_diag.get('safety_override', '') if leader_diag else ''
        stop_reason = leader_diag.get('stop_reason', 'NONE') if leader_diag else 'NONE'
        if stop_reason and stop_reason != 'NONE':
            status_text = f"STOP: {stop_reason[:16]}"
            status_col = (255, 80, 80)
        elif safety_ovr:
            status_text = f"Safety: {safety_ovr[:18]}"
            status_col = (255, 215, 80) if "traffic_light" in safety_ovr or "obstacle" in safety_ovr else (180, 210, 240)
        else:
            status_text = "Safety: PURE PURSUIT"
            status_col = (180, 210, 240)
        txt_fb(status_text, col2_x, 82, status_col, self.font_s)
        txt_fb(f"Traffic Lt: {tl_label}", col2_x, 98, tl_color, self.font_s)

        if leader and getattr(leader, "is_alive", False):
            try:
                ctrl = leader.get_control()
                thr, brk = ctrl.throttle, ctrl.brake
                txt_fb(f"THR: {thr:4.2f}", col2_x, 116, (180, 220, 180), self.font_s)
                pygame.draw.rect(self.active_strip, (60, 220, 90), (col2_x + 70, 118, int(60 * thr), 7))
                txt_fb(f"BRK: {brk:4.2f}", col2_x, 130, (220, 180, 180), self.font_s)
                pygame.draw.rect(self.active_strip, (220, 60, 60), (col2_x + 70, 132, int(60 * brk), 7))
            except Exception:
                pass

        col3_x = self.col3_x
        if mappo_actions:
            for mi, (role, act) in enumerate(mappo_actions.items()):
                acolor = self.ACTION_COLORS.get(act, (200, 200, 200))
                txt_fb(f"{role:<8}: {act}", col3_x, 38 + mi * 22, acolor, self.font_s)
        else:
            txt_fb("Waiting for rollout...", col3_x, 44, (130, 140, 160), self.font_s)

        col4_x = self.col4_x
        ghost_active = ghost_status.get("active", False) if ghost_status else False
        self.active_strip.blit(
            self.ghost_head_active if ghost_active else self.ghost_head_inactive,
            (col4_x, 8)
        )
        if ghost_status:
            if ghost_active:
                tid = ghost_status.get("target_id")
                disruption = ghost_status.get("disruption_name", "?")
                rem = ghost_status.get("remaining_seconds", 0.0)
                txt_fb(f"POSSESSED #{tid}", col4_x, 40, (255, 90, 90), self.font_s)
                txt_fb(f"{disruption[:28]}", col4_x, 58, (255, 170, 255), self.font_s)
                txt_fb(f"Time left: {rem:.1f}s", col4_x, 76, (255, 210, 100), self.font_s)
            else:
                cooldown = ghost_status.get("cooldown_remaining", 0.0)
                txt_fb("All → NORMAL", col4_x, 40, (90, 255, 140), self.font_s)
                txt_fb(f"Lurking... {cooldown:.1f}s", col4_x, 58, (160, 200, 200), self.font_s)

        live_npcs, moving_npcs, stopped_npcs, avg_spd = traffic_summary(vehicles) if vehicles else (0, 0, 0, 0.0)
        live_w, moving_w = walker_summary(walkers) if walkers else (0, 0)
        env_x = col4_x
        txt_fb(f"NPCs: {live_npcs}  Mv:{moving_npcs}  St:{stopped_npcs}", env_x, 100, (160, 190, 215), self.font_s)
        txt_fb(f"Avg NPC: {avg_spd:.1f} km/h", env_x, 118, (160, 190, 215), self.font_s)
        txt_fb(f"Walkers: {live_w} live / {moving_w} moving", env_x, 136, (160, 190, 215), self.font_s)

        display.blit(self.active_strip, (0, STRIP_Y))


# ── CARLA SIMULATION WORKER THREAD ──────────────────────────────────────────
class CarlaSimulationThread(threading.Thread):
    """Sole owner of CARLA client, world, synchronous tick, and control pipeline."""

    def __init__(
        self,
        args: argparse.Namespace,
        snapshot_holder: SnapshotHolder,
        command_queue: queue.Queue,
        stop_event: threading.Event,
        ready_event: threading.Event,
        error_holder: List[Exception],
        window_width: int,
        camera_h: int
    ):
        super().__init__(name="CarlaSimulationThread", daemon=True)
        self.args = args
        self.snapshot_holder = snapshot_holder
        self.command_queue = command_queue
        self.stop_event = stop_event
        self.ready_event = ready_event
        self.error_holder = error_holder
        self.window_width = window_width
        self.camera_h = camera_h
        self.camera: Optional[CameraManager] = None

    def run(self) -> None:
        args = self.args
        platoon: List[carla.Vehicle] = []
        npcs: List[carla.Vehicle] = []
        walkers: List[carla.Actor] = []
        controllers: List[carla.Actor] = []
        world: Optional[carla.World] = None
        tm = None
        original_settings = None
        mission_controller: Optional[MissionPlatoonController] = None
        safety_monitor: Optional[PlatoonSafetyMonitor] = None
        uncertainty_manager: Optional[UncertaintyManager] = None
        timeline_director: Optional[UncertaintyTimelineDirector] = None
        module2_manager: Optional[Module2Manager] = None
        camera: Optional[CameraManager] = None
        hw_monitor = HardwareMonitor()

        try:
            client = carla.Client(args.host, args.port)
            client.set_timeout(30.0)
            world = client.get_world()
            current_town = world.get_map().name.split("/")[-1]
            if args.town and args.town.lower() not in ("auto", "current") and args.town != current_town:
                logger.info("Loading %s ...", args.town)
                client.set_timeout(60.0)
                world = client.load_world(args.town)
                client.set_timeout(30.0)
                current_town = world.get_map().name.split("/")[-1]
                time.sleep(2.0)

            tm = client.get_trafficmanager(args.tm_port)
            original_settings = configure_world_timing(world, tm, sync=not args.async_world)

            if args.clean:
                cleanup_existing_actors(world)

            tm.set_global_distance_to_leading_vehicle(3.5)
            tm.global_percentage_speed_difference(0.0)

            spawn_points = world.get_map().get_spawn_points()
            if not spawn_points:
                raise RuntimeError("No vehicle spawn points found.")

            # Spawn Platoon
            logger.info("[PLATOON] Creating platoon...")
            platoon = spawn_platoon(world, tm, spawn_points, args.platoon_size)
            leader = platoon[0]
            advance_world(world, sync=not args.async_world)
            logger.info("[PLATOON] Platoon ready at (%.1f, %.1f).", leader.get_location().x, leader.get_location().y)
            for index, veh in enumerate(platoon):
                try:
                    veh.set_autopilot(False)
                except Exception:
                    pass
                logger.info("[AUTOPILOT ENFORCED] Platoon vehicle index %d (ID %d): CARLA Autopilot explicitly enforced to False", index, veh.id)

            # Spawn Uncertainty World Traffic
            logger.info("[WORLD] Initializing uncertain world traffic...")
            protected_locations = [p.get_location() for p in platoon if p and p.is_alive]
            available_sps = [sp for sp in spawn_points if all(sp.location.distance(loc) > 32.0 for loc in protected_locations)]
            npcs = spawn_npc_traffic(client, world, available_sps, count=args.vehicles, tm_port=args.tm_port)
            walkers, controllers = spawn_pedestrians(world, count=args.pedestrians)
            logger.info("[WORLD] Ready.")

            # Initialize Uncertainty System & Ghost Driver
            logger.info("[GHOST] Initializing Ghost-driver system...")
            uncertainty_manager = UncertaintyManager(seed=42)
            uncertainty_manager.enable_module("traffic_jam", False)
            uncertainty_manager.enable_module("accident", False)
            uncertainty_manager.enable_module("events", False)
            uncertainty_manager.setup(world=world, ego_vehicle=leader, npc_vehicles=npcs, traffic_manager=tm)
            apply_platoon_friction(
                uncertainty_manager, platoon, uncertainty_manager.weather.current_preset
            )

            def friction_for(vehicle):
                try:
                    val = uncertainty_manager.friction.get_friction(vehicle)
                    return 1.0 if val is None else val
                except Exception:
                    return 1.0

            mission_controller = MissionPlatoonController(
                world, platoon, cruise_speed_kmh=args.platoon_speed,
                friction_provider=friction_for,
            )
            safety_monitor = PlatoonSafetyMonitor(world, platoon)

            uncertainty_manager.behavior.assign_random_profiles(
                npcs, tm, weights={
                    DriverProfile.NORMAL: 0.45,
                    DriverProfile.AGGRESSIVE: 0.15,
                    DriverProfile.CONSERVATIVE: 0.15,
                    DriverProfile.DISTRACTED: 0.15,
                    DriverProfile.ERRATIC: 0.10,
                }
            )

            timeline_director = UncertaintyTimelineDirector(
                world=world, ego_vehicle=leader, npc_vehicles=npcs,
                walkers=walkers, walker_controllers=controllers,
                traffic_manager=tm, uncertainty_manager=uncertainty_manager,
                platoon_vehicles=platoon
            )
            timeline_director.ghost_driver.min_possession = 15.0
            timeline_director.ghost_driver.max_possession = 20.0
            timeline_director.ghost_driver.min_cooldown = 5.0
            timeline_director.ghost_driver.max_cooldown = 20.0
            timeline_director.ghost_driver.set_mission_controller(mission_controller)
            if getattr(args, "ghost_scenario", "auto") != "auto":
                timeline_director.ghost_driver.set_scripted_scenario(args.ghost_scenario, seed=getattr(args, "ghost_seed", 42))
            logger.info(
                "[GHOST] Ghost-driver system initialized (scenario=%s, seed=%d).",
                getattr(args, "ghost_scenario", "auto"), getattr(args, "ghost_seed", 42)
            )

            # Initialize Module 2 MAPPO System
            logger.info("[MODULE 2] Initializing MAPPO System...")
            module2_manager = Module2Manager(
                uncertainty_manager, len(platoon), mode=args.mode,
                checkpoint_path=getattr(args, "checkpoint", None),
            )
            module2_manager.set_control_context(mission_controller)
            logger.info("[MODULE 2] MAPPO System Initialized in %s mode.", args.mode.upper())

            # Checkpoint Verification Banner
            ckpt_path = os.path.abspath(module2_manager.trainer.get_checkpoint_path())
            ckpt_hash = getattr(module2_manager.trainer, "active_checkpoint_hash", "--------")
            obs_dim = module2_manager.obs_builder.obs_dim
            action_dim = 5
            update_step = module2_manager.trainer.update_step
            is_det = (args.mode == "evaluate")
            sep = "=" * 62
            logger.info("\n%s\n  MAPPO CONFIGURATION & CHECKPOINT VERIFICATION\n%s", sep, sep)
            logger.info("  CHECKPOINT PATH:           %s", ckpt_path)
            logger.info("  CHECKPOINT SHA-256:        %s", ckpt_hash)
            logger.info("  OBSERVATION DIM:           %d", obs_dim)
            logger.info("  ACTION DIM:                %d", action_dim)
            logger.info("  UPDATE STEP:               %d", update_step)
            logger.info("  DETERMINISTIC EVALUATION:  %s", is_det)
            logger.info("  ACTOR LOADED:              %s", module2_manager.trainer.actor is not None)
            logger.info("  CRITIC LOADED:             %s", module2_manager.trainer.critic is not None)
            if update_step <= 2:
                logger.warning("  WARNING: CHECKPOINT IS NOT SUFFICIENTLY TRAINED (update_step=%d)", update_step)
            logger.info("%s\n", sep)

            action_counts = {
                "CRUISE": 0, "ACCELERATE": 0, "DECELERATE": 0,
                "CHANGE_LEFT": 0, "CHANGE_RIGHT": 0,
            }
            tactical_actions = ["CRUISE"] * len(platoon)
            actions_dict = {"Leader" if i == 0 else "F%d" % i: "CRUISE" for i in range(len(platoon))}

            camera_vehicle_idx = 0
            camera = CameraManager(leader, self.window_width, self.camera_h)
            self.camera = camera
            control_active = True

            sim_start_time = world.get_snapshot().timestamp.elapsed_seconds
            last_motion_log = 0.0
            last_weather_time = 0.0
            was_ghost_active = False
            weather_interval_sec = args.weather * 60.0

            # Performance counters
            perf_sim_tick_total = 0.0
            perf_mappo_total = 0.0
            perf_controller_total = 0.0
            perf_ticks_count = 0
            perf_last_log_time = time.perf_counter()
            perf_sim_fps = 30.0

            # Signal ready to UI thread
            self.ready_event.set()
            logger.info("[SIMULATION] Simulation thread active (30 Hz target). Entering authoritative control loop...")

            # 30 Hz simulation pacing
            sim_target_dt = 1.0 / 30.0
            last_sim_time = time.perf_counter()
            cached_npc_stats = (0, 0, 0, 0.0)
            cached_walker_stats = (0, 0)
            last_stats_time = 0.0

            while not self.stop_event.is_set():
                t_loop_start = time.perf_counter()

                # 1. Process commands from UI thread
                while not self.command_queue.empty():
                    try:
                        cmd, param = self.command_queue.get_nowait()
                        if cmd == "quit":
                            self.stop_event.set()
                            break
                        elif cmd == "toggle_control":
                            control_active = not control_active
                            if not control_active:
                                brake_platoon(platoon)
                            logger.info("MAPPO mission control: %s", "ON" if control_active else "OFF")
                        elif cmd == "toggle_camera_view":
                            if camera:
                                camera.toggle()
                        elif cmd == "switch_camera_vehicle":
                            camera_vehicle_idx = (camera_vehicle_idx + 1) % len(platoon)
                            if camera:
                                camera.destroy()
                            camera = CameraManager(platoon[camera_vehicle_idx], self.window_width, self.camera_h)
                            self.camera = camera
                            logger.info("Camera attached to: %s", "Leader" if camera_vehicle_idx == 0 else f"Follower {camera_vehicle_idx}")
                    except queue.Empty:
                        break

                if self.stop_event.is_set():
                    break

                snapshot = world.get_snapshot()
                elapsed_sim = snapshot.timestamp.elapsed_seconds - sim_start_time
                delta_sec = max(0.0, snapshot.timestamp.delta_seconds)

                if args.duration > 0 and elapsed_sim >= args.duration:
                    self.stop_event.set()
                    break

                # Dynamic weather shift
                if elapsed_sim - last_weather_time >= weather_interval_sec:
                    last_weather_time = elapsed_sim
                    import random
                    from uncertainty.weather_uncertainty import WeatherPreset
                    presets = [WeatherPreset.CLEAR, WeatherPreset.CLOUDY, WeatherPreset.LIGHT_RAIN, WeatherPreset.HEAVY_RAIN, WeatherPreset.HEAVY_FOG]
                    new_weather = random.choice(presets)
                    uncertainty_manager.weather.set_weather(world, new_weather)
                    apply_platoon_friction(uncertainty_manager, platoon, new_weather)
                    logger.info("[WEATHER] Interval reached. Weather dynamically shifted to %s", new_weather.name)

                # Uncertainty & Ghost update
                uncertainty_manager.tick(delta_seconds=delta_sec, current_time=elapsed_sim)
                timeline_director.tick(elapsed_sim_time=elapsed_sim, delta_sec=delta_sec)
                ghost_telemetry = timeline_director.ghost_driver.get_status(elapsed_sim)
                is_ghost_active = ghost_telemetry.get("active", False)
                if is_ghost_active and not was_ghost_active:
                    logger.info("[GHOST] Vehicle %s -> POSSESSED", ghost_telemetry.get("target_id"))
                elif not is_ghost_active and was_ghost_active:
                    logger.info("[GHOST] Vehicle Released -> RELEASED")
                was_ghost_active = is_ghost_active

                # MAPPO tactical update (15 Hz)
                t_m0 = time.perf_counter()
                if control_active:
                    if getattr(args, "diagnostic_control_test", False):
                        diag_cycle = int(elapsed_sim // 5.0) % 3
                        forced_action = ["ACCELERATE", "DECELERATE", "CRUISE"][diag_cycle]
                        tactical_actions = [forced_action] * len(platoon)
                        actions_dict = {"Leader" if i == 0 else "F%d" % i: forced_action for i in range(len(platoon))}
                        mission_controller.set_tactical_actions(tactical_actions)
                        for act in tactical_actions:
                            action_counts[act] = action_counts.get(act, 0) + 1
                    else:
                        tactical_actions, actions_dict = module2_manager.update(platoon, elapsed_sim)
                        if module2_manager.episode_done:
                            logger.warning("[MODULE 2] Episode terminated by actor loss or platoon collision.")
                            self.stop_event.set()
                            break
                        if module2_manager.action_updated:
                            mission_controller.set_tactical_actions(tactical_actions)
                            for act in tactical_actions:
                                action_counts[act] = action_counts.get(act, 0) + 1
                            raw_acts = getattr(module2_manager, "current_raw_actions", None)
                            logger.info(
                                "[MAPPO DECISION] t=%.2fs | Raw Actions: %s | Post-Hysteresis Tactical: %s",
                                elapsed_sim,
                                list(raw_acts) if raw_acts is not None else tactical_actions,
                                tactical_actions,
                            )
                    module2_manager.train_if_ready()
                t_m1 = time.perf_counter()
                perf_mappo_total += (t_m1 - t_m0)

                # Low-level controller tick (30 Hz)
                t_c0 = time.perf_counter()
                if control_active:
                    mission_controller.tick(elapsed_sim)
                else:
                    brake_platoon(platoon)
                t_c1 = time.perf_counter()
                perf_controller_total += (t_c1 - t_c0)

                # Step CARLA Physics (Authoritative synchronous tick)
                t_s0 = time.perf_counter()
                advance_world(world, sync=not args.async_world)
                t_s1 = time.perf_counter()
                perf_sim_tick_total += (t_s1 - t_s0)
                perf_ticks_count += 1

                # Ghost debug bounding box
                if is_ghost_active:
                    tid = ghost_telemetry.get("target_id")
                    ghost_actor = world.get_actor(tid)
                    if ghost_actor and ghost_actor.is_alive:
                        t_box = ghost_actor.get_transform()
                        b = ghost_actor.bounding_box
                        extent = carla.Vector3D(b.extent.x + 0.15, b.extent.y + 0.15, b.extent.z + 0.15)
                        box = carla.BoundingBox(t_box.location + b.location, extent)
                        world.debug.draw_box(box, t_box.rotation, thickness=0.15, color=carla.Color(255, 0, 0), life_time=0.1)

                # Update traffic and walker stats periodically (every 0.5s) to avoid RPC overhead
                t_now = time.perf_counter()
                if t_now - last_stats_time >= 0.5:
                    last_stats_time = t_now
                    cached_npc_stats = traffic_summary(npcs)
                    cached_walker_stats = walker_summary(walkers)
                    hw_monitor.sample()

                # Build and publish telemetry snapshot for the UI thread
                snap = TelemetrySnapshot()
                snap.platoon_rows = platoon_rows(platoon)
                snap.npc_stats = cached_npc_stats
                snap.walker_stats = cached_walker_stats
                snap.control_active = control_active
                snap.camera_name = camera.name if camera else "None"
                snap.elapsed_sim = elapsed_sim
                snap.town_name = current_town
                snap.synchronous = not args.async_world
                snap.sim_fps = perf_sim_fps
                snap.ghost_status = ghost_telemetry
                snap.mappo_actions = actions_dict
                leader_diag = mission_controller.diagnostics(leader, 0) if (mission_controller and leader) else {}
                snap.leader_diag = leader_diag
                snap.cpu_util = hw_monitor.cpu_pct
                snap.gpu_util = hw_monitor.gpu_util
                snap.gpu_mem_used = hw_monitor.gpu_mem_used
                snap.gpu_mem_total = hw_monitor.gpu_mem_total
                snap.ckpt_step = module2_manager.trainer.update_step
                snap.ckpt_hash = getattr(module2_manager.trainer, "active_checkpoint_hash", "--------")[:8]
                snap.platoon_state = str(mission_controller.state).split(".")[-1] if mission_controller else "NORMAL"
                snap.overrides = {
                    ("Leader" if i == 0 else f"F{i}"): mission_controller.diagnostics(v, i).get("safety_override", "NONE")
                    for i, v in enumerate(platoon) if v and v.is_alive
                }

                # Traffic light status for leader (using controller's geometric projection)
                if leader and leader.is_alive:
                    try:
                        ctl_tl = leader_diag.get("traffic_light", "NONE")
                        if ctl_tl in ("RED", "YELLOW", "GREEN"):
                            if ctl_tl == "RED":
                                snap.tl_label, snap.tl_color = "RED  ●", (255, 80, 80)
                            elif ctl_tl == "YELLOW":
                                snap.tl_label, snap.tl_color = "AMBER ●", (255, 210, 50)
                            else:
                                snap.tl_label, snap.tl_color = "GREEN ●", (70, 255, 120)
                        elif ctl_tl in ("CLEARED_LIGHT", "INSIDE_JUNCTION", "ADJACENT_SIGNAL", "NONE"):
                            snap.tl_label, snap.tl_color = "CLEAR ○", (120, 200, 160)
                        else:
                            tl = leader.get_traffic_light()
                            if tl is not None:
                                state = tl.get_state()
                                if state == carla.TrafficLightState.Red:
                                    snap.tl_label, snap.tl_color = "RED  ●", (255, 80, 80)
                                elif state == carla.TrafficLightState.Yellow:
                                    snap.tl_label, snap.tl_color = "AMBER ●", (255, 210, 50)
                                elif state == carla.TrafficLightState.Green:
                                    snap.tl_label, snap.tl_color = "GREEN ●", (70, 255, 120)
                        ctrl = leader.get_control()
                        snap.leader_thr = ctrl.throttle
                        snap.leader_brk = ctrl.brake
                    except Exception:
                        pass

                self.snapshot_holder.set(snap)

                # Periodic motion logging (every 2.0s)
                if elapsed_sim - last_motion_log >= 2.0:
                    last_motion_log = elapsed_sim
                    logger.info("\n[TICK %d] [DEMO] Simulation Time: %.1fs", world.get_snapshot().frame, elapsed_sim)
                    raw_acts = getattr(module2_manager, "current_raw_actions", None)
                    if raw_acts is None:
                        raw_acts = [None] * len(platoon)
                    for index, row in enumerate(snap.platoon_rows):
                        veh = platoon[index]
                        diag = mission_controller.diagnostics(veh, index)
                        ctrl = veh.get_control() if veh and veh.is_alive else None
                        tag = "Leader (F0)" if index == 0 else "Follower %d" % index
                        raw_a = raw_acts[index] if index < len(raw_acts) else "?"
                        tact_a = tactical_actions[index] if index < len(tactical_actions) else "?"

                        logger.info(
                            "  [AUTOPILOT] veh_id=%d (%s) CARLA_autopilot=False (MAPPO Mission Controller Authoritative)",
                            row["id"], tag
                        )
                        logger.info(
                            "  [MAPPO] veh_id=%d obs_dim=%d raw_action=%s action_name=%s",
                            row["id"], module2_manager.obs_builder.obs_dim, raw_a, tact_a
                        )
                        logger.info(
                            "  [ACTION_MANAGER] veh_id=%d raw_action=%s -> applied_tactical_action=%s",
                            row["id"], raw_a, tact_a
                        )
                        if ctrl:
                            logger.info(
                                "  [CONTROL] veh_id=%d speed=%.1f km/h target=%.1f km/h thr=%.2f brk=%.2f steer=%.3f gap=%.1f m wp=%s lat_err=%.2f m tl=%s override=%s",
                                row["id"], row["speed"], diag.get("target_speed", 0.0) * 3.6,
                                ctrl.throttle, ctrl.brake, ctrl.steer,
                                diag.get("predecessor_gap", 0.0), diag.get("waypoint_index"),
                                diag.get("lateral_error", 0.0), diag.get("traffic_light"),
                                diag.get("safety_override", "NONE")
                            )
                    logger.info(
                        "  MAPPO ACTION TOTALS -> CRUISE: %d | ACCELERATE: %d | DECELERATE: %d | CHANGE_LEFT: %d | CHANGE_RIGHT: %d",
                        action_counts.get("CRUISE", 0), action_counts.get("ACCELERATE", 0), action_counts.get("DECELERATE", 0),
                        action_counts.get("CHANGE_LEFT", 0), action_counts.get("CHANGE_RIGHT", 0)
                    )
                    if is_ghost_active:
                        tid = ghost_telemetry.get("target_id")
                        logger.info("Vehicle %s | Platoon: NO | Ghost: YES", tid)

                # Periodic performance summary
                t_perf_now = time.perf_counter()
                dt_perf = t_perf_now - perf_last_log_time
                if dt_perf >= 2.0 and perf_ticks_count > 0:
                    perf_sim_fps = perf_ticks_count / dt_perf
                    avg_tick = (perf_sim_tick_total / perf_ticks_count) * 1000.0
                    avg_mappo = (perf_mappo_total / perf_ticks_count) * 1000.0
                    avg_ctrl = (perf_controller_total / perf_ticks_count) * 1000.0
                    gpu_str = f"{hw_monitor.gpu_util}% ({hw_monitor.gpu_mem_used}MB)" if hw_monitor.gpu_util is not None else "N/A"
                    logger.info(
                        "[PERF PROFILE] Sim FPS: %4.1f | CARLA Tick: %5.2fms | MAPPO: %4.2fms | Ctrl: %4.2fms | CPU: %4.1f%% | GPU: %s",
                        perf_sim_fps, avg_tick, avg_mappo, avg_ctrl, hw_monitor.cpu_pct, gpu_str
                    )
                    perf_sim_tick_total = 0.0
                    perf_mappo_total = 0.0
                    perf_controller_total = 0.0
                    perf_ticks_count = 0
                    perf_last_log_time = t_perf_now

                # Precise 30 Hz loop pacing
                t_step = time.perf_counter() - t_loop_start
                sleep_sec = sim_target_dt - t_step
                if sleep_sec > 0.001:
                    time.sleep(sleep_sec)

        except Exception as exc:
            logger.exception("Exception in CARLA simulation thread: %s", exc)
            self.error_holder.append(exc)
            self.ready_event.set()
            self.stop_event.set()

        finally:
            logger.info("Cleaning up CARLA simulation thread...")
            if module2_manager and platoon and module2_manager.mode == "train":
                try:
                    module2_manager.record_transition(platoon)
                    module2_manager.train_if_ready()
                except Exception:
                    logger.exception("Could not flush the final MAPPO transition")
            if safety_monitor:
                safety_monitor.destroy()
            if camera:
                camera.destroy()
            if timeline_director:
                timeline_director.cleanup()
            if uncertainty_manager:
                uncertainty_manager.cleanup()
            for controller in controllers:
                try:
                    if controller and controller.is_alive:
                        controller.stop()
                        controller.destroy()
                except Exception:
                    pass
            for actor in walkers + npcs + platoon:
                try:
                    if actor and actor.is_alive:
                        actor.destroy()
                except Exception:
                    pass
            if world and tm and original_settings:
                restore_world_timing(world, tm, original_settings)
            self.stop_event.set()
            logger.info("CARLA simulation thread terminated cleanly.")


# ── MAIN APPLICATION RUNNER (UI DISPLAY THREAD) ─────────────────────────────
def run(args: argparse.Namespace) -> None:
    setup_logging(args.log_file)
    server_process = None
    if args.launch_server and not wait_for_server(args.host, args.port, 2):
        server_process = maybe_launch_carla(args.town, args.port, args.quality)

    if not wait_for_server(args.host, args.port, args.wait_seconds):
        raise RuntimeError("CARLA server did not respond.")

    # Pygame display initialization on Main Thread
    pygame.init()
    pygame.display.set_caption("CARLA Uncertain Platoon Simulation")
    window_width = args.width if args.width else WINDOW_WIDTH
    window_height = args.height if args.height else WINDOW_HEIGHT

    flags = pygame.DOUBLEBUF
    if getattr(args, "fullscreen", False):
        flags |= pygame.FULLSCREEN
        infoObject = pygame.display.Info()
        window_width = infoObject.current_w
        window_height = infoObject.current_h
    elif hasattr(pygame, "HWSURFACE"):
        flags |= pygame.HWSURFACE

    display = pygame.display.set_mode((window_width, window_height), flags)
    clock = pygame.time.Clock()
    camera_h = window_height - UnifiedHUD.STRIP_H
    hud = UnifiedHUD(window_width, window_height)

    # Inter-thread communication primitives
    snapshot_holder = SnapshotHolder()
    command_queue: queue.Queue = queue.Queue(maxsize=32)
    stop_event = threading.Event()
    ready_event = threading.Event()
    error_holder: List[Exception] = []

    # Launch authoritative CARLA Simulation Thread
    sim_thread = CarlaSimulationThread(
        args=args,
        snapshot_holder=snapshot_holder,
        command_queue=command_queue,
        stop_event=stop_event,
        ready_event=ready_event,
        error_holder=error_holder,
        window_width=window_width,
        camera_h=camera_h
    )
    sim_thread.start()

    # Wait for simulation to connect and load the world while pumping Windows events
    logger.info("Waiting for CARLA world and vehicle platoon initialization...")
    while not ready_event.is_set() and not stop_event.is_set():
        pygame.event.pump()
        time.sleep(0.05)

    if error_holder:
        pygame.quit()
        if server_process and args.kill_launched_server:
            stop_launched_carla(server_process)
        raise error_holder[0]

    render_fps = max(1, getattr(args, "render_fps", 20))
    logger.info("[UI] Main display thread active (UI target: %d FPS). Entering render loop...", render_fps)

    try:
        while not stop_event.is_set():
            # Process Pygame window events on main thread
            for event in pygame.event.get():
                if event.type == QUIT:
                    stop_event.set()
                    command_queue.put(("quit", None))
                elif event.type == KEYDOWN:
                    if event.key in (K_ESCAPE, K_q):
                        stop_event.set()
                        command_queue.put(("quit", None))
                    elif event.key == K_p:
                        command_queue.put(("toggle_control", None))
                    elif event.key == K_c:
                        command_queue.put(("toggle_camera_view", None))
                    elif event.key == K_TAB:
                        command_queue.put(("switch_camera_vehicle", None))

            # Retrieve latest pre-collected snapshot and camera surface
            snap = snapshot_holder.get()
            cam = sim_thread.camera

            # Render camera frame directly (zero-copy surface from CARLA background thread)
            if cam:
                cam.render(display)

            # Render UnifiedHUD from snapshot (zero CARLA RPCs!)
            hud.render(
                display, None, None, None, None,
                snap.control_active, snap.camera_name,
                snap.elapsed_sim, snap.town_name, snap.synchronous,
                snap.sim_fps,
                ghost_status=snap.ghost_status,
                mappo_actions=snap.mappo_actions,
                leader_diag=snap.leader_diag,
                ui_fps=clock.get_fps(),
                snapshot=snap
            )

            # Flip hardware display buffer
            pygame.display.flip()

            # Pace UI display independently from 30 Hz simulation
            clock.tick(render_fps)

    except Exception as exc:
        logger.warning("Exception in UI display loop: %s", exc)
    finally:
        stop_event.set()
        command_queue.put(("quit", None))
        sim_thread.join(timeout=10.0)
        pygame.quit()
        if server_process and args.kill_launched_server:
            stop_launched_carla(server_process)
        logger.info("Application shutdown complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Uncertain Platoon Simulation")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--town", default="Town01", help="Town01, Town02, ... or auto")
    parser.add_argument("--vehicles", type=int, default=12)
    parser.add_argument("--pedestrians", type=int, default=0)
    parser.add_argument("--platoon-size", type=int, default=3)
    parser.add_argument("--platoon-speed", type=float, default=28.0)
    parser.add_argument("--wait-seconds", type=int, default=60)
    parser.add_argument("--quality", default="Low", choices=["Low", "Epic"])
    parser.add_argument("--launch-server", action="store_true")
    parser.add_argument("--kill-launched-server", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--async-world", action="store_true")
    parser.add_argument("--log-file", default=os.path.join(SCRIPT_DIR, "logs", "uncertain_platoon.log"))
    parser.add_argument("--no-clean", dest="clean", action="store_false")
    parser.add_argument("--weather", type=float, default=5.0, help="Weather change interval in minutes")
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train",
                        help="train updates MAPPO after rollouts; evaluate keeps the loaded policy frozen")
    parser.add_argument("--fullscreen", action="store_true", default=False,
                        help="Run Pygame in fullscreen mode (default: windowed)")
    parser.add_argument("--windowed", action="store_true", default=True,
                        help="Run Pygame in windowed mode (default: True)")
    parser.add_argument("--width", type=int, default=1280,
                        help="Window width in pixels (default: 1280)")
    parser.add_argument("--height", type=int, default=720,
                        help="Window height in pixels (default: 720)")
    parser.add_argument("--render-fps", type=int, default=20,
                        help="Dashboard UI render framerate limit (default: 20 FPS to prevent CARLA UE4 stutter)")
    parser.add_argument("--diagnostic-control-test", action="store_true",
                        help="Run diagnostic control response test cycling ACCELERATE/DECELERATE/CRUISE to verify physical throttle/brake dynamics")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to specific MAPPO checkpoint .pt file to load")
    parser.add_argument("--ghost-scenario",
                        choices=("auto", "normal", "gap_intrusion", "leader_cut_in",
                                 "follower_cut_in", "approach_behind", "adjacent_block",
                                 "clearing", "persistent"),
                        default="auto",
                        help="Deterministic scripted scenario for ghost driver (default: auto)")
    parser.add_argument("--ghost-seed", type=int, default=42,
                        help="Fixed random seed for deterministic ghost driver intrusion (default: 42)")
    parser.set_defaults(clean=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
