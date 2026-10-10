#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
CARLA 0.9.9 - LIVING REAL-WORLD UNCERTAINTY SIMULATION
================================================================================
A 1-click autonomous & manual driving simulation in CARLA 0.9.9 featuring:
  1. Populated living city: 30-35 NPC vehicles driven with diverse human driver
     profiles (Aggressive, Normal, Distracted, Conservative, Erratic), with
     pedestrians disabled for vehicle-only platoon uncertainty.
  2. Ego vehicle with 3D Pygame HUD (1024x576 resolution, 4GB VRAM optimized).
  3. AI Autopilot enabled by default via Traffic Manager; seamless manual
     driving override anytime via WASD / Arrow Keys, and 'P' to toggle.
  4. Dedicated Ghost Driver uncertainty: exactly one nearby NPC vehicle at a
     time is possessed inside the 50 m platoon bubble, then safely restored.
  5. Clean, robust cleanup on ESC / close.
================================================================================
"""

from __future__ import annotations

import glob
import logging
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Path configuration & CARLA egg setup
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CARLA_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, '..'))

try:
    sys.path.append(
        glob.glob(
            os.path.join(
                SCRIPT_DIR,
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

# Include PythonAPI root for importing uncertainty package
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import carla
import numpy as np
import pygame
from pygame.locals import (
    K_DOWN,
    K_ESCAPE,
    K_LEFT,
    K_RIGHT,
    K_SPACE,
    K_UP,
    K_a,
    K_c,
    K_d,
    K_p,
    K_q,
    K_s,
    K_w,
    KEYDOWN,
    KEYUP,
    QUIT,
)

# Import uncertainty framework modules
from uncertainty.accident_manager import AccidentManager, AccidentType
from uncertainty.friction_uncertainty import FrictionCondition, FrictionManager
from uncertainty.ghost_driver import GhostDriver
from uncertainty.human_behavior import DriverProfile, HumanBehaviorManager
from uncertainty.uncertainty_manager import UncertaintyManager
from uncertainty.weather_uncertainty import WeatherManager, WeatherPreset

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    level=logging.INFO
)
logger = logging.getLogger('RealWorldSimulation')

# Simulation Constants
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 576
TARGET_FPS = 30
NUM_VEHICLES = 32
NUM_PEDESTRIANS = 0  # Vehicle-only uncertainty; walkers disabled by default.
TM_PORT = 8000
DEFAULT_MAP = 'Town01'


# Helper for CARLA 0.9.9 Traffic Manager speed difference API
def set_tm_speed_diff(tm: Any, vehicle: carla.Vehicle, diff: float) -> None:
    """Set speed difference percentage supporting both CARLA 0.9.9 and newer TM APIs."""
    if hasattr(tm, "vehicle_percentage_speed_difference"):
        tm.vehicle_percentage_speed_difference(vehicle, float(diff))
    elif hasattr(tm, "set_percentage_speed_difference"):
        tm.set_percentage_speed_difference(vehicle, float(diff))


# =============================================================================
# SENSOR MANAGERS (Camera & Collision)
# =============================================================================

class CameraManager:
    """Manages the RGB camera attached to the Ego vehicle and renders to Pygame."""

    def __init__(self, parent_actor: carla.Vehicle, width: int, height: int):
        self._parent = parent_actor
        self.width = width
        self.height = height
        self.surface: Optional[pygame.Surface] = None
        self.sensor: Optional[carla.Sensor] = None
        self.view_mode = 0  # 0: Chase (3rd Person), 1: Cockpit (1st Person)

        self.views = [
            (
                "Chase View (3rd Person)",
                carla.Transform(
                    carla.Location(x=-5.5, z=2.4),
                    carla.Rotation(pitch=-7.5)
                )
            ),
            (
                "Platoon Overview (High Chase)",
                carla.Transform(
                    carla.Location(x=-13.5, z=6.0),
                    carla.Rotation(pitch=-18.0)
                )
            ),
            (
                "Cockpit View (1st Person)",
                carla.Transform(
                    carla.Location(x=0.7, y=-0.32, z=1.25),
                    carla.Rotation(pitch=-2.0)
                )
            )
        ]

        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(self.width))
        bp.set_attribute('image_size_y', str(self.height))
        bp.set_attribute('fov', '90')

        self.sensor = world.spawn_actor(
            bp,
            self.views[self.view_mode][1],
            attach_to=self._parent
        )

        import weakref
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CameraManager._on_image(weak_self, image))

    @staticmethod
    def _on_image(weak_self, image: carla.Image):
        self = weak_self()
        if not self:
            return
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        # BGRA -> RGB
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    def toggle_view(self) -> str:
        """Switch between Chase and Cockpit camera views."""
        self.view_mode = (self.view_mode + 1) % len(self.views)
        name, transform = self.views[self.view_mode]
        if self.sensor and self.sensor.is_alive:
            self.sensor.set_transform(transform)
        return name

    def render(self, display: pygame.Surface):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    def destroy(self):
        if self.sensor and self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()
            self.sensor = None


class CollisionSensor:
    """Detects collisions involving the ego vehicle."""

    def __init__(self, parent_actor: carla.Vehicle):
        self._parent = parent_actor
        self.last_collision_time = 0.0
        self.collision_msg = ""
        self.sensor: Optional[carla.Sensor] = None

        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(),
            attach_to=self._parent
        )

        import weakref
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        other_name = event.other_actor.type_id if event.other_actor else "Unknown Obstacle"
        # Extract clean actor name (e.g. vehicle.audi.a2 -> Audi A2)
        parts = other_name.split('.')
        clean_name = f"{parts[1].capitalize()} {parts[2].capitalize()}" if len(parts) >= 3 else other_name
        self.last_collision_time = time.time()
        self.collision_msg = f"COLLISION with {clean_name}!"
        logger.warning("Collision detected: %s", self.collision_msg)

    def is_collision_active(self, current_time: float, window: float = 3.5) -> bool:
        return (current_time - self.last_collision_time) < window

    def destroy(self):
        if self.sensor and self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()
            self.sensor = None


# =============================================================================
# SIMULATION HUD OVERLAY
# =============================================================================

class SimulationHUD:
    """Renders a modern, highly informative on-screen HUD for CARLA 0.9.9."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        pygame.font.init()

        # Fonts
        self.font_title = pygame.font.SysFont('consolas', 18, bold=True)
        self.font_large = pygame.font.SysFont('arial', 30, bold=True)
        self.font_main = pygame.font.SysFont('consolas', 14, bold=True)
        self.font_small = pygame.font.SysFont('consolas', 12, bold=False)
        self.font_banner = pygame.font.SysFont('arial', 17, bold=True)

    def render(
        self,
        display: pygame.Surface,
        ego_vehicle: carla.Vehicle,
        autopilot: bool,
        control: carla.VehicleControl,
        weather_info: Dict[str, Any],
        friction_mu: float,
        active_alert: Optional[Tuple[str, str, float]],
        collision_sensor: CollisionSensor,
        timeline_phase: str,
        elapsed_sim_time: float,
        fps: float,
        town_name: str,
        npc_count: int,
        ped_count: int,
        camera_name: str,
        ghost_status: Optional[Dict[str, Any]] = None
    ):
        """Draw all HUD elements over the 3D viewport."""

        # ---------------------------------------------------------------------
        # 1. TOP HEADER BAR
        # ---------------------------------------------------------------------
        header_surf = pygame.Surface((self.width, 36), pygame.SRCALPHA)
        header_surf.fill((12, 16, 24, 210))
        display.blit(header_surf, (0, 0))

        # Title (Left)
        title_text = self.font_title.render(
            "⚡ CARLA 0.9.9  |  LIVING REAL-WORLD UNCERTAINTY SIMULATION",
            True,
            (240, 245, 255)
        )
        display.blit(title_text, (16, 8))

        # Map & Simulation Time (Right)
        mins = int(elapsed_sim_time) // 60
        secs = int(elapsed_sim_time) % 60
        time_str = f"{mins:02d}:{secs:02d}"
        info_text = self.font_main.render(
            f"MAP: {town_name}  |  TIME: {time_str}  |  {fps:.0f} FPS",
            True,
            (180, 210, 240)
        )
        display.blit(info_text, (self.width - info_text.get_width() - 16, 10))

        # ---------------------------------------------------------------------
        # 2. PROMINENT HAZARD ALERT BANNER (Center-Top)
        # ---------------------------------------------------------------------
        now = time.time()
        is_collision = collision_sensor.is_collision_active(now)

        if is_collision:
            banner_title = "⚠️ CRITICAL IMPACT DETECTED"
            banner_desc = collision_sensor.collision_msg
            banner_bg = (90, 15, 15, 240)
            border_color = (255, 50, 50)
            text_color = (255, 255, 255)
            self._draw_banner(display, banner_title, banner_desc, banner_bg, border_color, text_color)
        elif ghost_status is not None and ghost_status.get("active"):
            target_id = ghost_status.get("target_id")
            m_name = ghost_status.get("disruption_name", "ADVERSARIAL INJECTION")
            active_for = ghost_status.get("elapsed_seconds", 0.0)
            dist = ghost_status.get("distance_to_platoon", ghost_status.get("distance_to_ego", 0.0))
            banner_title = f"👻 GHOST DRIVER: {m_name.upper()}"
            banner_desc = f"Vehicle #{target_id} possessed in 50m bubble! Dist: {dist:.1f}m | Active: {active_for:.1f}s"
            banner_bg = (55, 18, 68, 240)
            border_color = (215, 80, 255)
            text_color = (255, 240, 255)
            self._draw_banner(display, banner_title, banner_desc, banner_bg, border_color, text_color)
        elif active_alert is not None:
            alert_title, alert_desc, end_time = active_alert
            rem = max(0.0, end_time - elapsed_sim_time)
            desc_with_time = f"{alert_desc} [Active: {rem:.0f}s remaining]" if rem > 0 else alert_desc
            banner_bg = (40, 25, 10, 230)
            border_color = (255, 170, 20)
            text_color = (255, 245, 210)
            self._draw_banner(display, alert_title, desc_with_time, banner_bg, border_color, text_color)
        else:
            # Nominal conditions badge
            self._draw_nominal_pill(display, timeline_phase)

        # ---------------------------------------------------------------------
        # 3. ENVIRONMENT & ROAD GRIP STATUS CARD (Top-Right)
        # ---------------------------------------------------------------------
        card_w, card_h = 245, 160
        card_x = self.width - card_w - 16
        card_y = 48
        card_surf = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
        card_surf.fill((14, 18, 26, 215))
        pygame.draw.rect(card_surf, (50, 75, 110), card_surf.get_rect(), width=1, border_radius=6)
        display.blit(card_surf, (card_x, card_y))

        # Header
        cond_header = self.font_main.render("ENVIRONMENT & ROAD GRIP", True, (140, 185, 245))
        display.blit(cond_header, (card_x + 12, card_y + 10))

        # Weather string
        rain = weather_info.get('rain_intensity', 0.0)
        fog = weather_info.get('fog_intensity', 0.0)
        wetness = weather_info.get('wetness', 0.0)

        if rain > 50:
            w_desc = "Heavy Rainstorm (Wet)"
            w_color = (120, 190, 255)
        elif rain > 15:
            w_desc = "Moderate Rain"
            w_color = (140, 210, 255)
        elif fog > 35:
            w_desc = "Dense Fog (Low Vis)"
            w_color = (200, 215, 230)
        else:
            w_desc = "Sunny & Clear"
            w_color = (255, 230, 140)

        w_text = self.font_small.render(f"Weather: {w_desc}", True, w_color)
        display.blit(w_text, (card_x + 12, card_y + 34))

        # Road Grip (mu)
        if friction_mu >= 0.75:
            grip_label = f"Dry Road  (μ = {friction_mu:.2f})"
            grip_color = (80, 240, 130)
            grip_status = "Optimal Traction"
        elif friction_mu >= 0.50:
            grip_label = f"Wet Road  (μ = {friction_mu:.2f})"
            grip_color = (255, 200, 60)
            grip_status = "Reduced Grip"
        else:
            grip_label = f"Slippery  (μ = {friction_mu:.2f})"
            grip_color = (255, 80, 80)
            grip_status = "HYDROPLANING RISK!"

        grip_text = self.font_main.render(f"Grip: {grip_label}", True, grip_color)
        display.blit(grip_text, (card_x + 12, card_y + 56))

        grip_sub = self.font_small.render(f"Status: {grip_status}", True, (210, 220, 230))
        display.blit(grip_sub, (card_x + 12, card_y + 78))

        # Traffic & Walkers count
        pop_text = self.font_small.render(
            f"City: {npc_count} Vehicles | {ped_count} Walkers",
            True,
            (160, 180, 200)
        )
        display.blit(pop_text, (card_x + 12, card_y + 98))

        # Active Timeline Phase
        phase_text = self.font_small.render(
            f"Phase: {timeline_phase}",
            True,
            (140, 220, 255)
        )
        display.blit(phase_text, (card_x + 12, card_y + 118))

        # Ghost Driver Proximity Status
        if ghost_status and ghost_status.get("active"):
            target_id = ghost_status.get("target_id")
            active_for = ghost_status.get("elapsed_seconds", 0.0)
            g_str = f"Ghost: POSSESSING #{target_id} ({active_for:.1f}s)"
            g_col = (255, 120, 240)
        else:
            cd = ghost_status.get("cooldown_remaining", 0.0) if ghost_status else 0.0
            g_str = f"Ghost: LURKING (Next: {cd:.0f}s)"
            g_col = (170, 195, 225)
        g_text = self.font_small.render(g_str, True, g_col)
        display.blit(g_text, (card_x + 12, card_y + 138))

        # ---------------------------------------------------------------------
        # 4. VEHICLE DASHBOARD TELEMETRY (Bottom-Left)
        # ---------------------------------------------------------------------
        dash_w, dash_h = 240, 148
        dash_x = 16
        dash_y = self.height - dash_h - 16
        dash_surf = pygame.Surface((dash_w, dash_h), pygame.SRCALPHA)
        dash_surf.fill((14, 18, 26, 215))
        pygame.draw.rect(dash_surf, (50, 75, 110), dash_surf.get_rect(), width=1, border_radius=6)
        display.blit(dash_surf, (dash_x, dash_y))

        # Speed calculation
        v = ego_vehicle.get_velocity()
        speed_kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)

        # Big Speedometer Number
        speed_val = self.font_large.render(f"{int(speed_kmh):3d}", True, (255, 255, 255))
        speed_unit = self.font_main.render("km/h", True, (160, 190, 220))
        display.blit(speed_val, (dash_x + 12, dash_y + 8))
        display.blit(speed_unit, (dash_x + 82, dash_y + 20))

        # Gear Indicator
        gear_str = "R" if control.reverse else ("D" if control.gear > 0 else "N")
        gear_badge = self.font_main.render(f"GEAR: [{gear_str}]", True, (255, 215, 100))
        display.blit(gear_badge, (dash_x + 145, dash_y + 20))

        # Controls Gauges: Throttle, Brake, Steer
        # Throttle Bar (Green)
        self._draw_bar(display, dash_x + 14, dash_y + 54, 130, 12, control.throttle, (60, 220, 100), "THROTTLE")
        # Brake Bar (Red)
        self._draw_bar(display, dash_x + 14, dash_y + 76, 130, 12, control.brake, (255, 70, 70), "BRAKE")
        # Steer indicator (Cyan)
        self._draw_steer_gauge(display, dash_x + 14, dash_y + 98, 130, 12, control.steer)

        # Camera Mode indicator
        cam_text = self.font_small.render(f"Cam: {camera_name} [C]", True, (150, 180, 210))
        display.blit(cam_text, (dash_x + 14, dash_y + 124))

        # ---------------------------------------------------------------------
        # 5. AUTOPILOT / MANUAL MODE BADGE (Bottom-Center)
        # ---------------------------------------------------------------------
        badge_w, badge_h = 320, 48
        badge_x = (self.width - badge_w) // 2
        badge_y = self.height - badge_h - 16

        badge_surf = pygame.Surface((badge_w, badge_h), pygame.SRCALPHA)
        if autopilot:
            badge_surf.fill((10, 42, 22, 225))
            pygame.draw.rect(badge_surf, (45, 210, 110), badge_surf.get_rect(), width=2, border_radius=8)
            title = self.font_main.render("● AI AUTOPILOT ENGAGED (Traffic Manager)", True, (100, 255, 160))
            sub = self.font_small.render("Press 'P' or use WASD to take Manual Control", True, (200, 240, 215))
        else:
            badge_surf.fill((50, 28, 10, 225))
            pygame.draw.rect(badge_surf, (245, 150, 30), badge_surf.get_rect(), width=2, border_radius=8)
            title = self.font_main.render("▲ MANUAL CONTROL OVERRIDE (Driver WASD)", True, (255, 190, 80))
            sub = self.font_small.render("Press 'P' anytime to return to AI Autopilot", True, (255, 235, 200))

        display.blit(badge_surf, (badge_x, badge_y))
        display.blit(title, (badge_x + 16, badge_y + 8))
        display.blit(sub, (badge_x + 16, badge_y + 26))

        # ---------------------------------------------------------------------
        # 6. QUICK CONTROLS BAR (Bottom-Right)
        # ---------------------------------------------------------------------
        kb_w, kb_h = 245, 48
        kb_x = self.width - kb_w - 16
        kb_y = self.height - kb_h - 16
        kb_surf = pygame.Surface((kb_w, kb_h), pygame.SRCALPHA)
        kb_surf.fill((14, 18, 26, 215))
        pygame.draw.rect(kb_surf, (50, 75, 110), kb_surf.get_rect(), width=1, border_radius=6)
        display.blit(kb_surf, (kb_x, kb_y))

        kb_l1 = self.font_small.render("[W/A/S/D] Drive  |  [P] Autopilot Toggle", True, (200, 220, 240))
        kb_l2 = self.font_small.render("[C] Switch Camera  |  [ESC] Exit & Clean", True, (160, 185, 210))
        display.blit(kb_l1, (kb_x + 10, kb_y + 8))
        display.blit(kb_l2, (kb_x + 10, kb_y + 26))

    def _draw_banner(
        self,
        display: pygame.Surface,
        title: str,
        desc: str,
        bg_color: Tuple[int, int, int, int],
        border_color: Tuple[int, int, int],
        text_color: Tuple[int, int, int]
    ):
        """Draw the large, attention-grabbing center hazard banner."""
        banner_w = 600
        banner_h = 56
        bx = (self.width - banner_w) // 2
        by = 48

        if border_color[0] >= 200 and border_color[2] >= 220:
            for spread, alpha in ((8, 42), (4, 70)):
                glow_rect = pygame.Rect(bx - spread, by - spread, banner_w + spread * 2, banner_h + spread * 2)
                glow_surf = pygame.Surface((glow_rect.width, glow_rect.height), pygame.SRCALPHA)
                pygame.draw.rect(
                    glow_surf,
                    (160, 50, 255, alpha),
                    glow_surf.get_rect(),
                    width=2,
                    border_radius=10 + spread,
                )
                display.blit(glow_surf, glow_rect.topleft)

        banner_surf = pygame.Surface((banner_w, banner_h), pygame.SRCALPHA)
        banner_surf.fill(bg_color)
        pygame.draw.rect(banner_surf, border_color, banner_surf.get_rect(), width=2, border_radius=8)
        display.blit(banner_surf, (bx, by))

        t_surf = self.font_banner.render(title, True, border_color)
        d_surf = self.font_main.render(desc, True, text_color)

        display.blit(t_surf, (bx + (banner_w - t_surf.get_width()) // 2, by + 7))
        display.blit(d_surf, (bx + (banner_w - d_surf.get_width()) // 2, by + 30))

    def _draw_nominal_pill(self, display: pygame.Surface, phase_name: str):
        """Draw a calm, nominal driving status badge."""
        pill_w = 440
        pill_h = 32
        px = (self.width - pill_w) // 2
        py = 48

        pill_surf = pygame.Surface((pill_w, pill_h), pygame.SRCALPHA)
        pill_surf.fill((16, 26, 38, 190))
        pygame.draw.rect(pill_surf, (40, 120, 180), pill_surf.get_rect(), width=1, border_radius=6)
        display.blit(pill_surf, (px, py))

        text = self.font_small.render(f"✓ {phase_name.upper()}  |  ALL TELEMETRY NOMINAL", True, (130, 205, 255))
        display.blit(text, (px + (pill_w - text.get_width()) // 2, py + 8))

    def _draw_bar(
        self,
        display: pygame.Surface,
        x: int,
        y: int,
        w: int,
        h: int,
        val: float,
        color: Tuple[int, int, int],
        label: str
    ):
        lbl = self.font_small.render(f"{label[:4]}", True, (170, 190, 210))
        display.blit(lbl, (x, y))
        bar_x = x + 38
        pygame.draw.rect(display, (35, 45, 60), (bar_x, y, w, h), border_radius=3)
        fill_w = int(max(0.0, min(1.0, val)) * w)
        if fill_w > 0:
            pygame.draw.rect(display, color, (bar_x, y, fill_w, h), border_radius=3)

    def _draw_steer_gauge(
        self,
        display: pygame.Surface,
        x: int,
        y: int,
        w: int,
        h: int,
        steer_val: float
    ):
        lbl = self.font_small.render("STR", True, (170, 190, 210))
        display.blit(lbl, (x, y))
        bar_x = x + 38
        pygame.draw.rect(display, (35, 45, 60), (bar_x, y, w, h), border_radius=3)
        # Center line
        mid_x = bar_x + (w // 2)
        pygame.draw.line(display, (120, 140, 160), (mid_x, y), (mid_x, y + h))

        # Steer point
        clamped = max(-1.0, min(1.0, steer_val))
        steer_px = int(mid_x + clamped * (w // 2))
        pygame.draw.circle(display, (80, 220, 255), (steer_px, y + h // 2), 5)


# =============================================================================
# CONTINUOUS MULTI-PHASE UNCERTAINTY TIMELINE DIRECTOR
# =============================================================================

class UncertaintyTimelineDirector:
    """Coordinates the progressive 5-phase real-world uncertainty timeline."""

    def __init__(
        self,
        world: carla.World,
        ego_vehicle: carla.Vehicle,
        npc_vehicles: List[carla.Vehicle],
        walkers: List[carla.Actor],
        walker_controllers: List[carla.Actor],
        traffic_manager: carla.TrafficManager,
        uncertainty_manager: UncertaintyManager,
        platoon_vehicles: Optional[List[carla.Vehicle]] = None
    ):
        self.world = world
        self.ego = ego_vehicle
        self.platoon = platoon_vehicles or [ego_vehicle]
        self.npcs = npc_vehicles
        self.walkers = walkers
        self.controllers = walker_controllers
        self.tm = traffic_manager
        self.mgr = uncertainty_manager

        # Sub-managers for quick direct actions
        self.weather_mgr: WeatherManager = self.mgr.weather
        self.friction_mgr: FrictionManager = self.mgr.friction
        self.behavior_mgr: HumanBehaviorManager = self.mgr.behavior
        self.accident_mgr: AccidentManager = self.mgr.accident

        # Current Alert State: (title, desc, expiration_sim_time)
        self.active_alert: Optional[Tuple[str, str, float]] = None
        self.current_phase_name = "Phase 1: Baseline City Flow"

        # Milestones executed tracker
        self.events_done: Dict[str, bool] = {}

        # Recoverable Tactical Actions Tracker: (npc, release_time)
        self.braked_npcs: List[Tuple[carla.Vehicle, float]] = []

        # Temporary Hazard Actors Tracker: (actor, expiration_time)
        self.hazard_actors: List[Tuple[carla.Actor, float]] = []

        # Walker continuous navigation timer
        self.last_walker_refresh = 0.0

        # NPC vehicle anti-stuck watchdog
        self.npc_stopped_time: Dict[int, float] = {}
        self.last_unstick_check = 0.0

        # Phase 5 Cyclical Timer
        self.last_phase5_event_time = 125.0
        self.phase5_cycle_index = 0

        # Initialize Ghost Driver: one vehicle at a time inside the 50 m platoon bubble.
        self.ghost_driver = GhostDriver(
            world=self.world,
            traffic_manager=self.tm,
            ego_vehicle=self.ego,
            npc_vehicles=self.npcs,
            platoon_vehicles=self.platoon,
            walkers=self.walkers,
            walker_controllers=self.controllers,
            min_cooldown=18.0,
            max_cooldown=32.0,
            min_possession=5.0,
            max_possession=10.0,
            proximity_radius=50.0,
            seed=42
        )

        # Start with CLEAR weather and DRY friction
        self.weather_mgr.set_weather(self.world, WeatherPreset.CLEAR)
        self.friction_mgr.set_friction(self.ego, 0.85)

    def trigger_alert(self, title: str, desc: str, duration: float, sim_time: float):
        """Display an on-screen alert banner for a set duration."""
        self.active_alert = (title, desc, sim_time + duration)
        logger.info("[ALERT TRIGGERED @ t=%.1fs] %s - %s", sim_time, title, desc)

    def tick(self, elapsed_sim_time: float, delta_sec: float = 0.033):
        """Update timeline based on elapsed simulation time."""

        # 0. Update Ghost Driver state machine
        ghost_alert = self.ghost_driver.tick(delta_sec, elapsed_sim_time)
        if ghost_alert is not None:
            g_title, g_desc, g_dur = ghost_alert
            self.trigger_alert(g_title, g_desc, duration=g_dur, sim_time=elapsed_sim_time)

        # 1. Clear expired banner alert
        if self.active_alert is not None:
            _, _, end_t = self.active_alert
            if elapsed_sim_time >= end_t:
                self.active_alert = None

        # 2. Check and recover temporary braked NPCs (so cars resume driving naturally!)
        remaining_braked = []
        for npc, release_t in self.braked_npcs:
            if elapsed_sim_time >= release_t:
                if npc and npc.is_alive:
                    try:
                        npc.apply_control(carla.VehicleControl(brake=0.0, throttle=0.4))
                        npc.set_autopilot(True, self.tm.get_port())
                        set_tm_speed_diff(self.tm, npc, 0.0)
                        self.tm.distance_to_leading_vehicle(npc, 3.5)
                        logger.info("NPC vehicle %d recovered from brake, resuming autonomous flow.", npc.id)
                    except Exception:
                        pass
            else:
                remaining_braked.append((npc, release_t))
        self.braked_npcs = remaining_braked

        # 3. Check and clear expired temporary hazard obstacles
        remaining_hazards = []
        for actor, expire_t in self.hazard_actors:
            if elapsed_sim_time >= expire_t:
                if actor and actor.is_alive:
                    try:
                        actor.destroy()
                        logger.info("Temporary hazard actor %d cleared.", actor.id)
                    except Exception:
                        pass
            else:
                remaining_hazards.append((actor, expire_t))
        self.hazard_actors = remaining_hazards

        # 4. Keep pedestrians walking continuously (if arrived and stopped, assign new destination)
        if self.controllers and elapsed_sim_time - self.last_walker_refresh >= 4.0:
            self.last_walker_refresh = elapsed_sim_time
            for con in self.controllers:
                if con and con.is_alive:
                    try:
                        parent_walker = con.parent
                        needs_new_dest = True
                        if parent_walker and parent_walker.is_alive:
                            vel = parent_walker.get_velocity()
                            speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                            if speed > 0.2:
                                needs_new_dest = False
                        if needs_new_dest:
                            dest = self.world.get_random_location_from_navigation()
                            if dest:
                                con.go_to_location(dest)
                                con.set_max_speed(1.2 + random.random() * 0.6)
                    except Exception:
                        pass

        # 5. Anti-gridlock watchdog (re-kickstart any car stuck at 0 km/h for > 22s)
        if elapsed_sim_time - self.last_unstick_check >= 5.0:
            self.last_unstick_check = elapsed_sim_time
            for npc in self.npcs:
                if npc and npc.is_alive:
                    try:
                        v = npc.get_velocity()
                        spd = math.sqrt(v.x**2 + v.y**2 + v.z**2)
                        if spd < 0.1:
                            self.npc_stopped_time[npc.id] = self.npc_stopped_time.get(npc.id, 0.0) + 5.0
                            if self.npc_stopped_time[npc.id] >= 22.0:
                                npc.apply_control(carla.VehicleControl(brake=0.0, throttle=0.35))
                                npc.set_autopilot(True, self.tm.get_port())
                                self.tm.auto_lane_change(npc, True)
                                self.tm.distance_to_leading_vehicle(npc, 3.5)
                                self.npc_stopped_time[npc.id] = 0.0
                        else:
                            self.npc_stopped_time[npc.id] = 0.0
                    except Exception:
                        pass

        # Ghost-only vehicle uncertainty mode: the dedicated GhostDriver above
        # owns every adversarial vehicle disturbance. The older scripted events
        # remain below for reference but are intentionally bypassed so only one
        # nearby car is disturbed at any moment.
        self._update_ghost_phase_name(elapsed_sim_time)
        return

        # ---------------------------------------------------------------------
        # PHASE 1: Normal City Driving (0s - 25s)
        # ---------------------------------------------------------------------
        if elapsed_sim_time < 25.0:
            self.current_phase_name = "Phase 1: Normal City Driving (Dry Road)"

        # ---------------------------------------------------------------------
        # PHASE 2: Human Error & Behavioral Uncertainty (25s - 55s)
        # ---------------------------------------------------------------------
        elif 25.0 <= elapsed_sim_time < 55.0:
            self.current_phase_name = "Phase 2: Human Driver Error & Cut-In"

            # Event 2A: Aggressive Cut-In at t=25s
            if "p2_cutin" not in self.events_done:
                self.events_done["p2_cutin"] = True
                self._execute_cut_in(elapsed_sim_time)

            # Event 2B: Sudden Lead Vehicle Braking at t=40s
            if elapsed_sim_time >= 40.0 and "p2_lead_brake" not in self.events_done:
                self.events_done["p2_lead_brake"] = True
                self._execute_sudden_brake(elapsed_sim_time)

        # ---------------------------------------------------------------------
        # PHASE 3: Weather & Road Friction Shift (55s - 90s)
        # ---------------------------------------------------------------------
        elif 55.0 <= elapsed_sim_time < 90.0:
            self.current_phase_name = "Phase 3: Storm & Road Friction Drop"

            # Smoothly transition weather to HEAVY_RAIN between 55s and 72s
            if elapsed_sim_time < 72.0:
                progress = (elapsed_sim_time - 55.0) / 17.0
                self.weather_mgr.transition_weather(
                    self.world,
                    WeatherPreset.HEAVY_RAIN,
                    progress
                )
                if "p3_rain_alert" not in self.events_done:
                    self.events_done["p3_rain_alert"] = True
                    self.trigger_alert(
                        "⚠️ INCLEMENT WEATHER: HEAVY RAINSTORM",
                        "Precipitation & road wetness rising rapidly. Visibility compromised.",
                        duration=14.0,
                        sim_time=elapsed_sim_time
                    )

            # Drop friction at t=68s to mu = 0.40 (slippery)
            if elapsed_sim_time >= 68.0 and "p3_friction_drop" not in self.events_done:
                self.events_done["p3_friction_drop"] = True
                self.friction_mgr.set_friction(self.ego, 0.40)
                # Also apply to nearby NPCs
                for npc in self.npcs[:12]:
                    try:
                        self.friction_mgr.set_friction(npc, 0.40)
                    except Exception:
                        pass
                self.trigger_alert(
                    "⚠️ LOW ROAD GRIP: μ = 0.40 (SLIPPERY)",
                    "Wet asphalt traction loss! Stopping distance doubled. Hydroplaning hazard.",
                    duration=18.0,
                    sim_time=elapsed_sim_time
                )

        # ---------------------------------------------------------------------
        # PHASE 4: Road Hazards & VRU Crossing (90s - 125s)
        # ---------------------------------------------------------------------
        elif 90.0 <= elapsed_sim_time < 125.0:
            self.current_phase_name = "Phase 4: Road Hazards & VRU (Jaywalker)"

            # Event 4A: Disabled vehicle blocking lane at t=90s
            if "p4_breakdown" not in self.events_done:
                self.events_done["p4_breakdown"] = True
                self._execute_breakdown_ahead(elapsed_sim_time)

            # Event 4B: Jaywalker / Pedestrian crossing at t=108s
            if elapsed_sim_time >= 108.0 and "p4_jaywalker" not in self.events_done:
                self.events_done["p4_jaywalker"] = True
                self._execute_crossing_pedestrian(elapsed_sim_time)

        # ---------------------------------------------------------------------
        # PHASE 5: Continuous Cyclical Uncertainty Engine (125s+)
        # ---------------------------------------------------------------------
        else:
            self.current_phase_name = "Phase 5: Continuous Dynamic Real-World Engine"

            # Periodically fire a dynamic event every 25-30 seconds
            if elapsed_sim_time - self.last_phase5_event_time >= 26.0:
                self.last_phase5_event_time = elapsed_sim_time
                self._execute_cyclical_uncertainty(elapsed_sim_time)

    def _update_ghost_phase_name(self, elapsed_sim_time: float):
        """Keep HUD phases meaningful while GhostDriver is the sole disturbance source."""
        if elapsed_sim_time < 25.0:
            self.current_phase_name = "Phase 1: Smooth Traffic Manager Baseline"
        elif elapsed_sim_time < 55.0:
            self.current_phase_name = "Phase 2: Ghost Driver 50m Bubble Armed"
        elif elapsed_sim_time < 90.0:
            self.current_phase_name = "Phase 3: Platoon Stability Under Ghost Events"
        elif elapsed_sim_time < 125.0:
            self.current_phase_name = "Phase 4: Vehicle-Only Uncertainty"
        else:
            self.current_phase_name = "Phase 5: Continuous Ghost Driver Rotation"

    # -------------------------------------------------------------------------
    # TACTICAL EVENT IMPLEMENTATIONS
    # -------------------------------------------------------------------------

    def _execute_cut_in(self, sim_time: float):
        """Direct a nearby NPC to overtake and sharply cut into ego's lane."""
        ego_tf = self.ego.get_transform()
        ego_loc = ego_tf.location

        # Find closest NPC within 10 to 45 meters
        target_npc = None
        min_dist = float('inf')
        for npc in self.npcs:
            if not npc.is_alive:
                continue
            dist = ego_loc.distance(npc.get_location())
            if 8.0 < dist < 45.0 and dist < min_dist:
                min_dist = dist
                target_npc = npc

        if target_npc is not None:
            # Command NPC to accelerate past ego and force lane change
            set_tm_speed_diff(self.tm, target_npc, -20.0)
            self.tm.distance_to_leading_vehicle(target_npc, 2.0)
            self.tm.force_lane_change(target_npc, random.choice([True, False]))
            # Restore normal speed diff after 6 seconds
            self.braked_npcs.append((target_npc, sim_time + 6.0))

            self.trigger_alert(
                "⚠️ RECKLESS OVERTAKE & CUT-IN DETECTED",
                f"Aggressive driver (ID: {target_npc.id}) cutting into your lane!",
                duration=10.0,
                sim_time=sim_time
            )
            logger.info("Executed aggressive cut-in with NPC %d (dist: %.1fm)", target_npc.id, min_dist)
        else:
            self.trigger_alert(
                "⚠️ BEHAVIORAL UNCERTAINTY ACTIVE",
                "Reckless driving patterns detected in surrounding traffic.",
                duration=8.0,
                sim_time=sim_time
            )

    def _execute_sudden_brake(self, sim_time: float):
        """Cause an NPC ahead of the ego vehicle to brake for 3.5 seconds, then recover."""
        ego_tf = self.ego.get_transform()
        ego_fwd = ego_tf.get_forward_vector()
        ego_loc = ego_tf.location

        # Find an NPC directly in front of ego (positive dot product)
        lead_npc = None
        min_dist = float('inf')
        for npc in self.npcs:
            if not npc.is_alive:
                continue
            vec = npc.get_location() - ego_loc
            dist = math.sqrt(vec.x**2 + vec.y**2)
            if 6.0 < dist < 35.0:
                dot = (vec.x * ego_fwd.x + vec.y * ego_fwd.y) / max(0.001, dist)
                if dot > 0.6 and dist < min_dist:  # directly ahead
                    min_dist = dist
                    lead_npc = npc

        if lead_npc is not None:
            # Apply temporary hard braking (3.5 seconds) then auto-recover
            lead_npc.apply_control(carla.VehicleControl(brake=0.85))
            self.braked_npcs.append((lead_npc, sim_time + 3.5))
            self.trigger_alert(
                "⚠️ LEAD VEHICLE EMERGENCY BRAKING!",
                f"Vehicle ahead (dist: {min_dist:.1f}m) emergency decelerating!",
                duration=7.0,
                sim_time=sim_time
            )
            logger.info("Lead vehicle %d emergency braked for 3.5s at distance %.1fm", lead_npc.id, min_dist)
        else:
            active_npcs = [n for n in self.npcs if n.is_alive]
            if active_npcs:
                chosen = random.choice(active_npcs)
                chosen.apply_control(carla.VehicleControl(brake=0.85))
                self.braked_npcs.append((chosen, sim_time + 3.5))
                self.trigger_alert(
                    "⚠️ SUDDEN BRAKING IN TRAFFIC",
                    "A nearby vehicle performed an unexpected emergency stop.",
                    duration=6.0,
                    sim_time=sim_time
                )

    def _execute_breakdown_ahead(self, sim_time: float):
        """Spawn a broken-down vehicle with hazard blinkers on the road shoulder ahead."""
        ego_loc = self.ego.get_location()
        carla_map = self.world.get_map()
        ego_wp = carla_map.get_waypoint(ego_loc)

        # Clear any prior hazard so roads don't accumulate permanent blockages
        self.cleanup()

        # Trace 45-55 meters ahead along road
        next_wps = ego_wp.next(48.0)
        spawn_tf = None
        if next_wps:
            target_wp = next_wps[0]
            # Offset to shoulder / right side so vehicles can navigate around it smoothly
            right_vec = target_wp.transform.get_right_vector()
            spawn_tf = carla.Transform(
                target_wp.transform.location + carla.Location(x=right_vec.x * 1.6, y=right_vec.y * 1.6, z=0.3),
                target_wp.transform.rotation
            )

        if spawn_tf is not None:
            bp_lib = self.world.get_blueprint_library()
            sedans = [bp for bp in bp_lib.filter('vehicle.*') if int(bp.get_attribute('number_of_wheels')) >= 4]
            bp = random.choice(sedans)
            if bp.has_attribute('color'):
                bp.set_attribute('color', '200,40,40')

            stalled_veh = self.world.try_spawn_actor(bp, spawn_tf)
            if stalled_veh is not None:
                # Apply handbrake
                stalled_veh.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
                # Activate hazard blinkers (CARLA 0.9.9 compatible)
                try:
                    vls = carla.VehicleLightState
                    stalled_veh.set_light_state(
                        carla.VehicleLightState(vls.Position | vls.LowBeam | vls.BlinkerLeft | vls.BlinkerRight)
                    )
                except Exception:
                    pass

                self.hazard_actors.append((stalled_veh, sim_time + 22.0))
                self.trigger_alert(
                    "⚠️ ROAD HAZARD: DISABLED VEHICLE AHEAD",
                    "Stationary broken-down car on lane shoulder with hazard blinkers!",
                    duration=16.0,
                    sim_time=sim_time
                )
                logger.info("Spawned breakdown vehicle %d ahead at shoulder (clears in 22s)", stalled_veh.id)
                return

        # Fallback alert
        self.trigger_alert(
            "⚠️ OBSTACLE HAZARD IN TRAVEL PATH",
            "Stationary blockage reported in forward driving corridor.",
            duration=10.0,
            sim_time=sim_time
        )

    def _execute_crossing_pedestrian(self, sim_time: float):
        """Direct a single pedestrian to walk across the street ahead of the ego vehicle."""
        ego_loc = self.ego.get_location()

        # Find closest walker controller and assign cross-street destination
        closest_con = None
        min_dist = float('inf')
        for con in self.controllers:
            if not con.is_alive:
                continue
            dist = ego_loc.distance(con.get_location())
            if 10.0 < dist < 45.0 and dist < min_dist:
                min_dist = dist
                closest_con = con

        if closest_con is not None:
            cross_target = self.world.get_random_location_from_navigation()
            if cross_target:
                closest_con.go_to_location(cross_target)
                closest_con.set_max_speed(1.8)
            self.trigger_alert(
                "⚠️ VULNERABLE ROAD USER: JAYWALKER",
                f"Pedestrian crossing roadway ahead (dist: {min_dist:.1f}m)! Yield immediately.",
                duration=10.0,
                sim_time=sim_time
            )
            logger.info("Directed walker controller %d to cross road", closest_con.id)
        else:
            self.trigger_alert(
                "⚠️ PEDESTRIAN CROSSING ALERT",
                "Pedestrians detected crossing roadway at upcoming junction.",
                duration=8.0,
                sim_time=sim_time
            )

    def _execute_cyclical_uncertainty(self, sim_time: float):
        """Cycle seamlessly through randomized real-world driving uncertainties."""
        events = [
            "cut_in",
            "sudden_brake",
            "weather_shift",
            "friction_shift",
            "jaywalker",
            "traffic_jam"
        ]
        choice = events[self.phase5_cycle_index % len(events)]
        self.phase5_cycle_index += 1

        if choice == "cut_in":
            self._execute_cut_in(sim_time)
        elif choice == "sudden_brake":
            self._execute_sudden_brake(sim_time)
        elif choice == "weather_shift":
            presets = [WeatherPreset.HEAVY_FOG, WeatherPreset.RAIN_FOG, WeatherPreset.CLOUDY, WeatherPreset.LIGHT_RAIN]
            preset = random.choice(presets)
            self.weather_mgr.set_weather(self.world, preset)
            self.trigger_alert(
                f"⚠️ WEATHER SHIFT: {preset.name.replace('_', ' ')}",
                "Atmospheric conditions shifting dynamically.",
                duration=14.0,
                sim_time=sim_time
            )
        elif choice == "friction_shift":
            new_mu = random.choice([0.40, 0.55, 0.85])
            self.friction_mgr.set_friction(self.ego, new_mu)
            cond_str = "DRY (μ=0.85)" if new_mu > 0.7 else ("WET (μ=0.55)" if new_mu > 0.45 else "SLIPPERY (μ=0.40)")
            self.trigger_alert(
                f"⚠️ SURFACE GRIP CHANGE: {cond_str}",
                f"Road-tire adhesion updated to mu = {new_mu:.2f}.",
                duration=12.0,
                sim_time=sim_time
            )
        elif choice == "jaywalker":
            self._execute_crossing_pedestrian(sim_time)
        elif choice == "traffic_jam":
            self.trigger_alert(
                "⚠️ TRAFFIC DENSITY SURGE",
                "Surrounding vehicles decelerating due to localized congestion.",
                duration=12.0,
                sim_time=sim_time
            )
            # Temporarily slow random group of NPCs
            for npc in random.sample(self.npcs, min(8, len(self.npcs))):
                if npc.is_alive:
                    set_tm_speed_diff(self.tm, npc, 40.0)

    def cleanup(self):
        """Clean up all dynamically spawned hazard obstacles."""
        for actor, _ in self.hazard_actors:
            if actor and actor.is_alive:
                try:
                    actor.destroy()
                except Exception:
                    pass
        self.hazard_actors.clear()
        if hasattr(self, 'ghost_driver') and self.ghost_driver:
            try:
                self.ghost_driver.cleanup()
            except Exception:
                pass


# =============================================================================
# SPAWNING UTILITIES
# =============================================================================

def cleanup_existing_actors(world: carla.World):
    """Purge any orphaned actors from previous interrupted or crashed sessions."""
    actor_list = world.get_actors()
    destroyed_count = 0
    # 1. Sensors
    for sensor in actor_list.filter('sensor.*'):
        try:
            sensor.stop()
            sensor.destroy()
            destroyed_count += 1
        except Exception:
            pass
    # 2. Controllers
    for ctrl in actor_list.filter('controller.ai.walker'):
        try:
            ctrl.stop()
            ctrl.destroy()
            destroyed_count += 1
        except Exception:
            pass
    # 3. Walkers
    for walker in actor_list.filter('walker.pedestrian.*'):
        try:
            walker.destroy()
            destroyed_count += 1
        except Exception:
            pass
    # 4. Vehicles
    for vehicle in actor_list.filter('vehicle.*'):
        try:
            vehicle.destroy()
            destroyed_count += 1
        except Exception:
            pass
    if destroyed_count > 0:
        logger.info("Cleaned up %d orphaned actors from previous sessions.", destroyed_count)
        try:
            world.wait_for_tick(seconds=2.0)
        except Exception:
            pass


def spawn_ego_vehicle(world: carla.World, spawn_point: carla.Transform) -> Optional[carla.Vehicle]:
    """Spawn the player (hero) vehicle."""
    bp_lib = world.get_blueprint_library()
    candidates = ['vehicle.tesla.model3', 'vehicle.audi.a2', 'vehicle.lincoln.mkz2017', 'vehicle.ford.mustang']
    ego_bp = None
    for cand in candidates:
        matches = list(bp_lib.filter(cand))
        if matches:
            ego_bp = matches[0]
            break

    if ego_bp is None:
        sedans = [bp for bp in bp_lib.filter('vehicle.*') if int(bp.get_attribute('number_of_wheels')) >= 4]
        ego_bp = sedans[0] if sedans else bp_lib.filter('vehicle.*')[0]

    if ego_bp.has_attribute('color'):
        ego_bp.set_attribute('color', '230,25,25')  # Vibrant Red
    ego_bp.set_attribute('role_name', 'hero')

    return world.try_spawn_actor(ego_bp, spawn_point)


def spawn_platoon(
    world: carla.World,
    spawn_point: carla.Transform,
    count: int = 4,
    tm: Any = None,
    tm_port: int = TM_PORT
) -> List[carla.Vehicle]:
    """
    Spawn a cohesive 4-vehicle autonomous platoon.
    Vehicle 0: Platoon Leader (Hero Ego Vehicle, Red)
    Vehicles 1-3: Platoon Followers (Metallic Blue, 7.5m inter-vehicle spacing)
    """
    bp_lib = world.get_blueprint_library()
    candidates = ['vehicle.tesla.model3', 'vehicle.audi.a2', 'vehicle.lincoln.mkz2017', 'vehicle.ford.mustang']
    ego_bp = None
    for cand in candidates:
        matches = list(bp_lib.filter(cand))
        if matches:
            ego_bp = matches[0]
            break
    if ego_bp is None:
        sedans = [bp for bp in bp_lib.filter('vehicle.*') if int(bp.get_attribute('number_of_wheels')) >= 4]
        ego_bp = sedans[0] if sedans else bp_lib.filter('vehicle.*')[0]

    if ego_bp.has_attribute('color'):
        ego_bp.set_attribute('color', '230,25,25')  # Vibrant Red
    ego_bp.set_attribute('role_name', 'hero')

    leader = world.try_spawn_actor(ego_bp, spawn_point)
    if leader is None:
        return []

    platoon: List[carla.Vehicle] = [leader]

    carla_map = world.get_map()
    curr_wp = carla_map.get_waypoint(spawn_point.location)

    follower_bps = [
        bp for bp in bp_lib.filter('vehicle.*')
        if int(bp.get_attribute('number_of_wheels')) == 4
        and not bp.id.endswith('isetta')
        and not bp.id.endswith('carlacola')
    ]
    if not follower_bps:
        follower_bps = [ego_bp]

    follower_colors = ['35,95,225', '45,135,245', '30,75,195']  # Distinctive Platoon Blue
    for i in range(1, count):
        prev_wps = curr_wp.previous(7.5)
        if not prev_wps:
            break
        curr_wp = prev_wps[0]
        f_loc = curr_wp.transform.location + carla.Location(z=0.35)
        f_tf = carla.Transform(f_loc, curr_wp.transform.rotation)

        f_bp = random.choice(follower_bps)
        if f_bp.has_attribute('color'):
            f_bp.set_attribute('color', follower_colors[(i - 1) % len(follower_colors)])
        f_bp.set_attribute('role_name', f'platoon_follower_{i}')

        follower = world.try_spawn_actor(f_bp, f_tf)
        if follower is not None:
            follower.set_autopilot(True, tm_port)
            if tm is not None:
                try:
                    tm.auto_lane_change(follower, False)  # Lock in platoon lane
                    tm.distance_to_leading_vehicle(follower, 4.5)  # Tight 4.5m platoon headway
                    tm.ignore_lights_percentage(follower, 0.0)
                    tm.ignore_vehicles_percentage(follower, 0.0)
                    set_tm_speed_diff(tm, follower, -5.0)  # Follow closely
                except Exception:
                    pass
            platoon.append(follower)
            logger.info("Platoon Follower #%d spawned (ID: %d)", i, follower.id)

    logger.info("Platoon successfully formed with %d vehicles!", len(platoon))
    return platoon


def spawn_npc_traffic(
    client: carla.Client,
    world: carla.World,
    spawn_points: List[carla.Transform],
    count: int = NUM_VEHICLES,
    tm_port: int = TM_PORT
) -> List[carla.Vehicle]:
    """Spawn and initialize NPC vehicles connected to Traffic Manager."""
    logger.info("Spawning %d NPC vehicles via batch command API...", count)
    bp_lib = world.get_blueprint_library()
    # Safe 4-wheel vehicles only - no bikes, motorbikes that cause gridlock
    car_bps = [
        bp for bp in bp_lib.filter('vehicle.*')
        if int(bp.get_attribute('number_of_wheels')) == 4
        and not bp.id.endswith('isetta')
        and not bp.id.endswith('carlacola')
        and not bp.id.endswith('cybertruck')
        and not bp.id.endswith('t2')
    ]
    if not car_bps:
        car_bps = [bp for bp in bp_lib.filter('vehicle.*') if int(bp.get_attribute('number_of_wheels')) == 4]

    tm = client.get_trafficmanager(tm_port)
    shuffled = list(spawn_points)
    random.shuffle(shuffled)

    # Build batch using official CARLA command API: SpawnActor.then(SetAutopilot)
    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor

    batch = []
    for sp in shuffled[:min(len(shuffled), count * 3)]:  # Try 3x to cover collision failures
        if len(batch) >= count:
            break
        bp = random.choice(car_bps)
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        if bp.has_attribute('driver_id'):
            driver_id = random.choice(bp.get_attribute('driver_id').recommended_values)
            bp.set_attribute('driver_id', driver_id)
        bp.set_attribute('role_name', 'autopilot')
        batch.append(SpawnActor(bp, sp).then(SetAutopilot(FutureActor, True, tm_port)))

    npc_vehicles: List[carla.Vehicle] = []
    responses = client.apply_batch_sync(batch, False)
    for response in responses:
        if response.error:
            logger.debug("Spawn skipped (collision): %s", response.error)
        else:
            actor = world.get_actor(response.actor_id)
            if actor:
                npc_vehicles.append(actor)

    # Wait one server tick so Traffic Manager autopilot fully engages
    try:
        world.wait_for_tick(seconds=2.0)
    except Exception:
        pass

    # Configure TM per-vehicle for smooth flowing traffic
    for npc in npc_vehicles:
        try:
            tm.auto_lane_change(npc, True)
            tm.distance_to_leading_vehicle(npc, 3.0)
            tm.ignore_lights_percentage(npc, 0.0)
            tm.ignore_signs_percentage(npc, 0.0)
            tm.ignore_vehicles_percentage(npc, 0.0)
            tm.ignore_walkers_percentage(npc, 0.0)
            set_tm_speed_diff(tm, npc, 0.0)
        except Exception:
            pass

    logger.info("Successfully spawned %d NPC vehicles (requested %d).", len(npc_vehicles), count)
    return npc_vehicles


def spawn_pedestrians(
    world: carla.World,
    count: int = NUM_PEDESTRIANS
) -> Tuple[List[carla.Actor], List[carla.Actor]]:
    """Spawn walking pedestrians and AI walker controllers."""
    if count <= 0:
        logger.info("Pedestrians disabled (NUM_PEDESTRIANS = 0).")
        try:
            world.set_pedestrians_cross_factor(0.0)
        except Exception:
            pass
        return [], []

    logger.info("Spawning %d pedestrians with navigation AI...", count)
    bp_lib = world.get_blueprint_library()
    walker_bps = list(bp_lib.filter('walker.pedestrian.*'))
    ctrl_bp = bp_lib.find('controller.ai.walker')

    walkers: List[carla.Actor] = []
    controllers: List[carla.Actor] = []

    # Keep pedestrians strictly on sidewalks during nominal flow
    world.set_pedestrians_cross_factor(0.0)

    for _ in range(count * 3):
        if len(walkers) >= count:
            break
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        sp = carla.Transform(loc)
        bp = random.choice(walker_bps)
        if bp.has_attribute('is_invincible'):
            bp.set_attribute('is_invincible', 'false')
        w = world.try_spawn_actor(bp, sp)
        if w is None:
            continue
        walkers.append(w)

    is_sync = world.get_settings().synchronous_mode
    if is_sync and walkers:
        try:
            world.tick()
        except Exception:
            pass

    for w in walkers:
        try:
            c = world.try_spawn_actor(ctrl_bp, carla.Transform(), w)
            if c is not None:
                controllers.append(c)
        except Exception:
            pass

    if is_sync and controllers:
        try:
            world.tick()
        except Exception:
            pass

    for c in controllers:
        try:
            c.start()
            dest = world.get_random_location_from_navigation()
            if dest is not None:
                c.go_to_location(dest)
            speed = 1.2 + random.random() * 0.6
            c.set_max_speed(speed)
        except Exception:
            pass

    logger.info("Successfully spawned %d pedestrians with %d controllers.", len(walkers), len(controllers))
    return walkers, controllers


# =============================================================================
# KEYBOARD DRIVER INPUT HANDLER
# =============================================================================

class KeyboardDriver:
    """Handles smooth keyboard driving input (WASD / Arrow Keys) with auto-override."""

    def __init__(self):
        self.control = carla.VehicleControl()
        self.steer_cache = 0.0

    def parse_input(
        self,
        keys,
        milliseconds: int,
        autopilot_active: bool
    ) -> Tuple[carla.VehicleControl, bool]:
        """
        Process keyboard events and return updated VehicleControl.
        Returns: (VehicleControl, user_initiated_manual_action)
        """
        user_acting = False

        # Throttle / Reverse / Brake
        throttle_delta = 2.0 * (milliseconds / 1000.0)
        steer_delta = 3.5 * (milliseconds / 1000.0)

        # Forward
        if keys[K_UP] or keys[K_w]:
            user_acting = True
            self.control.throttle = min(1.0, self.control.throttle + throttle_delta)
            self.control.reverse = False
            self.control.brake = 0.0
        else:
            self.control.throttle = max(0.0, self.control.throttle - 2.5 * throttle_delta)

        # Brake / Reverse
        if keys[K_DOWN] or keys[K_s]:
            user_acting = True
            if self.control.reverse:
                # Accelerating backwards
                self.control.throttle = min(1.0, self.control.throttle + throttle_delta)
                self.control.brake = 0.0
            else:
                # Braking
                self.control.brake = min(1.0, self.control.brake + 3.0 * throttle_delta)
                if self.control.brake >= 0.95 and self.control.throttle == 0:
                    self.control.reverse = True  # Shift into reverse when stopped
        else:
            self.control.brake = max(0.0, self.control.brake - 3.0 * throttle_delta)

        # Steering
        if keys[K_LEFT] or keys[K_a]:
            user_acting = True
            if self.steer_cache > 0:
                self.steer_cache = 0
            else:
                self.steer_cache -= steer_delta
        elif keys[K_RIGHT] or keys[K_d]:
            user_acting = True
            if self.steer_cache < 0:
                self.steer_cache = 0
            else:
                self.steer_cache += steer_delta
        else:
            # Auto-center steering
            if self.steer_cache > 0:
                self.steer_cache = max(0.0, self.steer_cache - 2.5 * steer_delta)
            elif self.steer_cache < 0:
                self.steer_cache = min(0.0, self.steer_cache + 2.5 * steer_delta)

        self.steer_cache = min(0.7, max(-0.7, self.steer_cache))
        self.control.steer = round(self.steer_cache, 2)

        # Handbrake
        self.control.hand_brake = bool(keys[K_SPACE])
        if self.control.hand_brake:
            user_acting = True

        return self.control, user_acting


# =============================================================================
# MAIN SIMULATION RUNNER
# =============================================================================

def run_simulation(
    host: str = '127.0.0.1',
    port: int = 2000,
    town: str = 'auto',
    num_vehicles: int = NUM_VEHICLES,
    num_pedestrians: int = NUM_PEDESTRIANS
):
    """Initializes and runs the full Living Real-World Uncertainty Simulation."""

    logger.info("Initializing Living Real-World Uncertainty Simulation on %s:%d...", host, port)

    # -------------------------------------------------------------------------
    # PRE-FLIGHT: Wait for CARLA server before opening any window
    # -------------------------------------------------------------------------
    import socket as _socket
    logger.info("Waiting for CARLA server on %s:%d ...", host, port)
    for _attempt in range(60):
        _s = _socket.socket()
        _s.settimeout(1.0)
        _res = _s.connect_ex((host, port))
        _s.close()
        if _res == 0:
            logger.info("CARLA server is UP! Connecting...")
            break
        if _attempt % 5 == 0:
            logger.info("  ...CARLA not ready yet (%ds elapsed). Is CarlaUE4.exe running?", _attempt)
        time.sleep(1.0)
    else:
        logger.error("CARLA server did not respond on %s:%d after 60s. Please launch CarlaUE4.exe first!", host, port)
        return

    # 1. Initialize CARLA Client
    client = carla.Client(host, port)
    client.set_timeout(30.0)

    # Clean up tracking references for safe teardown
    ego_vehicle: Optional[carla.Vehicle] = None
    camera_manager: Optional[CameraManager] = None
    collision_sensor: Optional[CollisionSensor] = None
    npc_vehicles: List[carla.Vehicle] = []
    walkers: List[carla.Actor] = []
    controllers: List[carla.Actor] = []
    uncertainty_manager: Optional[UncertaintyManager] = None
    timeline_director: Optional[UncertaintyTimelineDirector] = None

    # Pygame display setup - AFTER CARLA is confirmed running
    pygame.init()
    pygame.font.init()
    pygame.display.set_caption("CARLA 0.9.9 - Living Real-World Uncertainty Simulation")
    display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
    clock = pygame.time.Clock()

    try:
        world = client.get_world()
        current_map = world.get_map().name.split('/')[-1]
        logger.info("Connected to CARLA simulator! Active Map: %s", current_map)

        # Dynamic Map Handling: If specific town requested and differs from current, load it
        if town and town.lower() not in ('auto', 'current', 'none', '') and town not in current_map:
            try:
                logger.info("Switching map from %s to %s (allowing up to 60s for asset streaming)...", current_map, town)
                client.set_timeout(60.0)
                world = client.load_world(town)
                current_map = town
                client.set_timeout(30.0)
                logger.info("Successfully loaded %s!", town)
            except Exception as e:
                logger.warning("Could not switch to %s (%s). Continuing on %s.", town, e, current_map)

        # 1. Clean up any leftover actors from prior crashed/interrupted sessions
        cleanup_existing_actors(world)

        # Configure Traffic Manager
        traffic_manager = client.get_trafficmanager(TM_PORT)
        traffic_manager.set_global_distance_to_leading_vehicle(3.5)
        traffic_manager.global_percentage_speed_difference(0.0)

        # Get spawn points
        spawn_points = world.get_map().get_spawn_points()
        if len(spawn_points) < 5:
            raise RuntimeError(f"Insufficient spawn points in map: {len(spawn_points)}")

        # 2. Spawn 4-Vehicle Platoon (Leader Hero Ego + 3 Followers)
        platoon: List[carla.Vehicle] = []
        for sp in spawn_points:
            platoon = spawn_platoon(world, sp, count=4, tm=traffic_manager, tm_port=TM_PORT)
            if len(platoon) >= 4:
                break
            for p in platoon:
                try:
                    p.destroy()
                except Exception:
                    pass
            platoon = []

        if not platoon:
            ego_vehicle = spawn_ego_vehicle(world, spawn_points[0])
            if ego_vehicle is None:
                raise RuntimeError("Failed to spawn Ego Vehicle!")
            platoon = [ego_vehicle]
        else:
            ego_vehicle = platoon[0]

        logger.info("Platoon operational: 1 Leader (Ego #%d) + %d Followers.", ego_vehicle.id, len(platoon) - 1)

        # 3. Attach Sensors
        camera_manager = CameraManager(ego_vehicle, WINDOW_WIDTH, WINDOW_HEIGHT)
        collision_sensor = CollisionSensor(ego_vehicle)

        # 4. Spawn Living City (NPC Vehicles + Walkers)
        npc_vehicles = spawn_npc_traffic(client, world, spawn_points[1:], count=num_vehicles, tm_port=TM_PORT)
        walkers, controllers = spawn_pedestrians(world, count=num_pedestrians)

        # Warm up Traffic Manager so all vehicles begin moving smoothly
        logger.info("Warming up Traffic Manager (4 ticks)...")
        for _ in range(4):
            try:
                world.wait_for_tick(seconds=2.0)
            except Exception:
                pass

        # 5. Initialize Uncertainty Framework
        uncertainty_manager = UncertaintyManager(seed=42)
        # Explicitly disable background rogue jam/accident/event generators!
        uncertainty_manager.enable_module("traffic_jam", False)
        uncertainty_manager.enable_module("accident", False)
        uncertainty_manager.enable_module("events", False)
        uncertainty_manager.setup(
            world=world,
            ego_vehicle=ego_vehicle,
            npc_vehicles=npc_vehicles,
            traffic_manager=traffic_manager
        )

        # Assign diverse human driver behavioral profiles to NPCs
        assigned_profiles = uncertainty_manager.behavior.assign_random_profiles(
            npc_vehicles,
            traffic_manager,
            weights={
                DriverProfile.NORMAL: 0.45,
                DriverProfile.AGGRESSIVE: 0.15,
                DriverProfile.CONSERVATIVE: 0.15,
                DriverProfile.DISTRACTED: 0.15,
                DriverProfile.ERRATIC: 0.10,
            }
        )
        logger.info("Assigned human driver profiles: %d vehicles configured.", len(assigned_profiles))

        # 6. Initialize Timeline Director & HUD
        timeline_director = UncertaintyTimelineDirector(
            world=world,
            ego_vehicle=ego_vehicle,
            npc_vehicles=npc_vehicles,
            walkers=walkers,
            walker_controllers=controllers,
            traffic_manager=traffic_manager,
            uncertainty_manager=uncertainty_manager,
            platoon_vehicles=platoon
        )

        hud = SimulationHUD(WINDOW_WIDTH, WINDOW_HEIGHT)
        keyboard_driver = KeyboardDriver()

        # Autopilot active by default on Ego vehicle
        autopilot = True
        ego_vehicle.set_autopilot(True, TM_PORT)
        logger.info("AI Autopilot engaged on Ego vehicle via Traffic Manager.")

        # 7. Main Simulation Loop
        sim_start_time = time.time()
        last_tick_time = time.time()
        running = True

        logger.info("Simulation running! Press 'P' for manual driving, 'ESC' to quit.")

        while running:
            dt = clock.tick(TARGET_FPS)
            current_wall_time = time.time()
            elapsed_sim = current_wall_time - sim_start_time
            delta_sec = current_wall_time - last_tick_time
            last_tick_time = current_wall_time

            # -----------------------------------------------------------------
            # Pygame Event Processing
            # -----------------------------------------------------------------
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE or event.key == K_q:
                        running = False
                    elif event.key == K_p:
                        # Toggle Autopilot
                        autopilot = not autopilot
                        ego_vehicle.set_autopilot(autopilot, TM_PORT)
                        logger.info("Autopilot toggled: %s", "ON" if autopilot else "OFF")
                    elif event.key == K_c:
                        # Toggle Camera View
                        cam_mode_name = camera_manager.toggle_view()
                        logger.info("Camera view switched to: %s", cam_mode_name)

            # -----------------------------------------------------------------
            # Driving Controls Handling
            # -----------------------------------------------------------------
            keys = pygame.key.get_pressed()
            veh_control, user_acted = keyboard_driver.parse_input(keys, dt, autopilot)

            if user_acted and autopilot:
                # Auto-override: Take manual control immediately on user input
                autopilot = False
                ego_vehicle.set_autopilot(False)
                logger.info("Manual user override detected. Autopilot disabled.")

            if not autopilot:
                # Apply manual control with uncertainty (actuator noise/delay)
                uncertainty_manager.apply_control_with_uncertainty(ego_vehicle, veh_control)
                applied_control = veh_control
            else:
                applied_control = ego_vehicle.get_control()

            # -----------------------------------------------------------------
            # Uncertainty Framework & Timeline Progression
            # -----------------------------------------------------------------
            uncertainty_manager.tick(delta_seconds=delta_sec, current_time=elapsed_sim)
            timeline_director.tick(elapsed_sim_time=elapsed_sim, delta_sec=delta_sec)

            # Query live state
            weather_data = uncertainty_manager.weather.get_weather_info(world)
            current_mu = uncertainty_manager.friction.get_friction(ego_vehicle) or 0.85
            ghost_telemetry = timeline_director.ghost_driver.get_status(elapsed_sim)

            # -----------------------------------------------------------------
            # 3D Viewport & HUD Rendering
            # -----------------------------------------------------------------
            display.fill((10, 10, 10))
            camera_manager.render(display)

            hud.render(
                display=display,
                ego_vehicle=ego_vehicle,
                autopilot=autopilot,
                control=applied_control,
                weather_info=weather_data,
                friction_mu=current_mu,
                active_alert=timeline_director.active_alert,
                collision_sensor=collision_sensor,
                timeline_phase=timeline_director.current_phase_name,
                elapsed_sim_time=elapsed_sim,
                fps=clock.get_fps(),
                town_name=current_map,
                npc_count=len([n for n in npc_vehicles if n.is_alive]),
                ped_count=len([w for w in walkers if w.is_alive]),
                camera_name=camera_manager.views[camera_manager.view_mode][0].split()[0],
                ghost_status=ghost_telemetry
            )

            pygame.display.flip()

    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user.")
    except Exception as exc:
        logger.exception("Unexpected error in simulation loop: %s", exc)
    finally:
        # ---------------------------------------------------------------------
        # CLEANUP & TEARDOWN
        # ---------------------------------------------------------------------
        logger.info("=" * 60)
        logger.info("Performing clean simulation teardown...")

        # 1. Destroy Sensors
        if camera_manager:
            camera_manager.destroy()
        if collision_sensor:
            collision_sensor.destroy()

        # 2. Clean Timeline hazards
        if timeline_director:
            timeline_director.cleanup()

        # 3. Clean Uncertainty Manager
        if uncertainty_manager:
            try:
                uncertainty_manager.cleanup()
            except Exception:
                pass

        # 4. Destroy Pedestrians and AI Controllers
        logger.info("Destroying %d pedestrians & controllers...", len(controllers) + len(walkers))
        for con in controllers:
            if con and con.is_alive:
                try:
                    con.stop()
                    con.destroy()
                except Exception:
                    pass
        for w in walkers:
            if w and w.is_alive:
                try:
                    w.destroy()
                except Exception:
                    pass

        # 5. Destroy NPC Vehicles
        logger.info("Destroying %d NPC vehicles...", len(npc_vehicles))
        for npc in npc_vehicles:
            if npc and npc.is_alive:
                try:
                    npc.destroy()
                except Exception:
                    pass

        # 6. Destroy Ego Vehicle & Platoon Followers
        if ego_vehicle and ego_vehicle.is_alive:
            try:
                ego_vehicle.destroy()
                logger.info("Ego vehicle destroyed.")
            except Exception:
                pass
        for follower in platoon[1:]:
            if follower and follower.is_alive:
                try:
                    follower.destroy()
                except Exception:
                    pass

        # 7. Restore Clear Weather
        try:
            world.set_weather(carla.WeatherParameters.ClearNoon)
        except Exception:
            pass

        pygame.quit()
        logger.info("Cleanup complete. CARLA environment is clean and ready.")
        logger.info("=" * 60)

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="CARLA 0.9.9 - Living Real-World Uncertainty Simulation"
    )
    parser.add_argument('--host', default='127.0.0.1', help='CARLA server IP (default: 127.0.0.1)')
    parser.add_argument('-p', '--port', type=int, default=2000, help='CARLA server port (default: 2000)')
    parser.add_argument('--town', default='auto', help='Town map name (default: auto = adapt to whatever map is running, or Town01-Town07)')
    parser.add_argument('--vehicles', type=int, default=NUM_VEHICLES, help=f'Number of NPC vehicles (default: {NUM_VEHICLES})')
    parser.add_argument('--pedestrians', type=int, default=NUM_PEDESTRIANS, help=f'Number of pedestrians (default: {NUM_PEDESTRIANS})')

    args = parser.parse_args()

    run_simulation(
        host=args.host,
        port=args.port,
        town=args.town,
        num_vehicles=args.vehicles,
        num_pedestrians=args.pedestrians
    )


if __name__ == '__main__':
    main()
