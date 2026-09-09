#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified Uncertain Platoon Simulation
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import carla
import pygame
from pygame.locals import K_c, K_ESCAPE, K_p, K_q, K_TAB, KEYDOWN, QUIT

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

# Ensure SCRIPT_DIR is in sys.path
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Import from autopilot_platoon_window
from autopilot_platoon_window import (
    setup_logging, wait_for_server, maybe_launch_carla,
    configure_world_timing, restore_world_timing, advance_world,
    cleanup_existing_actors, spawn_platoon, CameraManager,
    PlatoonPathController, RoadLeaderController, MiniHUD,
    actor_speed_kmh, brake_platoon, traffic_summary, walker_summary, platoon_rows,
    WINDOW_WIDTH, WINDOW_HEIGHT, TARGET_FPS, DEFAULT_LOG_FILE, TM_PORT
)

# Import from real_world_simulation
from real_world_simulation import (
    spawn_npc_traffic, spawn_pedestrians, UncertaintyTimelineDirector
)
from uncertainty.uncertainty_manager import UncertaintyManager
from uncertainty.human_behavior import DriverProfile

if CARLA_ROOT not in sys.path:
    sys.path.insert(0, CARLA_ROOT)

from Module2.module2_manager import Module2Manager

logger = logging.getLogger("Module2PlatoonMain")

class UnifiedHUD(MiniHUD):
    """Extension of MiniHUD to include Ghost Driver status."""
    def render(
        self, display, leader, platoon, vehicles, walkers,
        autopilot, camera_name, elapsed, town_name, synchronous, fps, ghost_status=None
    ) -> None:
        super().render(display, leader, platoon, vehicles, walkers, autopilot, camera_name, elapsed, town_name, synchronous, fps)
        
        # Add ghost driver status
        if ghost_status:
            font = pygame.font.SysFont("consolas", 14, bold=True)
            active = ghost_status.get("active", False)
            bg_color = (80, 15, 60, 240) if active else (15, 30, 45, 240)
            text_color = (255, 150, 255) if active else (150, 200, 255)
            
            box_w = 430
            box_h = 70
            bx = 16
            by = self.height - 230
            
            surf = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
            surf.fill(bg_color)
            pygame.draw.rect(surf, text_color, surf.get_rect(), width=1, border_radius=6)
            display.blit(surf, (bx, by))
            
            title = font.render("GHOST DRIVER SYSTEM", True, text_color)
            display.blit(title, (bx + 12, by + 10))
            
            if active:
                target_id = ghost_status.get("target_id")
                disruption = ghost_status.get("disruption_name", "Unknown")
                rem_sec = ghost_status.get("remaining_seconds", 0.0)
                msg1 = f"Vehicle #{target_id} -> POSSESSED"
                msg2 = f"Action: {disruption} ({rem_sec:.1f}s left)"
                c1 = (255, 100, 100)
            else:
                target_id = "None"
                cooldown = ghost_status.get("cooldown_remaining", 0.0)
                msg1 = "All vehicles -> NORMAL"
                msg2 = f"Lurking... (next in {cooldown:.1f}s)"
                c1 = (100, 255, 100)

            display.blit(font.render(msg1, True, c1), (bx + 12, by + 30))
            display.blit(font.render(msg2, True, (200, 200, 200)), (bx + 12, by + 48))

def run(args: argparse.Namespace) -> None:
    setup_logging(args.log_file)
    server_process = None
    if args.launch_server and not wait_for_server(args.host, args.port, 2):
        server_process = maybe_launch_carla(args.town, args.port, args.quality)

    if not wait_for_server(args.host, args.port, args.wait_seconds):
        raise RuntimeError("CARLA server did not respond.")

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    pygame.init()
    pygame.display.set_caption("CARLA Uncertain Platoon Simulation")
    window_width = args.width if args.width else WINDOW_WIDTH
    window_height = args.height if args.height else WINDOW_HEIGHT
    
    flags = pygame.HWSURFACE | pygame.DOUBLEBUF
    if not args.windowed:
        pass # add fullscreen if we want, but windowed is standard for Pygame unless FULLSCREEN is specified
    
    display = pygame.display.set_mode((window_width, window_height), flags)
    clock = pygame.time.Clock()

    camera = None
    platoon = []
    npcs = []
    walkers = []
    controllers = []
    world = None
    tm = None
    original_settings = None
    leader_start_loc = None
    
    platoon_controller = PlatoonPathController(desired_clearance=7.5)
    leader_controller = None
    uncertainty_manager = None
    timeline_director = None

    try:
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
        leader_controller = RoadLeaderController(world.get_map(), target_speed_kmh=args.platoon_speed)
        logger.info("[PLATOON] Platoon ready.")

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
        
        # Override Ghost Driver settings as requested
        timeline_director.ghost_driver.min_possession = 15.0
        timeline_director.ghost_driver.max_possession = 20.0
        timeline_director.ghost_driver.min_cooldown = 5.0
        timeline_director.ghost_driver.max_cooldown = 20.0
        
        logger.info("[GHOST] Ghost-driver system initialized.")

        # Initialize Module 2 MAPPO System
        logger.info("[MODULE 2] Initializing MAPPO System...")
        module2_manager = Module2Manager(uncertainty_manager, len(platoon))
        logger.info("[MODULE 2] MAPPO System Initialized.")

        # Warm up controllers
        for _ in range(4):
            leader_controller.update(leader)
            platoon_controller.update(platoon)
            advance_world(world, sync=not args.async_world)

        camera_vehicle_idx = 0
        camera = CameraManager(leader, window_width, window_height)
        hud = UnifiedHUD(window_width, window_height)
        autopilot = True
        
        started = time.time()
        last_tick_time = started
        leader_start_loc = leader.get_location()
        last_motion_log = 0.0
        running = True
        
        # Weather and ghost state tracking
        weather_interval_sec = args.weather * 60.0
        last_weather_time = 0.0
        was_ghost_active = False

        logger.info("[SIMULATION] Starting unified simulation loop...")

        while running:
            current_time = time.time()
            elapsed_sim = current_time - started
            delta_sec = current_time - last_tick_time
            last_tick_time = current_time

            if args.duration > 0 and elapsed_sim >= args.duration:
                running = False
                
            # Periodic weather update
            if elapsed_sim - last_weather_time >= weather_interval_sec:
                last_weather_time = elapsed_sim
                import random
                from uncertainty.weather_uncertainty import WeatherPreset
                presets = [WeatherPreset.CLEAR, WeatherPreset.CLOUDY, WeatherPreset.LIGHT_RAIN, WeatherPreset.HEAVY_RAIN, WeatherPreset.HEAVY_FOG]
                new_weather = random.choice(presets)
                uncertainty_manager.weather.set_weather(world, new_weather)
                logger.info("[WEATHER] Interval reached. Weather dynamically shifted to %s", new_weather.name)

            clock.tick(TARGET_FPS)
            
            # Events
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key in (K_ESCAPE, K_q):
                        running = False
                    elif event.key == K_p:
                        autopilot = not autopilot
                        if not autopilot:
                            brake_platoon(platoon)
                        logger.info("Platoon waypoint autopilot: %s", "ON" if autopilot else "OFF")
                    elif event.key == K_c and camera:
                        camera.toggle()
                    elif event.key == K_TAB and camera:
                        camera_vehicle_idx = (camera_vehicle_idx + 1) % len(platoon)
                        camera.destroy()
                        camera = CameraManager(platoon[camera_vehicle_idx], window_width, window_height)
                        logger.info("Camera attached to vehicle: %s", "Leader" if camera_vehicle_idx == 0 else f"Follower {camera_vehicle_idx}")

            # 1. Update Uncertainty & Ghost
            uncertainty_manager.tick(delta_seconds=delta_sec, current_time=elapsed_sim)
            timeline_director.tick(elapsed_sim_time=elapsed_sim, delta_sec=delta_sec)
            ghost_telemetry = timeline_director.ghost_driver.get_status(elapsed_sim)
            
            is_ghost_active = ghost_telemetry.get("active", False)
            if is_ghost_active and not was_ghost_active:
                logger.info("[GHOST] Vehicle %s -> POSSESSED", ghost_telemetry.get("target_id"))
            elif not is_ghost_active and was_ghost_active:
                logger.info("[GHOST] Vehicle Released -> RELEASED")
            was_ghost_active = is_ghost_active

            # 2. Update Platoon Behavior
            if autopilot:
                # MAPPO Module 2 Hook
                tactical_actions, actions_dict = module2_manager.update(platoon, elapsed_sim)
                module2_manager.apply_actions(tactical_actions, leader_controller, platoon_controller)
                module2_manager.record_transition(platoon)
                module2_manager.train_if_ready()

                leader_controller.update(leader)
                platoon_controller.update(platoon)
            else:
                brake_platoon(platoon)
                
            # 3. Step Physics (Unified Tick)
            advance_world(world, sync=not args.async_world)

            # 4. Logging
            if elapsed_sim - last_motion_log >= 2.0:
                last_motion_log = elapsed_sim
                logger.info("\n[TICK %d]", world.get_snapshot().frame)
                for row in platoon_rows(platoon):
                    logger.info("Vehicle %d | Platoon: YES | Ghost: NO", row["id"])
                
                # Check for ghost possessed vehicle
                if is_ghost_active:
                    tid = ghost_telemetry.get("target_id")
                    logger.info("Vehicle %s | Platoon: NO | Ghost: YES", tid)
                
            # Draw visual red cuboid around the ghost vehicle
            if is_ghost_active:
                tid = ghost_telemetry.get("target_id")
                ghost_actor = world.get_actor(tid)
                if ghost_actor and ghost_actor.is_alive:
                    t = ghost_actor.get_transform()
                    b = ghost_actor.bounding_box
                    # Add a slight padding to the bounding box so it's clearly visible around the car
                    extent = carla.Vector3D(b.extent.x + 0.15, b.extent.y + 0.15, b.extent.z + 0.15)
                    box = carla.BoundingBox(t.location + b.location, extent)
                    # Draw a red box that lasts for just this tick (0.05 seconds)
                    world.debug.draw_box(box, t.rotation, thickness=0.15, color=carla.Color(255, 0, 0), life_time=0.1)
                
            # 5. Visualization
            display.fill((8, 8, 10))
            if camera:
                camera.render(display)
            
            hud.render(
                display, leader, platoon, npcs, walkers,
                autopilot, camera.name if camera else "None",
                elapsed_sim, current_town, not args.async_world,
                clock.get_fps(), ghost_status=ghost_telemetry
            )
            pygame.display.flip()

    finally:
        logger.info("Cleaning up...")
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
        restore_world_timing(world, tm, original_settings)
        pygame.quit()
        if server_process and args.kill_launched_server:
            server_process.terminate()
        logger.info("Done.")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Uncertain Platoon Simulation")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--town", default="Town01", help="Town01, Town02, ... or auto")
    parser.add_argument("--vehicles", type=int, default=32)
    parser.add_argument("--pedestrians", type=int, default=20)
    parser.add_argument("--platoon-size", type=int, default=4)
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
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.set_defaults(clean=True)
    return parser.parse_args()

if __name__ == "__main__":
    run(parse_args())

