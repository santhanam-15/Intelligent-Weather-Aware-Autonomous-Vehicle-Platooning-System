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
import math

import carla
import pygame
from pygame.locals import K_c, K_ESCAPE, K_p, K_q, KEYDOWN, QUIT

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

# Import Module 2
import sys
if os.path.join(CARLA_ROOT, 'module_2') not in sys.path:
    sys.path.append(CARLA_ROOT)

from module_2.mappo_agent import WeatherAwareMAPPO
from module_2.training_config import TrainingConfig
from module_2.observation_builder import ObservationBuilder
from module_2.reward import RewardCalculator
from module_2.action_space import TacticalAction

logger = logging.getLogger("UncertainPlatoonMain")

class UnifiedHUD(MiniHUD):
    """Extension of MiniHUD to include Ghost Driver status and DRL Dashboard."""
    def render(
        self,
        display: pygame.Surface,
        leader: carla.Vehicle,
        platoon: List[carla.Vehicle],
        vehicles: List[carla.Actor],
        walkers: List[carla.Actor],
        autopilot: bool,
        camera_name: str,
        elapsed: float,
        town_name: str,
        synchronous: bool,
        fps: float,
        ghost_status: dict = None
    ) -> None:
        if ghost_status is None:
            ghost_status = {}
            
        pt_cache = ghost_status.get("platoon_telemetry", {})

        # 1. Top FPS Banner (Super lightweight, replaces heavy super().render)
        mins = int(elapsed) // 60
        secs = int(elapsed) % 60
        top = pygame.Surface((self.width, 38), pygame.SRCALPHA)
        top.fill((10, 14, 22, 215))
        display.blit(top, (0, 0))
        display.blit(self.title.render("CARLA AUTOPILOT PLATOON SYSTEM", True, (245, 248, 255)), (16, 9))
        mode = "SYNCHRONOUS 30 FPS" if synchronous else "ASYNC WORLD"
        right_text = self.font.render(f"{town_name} | {mins:02d}:{secs:02d} | {fps:02.0f} FPS | {mode}", True, (180, 215, 245))
        display.blit(right_text, (self.width - right_text.get_width() - 16, 10))

        # Banner Top Center
        banner_w = 650
        banner_h = 42
        bx = (self.width - banner_w) // 2
        by = 48
        banner = pygame.Surface((banner_w, banner_h), pygame.SRCALPHA)
        banner.fill((8, 42, 24, 225) if autopilot else (60, 28, 10, 225))
        pygame.draw.rect(banner, (70, 235, 130) if autopilot else (245, 150, 45), banner.get_rect(), width=2, border_radius=7)
        display.blit(banner, (bx, by))
        
        leader_speed = pt_cache[leader.id]["speed_kmh"] if leader and leader.id in pt_cache else 0.0
        banner_text = f"PLATOON AUTOPILOT {'ACTIVE' if autopilot else 'PAUSED'} | {len(platoon)} VEHICLES | LEADER #{leader.id if leader else 'N/A'} | {leader_speed:4.1f} km/h"
        text = self.font.render(banner_text, True, (210, 255, 225) if autopilot else (255, 230, 200))
        display.blit(text, (bx + (banner_w - text.get_width()) // 2, by + 10))

        # 2. Ghost Driver System Box
        box_w = 400
        box_h = 70
        bx = 16
        by = self.height - 230
        
        surf_ghost = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        surf_ghost.fill((20, 25, 40, 240))
        pygame.draw.rect(surf_ghost, (100, 150, 255), surf_ghost.get_rect(), width=1, border_radius=6)
        display.blit(surf_ghost, (bx, by))
        
        font = pygame.font.SysFont("consolas", 13, bold=True)
        display.blit(font.render("GHOST DRIVER SYSTEM", True, (150, 200, 255)), (bx + 12, by + 10))
        
        active = ghost_status.get("active", False)
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

        # 3. Draw Detailed Platoon DRL Dashboard at the bottom
        if "drl_actions" in ghost_status:
            dash_h = 130
            dash_w = self.width - 32
            dash_x = 16
            dash_y = self.height - dash_h - 16
            
            surf_dash = pygame.Surface((dash_w, dash_h), pygame.SRCALPHA)
            surf_dash.fill((15, 25, 35, 240))
            pygame.draw.rect(surf_dash, (100, 200, 255), surf_dash.get_rect(), width=1, border_radius=6)
            display.blit(surf_dash, (dash_x, dash_y))
            
            winfo = ghost_status.get("weather_info", {})
            w_preset = winfo.get("preset_name", "CLEAR") if winfo else "CLEAR"
            mapping = {"CLEAR": 1.0, "CLOUDY": 0.95, "LIGHT_RAIN": 0.8, "MODERATE_RAIN": 0.7, "HEAVY_RAIN": 0.55, "FOG": 0.9, "HEAVY_FOG": 0.8, "RAIN_FOG": 0.6}
            mu = mapping.get(w_preset, 1.0)
            
            header_font = pygame.font.SysFont("consolas", 14, bold=True)
            val_font = pygame.font.SysFont("consolas", 13)
            
            header_str = f"MODULE 2 - WEATHER-AWARE DRL COOPERATIVE DECISION | WEATHER: {w_preset} | FRICTION (μ): {mu:.2f}"
            display.blit(header_font.render(header_str, True, (100, 255, 100)), (dash_x + 12, dash_y + 8))
            pygame.draw.line(display, (100, 200, 255), (dash_x, dash_y + 30), (dash_x + dash_w, dash_y + 30), 1)
            
            actions = ghost_status.get("drl_actions", {})
            action_names = ["CRUISE", "ACCELERATE", "DECELERATE", "LEFT", "RIGHT"]
            
            num_veh = len(platoon)
            if num_veh > 0:
                col_w = dash_w // num_veh
                
                for i, vehicle in enumerate(platoon):
                    col_x = dash_x + i * col_w
                    
                    if i > 0:
                        pygame.draw.line(display, (60, 80, 100), (col_x, dash_y + 30), (col_x, dash_y + dash_h), 1)
                    
                    if not vehicle or not vehicle.is_alive or vehicle.id not in pt_cache:
                        display.blit(header_font.render("MISSING", True, (255, 100, 100)), (col_x + 10, dash_y + 35))
                        continue
                        
                    role = "LEADER" if i == 0 else f"FOLLOWER {i}"
                    vid = vehicle.id
                    act_idx = actions.get(vid, 0)
                    decision = action_names[act_idx] if 0 <= act_idx < len(action_names) else "UNKNOWN"
                    
                    c_data = pt_cache[vid]
                    speed = c_data["speed_kmh"]
                    steer = c_data["steer"]
                    
                    gap_str = "N/A"
                    if i > 0 and platoon[i-1] and platoon[i-1].is_alive and platoon[i-1].id in pt_cache:
                        prev_data = pt_cache[platoon[i-1].id]
                        loc = c_data["location"]
                        prev_loc = prev_data["location"]
                        gap = loc.distance(prev_loc) - c_data["bb_x"] - prev_data["bb_x"]
                        gap_str = f"{gap:.1f}m"
                        
                    color_title = (255, 200, 50) if i == 0 else (150, 200, 255)
                    display.blit(header_font.render(f"{role} (ID:{vid})", True, color_title), (col_x + 10, dash_y + 38))
                    display.blit(val_font.render(f"SPEED   : {speed:.1f} km/h", True, (200, 200, 200)), (col_x + 10, dash_y + 60))
                    
                    color_act = (100, 255, 100) if decision == "CRUISE" else ((255, 150, 150) if decision in ["DECELERATE"] else (255, 255, 100))
                    display.blit(val_font.render(f"DECISION: {decision}", True, color_act), (col_x + 10, dash_y + 78))
                    
                    display.blit(val_font.render(f"GAP     : {gap_str}", True, (200, 200, 200)), (col_x + 10, dash_y + 96))
                    display.blit(val_font.render(f"STEERING: {steer:+.2f}", True, (200, 200, 200)), (col_x + 10, dash_y + 114))


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

        # Warm up controllers
        for _ in range(4):
            leader_controller.update(leader)
            platoon_controller.update(platoon)
            advance_world(world, sync=not args.async_world)

        camera = CameraManager(leader, window_width, window_height)
        hud = UnifiedHUD(window_width, window_height)
        autopilot = True
        
        # --- MODULE 2 INITIALIZATION ---
        logger.info("[MODULE 2] Initializing Weather-Aware MAPPO...")
        drl_config = TrainingConfig()
        drl_config.num_agents = len(platoon)
        obs_builder = ObservationBuilder(desired_clearance=7.5)
        reward_calc = RewardCalculator(desired_clearance=7.5)
        weather_drl = WeatherAwareMAPPO(
            obs_dim=obs_builder.obs_dim, 
            global_state_dim=19, # 5 * 3 + 4
            action_dim=5, 
            config=drl_config
        )
        drl_step_count = 0
        drl_active = getattr(args, 'drl', True) # Enable by default for test
        logger.info("[MODULE 2] Initialization complete.")
        # -------------------------------
        
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
                leader_controller.update(leader)
                platoon_controller.update(platoon)
                
                # --- CACHE TELEMETRY ONCE PER FRAME TO FIX RPC LAG ---
                platoon_telemetry = {}
                for idx, v in enumerate(platoon):
                    if v and v.is_alive:
                        t = v.get_transform()
                        vel = v.get_velocity()
                        ctrl = v.get_control()
                        bb_x = v.bounding_box.extent.x
                        platoon_telemetry[v.id] = {
                            "transform": t,
                            "location": t.location,
                            "rotation": t.rotation,
                            "velocity": vel,
                            "speed_kmh": 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2),
                            "steer": ctrl.steer,
                            "bb_x": bb_x,
                            "idx": idx
                        }
                ghost_telemetry["platoon_telemetry"] = platoon_telemetry
                # -----------------------------------------------------
                
                # --- MODULE 2 EXECUTION ---
                if drl_active:
                    weather_info = uncertainty_manager.weather.get_weather_info(world) if uncertainty_manager and hasattr(uncertainty_manager, 'weather') else None
                    
                    # 1. Get observations (passing cache to avoid RPC)
                    observations = obs_builder.build_local_observations(platoon, weather_info, cache=platoon_telemetry)
                    global_state = obs_builder.build_global_state(platoon, weather_info, cache=platoon_telemetry)
                    
                    # 2. Select actions
                    actions, log_probs = weather_drl.select_actions(observations)
                    
                    # 3. Calculate value
                    value = weather_drl.get_value(global_state)
                    
                    # 4. Calculate rewards
                    rewards = reward_calc.calculate_reward(platoon, actions, weather_info, collision_monitor=None, cache=platoon_telemetry)

                    
                    # 5. Store in buffer
                    if len(platoon) > 0:
                        avg_reward = sum(rewards.values()) / max(1, len(rewards))
                        vid = platoon[0].id if platoon[0] else 0
                        weather_drl.buffer.add(
                            observations.get(vid, [0.0]*obs_builder.obs_dim), 
                            global_state, 
                            actions.get(vid, 0), 
                            log_probs.get(vid, 0.0), 
                            avg_reward, 
                            value, 
                            False
                        )
                        
                    drl_step_count += 1
                    if drl_step_count % drl_config.rollout_length == 0:
                        metrics = weather_drl.train_step()
                        msg = f"[MODULE 2 MAPPO] Step: {drl_step_count} | actor_loss={metrics.get('actor_loss', 0):.3f} | critic_loss={metrics.get('critic_loss', 0):.3f} | entropy={metrics.get('entropy', 0):.3f}\n"
                        logger.info(msg.strip())
                        
                        # Write to dedicated DRL log file
                        drl_log_path = os.path.join(SCRIPT_DIR, "logs", "mappo_training.log")
                        with open(drl_log_path, "a") as f:
                            f.write(msg)
                            
                        # Save checkpoint optionally
                        models_dir = os.path.join(SCRIPT_DIR, "models")
                        if not os.path.exists(models_dir):
                            os.makedirs(models_dir)
                        if (drl_step_count // drl_config.rollout_length) % drl_config.model_save_interval == 0:
                            save_path = os.path.join(models_dir, "weather_mappo_latest.pt")
                            weather_drl.save_checkpoint(save_path)
                            with open(drl_log_path, "a") as f:
                                f.write(f"[SAVED] MAPPO Model Checkpoint saved to: {save_path}\n")
                        
                    # Inject debug info into telemetry for HUD
                    ghost_telemetry["drl_actions"] = actions
                    ghost_telemetry["weather_info"] = weather_info
                    ghost_telemetry["avg_reward"] = avg_reward
                # --------------------------
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

