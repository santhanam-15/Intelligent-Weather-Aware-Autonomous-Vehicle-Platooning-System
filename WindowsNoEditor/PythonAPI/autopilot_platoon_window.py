#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone CARLA Pygame demo:
  - connects to/optionally launches CARLA
  - loads a town
  - spawns a Traffic Manager autopilot platoon
  - spawns NPC vehicles and pedestrians
  - opens a Pygame camera window
  - cleans up all spawned actors on exit
"""

from __future__ import annotations

import argparse
from collections import deque
import glob
import logging
import math
import os
import random
import socket
import subprocess
import sys
import threading
import time
import weakref
from typing import Any, Deque, Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CARLA_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
CARLA_EXE = os.path.join(CARLA_ROOT, "CarlaUE4.exe")

platform_tag = "win-amd64" if os.name == "nt" else "linux-x86_64"
egg_pattern = os.path.join(
    SCRIPT_DIR,
    "carla",
    "dist",
    "carla-*%d.%d-%s.egg" % (sys.version_info.major, sys.version_info.minor, platform_tag),
)
egg_matches = glob.glob(egg_pattern)
if not egg_matches:
    any_eggs = glob.glob(os.path.join(SCRIPT_DIR, "carla", "dist", "carla-*.egg"))
    available = ", ".join(os.path.basename(path) for path in any_eggs) or "none"
    raise RuntimeError(
        "No CARLA egg matches Python %d.%d. Available eggs: %s. "
        "For CARLA 0.9.9 on this machine, run with `py -3.7`."
        % (sys.version_info.major, sys.version_info.minor, available)
    )
sys.path.insert(0, egg_matches[0])

import carla
import numpy as np
import pygame
from pygame.locals import K_c, K_ESCAPE, K_p, K_q, KEYDOWN, QUIT

if not hasattr(carla, "Client"):
    raise RuntimeError(
        "Imported a CARLA helper folder instead of the compiled CARLA client. "
        "Run from the launcher or use `py -3.7 autopilot_platoon_window.py`."
    )

from uncertainty.ghost_driver import GhostDriver


logger = logging.getLogger("AutopilotPlatoonWindow")

WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 576
TARGET_FPS = 30
TM_PORT = 8000
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
DEFAULT_LOG_FILE = os.path.join(LOG_DIR, "autopilot_platoon_window.log")


def setup_logging(log_file: str = DEFAULT_LOG_FILE) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)

    root.setLevel(logging.INFO)
    root.addHandler(console_handler)
    root.addHandler(file_handler)
    logger.info("Live log file: %s", log_file)


def set_tm_speed_diff(tm, vehicle: carla.Vehicle, diff: float) -> None:
    try:
        if hasattr(tm, "vehicle_percentage_speed_difference"):
            tm.vehicle_percentage_speed_difference(vehicle, float(diff))
        elif hasattr(tm, "set_percentage_speed_difference"):
            tm.set_percentage_speed_difference(vehicle, float(diff))
    except Exception:
        pass


def wait_for_server(host: str, port: int, seconds: int) -> bool:
    logger.info("Waiting for CARLA server at %s:%d ...", host, port)
    for attempt in range(seconds):
        sock = socket.socket()
        sock.settimeout(1.0)
        result = sock.connect_ex((host, port))
        sock.close()
        if result == 0:
            logger.info("CARLA server is ready.")
            return True
        if attempt % 5 == 0:
            logger.info("  ...still waiting (%ds)", attempt)
        time.sleep(1.0)
    return False


def configure_world_timing(world: carla.World, tm: carla.TrafficManager, sync: bool) -> carla.WorldSettings:
    """Put the simulator in a known timing mode and return original settings."""
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = bool(sync)
    settings.fixed_delta_seconds = 1.0 / TARGET_FPS if sync else None
    world.apply_settings(settings)

    try:
        tm.set_synchronous_mode(bool(sync))
    except Exception:
        pass

    logger.info("World timing: %s", "synchronous client ticks" if sync else "asynchronous server ticks")
    return original_settings


def restore_world_timing(world: Optional[carla.World], tm: Optional[carla.TrafficManager], original_settings) -> None:
    if tm is not None:
        try:
            tm.set_synchronous_mode(False)
        except Exception:
            pass
    if world is not None and original_settings is not None:
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass


def advance_world(world: carla.World, sync: bool) -> None:
    try:
        if sync:
            world.tick()
        else:
            world.wait_for_tick(seconds=2.0)
    except Exception:
        pass


def maybe_launch_carla(town: str, port: int, quality: str) -> Optional[subprocess.Popen]:
    if not os.path.exists(CARLA_EXE):
        logger.warning("Could not find CarlaUE4.exe at %s", CARLA_EXE)
        return None

    args = [
        CARLA_EXE,
        f"/Game/Carla/Maps/{town}",
        f"-carla-rpc-port={port}",
        "-windowed",
        "-ResX=1280",
        "-ResY=720",
        f"-quality-level={quality}",
    ]
    logger.info("Launching CARLA: %s", " ".join(args))
    return subprocess.Popen(args, cwd=CARLA_ROOT)


def stop_launched_carla(server_process: Optional[subprocess.Popen]) -> None:
    """Stop only the CARLA process tree started by this runner."""
    if server_process is None:
        return
    try:
        if server_process.poll() is None:
            server_process.terminate()
            server_process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        try:
            server_process.kill()
            server_process.wait(timeout=5.0)
        except Exception:
            logger.warning("CARLA did not exit after forced termination", exc_info=True)
    except Exception:
        logger.warning("Could not terminate launched CARLA process", exc_info=True)

    # On Windows the launcher can leave the Unreal child process alive after
    # the parent exits.  Restrict taskkill to this exact process tree and only
    # when the caller explicitly requested --kill-launched-server.
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(server_process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            # Unreal may detach the actual game process from the Popen
            # wrapper, so the PID-tree command alone is not sufficient.
            cleanup_result = subprocess.run(
                ["taskkill", "/IM", "CarlaUE4-Win64-Shipping.exe", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if cleanup_result.returncode != 0:
                # Some Windows CARLA builds deny taskkill but allow the
                # same-user PowerShell process stop operation.
                subprocess.run(
                    [
                        "powershell.exe",
                        "-NoProfile",
                        "-NonInteractive",
                        "-Command",
                        "Get-Process -Name 'CarlaUE4-Win64-Shipping' "
                        "-ErrorAction SilentlyContinue | Stop-Process -Force",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
        except Exception:
            logger.debug("CARLA process-tree cleanup was unavailable", exc_info=True)


def cleanup_existing_actors(world: carla.World) -> None:
    destroyed = 0
    actors = world.get_actors()
    for pattern in ("sensor.*", "controller.ai.walker", "walker.pedestrian.*", "vehicle.*"):
        for actor in actors.filter(pattern):
            try:
                if pattern == "controller.ai.walker":
                    actor.stop()
                actor.destroy()
                destroyed += 1
            except Exception:
                pass
    if destroyed:
        logger.info("Cleaned %d old actors from the world.", destroyed)
        advance_world(world, sync=world.get_settings().synchronous_mode)


def choose_vehicle_blueprints(world: carla.World) -> List[carla.ActorBlueprint]:
    bp_lib = world.get_blueprint_library()
    blueprints = []
    for bp in bp_lib.filter("vehicle.*"):
        try:
            wheels = int(bp.get_attribute("number_of_wheels"))
        except Exception:
            wheels = 4
        if wheels == 4 and not bp.id.endswith(("isetta", "carlacola", "cybertruck", "t2")):
            blueprints.append(bp)
    return blueprints or list(bp_lib.filter("vehicle.*"))


def configure_vehicle_tm(
    tm: carla.TrafficManager,
    vehicle: carla.Vehicle,
    *,
    speed_diff: float = 0.0,
    gap: float = 3.5,
    ignore_rules: bool = False,
) -> None:
    try:
        vehicle.set_autopilot(True, tm.get_port())
        tm.auto_lane_change(vehicle, False)
        tm.distance_to_leading_vehicle(vehicle, gap)
        tm.ignore_lights_percentage(vehicle, 100.0 if ignore_rules else 0.0)
        tm.ignore_signs_percentage(vehicle, 100.0 if ignore_rules else 0.0)
        tm.ignore_vehicles_percentage(vehicle, 0.0)
        tm.ignore_walkers_percentage(vehicle, 0.0)
        set_tm_speed_diff(tm, vehicle, speed_diff)
    except Exception:
        pass


def spawn_platoon(
    world: carla.World,
    tm: carla.TrafficManager,
    spawn_points: List[carla.Transform],
    count: int,
) -> List[carla.Vehicle]:
    bp_lib = world.get_blueprint_library()
    carla_map = world.get_map()
    leader_bp = bp_lib.find("vehicle.tesla.model3") if bp_lib.filter("vehicle.tesla.model3") else choose_vehicle_blueprints(world)[0]
    if leader_bp.has_attribute("color"):
        leader_bp.set_attribute("color", "230,25,25")
    leader_bp.set_attribute("role_name", "hero")

    vehicle_bps = choose_vehicle_blueprints(world)
    random.shuffle(spawn_points)

    for spawn_point in spawn_points:
        leader = world.try_spawn_actor(leader_bp, spawn_point)
        if leader is None:
            continue

        platoon = [leader]
        try:
            leader.set_autopilot(False)
            leader.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0))
        except Exception:
            pass
        logger.info(
            "Platoon leader #%d spawned: %s | waypoint autopilot controller ON",
            leader.id,
            actor_type_name(leader),
        )

        current_wp = carla_map.get_waypoint(spawn_point.location)
        for index in range(1, count):
            previous = [
                wp for wp in current_wp.previous(9.0)
                if wp.road_id == current_wp.road_id and wp.lane_id == current_wp.lane_id
            ]
            if not previous:
                break
            current_wp = previous[0]

            bp = random.choice(vehicle_bps)
            if bp.has_attribute("color"):
                color = ["35,95,225", "45,135,245", "30,75,195"][(index - 1) % 3]
                bp.set_attribute("color", color)
            bp.set_attribute("role_name", f"platoon_follower_{index}")

            loc = current_wp.transform.location + carla.Location(z=0.55)
            follower = world.try_spawn_actor(bp, carla.Transform(loc, current_wp.transform.rotation))
            if follower is None:
                break

            try:
                follower.set_autopilot(False)
                follower.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0))
            except Exception:
                pass
            platoon.append(follower)
            logger.info(
                "Platoon follower %d #%d spawned: %s | trajectory follower ON | target clearance 7.5m",
                index,
                follower.id,
                actor_type_name(follower),
            )

        if len(platoon) == count:
            logger.info("Spawned autopilot platoon: %s", [v.id for v in platoon])
            return platoon

        for vehicle in platoon:
            try:
                vehicle.destroy()
            except Exception:
                pass

        # Destruction is deferred by CARLA until the next world tick.  Advance
        # the world before retrying so failed partial groups cannot remain in
        # the actor set or occupy spawn points during the next attempt.
        try:
            advance_world(world, sync=world.get_settings().synchronous_mode)
        except Exception:
            pass

    raise RuntimeError("Could not spawn a complete platoon. Try another town or reduce --platoon-size.")


def spawn_npc_vehicles(
    client: carla.Client,
    world: carla.World,
    tm: carla.TrafficManager,
    spawn_points: List[carla.Transform],
    count: int,
    protected_actors: List[carla.Vehicle],
) -> List[carla.Vehicle]:
    protected_locations = [actor.get_location() for actor in protected_actors if actor and actor.is_alive]
    usable_points = []
    for spawn_point in spawn_points:
        if all(spawn_point.location.distance(loc) > 32.0 for loc in protected_locations):
            usable_points.append(spawn_point)

    blueprints = choose_vehicle_blueprints(world)
    random.shuffle(usable_points)

    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor

    batch = []
    for spawn_point in usable_points[: count * 3]:
        if len(batch) >= count:
            break
        bp = random.choice(blueprints)
        if bp.has_attribute("color"):
            bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
        if bp.has_attribute("driver_id"):
            bp.set_attribute("driver_id", random.choice(bp.get_attribute("driver_id").recommended_values))
        bp.set_attribute("role_name", "autopilot")
        batch.append(SpawnActor(bp, spawn_point).then(SetAutopilot(FutureActor, True, tm.get_port())))

    vehicles = []
    for response in client.apply_batch_sync(batch, True):
        if response.error:
            continue
        actor = world.get_actor(response.actor_id)
        if actor:
            configure_vehicle_tm(tm, actor, speed_diff=random.uniform(-5.0, 8.0), gap=random.uniform(3.0, 5.0))
            vehicles.append(actor)

    logger.info("Spawned %d NPC autopilot vehicles.", len(vehicles))
    return vehicles


def spawn_pedestrians(world: carla.World, count: int) -> Tuple[List[carla.Actor], List[carla.Actor]]:
    if count <= 0:
        logger.info("Pedestrians disabled.")
        return [], []

    bp_lib = world.get_blueprint_library()
    walker_bps = list(bp_lib.filter("walker.pedestrian.*"))
    controller_bp = bp_lib.find("controller.ai.walker")
    walkers: List[carla.Actor] = []
    controllers: List[carla.Actor] = []

    try:
        world.set_pedestrians_cross_factor(0.0)
    except Exception:
        pass

    for _ in range(count * 4):
        if len(walkers) >= count:
            break
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue

        bp = random.choice(walker_bps)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")

        walker = world.try_spawn_actor(bp, carla.Transform(loc))
        if walker is None:
            continue

        walkers.append(walker)

    is_sync = world.get_settings().synchronous_mode
    if is_sync and walkers:
        try:
            world.tick()
        except Exception:
            pass

    for walker in walkers:
        try:
            controller = world.try_spawn_actor(controller_bp, carla.Transform(), walker)
            if controller:
                controllers.append(controller)
        except Exception:
            pass

    if is_sync and controllers:
        try:
            world.tick()
        except Exception:
            pass

    for controller in controllers:
        try:
            controller.start()
            destination = world.get_random_location_from_navigation()
            if destination:
                controller.go_to_location(destination)
            controller.set_max_speed(random.uniform(1.1, 1.8))
        except Exception:
            pass

    logger.info("Spawned %d pedestrians with %d controllers.", len(walkers), len(controllers))
    return walkers, controllers


class CameraManager:
    def __init__(self, vehicle: carla.Vehicle, width: int, height: int):
        self.vehicle = vehicle
        self.width = width
        self.height = height
        self.surface: Optional[pygame.Surface] = pygame.Surface((width, height))
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._closed = False
        self.sensor: Optional[carla.Sensor] = None
        self.view_index = 0
        self.views = [
            ("Platoon Overview", carla.Transform(carla.Location(x=-34.0, z=12.0), carla.Rotation(pitch=-22.0))),
            ("Leader Chase", carla.Transform(carla.Location(x=-7.0, z=3.0), carla.Rotation(pitch=-10.0))),
            ("Cockpit", carla.Transform(carla.Location(x=0.75, y=-0.3, z=1.25), carla.Rotation(pitch=-2.0))),
        ]

        bp = vehicle.get_world().get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(width))
        bp.set_attribute("image_size_y", str(height))
        bp.set_attribute("fov", "110")

        self.sensor = vehicle.get_world().spawn_actor(bp, self.views[self.view_index][1], attach_to=vehicle)
        import weakref

        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CameraManager._on_image(weak_self, image))

    @staticmethod
    def _on_image(weak_self, image: carla.Image) -> None:
        self = weak_self()
        if not self or self._closed:
            return
        try:
            # Direct BGRA surface from CARLA buffer: eliminates CPU slice/swap overhead
            surf = pygame.image.frombuffer(image.raw_data, (image.width, image.height), 'BGRA')
            with self._frame_lock:
                if not self._closed:
                    self._latest_surface = surf
        except Exception:
            pass

    def toggle(self) -> str:
        self.view_index = (self.view_index + 1) % len(self.views)
        name, transform = self.views[self.view_index]
        if self.sensor and self.sensor.is_alive:
            self.sensor.set_transform(transform)
        return name

    @property
    def name(self) -> str:
        return self.views[self.view_index][0]

    def render(self, display: pygame.Surface) -> None:
        with self._frame_lock:
            surf = getattr(self, "_latest_surface", None)
        if surf is not None:
            display.blit(surf, (0, 0))

    def destroy(self) -> None:
        with self._frame_lock:
            self._closed = True
            self._latest_surface = None
            self._latest_frame = None
        sensor = self.sensor
        self.sensor = None
        if sensor:
            try:
                sensor.stop()
            except Exception:
                pass
            try:
                if sensor.is_alive:
                    sensor.destroy()
            except Exception:
                logger.debug("Camera sensor cleanup failed", exc_info=True)
        self.surface = None


class PlatoonSafetyMonitor:
    """Collision and lane-invasion logging for every platoon vehicle."""

    def __init__(self, world: carla.World, platoon: List[carla.Vehicle]):
        self.world = world
        self.platoon = platoon
        self.sensors: List[carla.Sensor] = []
        self.last_collision = "None"
        self.last_lane_event = "None"
        self.collision_count = 0
        self.lane_violation_count = 0
        self._attach_sensors()

    def _attach_sensors(self) -> None:
        bp_lib = self.world.get_blueprint_library()
        collision_bp = bp_lib.find("sensor.other.collision")
        lane_bp = bp_lib.find("sensor.other.lane_invasion")

        for vehicle in self.platoon:
            if not vehicle or not vehicle.is_alive:
                continue
            role = "Leader" if vehicle == self.platoon[0] else "Follower %d" % self.platoon.index(vehicle)
            try:
                sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
                weak_self = weakref.ref(self)
                sensor.listen(lambda event, role=role: PlatoonSafetyMonitor._on_collision(weak_self, event, role))
                self.sensors.append(sensor)
                logger.info("Collision sensor attached to %s vehicle #%d", role, vehicle.id)
            except Exception as exc:
                logger.warning("Failed to attach collision sensor to vehicle #%d: %s", vehicle.id, exc)

            try:
                sensor = self.world.spawn_actor(lane_bp, carla.Transform(), attach_to=vehicle)
                weak_self = weakref.ref(self)
                sensor.listen(lambda event, role=role: PlatoonSafetyMonitor._on_lane_invasion(weak_self, event, role))
                self.sensors.append(sensor)
                logger.info("Lane invasion sensor attached to %s vehicle #%d", role, vehicle.id)
            except Exception as exc:
                logger.warning("Failed to attach lane invasion sensor to vehicle #%d: %s", vehicle.id, exc)

    @staticmethod
    def _on_collision(weak_self, event, role: str) -> None:
        self = weak_self()
        if not self:
            return
        try:
            actor = event.actor
            other = event.other_actor
            if other and actor and other.id == actor.id:
                return
            impulse = event.normal_impulse
            intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
            other_name = other.type_id if other else "unknown"
            actor_id = actor.id if actor else 0
            other_id = other.id if other else 0
            self.collision_count += 1
            self.last_collision = "%s #%d hit %s #%d | impulse %.2f" % (role, actor_id, other_name, other_id, intensity)
            logger.error(
                "COLLISION DETECTED | %s vehicle #%d | other=%s #%d | impulse=%.2f | total_collisions=%d",
                role,
                actor_id,
                other_name,
                other_id,
                intensity,
                self.collision_count,
            )
        except Exception:
            pass

    @staticmethod
    def _on_lane_invasion(weak_self, event, role: str) -> None:
        self = weak_self()
        if not self:
            return
        try:
            actor = event.actor
            if not actor or not actor.is_alive:
                return
            markings = []
            try:
                markings = [str(marking.type) for marking in event.crossed_lane_markings]
            except Exception:
                markings = ["unknown"]
            actor_id = actor.id if actor else 0
        except Exception:
            return
        self.lane_violation_count += 1
        self.last_lane_event = "%s #%d crossed %s" % (role, actor_id, ",".join(markings))
        logger.warning(
            "LANE VIOLATION | %s vehicle #%d | crossed_markings=%s | total_lane_events=%d",
            role,
            actor_id,
            ",".join(markings),
            self.lane_violation_count,
        )

    def destroy(self) -> None:
        for sensor in self.sensors:
            try:
                if sensor and sensor.is_alive:
                    sensor.stop()
                    sensor.destroy()
            except Exception:
                pass
        self.sensors = []


def actor_speed_kmh(actor: carla.Actor) -> float:
    try:
        velocity = actor.get_velocity()
        return 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
    except Exception:
        return 0.0


def actor_location(actor: carla.Actor) -> Optional[carla.Location]:
    try:
        if actor and actor.is_alive:
            return actor.get_location()
    except Exception:
        pass
    return None


def actor_type_name(actor: carla.Actor) -> str:
    try:
        parts = actor.type_id.split(".")
        if len(parts) >= 3:
            return f"{parts[1]} {parts[2]}".title()
        return actor.type_id
    except Exception:
        return "Unknown"


def vehicle_half_length(actor: carla.Actor) -> float:
    try:
        return float(actor.bounding_box.extent.x)
    except Exception:
        return 2.35


def vehicle_clearance(ahead: carla.Actor, behind: carla.Actor) -> Optional[float]:
    ahead_loc = actor_location(ahead)
    behind_loc = actor_location(behind)
    if ahead_loc is None or behind_loc is None:
        return None
    center_distance = ahead_loc.distance(behind_loc)
    return center_distance - vehicle_half_length(ahead) - vehicle_half_length(behind)


def platoon_rows(platoon: List[carla.Vehicle]) -> List[dict]:
    rows = []
    previous = None
    for index, vehicle in enumerate(platoon):
        alive = bool(vehicle and vehicle.is_alive)
        clearance = None
        center_gap = None
        if previous is not None and alive:
            prev_loc = actor_location(previous)
            loc = actor_location(vehicle)
            if prev_loc and loc:
                center_gap = loc.distance(prev_loc)
                clearance = vehicle_clearance(previous, vehicle)
        rows.append(
            {
                "role": "LEADER" if index == 0 else f"FOLLOWER {index}",
                "id": vehicle.id if vehicle else 0,
                "model": actor_type_name(vehicle) if alive else "Missing",
                "speed": actor_speed_kmh(vehicle) if alive else 0.0,
                "gap": clearance,
                "center_gap": center_gap,
                "alive": alive,
            }
        )
        previous = vehicle
    return rows


def traffic_summary(vehicles: List[carla.Vehicle]) -> Tuple[int, int, int, float]:
    live = [vehicle for vehicle in vehicles if vehicle and vehicle.is_alive]
    speeds = [actor_speed_kmh(vehicle) for vehicle in live]
    moving = len([speed for speed in speeds if speed > 1.0])
    stopped = len(speeds) - moving
    avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
    return len(live), moving, stopped, avg_speed


def walker_summary(walkers: List[carla.Actor]) -> Tuple[int, int]:
    live = [walker for walker in walkers if walker and walker.is_alive]
    moving = len([walker for walker in live if actor_speed_kmh(walker) > 0.3])
    return len(live), moving


def steer_to_location(vehicle: carla.Vehicle, target_loc: carla.Location) -> float:
    try:
        transform = vehicle.get_transform()
        loc = transform.location
        dx = target_loc.x - loc.x
        dy = target_loc.y - loc.y
        yaw = math.radians(transform.rotation.yaw)
        local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
        local_y = -dx * math.sin(yaw) + dy * math.cos(yaw)
        # Never turn toward a target that has fallen behind the vehicle.  A
        # minimum forward look-ahead prevents a follower overshoot from
        # generating a near-90-degree steering reversal.
        angle = math.atan2(local_y, max(2.0, local_x))
        return max(-0.65, min(0.65, angle * 1.8))
    except Exception:
        return 0.0


def interpolate_location(a: carla.Location, b: carla.Location, ratio: float) -> carla.Location:
    ratio = max(0.0, min(1.0, ratio))
    return carla.Location(
        x=a.x + (b.x - a.x) * ratio,
        y=a.y + (b.y - a.y) * ratio,
        z=a.z + (b.z - a.z) * ratio,
    )


class PlatoonPathController:
    """Follower controller that replays the leader trajectory with distance delay."""

    def __init__(self, desired_clearance: float = 7.5, max_history: int = 900):
        self.desired_clearance = desired_clearance
        self.history: Deque[carla.Location] = deque(maxlen=max_history)
        self.last_steer: Dict[int, float] = {}
        self.vehicle_clearances: Dict[int, float] = {}

    def reset(self) -> None:
        self.history.clear()
        self.last_steer.clear()
        self.vehicle_clearances.clear()

    def _get_vehicle_clearance(self, vehicle: carla.Vehicle) -> float:
        return self.vehicle_clearances.setdefault(vehicle.id, self.desired_clearance)

    def apply_tactical_action(self, vehicle: carla.Vehicle, action: str) -> None:
        """Apply one follower's action without changing another follower."""
        if not vehicle:
            return
        current = self._get_vehicle_clearance(vehicle)
        target = {
            "DECELERATE": self.desired_clearance + 2.0,
            "ACCELERATE": max(4.0, self.desired_clearance - 1.0),
            "CRUISE": self.desired_clearance,
        }.get(action, self.desired_clearance)
        # Tactical actions set a target mode; they do not add an unbounded
        # clearance delta on every render frame.  Smooth the target so an
        # action change cannot produce a sudden longitudinal jump.
        clearance = 0.75 * current + 0.25 * target
        self.vehicle_clearances[vehicle.id] = max(4.0, min(25.0, clearance))

    def update(self, platoon: List[carla.Vehicle]) -> None:
        if len(platoon) < 2 or not platoon[0] or not platoon[0].is_alive:
            return

        leader_loc = actor_location(platoon[0])
        if leader_loc is None:
            return
        self._append_leader_location(leader_loc)

        for index in range(1, len(platoon)):
            follower = platoon[index]
            previous = platoon[index - 1]
            if not follower or not previous or not follower.is_alive or not previous.is_alive:
                continue

            desired_clearance = self._get_vehicle_clearance(follower)
            # Follow the immediate predecessor directly.  A delayed point on
            # the leader history can become invalid when an earlier follower
            # falls behind, causing the next follower to stop or leave the
            # intended lane.  Each follower must close the gap to its own
            # predecessor before considering the leader trajectory.
            target_loc = self._point_behind_vehicle(
                previous, follower, desired_clearance
            )
            if target_loc is None:
                target_loc = actor_location(previous)
            if target_loc is None:
                continue

            self._drive_follower(follower, previous, target_loc, desired_clearance)

    def _point_behind_vehicle(
        self,
        ahead: carla.Vehicle,
        follower: carla.Vehicle,
        desired_clearance: float,
    ) -> Optional[carla.Location]:
        """Return a short-horizon target behind the immediate predecessor."""
        ahead_loc = actor_location(ahead)
        if ahead_loc is None:
            return None
        try:
            forward = ahead.get_transform().get_forward_vector()
            distance = (
                vehicle_half_length(ahead)
                + vehicle_half_length(follower)
                + max(4.0, desired_clearance)
            )
            return carla.Location(
                x=ahead_loc.x - forward.x * distance,
                y=ahead_loc.y - forward.y * distance,
                z=ahead_loc.z,
            )
        except Exception:
            return None

    def _center_delay_for_index(self, platoon: List[carla.Vehicle], index: int) -> float:
        distance = 0.0
        for pair_idx in range(1, index + 1):
            ahead = platoon[pair_idx - 1]
            behind = platoon[pair_idx]
            distance += (
                vehicle_half_length(ahead)
                + vehicle_half_length(behind)
                + self._get_vehicle_clearance(behind)
            )
        return distance

    def _append_leader_location(self, loc: carla.Location) -> None:
        if not self.history:
            self.history.append(carla.Location(x=loc.x, y=loc.y, z=loc.z))
            return
        if self.history[-1].distance(loc) >= 0.25:
            self.history.append(carla.Location(x=loc.x, y=loc.y, z=loc.z))

    def _point_behind_leader(self, target_distance: float) -> Optional[carla.Location]:
        if len(self.history) < 2:
            return self.history[0] if self.history else None

        travelled = 0.0
        newest = self.history[-1]
        for idx in range(len(self.history) - 2, -1, -1):
            older = self.history[idx]
            segment = newest.distance(older)
            if travelled + segment >= target_distance:
                ratio = (target_distance - travelled) / max(0.001, segment)
                return interpolate_location(newest, older, ratio)
            travelled += segment
            newest = older
        return self.history[0]

    def _drive_follower(
        self,
        follower: carla.Vehicle,
        previous: carla.Vehicle,
        target_loc: carla.Location,
        desired_clearance: float,
    ) -> None:
        follower_loc = actor_location(follower)
        previous_loc = actor_location(previous)
        if follower_loc is None or previous_loc is None:
            return

        clearance = vehicle_clearance(previous, follower)
        if clearance is None:
            return
        path_error = follower_loc.distance(target_loc)
        previous_speed = actor_speed_kmh(previous)
        follower_speed = actor_speed_kmh(follower)
        closing_speed_mps = max(0.0, (follower_speed - previous_speed) / 3.6)
        time_to_close = clearance / closing_speed_mps if closing_speed_mps > 0.05 else 999.0

        raw_steer = steer_to_location(follower, target_loc)
        old_steer = self.last_steer.get(follower.id, 0.0)
        steer = old_steer * 0.85 + raw_steer * 0.15
        self.last_steer[follower.id] = steer

        throttle = 0.0
        brake = 0.0
        speed_excess = follower_speed - previous_speed
        if clearance < 1.5 or time_to_close < 0.9:
            brake = 1.0
        elif clearance < 3.0 or time_to_close < 1.4:
            brake = 0.82
        elif speed_excess > 4.0 and clearance < desired_clearance + 15.0:
            # Never keep accelerating when this follower is already faster
            # than the vehicle it must follow.
            brake = min(0.65, 0.18 + (speed_excess - 4.0) * 0.045)
        elif previous_speed < 2.0 and follower_speed > 3.0 and clearance < desired_clearance + 6.0:
            brake = 0.85
        elif time_to_close < 2.4:
            brake = 0.62
        elif clearance < desired_clearance * 0.82:
            # Proportional brake – too close
            brake = min(0.55, 0.12 + (desired_clearance * 0.82 - clearance) * 0.07)
        elif previous_speed < 1.0 and clearance < desired_clearance + 2.0:
            brake = 0.70
        elif clearance < desired_clearance * 1.10:
            # Dead-band: within ±10% of target → gentle coast, match speed
            speed_diff = previous_speed - follower_speed
            throttle = max(0.0, min(0.22, 0.08 + speed_diff * 0.012))
        else:
            # Too far → pull up proportionally
            gap_excess = clearance - desired_clearance
            gap_boost = min(0.28, gap_excess * 0.022)
            path_boost = min(0.12, path_error * 0.010)
            speed_boost = max(0.0, previous_speed - follower_speed) * 0.014
            throttle = min(0.68, 0.16 + gap_boost + path_boost + speed_boost)

        try:
            follower.apply_control(
                carla.VehicleControl(
                    throttle=float(throttle),
                    steer=float(max(-0.65, min(0.65, steer))),
                    brake=float(brake),
                    hand_brake=False,
                    reverse=False,
                )
            )
        except Exception:
            pass


class RoadLeaderController:
    """Simple road waypoint autopilot for the platoon leader."""

    LANE_IDLE = "IDLE"
    LANE_IN_PROGRESS = "IN_PROGRESS"
    LANE_CHANGE_DT = 1.0 / 30.0
    MIN_LANE_GAP = 15.0
    LANE_TIMEOUT = 4.0

    def __init__(self, carla_map: carla.Map, target_speed_kmh: float = 28.0):
        self.map = carla_map
        self.cruise_speed_kmh = float(target_speed_kmh)
        self.target_speed_kmh = float(target_speed_kmh)
        self.tactical_speed_offset = 0.0
        self.traffic_stop = False
        self.last_steer = 0.0
        self.requested_lane_change: Optional[str] = None
        self.lane_state = self.LANE_IDLE
        self.target_lane_id: Optional[int] = None
        self.lane_direction: Optional[str] = None
        self.lane_elapsed = 0.0
        self.lane_cooldown = 0.0

    def apply_tactical_action(self, action: str) -> None:
        """Set a bounded tactical mode; do not accumulate speed impulses."""
        if action == "CHANGE_LEFT":
            self.request_lane_change("LEFT")
            return
        if action == "CHANGE_RIGHT":
            self.request_lane_change("RIGHT")
            return
        self.tactical_speed_offset = {
            "ACCELERATE": 4.0,
            "DECELERATE": -5.0,
            "CRUISE": 0.0,
        }.get(action, 0.0)
        if not self.traffic_stop:
            self.target_speed_kmh = max(
                0.0,
                min(60.0, self.cruise_speed_kmh + self.tactical_speed_offset),
            )

    def set_traffic_stop(self, stopped: bool) -> None:
        """Apply/remove a red-light stop without losing the tactical mode."""
        self.traffic_stop = bool(stopped)
        if self.traffic_stop:
            self.target_speed_kmh = 0.0
        else:
            self.target_speed_kmh = max(
                0.0,
                min(60.0, self.cruise_speed_kmh + self.tactical_speed_offset),
            )

    def request_lane_change(self, direction: str) -> None:
        """Request a leader lane change; the waypoint controller executes it."""
        direction = str(direction).upper()
        if (
            direction in ("LEFT", "RIGHT")
            and self.lane_state == self.LANE_IDLE
            and self.lane_cooldown <= 0.0
            and self.requested_lane_change is None
        ):
            self.requested_lane_change = direction

    def _lane_change_is_safe(self, vehicle: carla.Vehicle, target_wp: carla.Waypoint) -> bool:
        try:
            for other in vehicle.get_world().get_actors().filter("vehicle.*"):
                if other.id == vehicle.id or not other.is_alive:
                    continue
                other_wp = self.map.get_waypoint(
                    other.get_location(),
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving,
                )
                if (
                    other_wp
                    and other_wp.road_id == target_wp.road_id
                    and other_wp.lane_id == target_wp.lane_id
                    and other.get_location().distance(target_wp.transform.location)
                    < self.MIN_LANE_GAP
                ):
                    return False
        except Exception:
            return False
        return True

    def _nearest_obstacle_ahead(self, vehicle: carla.Vehicle) -> float:
        """Returns clearance (metres) to nearest vehicle in forward cone. 50m if clear."""
        import math
        fwd = vehicle.get_transform().get_forward_vector()
        loc = vehicle.get_location()
        min_dist = 50.0
        try:
            actors = list(vehicle.get_world().get_actors().filter('vehicle.*'))
            actors += list(vehicle.get_world().get_actors().filter('walker.*'))
            for v in actors:
                if v.id == vehicle.id or not v.is_alive:
                    continue
                v_loc = v.get_location()
                dist = loc.distance(v_loc)
                if dist < min_dist:
                    dx = v_loc.x - loc.x
                    dy = v_loc.y - loc.y
                    mag = math.sqrt(dx * dx + dy * dy)
                    if mag > 0 and (dx * fwd.x + dy * fwd.y) / mag > 0.80:
                        min_dist = dist
        except Exception:
            pass
        return max(0.0, min_dist - 4.7)

    def update(self, vehicle: carla.Vehicle) -> None:
        if not vehicle or not vehicle.is_alive:
            return

        try:
            self.lane_cooldown = max(0.0, self.lane_cooldown - self.LANE_CHANGE_DT)
            if self.lane_state == self.LANE_IN_PROGRESS:
                self.lane_elapsed += self.LANE_CHANGE_DT
                if self.lane_elapsed > self.LANE_TIMEOUT:
                    self.lane_state = self.LANE_IDLE
                    self.target_lane_id = None
                    self.lane_direction = None
                    self.lane_elapsed = 0.0

            waypoint = self.map.get_waypoint(
                vehicle.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            lane_waypoint = waypoint

            if self.lane_state == self.LANE_IN_PROGRESS:
                if waypoint.lane_id == self.target_lane_id:
                    self.lane_state = self.LANE_IDLE
                    self.lane_cooldown = 2.0
                    self.target_lane_id = None
                    self.lane_direction = None
                    self.lane_elapsed = 0.0
                else:
                    direction = self.lane_direction
                    adjacent = (
                        waypoint.get_left_lane()
                        if direction == "LEFT"
                        else waypoint.get_right_lane()
                    )
                    if adjacent and adjacent.lane_id == self.target_lane_id:
                        lane_waypoint = adjacent
                    else:
                        self.lane_state = self.LANE_IDLE
                        self.target_lane_id = None
                        self.lane_direction = None
                        self.lane_elapsed = 0.0

            if self.lane_state == self.LANE_IDLE and self.requested_lane_change:
                adjacent = (
                    waypoint.get_left_lane()
                    if self.requested_lane_change == "LEFT"
                    else waypoint.get_right_lane()
                )
                if (
                    adjacent
                    and adjacent.lane_type == carla.LaneType.Driving
                    and self._lane_change_is_safe(vehicle, adjacent)
                ):
                    lane_waypoint = adjacent
                    self.target_lane_id = adjacent.lane_id
                    self.lane_direction = self.requested_lane_change
                    self.lane_state = self.LANE_IN_PROGRESS
                    self.lane_elapsed = 0.0
                else:
                    # Reject unsafe/unavailable requests rather than retrying
                    # them on every neural decision.
                    self.requested_lane_change = None
            self.requested_lane_change = None

            next_points = lane_waypoint.next(9.0)
            if not next_points:
                return
            target = self._choose_smooth_next(vehicle, next_points).transform.location

            speed = actor_speed_kmh(vehicle)
            raw_steer = steer_to_location(vehicle, target)
            steer = self.last_steer * 0.65 + raw_steer * 0.35
            self.last_steer = steer

            # --- Forward obstacle check (leader must brake for cut-ins) ---
            gap = self._nearest_obstacle_ahead(vehicle)
            if gap < 4.0:
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0,
                    steer=float(max(-0.6, min(0.6, steer))), hand_brake=False, reverse=False))
                return
            elif gap < 8.0:
                # Emergency proportional slow-down: ignore target_speed, just crawl
                throttle = 0.0
                brake = 0.75
                vehicle.apply_control(carla.VehicleControl(throttle=float(throttle), brake=float(brake),
                    steer=float(max(-0.6, min(0.6, steer))), hand_brake=False, reverse=False))
                return
            elif gap < 15.0:
                # Obstacle in range – cap speed at 15 km/h
                effective_target = min(self.target_speed_kmh, 15.0)
            else:
                effective_target = self.target_speed_kmh
            # ---

            throttle = 0.0
            brake = 0.0
            if speed < effective_target - 4.0:
                throttle = 0.58
            elif speed < effective_target:
                throttle = 0.34
            elif speed > effective_target + 6.0:
                brake = 0.22
            else:
                throttle = 0.18

            if abs(steer) > 0.45 and speed > 22.0:
                throttle = min(throttle, 0.20)
                brake = max(brake, 0.10)
            if brake > 0.0:
                throttle = 0.0

            vehicle.apply_control(
                carla.VehicleControl(
                    throttle=float(throttle),
                    steer=float(max(-0.6, min(0.6, steer))),
                    brake=float(brake),
                    hand_brake=False,
                    reverse=False,
                )
            )
        except Exception:
            pass

    def _choose_smooth_next(self, vehicle: carla.Vehicle, next_points: List[carla.Waypoint]) -> carla.Waypoint:
        if len(next_points) == 1:
            return next_points[0]

        yaw = vehicle.get_transform().rotation.yaw

        def yaw_error(wp: carla.Waypoint) -> float:
            diff = (wp.transform.rotation.yaw - yaw + 180.0) % 360.0 - 180.0
            return abs(diff)

        return min(next_points, key=yaw_error)


def brake_platoon(platoon: List[carla.Vehicle]) -> None:
    for vehicle in platoon:
        if vehicle and vehicle.is_alive:
            try:
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0, hand_brake=False))
            except Exception:
                pass


class MiniHUD:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        pygame.font.init()
        self.title = pygame.font.SysFont("consolas", 18, bold=True)
        self.font = pygame.font.SysFont("consolas", 14, bold=True)
        self.small = pygame.font.SysFont("consolas", 12)
        self.tiny = pygame.font.SysFont("consolas", 11)

    def render(
        self,
        display: pygame.Surface,
        leader: carla.Vehicle,
        platoon: List[carla.Vehicle],
        vehicles: List[carla.Vehicle],
        walkers: List[carla.Actor],
        autopilot: bool,
        camera_name: str,
        elapsed: float,
        town_name: str,
        synchronous: bool,
        fps: float,
    ) -> None:
        rows = platoon_rows(platoon)
        live_npcs, moving_npcs, stopped_npcs, avg_npc_speed = traffic_summary(vehicles)
        live_walkers, moving_walkers = walker_summary(walkers)
        leader_speed = rows[0]["speed"] if rows else 0.0
        mins = int(elapsed) // 60
        secs = int(elapsed) % 60

        top = pygame.Surface((self.width, 38), pygame.SRCALPHA)
        top.fill((10, 14, 22, 215))
        display.blit(top, (0, 0))
        display.blit(self.title.render("CARLA AUTOPILOT PLATOON SYSTEM", True, (245, 248, 255)), (16, 9))

        mode = "SYNCHRONOUS 30 FPS" if synchronous else "ASYNC WORLD"
        right_text = self.font.render(f"{town_name} | {mins:02d}:{secs:02d} | {fps:02.0f} FPS | {mode}", True, (180, 215, 245))
        display.blit(right_text, (self.width - right_text.get_width() - 16, 10))

        banner_w = 650
        banner_h = 42
        bx = (self.width - banner_w) // 2
        by = 48
        banner = pygame.Surface((banner_w, banner_h), pygame.SRCALPHA)
        banner.fill((8, 42, 24, 225) if autopilot else (60, 28, 10, 225))
        pygame.draw.rect(banner, (70, 235, 130) if autopilot else (245, 150, 45), banner.get_rect(), width=2, border_radius=7)
        display.blit(banner, (bx, by))
        banner_text = f"PLATOON AUTOPILOT {'ACTIVE' if autopilot else 'PAUSED'} | {len(rows)} VEHICLES | LEADER #{leader.id} | {leader_speed:4.1f} km/h"
        text = self.font.render(banner_text, True, (210, 255, 225) if autopilot else (255, 230, 200))
        display.blit(text, (bx + (banner_w - text.get_width()) // 2, by + 13))

        self._draw_platoon_card(display, 16, 104, rows)
        self._draw_environment_card(
            display,
            self.width - 310,
            104,
            live_npcs,
            moving_npcs,
            stopped_npcs,
            avg_npc_speed,
            live_walkers,
            moving_walkers,
            camera_name,
            autopilot,
        )
        self._draw_leader_dashboard(display, 16, self.height - 154, leader, leader_speed)

        help_bar = pygame.Surface((610, 34), pygame.SRCALPHA)
        help_bar.fill((10, 14, 22, 220))
        bx = (self.width - 610) // 2
        by = self.height - 46
        display.blit(help_bar, (bx, by))
        text = self.font.render("[P] Toggle leader autopilot  |  [C] Camera view  |  [ESC/Q] Exit  |  Live logs in second terminal", True, (220, 235, 255))
        display.blit(text, (bx + (610 - text.get_width()) // 2, by + 9))

    def _draw_platoon_card(self, display: pygame.Surface, x: int, y: int, rows: List[dict]) -> None:
        w = 430
        h = 178
        card = pygame.Surface((w, h), pygame.SRCALPHA)
        card.fill((12, 18, 28, 225))
        pygame.draw.rect(card, (65, 105, 155), card.get_rect(), width=1, border_radius=6)
        display.blit(card, (x, y))

        display.blit(self.font.render("PLATOON TELEMETRY", True, (140, 190, 255)), (x + 12, y + 10))
        headers = "ROLE          ID     SPEED       CLEAR     STATUS"
        display.blit(self.tiny.render(headers, True, (165, 190, 215)), (x + 12, y + 36))

        for idx, row in enumerate(rows[:6]):
            yy = y + 58 + idx * 25
            gap = "--" if row["gap"] is None else f"{row['gap']:4.1f}m"
            status = "MOVING" if row["speed"] > 1.0 else "WAITING"
            color = (105, 255, 165) if row["speed"] > 1.0 else (255, 210, 90)
            if row["gap"] is not None and row["gap"] < 2.5:
                status = "TOO CLOSE"
                color = (255, 95, 95)
            if not row["alive"]:
                status = "MISSING"
                color = (255, 95, 95)
            line = f"{row['role']:<12} #{row['id']:<4} {row['speed']:6.1f} km/h  {gap:<7} {status}"
            display.blit(self.small.render(line, True, color), (x + 12, yy))
            model = row["model"][:44]
            display.blit(self.tiny.render(model, True, (155, 175, 200)), (x + 28, yy + 13))

    def _draw_environment_card(
        self,
        display: pygame.Surface,
        x: int,
        y: int,
        live_npcs: int,
        moving_npcs: int,
        stopped_npcs: int,
        avg_npc_speed: float,
        live_walkers: int,
        moving_walkers: int,
        camera_name: str,
        autopilot: bool,
    ) -> None:
        w = 294
        h = 178
        card = pygame.Surface((w, h), pygame.SRCALPHA)
        card.fill((12, 18, 28, 225))
        pygame.draw.rect(card, (65, 105, 155), card.get_rect(), width=1, border_radius=6)
        display.blit(card, (x, y))

        lines = [
            ("ENVIRONMENT STATUS", (140, 190, 255), self.font),
            (f"Traffic: {live_npcs} vehicles", (220, 235, 245), self.small),
            (f"Moving: {moving_npcs} | Waiting: {stopped_npcs}", (170, 220, 255), self.small),
            (f"Avg NPC Speed: {avg_npc_speed:4.1f} km/h", (170, 220, 255), self.small),
            (f"Pedestrians: {live_walkers} live | {moving_walkers} moving", (220, 235, 245), self.small),
            ("Traffic Manager: ACTIVE", (105, 255, 165), self.small),
            (f"Leader Mode: {'AUTOPILOT' if autopilot else 'PAUSED'}", (105, 255, 165) if autopilot else (255, 210, 90), self.small),
            (f"Camera: {camera_name}", (220, 235, 245), self.small),
        ]

        yy = y + 10
        for text, color, font in lines:
            display.blit(font.render(text, True, color), (x + 12, yy))
            yy += 22 if font == self.font else 19

    def _draw_leader_dashboard(self, display: pygame.Surface, x: int, y: int, leader: carla.Vehicle, speed: float) -> None:
        w = 285
        h = 108
        card = pygame.Surface((w, h), pygame.SRCALPHA)
        card.fill((12, 18, 28, 225))
        pygame.draw.rect(card, (65, 105, 155), card.get_rect(), width=1, border_radius=6)
        display.blit(card, (x, y))

        display.blit(self.font.render("LEADER CONTROL", True, (140, 190, 255)), (x + 12, y + 10))
        display.blit(pygame.font.SysFont("arial", 30, bold=True).render(f"{int(speed):3d}", True, (255, 255, 255)), (x + 14, y + 38))
        display.blit(self.font.render("km/h", True, (170, 195, 220)), (x + 92, y + 50))

        control = leader.get_control()
        self._draw_bar(display, x + 150, y + 39, 100, 10, control.throttle, (80, 230, 120), "THR")
        self._draw_bar(display, x + 150, y + 60, 100, 10, control.brake, (255, 90, 90), "BRK")
        self._draw_bar(display, x + 150, y + 81, 100, 10, abs(control.steer), (80, 210, 255), "STR")

    def _draw_bar(
        self,
        display: pygame.Surface,
        x: int,
        y: int,
        w: int,
        h: int,
        val: float,
        color: Tuple[int, int, int],
        label: str,
    ) -> None:
        display.blit(self.tiny.render(label, True, (170, 190, 210)), (x, y - 2))
        bx = x + 30
        pygame.draw.rect(display, (35, 45, 60), (bx, y, w, h), border_radius=3)
        fill = int(max(0.0, min(1.0, val)) * w)
        if fill > 0:
            pygame.draw.rect(display, color, (bx, y, fill, h), border_radius=3)


def run(args: argparse.Namespace) -> None:
    setup_logging(args.log_file)
    server_process = None
    if args.launch_server and not wait_for_server(args.host, args.port, 2):
        server_process = maybe_launch_carla(args.town, args.port, args.quality)

    if not wait_for_server(args.host, args.port, args.wait_seconds):
        raise RuntimeError("CARLA server did not respond. Start CarlaUE4.exe or use --launch-server.")

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    pygame.init()
    pygame.display.set_caption("CARLA Autopilot Platoon Window")
    display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
    clock = pygame.time.Clock()

    camera = None
    platoon: List[carla.Vehicle] = []
    npcs: List[carla.Vehicle] = []
    walkers: List[carla.Actor] = []
    controllers: List[carla.Actor] = []
    world: Optional[carla.World] = None
    tm: Optional[carla.TrafficManager] = None
    original_settings = None
    leader_start_loc = None
    platoon_controller = PlatoonPathController(desired_clearance=7.5)
    leader_controller = None

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

        tm = client.get_trafficmanager(TM_PORT)
        original_settings = configure_world_timing(world, tm, sync=not args.async_world)

        if args.clean:
            cleanup_existing_actors(world)

        tm.set_global_distance_to_leading_vehicle(3.5)
        tm.global_percentage_speed_difference(0.0)

        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No vehicle spawn points found in this map.")

        platoon = spawn_platoon(world, tm, spawn_points, args.platoon_size)
        npcs = spawn_npc_vehicles(client, world, tm, spawn_points, args.vehicles, platoon)
        walkers, controllers = spawn_pedestrians(world, args.pedestrians)
        leader = platoon[0]
        leader_controller = RoadLeaderController(world.get_map(), target_speed_kmh=args.platoon_speed)

        for _ in range(4):
            leader_controller.update(leader)
            platoon_controller.update(platoon)
            advance_world(world, sync=not args.async_world)

        camera = CameraManager(leader, WINDOW_WIDTH, WINDOW_HEIGHT)
        hud = MiniHUD(WINDOW_WIDTH, WINDOW_HEIGHT)
        autopilot = True
        started = time.time()
        leader_start_loc = leader.get_location()
        last_motion_log = 0.0
        running = True

        logger.info("Window running. Press P for leader autopilot toggle, C for camera, ESC/Q to exit.")
        logger.info("Platoon path controller active: followers replay leader trajectory with 7.5m bumper clearance.")
        while running:
            elapsed = time.time() - started
            if args.duration > 0 and time.time() - started >= args.duration:
                logger.info("Duration %.1fs reached; exiting test run.", args.duration)
                running = False

            clock.tick(TARGET_FPS)
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
                        logger.info("Camera: %s", camera.toggle())

            if autopilot:
                leader_controller.update(leader)
                platoon_controller.update(platoon)
            else:
                brake_platoon(platoon)
            advance_world(world, sync=not args.async_world)
            if elapsed - last_motion_log >= 2.0:
                last_motion_log = elapsed
                speed = actor_speed_kmh(leader)
                displacement = leader.get_location().distance(leader_start_loc)
                logger.info("Leader motion check: %.1f km/h, displaced %.2f m", speed, displacement)
                for row in platoon_rows(platoon):
                    gap = "--" if row["gap"] is None else "%.1fm" % row["gap"]
                    logger.info(
                        "Platoon %s #%d | speed %.1f km/h | clearance %s | %s",
                        row["role"],
                        row["id"],
                        row["speed"],
                        gap,
                        "alive" if row["alive"] else "missing",
                    )
                live_npcs, moving_npcs, stopped_npcs, avg_npc_speed = traffic_summary(npcs)
                live_walkers, moving_walkers = walker_summary(walkers)
                logger.info(
                    "Traffic telemetry | NPC live=%d moving=%d waiting=%d avg=%.1f km/h | walkers live=%d moving=%d",
                    live_npcs,
                    moving_npcs,
                    stopped_npcs,
                    avg_npc_speed,
                    live_walkers,
                    moving_walkers,
                )

            display.fill((8, 8, 10))
            if camera:
                camera.render(display)
            hud.render(
                display,
                leader,
                platoon,
                npcs,
                walkers,
                autopilot,
                camera.name if camera else "None",
                time.time() - started,
                current_town,
                not args.async_world,
                clock.get_fps(),
            )
            pygame.display.flip()

    finally:
        if leader_start_loc is not None and platoon and platoon[0] and platoon[0].is_alive:
            try:
                logger.info("Final leader displacement: %.2f m", platoon[0].get_location().distance(leader_start_loc))
            except Exception:
                pass
        logger.info("Cleaning up spawned actors...")
        if camera:
            camera.destroy()
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
            stop_launched_carla(server_process)
        logger.info("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spawn a CARLA autopilot platoon in a Pygame window.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town01", help="Town01, Town02, ... or auto/current")
    parser.add_argument("--vehicles", type=int, default=32)
    parser.add_argument("--pedestrians", type=int, default=20)
    parser.add_argument("--platoon-size", type=int, default=4)
    parser.add_argument("--platoon-speed", type=float, default=28.0, help="Target platoon leader speed in km/h.")
    parser.add_argument("--wait-seconds", type=int, default=60)
    parser.add_argument("--quality", default="Low", choices=["Low", "Epic"])
    parser.add_argument("--launch-server", action="store_true", help="Launch WindowsNoEditor/CarlaUE4.exe if no server is running.")
    parser.add_argument("--kill-launched-server", action="store_true", help="Close CARLA on exit, only when launched by this script.")
    parser.add_argument("--duration", type=float, default=0.0, help="Auto-exit after N seconds; 0 means run until closed.")
    parser.add_argument("--async-world", action="store_true", help="Use CARLA's free-running async mode instead of client-driven ticks.")
    parser.add_argument("--log-file", default=DEFAULT_LOG_FILE, help="Path for live detailed telemetry logs.")
    parser.add_argument("--no-clean", dest="clean", action="store_false", help="Do not remove old actors before spawning.")
    parser.set_defaults(clean=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
