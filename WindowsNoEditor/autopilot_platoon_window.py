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
import glob
import logging
import math
import os
import random
import socket
import subprocess
import sys
import time
from typing import List, Optional, Tuple

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


logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger("AutopilotPlatoonWindow")

WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 576
TARGET_FPS = 30
TM_PORT = 8000


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
        try:
            world.wait_for_tick(seconds=2.0)
        except Exception:
            pass


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


def configure_vehicle_tm(tm: carla.TrafficManager, vehicle: carla.Vehicle, *, speed_diff: float = 0.0, gap: float = 3.5) -> None:
    try:
        vehicle.set_autopilot(True, tm.get_port())
        tm.auto_lane_change(vehicle, True)
        tm.distance_to_leading_vehicle(vehicle, gap)
        tm.ignore_lights_percentage(vehicle, 0.0)
        tm.ignore_signs_percentage(vehicle, 0.0)
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
        configure_vehicle_tm(tm, leader, speed_diff=0.0, gap=4.5)

        current_wp = carla_map.get_waypoint(spawn_point.location)
        for index in range(1, count):
            previous = current_wp.previous(8.0)
            if not previous:
                break
            current_wp = previous[0]

            bp = random.choice(vehicle_bps)
            if bp.has_attribute("color"):
                color = ["35,95,225", "45,135,245", "30,75,195"][(index - 1) % 3]
                bp.set_attribute("color", color)
            bp.set_attribute("role_name", f"platoon_follower_{index}")

            loc = current_wp.transform.location + carla.Location(z=0.35)
            follower = world.try_spawn_actor(bp, carla.Transform(loc, current_wp.transform.rotation))
            if follower is None:
                break

            configure_vehicle_tm(tm, follower, speed_diff=-5.0, gap=4.5)
            try:
                tm.auto_lane_change(follower, False)
            except Exception:
                pass
            platoon.append(follower)

        if len(platoon) == count:
            logger.info("Spawned autopilot platoon: %s", [v.id for v in platoon])
            return platoon

        for vehicle in platoon:
            try:
                vehicle.destroy()
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
        if all(spawn_point.location.distance(loc) > 12.0 for loc in protected_locations):
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
    for response in client.apply_batch_sync(batch, False):
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

        controller = world.try_spawn_actor(controller_bp, carla.Transform(), walker)
        if controller:
            controller.start()
            destination = world.get_random_location_from_navigation()
            if destination:
                controller.go_to_location(destination)
            controller.set_max_speed(random.uniform(1.1, 1.8))
            controllers.append(controller)
        walkers.append(walker)

    logger.info("Spawned %d pedestrians with %d controllers.", len(walkers), len(controllers))
    return walkers, controllers


class CameraManager:
    def __init__(self, vehicle: carla.Vehicle, width: int, height: int):
        self.vehicle = vehicle
        self.width = width
        self.height = height
        self.surface: Optional[pygame.Surface] = None
        self.sensor: Optional[carla.Sensor] = None
        self.view_index = 0
        self.views = [
            ("Chase", carla.Transform(carla.Location(x=-6.0, z=2.6), carla.Rotation(pitch=-8.0))),
            ("Platoon", carla.Transform(carla.Location(x=-14.0, z=6.2), carla.Rotation(pitch=-18.0))),
            ("Cockpit", carla.Transform(carla.Location(x=0.75, y=-0.3, z=1.25), carla.Rotation(pitch=-2.0))),
        ]

        bp = vehicle.get_world().get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(width))
        bp.set_attribute("image_size_y", str(height))
        bp.set_attribute("fov", "90")

        self.sensor = vehicle.get_world().spawn_actor(bp, self.views[self.view_index][1], attach_to=vehicle)
        import weakref

        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CameraManager._on_image(weak_self, image))

    @staticmethod
    def _on_image(weak_self, image: carla.Image) -> None:
        self = weak_self()
        if not self:
            return
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))[:, :, :3][:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

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
        if self.surface:
            display.blit(self.surface, (0, 0))

    def destroy(self) -> None:
        if self.sensor and self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()


class MiniHUD:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        pygame.font.init()
        self.title = pygame.font.SysFont("consolas", 18, bold=True)
        self.font = pygame.font.SysFont("consolas", 14, bold=True)
        self.small = pygame.font.SysFont("consolas", 12)

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
    ) -> None:
        top = pygame.Surface((self.width, 38), pygame.SRCALPHA)
        top.fill((10, 14, 22, 215))
        display.blit(top, (0, 0))
        display.blit(self.title.render("CARLA AUTOPILOT PLATOON WINDOW", True, (245, 248, 255)), (16, 9))

        velocity = leader.get_velocity()
        speed = 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        live_npcs = len([v for v in vehicles if v and v.is_alive])
        live_walkers = len([w for w in walkers if w and w.is_alive])

        panel = pygame.Surface((330, 118), pygame.SRCALPHA)
        panel.fill((12, 18, 28, 220))
        pygame.draw.rect(panel, (65, 105, 155), panel.get_rect(), width=1, border_radius=6)
        x = self.width - 346
        y = 52
        display.blit(panel, (x, y))

        lines = [
            f"Leader: #{leader.id} | Speed: {speed:5.1f} km/h",
            f"Platoon: {len([p for p in platoon if p and p.is_alive])} vehicles",
            f"Traffic: {live_npcs} NPC vehicles | {live_walkers} pedestrians",
            f"Mode: {'AUTOPILOT' if autopilot else 'MANUAL PAUSE'} | Cam: {camera_name}",
            f"Time: {int(elapsed // 60):02d}:{int(elapsed % 60):02d}",
        ]
        for idx, line in enumerate(lines):
            color = (110, 255, 165) if idx == 3 and autopilot else (210, 225, 240)
            display.blit(self.small.render(line, True, color), (x + 14, y + 12 + idx * 19))

        help_bar = pygame.Surface((470, 32), pygame.SRCALPHA)
        help_bar.fill((10, 14, 22, 210))
        bx = (self.width - 470) // 2
        by = self.height - 46
        display.blit(help_bar, (bx, by))
        text = self.font.render("[P] Toggle leader autopilot  |  [C] Camera  |  [ESC/Q] Exit", True, (220, 235, 255))
        display.blit(text, (bx + (470 - text.get_width()) // 2, by + 8))


def run(args: argparse.Namespace) -> None:
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

    try:
        world = client.get_world()
        current_town = world.get_map().name.split("/")[-1]
        if args.town and args.town.lower() not in ("auto", "current") and args.town != current_town:
            logger.info("Loading %s ...", args.town)
            client.set_timeout(60.0)
            world = client.load_world(args.town)
            client.set_timeout(30.0)
            time.sleep(2.0)

        if args.clean:
            cleanup_existing_actors(world)

        tm = client.get_trafficmanager(TM_PORT)
        tm.set_global_distance_to_leading_vehicle(3.5)
        tm.global_percentage_speed_difference(0.0)

        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No vehicle spawn points found in this map.")

        platoon = spawn_platoon(world, tm, spawn_points, args.platoon_size)
        npcs = spawn_npc_vehicles(client, world, tm, spawn_points, args.vehicles, platoon)
        walkers, controllers = spawn_pedestrians(world, args.pedestrians)

        for _ in range(4):
            try:
                world.wait_for_tick(seconds=2.0)
            except Exception:
                pass

        leader = platoon[0]
        camera = CameraManager(leader, WINDOW_WIDTH, WINDOW_HEIGHT)
        hud = MiniHUD(WINDOW_WIDTH, WINDOW_HEIGHT)
        autopilot = True
        started = time.time()
        running = True

        logger.info("Window running. Press P for leader autopilot toggle, C for camera, ESC/Q to exit.")
        while running:
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
                        leader.set_autopilot(autopilot, TM_PORT)
                        logger.info("Leader autopilot: %s", "ON" if autopilot else "OFF")
                    elif event.key == K_c and camera:
                        logger.info("Camera: %s", camera.toggle())

            display.fill((8, 8, 10))
            if camera:
                camera.render(display)
            hud.render(display, leader, platoon, npcs, walkers, autopilot, camera.name if camera else "None", time.time() - started)
            pygame.display.flip()

    finally:
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
        pygame.quit()
        if server_process and args.kill_launched_server:
            server_process.terminate()
        logger.info("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spawn a CARLA autopilot platoon in a Pygame window.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town01", help="Town01, Town02, ... or auto/current")
    parser.add_argument("--vehicles", type=int, default=32)
    parser.add_argument("--pedestrians", type=int, default=20)
    parser.add_argument("--platoon-size", type=int, default=4)
    parser.add_argument("--wait-seconds", type=int, default=60)
    parser.add_argument("--quality", default="Low", choices=["Low", "Epic"])
    parser.add_argument("--launch-server", action="store_true", help="Launch WindowsNoEditor/CarlaUE4.exe if no server is running.")
    parser.add_argument("--kill-launched-server", action="store_true", help="Close CARLA on exit, only when launched by this script.")
    parser.add_argument("--duration", type=float, default=0.0, help="Auto-exit after N seconds; 0 means run until closed.")
    parser.add_argument("--no-clean", dest="clean", action="store_false", help="Do not remove old actors before spawning.")
    parser.set_defaults(clean=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
