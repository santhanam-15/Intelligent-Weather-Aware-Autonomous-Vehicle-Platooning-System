#!/usr/bin/env python

"""
Test script for the CARLA 0.9.9 Human Driver Behavior module.

Demonstrates:
  1. Connecting to CARLA simulator and initializing Traffic Manager.
  2. Spawning 10 NPC vehicles using map spawn points.
  3. Setting all NPC vehicles to autopilot via Traffic Manager.
  4. Initializing HumanBehaviorManager with a reproducible seed.
  5. Assigning diverse behavioral profiles (NORMAL, AGGRESSIVE, CONSERVATIVE,
     DISTRACTED, ERRATIC, SLOW) across all vehicles.
  6. Printing each vehicle's assigned behavioral profile.
  7. Observing driver profiles in motion for 5 seconds.
  8. Triggering sudden braking on a specific vehicle.
  9. Triggering sudden acceleration on a second vehicle.
  10. Triggering a forced lane change on a third vehicle.
  11. Observing maneuvers for 5 additional seconds.
  12. Gracefully cleaning up and destroying all spawned vehicles.
"""

import argparse
import glob
import logging
import os
import random
import sys
import time
from typing import List, Optional

# Set up CARLA Python API egg
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# Add parent directory to path to import uncertainty package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import carla
from uncertainty.human_behavior import HumanBehaviorManager, DriverProfile

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)
logger = logging.getLogger('test_human_behavior')


def spawn_npc_vehicles(world: carla.World, count: int = 10, seed: int = 42) -> List[carla.Vehicle]:
    """Spawn specified number of NPC vehicles at map spawn points."""
    rng = random.Random(seed)
    blueprint_library = world.get_blueprint_library()
    vehicle_blueprints = list(blueprint_library.filter('vehicle.*'))
    
    # Exclude 2-wheelers if possible to ensure stable autonomous driving
    car_blueprints = [
        bp for bp in vehicle_blueprints
        if int(bp.get_attribute('number_of_wheels')) >= 4
    ]
    if not car_blueprints:
        car_blueprints = vehicle_blueprints

    spawn_points = world.get_map().get_spawn_points()
    rng.shuffle(spawn_points)

    spawned_vehicles: List[carla.Vehicle] = []
    for sp in spawn_points:
        if len(spawned_vehicles) >= count:
            break
        bp = rng.choice(car_blueprints)
        if bp.has_attribute('color'):
            color = rng.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        
        actor = world.try_spawn_actor(bp, sp)
        if actor is not None:
            spawned_vehicles.append(actor)

    return spawned_vehicles


def main():
    argparser = argparse.ArgumentParser(description="Test Human Behavior Module for CARLA 0.9.9")
    argparser.add_argument('--host', metavar='H', default='127.0.0.1',
                           help='IP of the CARLA host server (default: 127.0.0.1)')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int,
                           help='TCP port of CARLA host server (default: 2000)')
    argparser.add_argument('--tm-port', metavar='TM_P', default=8000, type=int,
                           help='Port to communicate with Traffic Manager (default: 8000)')
    argparser.add_argument('-n', '--num-vehicles', default=10, type=int,
                           help='Number of NPC vehicles to spawn (default: 10)')
    argparser.add_argument('--seed', metavar='S', default=42, type=int,
                           help='Random seed for reproducibility (default: 42)')
    args = argparser.parse_args()

    client: Optional[carla.Client] = None
    vehicles: List[carla.Vehicle] = []
    behavior_manager: Optional[HumanBehaviorManager] = None

    try:
        # Step 1: Connect to CARLA, get Traffic Manager
        print("=" * 60)
        print("CARLA 0.9.9 Human Behavior Module Test")
        print("Connecting to CARLA at %s:%d..." % (args.host, args.port))
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)

        world = client.get_world()
        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        print("Connected! Map: %s, TM Port: %d" % (world.get_map().name, args.tm_port))

        # Step 2: Spawn 10 NPC vehicles using spawn points
        print("\nStep 2: Spawning %d NPC vehicles..." % args.num_vehicles)
        vehicles = spawn_npc_vehicles(world, count=args.num_vehicles, seed=args.seed)
        print("Successfully spawned %d / %d NPC vehicles." % (len(vehicles), args.num_vehicles))

        if not vehicles:
            print("ERROR: Could not spawn any NPC vehicles. Aborting test.")
            return

        # Step 3: Set all to autopilot
        print("\nStep 3: Enabling autopilot via Traffic Manager on all vehicles...")
        tm_port = traffic_manager.get_port()
        for veh in vehicles:
            veh.set_autopilot(True, tm_port)
        print("Autopilot enabled for %d vehicles." % len(vehicles))

        # Step 4: Create HumanBehaviorManager(seed=42)
        print("\nStep 4: Initializing HumanBehaviorManager(seed=%d)..." % args.seed)
        behavior_manager = HumanBehaviorManager(seed=args.seed)
        print("HumanBehaviorManager initialized.")

        # Step 5: Assign random profiles to all vehicles
        print("\nStep 5: Assigning random behavioral profiles to all vehicles...")
        assigned_profiles = behavior_manager.assign_random_profiles(
            vehicles=vehicles,
            traffic_manager=traffic_manager
        )

        # Step 6: Print each vehicle's assigned profile
        print("\nStep 6: Assigned Driver Profiles Summary:")
        print("-" * 55)
        print(" %-12s | %-20s | %-12s" % ("Vehicle ID", "Type ID", "Profile"))
        print("-" * 55)
        for veh in vehicles:
            prof: Optional[DriverProfile] = assigned_profiles.get(veh.id)
            prof_name = prof.name if prof else "UNKNOWN"
            print(" %-12d | %-20s | %-12s" % (veh.id, veh.type_id[:20], prof_name))
        print("-" * 55)

        # Step 7: Wait 5 seconds to observe different behaviors
        print("\nStep 7: Observing autonomous behavior for 5 seconds...")
        time.sleep(5.0)

        # Step 8: Trigger sudden_brake on one vehicle, print which one
        if len(vehicles) >= 1:
            target_brake = vehicles[0]
            print("\nStep 8: Triggering SUDDEN BRAKE on Vehicle ID %d (%s)..." % (
                target_brake.id, target_brake.type_id
            ))
            behavior_manager.trigger_sudden_brake(target_brake)
            print("  Full emergency brake applied (brake=1.0) to vehicle %d." % target_brake.id)

        # Step 9: Trigger sudden_acceleration on another
        if len(vehicles) >= 2:
            target_accel = vehicles[1]
            print("\nStep 9: Triggering SUDDEN ACCELERATION on Vehicle ID %d (%s)..." % (
                target_accel.id, target_accel.type_id
            ))
            behavior_manager.trigger_sudden_acceleration(target_accel)
            print("  Full throttle applied (throttle=1.0) to vehicle %d." % target_accel.id)

        # Step 10: Trigger lane_change on another
        if len(vehicles) >= 3:
            target_lc = vehicles[2]
            print("\nStep 10: Triggering FORCED LANE CHANGE on Vehicle ID %d (%s)..." % (
                target_lc.id, target_lc.type_id
            ))
            direction = behavior_manager.trigger_lane_change(target_lc, traffic_manager)
            direction_str = "RIGHT" if direction else "LEFT"
            print("  Forced lane change towards %s applied to vehicle %d." % (
                direction_str, target_lc.id
            ))

        # Step 11: Wait 5 seconds more
        print("\nStep 11: Observing vehicles post-maneuvers for 5 seconds...")
        time.sleep(5.0)

        print("\nHuman behavior module test completed successfully!")

    except Exception as exc:
        logger.error("Error occurred during human behavior test: %s", exc, exc_info=True)

    finally:
        # Step 12: Cleanup: destroy all vehicles
        print("\nStep 12: Cleaning up and destroying all spawned vehicles...")
        if vehicles and client:
            client.apply_batch([carla.command.DestroyActor(v) for v in vehicles])
            print("Successfully destroyed %d NPC vehicles." % len(vehicles))
        print("Test script finished.")


if __name__ == '__main__':
    main()
