#!/usr/bin/env python

"""
Test script for the CARLA 0.9.9 Accident and Road Obstacle module.

Demonstrates:
  1. Connecting to CARLA simulator.
  2. Initializing AccidentManager with reproducible seed.
  3. Testing spawn_stopped_vehicle and inspecting spawned vehicle details.
  4. Testing spawn_accident with 3 vehicles (multi-vehicle collision).
  5. Testing block_lane with 'partial' blockage.
  6. Testing block_lane with 'full' blockage.
  7. Inspecting all active accident events via get_active_events().
  8. Graceful cleanup of all spawned actors via accident_manager.cleanup().
"""

import argparse
import glob
import logging
import os
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
from uncertainty.accident_manager import AccidentManager, AccidentType, AccidentEvent

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)
logger = logging.getLogger('test_accident')


def print_active_events(accident_manager: AccidentManager) -> None:
    """Print all currently active accident events and their details."""
    events: List[AccidentEvent] = accident_manager.get_active_events()
    print("\n================ Active Accident Events (Total: %d) ================" % len(events))
    if not events:
        print("  No active accident events.")
    for idx, ev in enumerate(events, 1):
        loc = ev.location.location if hasattr(ev.location, 'location') else ev.location
        loc_str = "x=%.1f, y=%.1f, z=%.1f" % (loc.x, loc.y, loc.z)
        actor_ids = [a.id for a in ev.vehicles if getattr(a, 'is_alive', True)]
        print("  [%d] Type: %s | Severity: %s | Start: %.1fs | Duration: %.1fs" % (
            idx, ev.accident_type.name, ev.severity, ev.start_time, ev.duration
        ))
        print("      Location: (%s) | Active Actor IDs (%d): %s" % (
            loc_str, len(actor_ids), actor_ids
        ))
    print("====================================================================\n")


def main():
    argparser = argparse.ArgumentParser(description="Test Accident and Obstacle Module for CARLA 0.9.9")
    argparser.add_argument('--host', metavar='H', default='127.0.0.1',
                           help='IP of the CARLA host server (default: 127.0.0.1)')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int,
                           help='TCP port of CARLA host server (default: 2000)')
    argparser.add_argument('--seed', metavar='S', default=42, type=int,
                           help='Random seed for reproducibility (default: 42)')
    args = argparser.parse_args()

    client: Optional[carla.Client] = None
    accident_manager: Optional[AccidentManager] = None

    try:
        # Step 1: Connect to CARLA
        print("=" * 60)
        print("CARLA 0.9.9 Accident & Road Obstacle Test")
        print("Connecting to CARLA at %s:%d..." % (args.host, args.port))
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)

        world = client.get_world()
        print("Connected to CARLA world: %s" % world.get_map().name)

        # Step 2: Create AccidentManager(seed=42)
        print("\nStep 2: Initializing AccidentManager(seed=%d)..." % args.seed)
        accident_manager = AccidentManager(seed=args.seed)
        print("AccidentManager initialized successfully.")

        # Step 3: Test spawn_stopped_vehicle, print location and info
        print("\nStep 3: Testing spawn_stopped_vehicle()...")
        stopped_event = accident_manager.spawn_stopped_vehicle(
            world=world,
            duration=60.0,
            severity="low"
        )
        if stopped_event:
            loc = stopped_event.location.location
            v_ids = [v.id for v in stopped_event.vehicles]
            print("  [SUCCESS] Spawned STOPPED_VEHICLE:")
            print("    Actor IDs : %s" % v_ids)
            print("    Location  : (x=%.2f, y=%.2f, z=%.2f, yaw=%.1f deg)" % (
                loc.x, loc.y, loc.z, stopped_event.location.rotation.yaw
            ))
            print("    Severity  : %s, Duration: %.1fs" % (stopped_event.severity, stopped_event.duration))
        else:
            print("  [WARNING] Failed to spawn stopped vehicle.")

        # Step 4: Wait 3 seconds
        print("\nStep 4: Waiting 3 seconds...")
        time.sleep(3.0)

        # Step 5: Test spawn_accident with 3 vehicles
        print("\nStep 5: Testing spawn_accident(num_vehicles=3)...")
        multi_event = accident_manager.spawn_accident(
            world=world,
            num_vehicles=3,
            duration=60.0
        )
        if multi_event:
            loc = multi_event.location.location
            v_ids = [v.id for v in multi_event.vehicles]
            print("  [SUCCESS] Spawned MULTI_VEHICLE_ACCIDENT:")
            print("    Actor IDs : %s (%d vehicles)" % (v_ids, len(v_ids)))
            print("    Location  : (x=%.2f, y=%.2f, z=%.2f)" % (loc.x, loc.y, loc.z))
            print("    Severity  : %s, Duration: %.1fs" % (multi_event.severity, multi_event.duration))
        else:
            print("  [WARNING] Failed to spawn multi-vehicle accident.")

        # Step 6: Wait 3 seconds
        print("\nStep 6: Waiting 3 seconds...")
        time.sleep(3.0)

        # Step 7: Test block_lane with 'partial' block
        print("\nStep 7: Testing block_lane(block_type='partial')...")
        partial_event = accident_manager.block_lane(
            world=world,
            block_type='partial',
            duration=60.0
        )
        if partial_event:
            loc = partial_event.location.location
            v_ids = [v.id for v in partial_event.vehicles]
            print("  [SUCCESS] Spawned PARTIAL_BLOCK:")
            print("    Actor IDs : %s" % v_ids)
            print("    Location  : (x=%.2f, y=%.2f, z=%.2f)" % (loc.x, loc.y, loc.z))
            print("    Severity  : %s, Duration: %.1fs" % (partial_event.severity, partial_event.duration))
        else:
            print("  [WARNING] Failed to spawn partial lane block.")

        # Step 8: Wait 3 seconds
        print("\nStep 8: Waiting 3 seconds...")
        time.sleep(3.0)

        # Step 9: Test block_lane with 'full' block
        print("\nStep 9: Testing block_lane(block_type='full')...")
        full_event = accident_manager.block_lane(
            world=world,
            block_type='full',
            duration=60.0
        )
        if full_event:
            loc = full_event.location.location
            v_ids = [v.id for v in full_event.vehicles]
            print("  [SUCCESS] Spawned FULL_BLOCK:")
            print("    Actor IDs : %s (%d vehicles)" % (v_ids, len(v_ids)))
            print("    Location  : (x=%.2f, y=%.2f, z=%.2f)" % (loc.x, loc.y, loc.z))
            print("    Severity  : %s, Duration: %.1fs" % (full_event.severity, full_event.duration))
        else:
            print("  [WARNING] Failed to spawn full lane block.")

        # Step 10: Print all active events
        print("\nStep 10: Printing all active accident events:")
        print_active_events(accident_manager)

        print("All accident scenarios created and verified.")

    except Exception as exc:
        logger.error("Error occurred during accident test: %s", exc, exc_info=True)

    finally:
        # Step 11: Cleanup all actors
        print("\nStep 11: Cleaning up all accident actors...")
        if accident_manager is not None:
            accident_manager.cleanup()
            print("AccidentManager cleanup() completed.")
        print("Test script finished.")


if __name__ == '__main__':
    main()
