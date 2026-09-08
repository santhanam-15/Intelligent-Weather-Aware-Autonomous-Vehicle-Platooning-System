#!/usr/bin/env python

"""
Test script for the CARLA 0.9.9 Traffic Jam uncertainty module.

Demonstrates:
  1. Connecting to CARLA and initializing Traffic Manager.
  2. Initializing TrafficJamManager with reproducible seed.
  3. Triggering a LIGHT jam (3 vehicles) and inspecting active jams.
  4. Triggering a HEAVY jam (5 vehicles) and inspecting active jams.
  5. Monitoring jam lifetime and expiring jams via update().
  6. Probabilistic jam triggering via maybe_trigger_jam().
  7. Graceful cleanup of spawned vehicles.
"""

import argparse
import glob
import logging
import os
import sys
import time
from typing import Optional

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
from uncertainty.traffic_uncertainty import TrafficJamManager, JamSeverity

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)
logger = logging.getLogger('test_traffic_jam')


def get_simulation_time(world: carla.World) -> float:
    """Retrieve current simulation timestamp in seconds from world snapshot."""
    try:
        return float(world.get_snapshot().timestamp.elapsed_seconds)
    except Exception:
        return time.time()


def print_active_jams(jam_manager: TrafficJamManager) -> None:
    """Print details of all currently active traffic jams."""
    active_jams = jam_manager.get_active_jams()
    print("\n--- Active Traffic Jams (Total: %d) ---" % len(active_jams))
    if not active_jams:
        print("  No active traffic jams.")
    for idx, jam in enumerate(active_jams, 1):
        sev = jam.get('severity')
        sev_name = sev.name if hasattr(sev, 'name') else str(sev)
        loc = jam.get('location')
        loc_str = "x=%.1f, y=%.1f, z=%.1f" % (loc.x, loc.y, loc.z) if loc else "N/A"
        vehicles = jam.get('vehicles', [])
        alive_vehicles = [v for v in vehicles if getattr(v, 'is_alive', True)]
        print("  [%d] Severity: %s | Start: %.2fs | Duration: %.2fs | Vehicles: %d (alive: %d) | Location: (%s)" % (
            idx, sev_name, jam.get('start_time', 0.0), jam.get('duration', 0.0),
            len(vehicles), len(alive_vehicles), loc_str
        ))
    print("----------------------------------------\n")


def main():
    argparser = argparse.ArgumentParser(description="Test Traffic Jam Module for CARLA 0.9.9")
    argparser.add_argument('--host', metavar='H', default='127.0.0.1',
                           help='IP of the CARLA host server (default: 127.0.0.1)')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int,
                           help='TCP port of CARLA host server (default: 2000)')
    argparser.add_argument('--tm-port', metavar='TM_P', default=8000, type=int,
                           help='Port to communicate with Traffic Manager (default: 8000)')
    argparser.add_argument('--seed', metavar='S', default=42, type=int,
                           help='Random seed for reproducibility (default: 42)')
    args = argparser.parse_args()

    client: Optional[carla.Client] = None
    jam_manager: Optional[TrafficJamManager] = None

    try:
        # Step 1: Connect to CARLA and get Traffic Manager
        print("=" * 60)
        print("CARLA 0.9.9 Traffic Jam Module Test")
        print("Connecting to CARLA at %s:%d..." % (args.host, args.port))
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)

        world = client.get_world()
        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        print("Connected! Map: %s, TM Port: %d" % (world.get_map().name, args.tm_port))

        # Step 2: Create TrafficJamManager with random seed
        print("\nStep 2: Initializing TrafficJamManager(seed=%d)..." % args.seed)
        jam_manager = TrafficJamManager(seed=args.seed)
        print("TrafficJamManager initialized successfully.")

        # Step 3: Trigger a LIGHT jam with 3 vehicles
        current_time = get_simulation_time(world)
        print("\nStep 3: Triggering LIGHT traffic jam (3 vehicles, duration=5.0s)...")
        light_jam = jam_manager.trigger_jam(
            world=world,
            traffic_manager=traffic_manager,
            severity=JamSeverity.LIGHT,
            duration=5.0,
            num_vehicles=3,
            start_time=current_time
        )
        print("Triggered LIGHT jam with %d vehicles at start time %.2fs" % (
            len(light_jam.get('vehicles', [])), light_jam.get('start_time', 0.0)
        ))
        print_active_jams(jam_manager)

        # Step 4: Wait a few seconds, trigger a HEAVY jam with 5 vehicles
        wait_sec = 2.0
        print("Waiting %.1f seconds before next event..." % wait_sec)
        time.sleep(wait_sec)

        current_time = get_simulation_time(world)
        print("\nStep 4: Triggering HEAVY traffic jam (5 vehicles, duration=6.0s)...")
        heavy_jam = jam_manager.trigger_jam(
            world=world,
            traffic_manager=traffic_manager,
            severity=JamSeverity.HEAVY,
            duration=6.0,
            num_vehicles=5,
            start_time=current_time
        )
        print("Triggered HEAVY jam with %d vehicles at start time %.2fs" % (
            len(heavy_jam.get('vehicles', [])), heavy_jam.get('start_time', 0.0)
        ))

        # Step 5: Print all active jams
        print("\nStep 5: Printing all active traffic jams:")
        print_active_jams(jam_manager)

        # Step 6: Wait for jam duration to expire, call update(), verify jams are cleaned up
        print("Step 6: Waiting for jams to expire (waiting 6.0 seconds)...")
        time.sleep(6.0)

        # Advance simulation time reading or current time
        current_time = get_simulation_time(world)
        # Ensure current_time has progressed beyond earliest jam expiration
        # If simulation clock runs slower or is paused, adjust time forward
        if current_time <= light_jam['start_time'] + light_jam['duration']:
            current_time = light_jam['start_time'] + light_jam['duration'] + 1.0

        print("Calling jam_manager.update(current_time=%.2fs)..." % current_time)
        expired = jam_manager.update(current_time)
        print("Expired jams removed: %d" % len(expired))
        print_active_jams(jam_manager)

        # Now advance further to expire all remaining jams
        current_time = heavy_jam['start_time'] + heavy_jam['duration'] + 1.0
        print("Expiring remaining jams at t=%.2fs..." % current_time)
        expired_remaining = jam_manager.update(current_time)
        print("Additional expired jams removed: %d" % len(expired_remaining))
        print_active_jams(jam_manager)
        active = jam_manager.get_active_jams()
        assert len(active) == 0, "Expected 0 active jams after expiration, got %d" % len(active)
        print("Verification PASSED: All expired jams successfully cleaned up.")

        # Step 7: Test maybe_trigger_jam with probability=0.5 a few times
        print("\nStep 7: Testing maybe_trigger_jam with probability=0.5 (5 trials)...")
        current_time = get_simulation_time(world)
        triggered_count = 0
        for trial in range(1, 6):
            cur_t = current_time + trial * 2.0
            jam = jam_manager.maybe_trigger_jam(
                world=world,
                traffic_manager=traffic_manager,
                current_time=cur_t,
                probability=0.5,
                duration=10.0,
                num_vehicles=3
            )
            if jam is not None:
                triggered_count += 1
                sev = jam['severity']
                sev_name = sev.name if hasattr(sev, 'name') else str(sev)
                print("  Trial %d: Jam TRIGGERED! Severity: %s, Vehicles: %d, Duration: %.1fs" % (
                    trial, sev_name, len(jam.get('vehicles', [])), jam.get('duration', 0.0)
                ))
            else:
                print("  Trial %d: No jam triggered (random roll exceeded probability)." % trial)

        print("maybe_trigger_jam test completed. Triggered %d / 5 times." % triggered_count)
        print_active_jams(jam_manager)

        print("\nAll traffic jam tests executed successfully!")

    except Exception as exc:
        logger.error("Error occurred during traffic jam test: %s", exc, exc_info=True)

    finally:
        # Step 8: Cleanup
        print("\nStep 8: Cleaning up actors...")
        if jam_manager is not None:
            jam_manager.cleanup()
            print("TrafficJamManager cleanup() completed.")
        print("Test script finished.")


if __name__ == '__main__':
    main()
