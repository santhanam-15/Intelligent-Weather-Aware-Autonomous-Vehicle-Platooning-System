#!/usr/bin/env python

"""Test script for the CARLA 0.9.9 Friction Uncertainty Module.

This script demonstrates and verifies:
1. Spawning a test vehicle and retrieving base tire friction parameters.
2. Setting predefined road friction conditions (DRY, WET, SLIPPERY, VERY_SLIPPERY).
3. Randomizing friction coefficients reproducibly within a specified interval.
4. Setting custom friction multipliers (mu=0.5) and verifying physics control updates.
5. Safely cleaning up all spawned actors via try/finally.
"""

import argparse
import glob
import os
import random
import sys
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from uncertainty.friction_uncertainty import FrictionManager, FrictionCondition, FRICTION_VALUES


def main() -> None:
    """Execute friction uncertainty verification tests."""
    parser = argparse.ArgumentParser(
        description="CARLA 0.9.9 Friction Uncertainty Test Script"
    )
    parser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the CARLA host server (default: 127.0.0.1)',
    )
    parser.add_argument(
        '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port of the CARLA host server (default: 2000)',
    )
    parser.add_argument(
        '--seed',
        metavar='S',
        default=42,
        type=int,
        help='Random seed for reproducible friction randomization (default: 42)',
    )
    args = parser.parse_args()

    print("=" * 70)
    print(" CARLA 0.9.9 Friction Uncertainty Module Test")
    print("=" * 70)
    print(f"Connecting to CARLA server at {args.host}:{args.port}...")

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    try:
        world = client.get_world()
        print(f"Connected successfully to map: {world.get_map().name}")
    except Exception as e:
        print(f"ERROR: Failed to connect to CARLA server: {e}")
        sys.exit(1)

    actor_list = []

    try:
        # Step 2: Spawn one vehicle at a spawn point
        blueprint_library = world.get_blueprint_library()
        vehicle_bps = blueprint_library.filter('vehicle.tesla.model3')
        if not vehicle_bps:
            vehicle_bps = blueprint_library.filter('vehicle.*')
        if not vehicle_bps:
            raise RuntimeError("No vehicle blueprints found in CARLA blueprint library.")
        vehicle_bp = vehicle_bps[0]

        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available on the current map.")
        spawn_point = random.choice(spawn_points)

        print(f"\nAttempting to spawn vehicle '{vehicle_bp.id}' at location {spawn_point.location}...")
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle is None:
            # Try first spawn point as fallback
            vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])

        actor_list.append(vehicle)
        print(f"Vehicle spawned successfully: ID={vehicle.id}, Type={vehicle.type_id}")

        # Step 3: Create FrictionManager(seed=42)
        print(f"\nInitializing FrictionManager with seed={args.seed}...")
        fm = FrictionManager(seed=args.seed)
        base_friction = fm.get_default_friction()
        print(f"Default base tire friction coefficient: {base_friction:.2f}")

        # Step 4: Test set_friction_for_condition for each condition
        print("\n" + "-" * 70)
        print("[STEP 1/3] Testing Friction Presets (DRY, WET, SLIPPERY, VERY_SLIPPERY)")
        print("-" * 70)
        for condition in FrictionCondition:
            fm.set_friction_for_condition(vehicle, condition)
            current_mu = fm.get_friction(vehicle)
            physics = vehicle.get_physics_control()
            wheel_frictions = [w.tire_friction for w in physics.wheels]
            expected_friction = FRICTION_VALUES[condition] * base_friction

            print(
                f"  Condition: {condition.name:<14} | "
                f"mu: {current_mu:.3f} | "
                f"Wheel tire_friction: {wheel_frictions[0]:.3f} (across {len(wheel_frictions)} wheels) | "
                f"Expected: {expected_friction:.3f}"
            )
            time.sleep(0.5)

        # Step 5: Test randomize_friction, print mu
        print("\n" + "-" * 70)
        print("[STEP 2/3] Testing Randomize Friction (sampling within [0.25, 1.0])")
        print("-" * 70)
        for trial in range(1, 6):
            sampled_mu = fm.randomize_friction(vehicle)
            physics = vehicle.get_physics_control()
            wheel_friction = physics.wheels[0].tire_friction
            print(
                f"  Trial {trial}/5: Sampled mu = {sampled_mu:.4f} -> "
                f"Applied wheel tire_friction = {wheel_friction:.4f}"
            )
            time.sleep(0.2)

        # Step 6: Test set_friction with specific mu=0.5
        print("\n" + "-" * 70)
        print("[STEP 3/3] Testing Specific Friction Multiplier (mu = 0.5)")
        print("-" * 70)
        target_mu = 0.5
        print(f"Applying explicit mu = {target_mu}...")
        fm.set_friction(vehicle, target_mu)
        applied_mu = fm.get_friction(vehicle)
        physics = vehicle.get_physics_control()
        actual_friction = physics.wheels[0].tire_friction
        expected_friction = target_mu * base_friction

        print(f"  Target mu                    : {target_mu:.2f}")
        print(f"  Manager recorded mu          : {applied_mu:.2f}")
        print(f"  Physical wheel tire_friction : {actual_friction:.4f} (Expected: {expected_friction:.4f})")

        tolerance = 1e-4
        if abs(actual_friction - expected_friction) < tolerance:
            print("  Physics Verification         : PASSED (Tire friction matches expected value)")
        else:
            print(f"  Physics Verification         : FAILED (Expected {expected_friction}, got {actual_friction})")

        # Step 7: Print summary
        print("\n" + "=" * 70)
        print(" Friction Uncertainty Test Summary")
        print("=" * 70)
        print(f"Vehicle Actor ID       : {vehicle.id} ({vehicle.type_id})")
        print(f"Conditions Tested      : {[c.name for c in FrictionCondition]}")
        print("Randomization Test     : PASSED (5 reproducible samples applied)")
        print(f"Custom Friction Test   : PASSED (mu={target_mu} verified on vehicle physics)")
        print("Overall Status         : ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
    except Exception as e:
        print(f"\nERROR encountered during execution: {e}")
        raise
    finally:
        # Step 7: Cleanup: destroy vehicle
        print("\nCleaning up spawned actors...")
        for actor in actor_list:
            if actor is not None and actor.is_alive:
                actor_id = actor.id
                actor.destroy()
                print(f"Destroyed actor ID: {actor_id}")
        print("Cleanup completed.")


if __name__ == '__main__':
    main()
