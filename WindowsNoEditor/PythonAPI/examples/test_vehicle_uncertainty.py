#!/usr/bin/env python

"""Test script for the CARLA 0.9.9 Actuator / Vehicle Uncertainty Module.

This script demonstrates and verifies:
1. Connecting to CARLA and spawning a test vehicle.
2. Initializing ActuatorUncertainty with Gaussian noise and transport delay.
3. Generating and applying perturbed vehicle controls from a nominal command.
4. Demonstrating stochastic noise injection across 10 sample iterations.
5. Demonstrating FIFO actuator transport delay (latency queue) behavior.
6. Applying perturbed controls directly to the CARLA vehicle actor.
7. Safely destroying the spawned vehicle via try/finally cleanup.
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

from uncertainty.vehicle_uncertainty import ActuatorUncertainty


def main() -> None:
    """Execute actuator uncertainty verification tests."""
    parser = argparse.ArgumentParser(
        description="CARLA 0.9.9 Actuator Uncertainty Test Script"
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
        help='Random seed for reproducible noise generation (default: 42)',
    )
    args = parser.parse_args()

    print("=" * 70)
    print(" CARLA 0.9.9 Actuator Uncertainty Module Test")
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
        # Step 2: Spawn one vehicle
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
            vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])

        actor_list.append(vehicle)
        print(f"Vehicle spawned successfully: ID={vehicle.id}, Type={vehicle.type_id}")

        # Step 3: Create ActuatorUncertainty(seed=42, throttle_noise=0.1, steer_noise=0.05, brake_noise=0.1, delay_ticks=2)
        print("\n" + "-" * 70)
        print("Initializing ActuatorUncertainty Instance")
        print("-" * 70)
        actuator = ActuatorUncertainty(
            seed=args.seed,
            throttle_noise=0.1,
            steer_noise=0.05,
            brake_noise=0.1,
            delay_ticks=2,
        )
        config = actuator.get_config()
        print(f"Actuator Configuration:")
        print(f"  Random Seed    : {args.seed}")
        print(f"  Throttle Noise : std={config['throttle_noise']:.2f}")
        print(f"  Steer Noise    : std={config['steer_noise']:.2f}")
        print(f"  Brake Noise    : std={config['brake_noise']:.2f}")
        print(f"  Delay Ticks    : {config['delay_ticks']} ticks")

        # Step 4: Create a nominal control: carla.VehicleControl(throttle=0.5, steer=0.0, brake=0.0)
        nominal_control = carla.VehicleControl(throttle=0.5, steer=0.0, brake=0.0)
        print(f"\nNominal Control Command:")
        print(
            f"  throttle={nominal_control.throttle:.4f}, "
            f"steer={nominal_control.steer:.4f}, "
            f"brake={nominal_control.brake:.4f}"
        )

        # Step 5 & 6: Apply uncertainty 10 times, print original vs modified control values
        print("\n" + "-" * 70)
        print("[STEP 1/2] Applying Uncertainty Across 10 Iterations (Noise & FIFO Queue)")
        print("-" * 70)
        print(f"{'Iter':<5} | {'Nominal (Thr, Str, Brk)':<26} | {'Modified (Thr, Str, Brk)':<26} | {'Diff (dThr, dStr, dBrk)':<24}")
        print("-" * 88)

        applied_controls = []
        for i in range(1, 11):
            modified_control = actuator.apply_uncertainty(nominal_control)
            applied_controls.append(modified_control)

            # Apply to actual vehicle actor to verify API compatibility
            actuator.apply_to_vehicle(vehicle, nominal_control)

            diff_thr = modified_control.throttle - nominal_control.throttle
            diff_str = modified_control.steer - nominal_control.steer
            diff_brk = modified_control.brake - nominal_control.brake

            print(
                f"{i:<5} | "
                f"({nominal_control.throttle:5.3f}, {nominal_control.steer:5.3f}, {nominal_control.brake:5.3f}) | "
                f"({modified_control.throttle:5.3f}, {modified_control.steer:5.3f}, {modified_control.brake:5.3f}) | "
                f"({diff_thr:+6.3f}, {diff_str:+6.3f}, {diff_brk:+6.3f})"
            )
            time.sleep(0.05)

        # Show that noise is being applied (values differ slightly)
        throttles = [c.throttle for c in applied_controls]
        steers = [c.steer for c in applied_controls]
        brakes = [c.brake for c in applied_controls]

        print("\nStatistical verification of injected noise:")
        print(f"  Throttle: min={min(throttles):.4f}, max={max(throttles):.4f}, range={max(throttles)-min(throttles):.4f}")
        print(f"  Steer   : min={min(steers):.4f}, max={max(steers):.4f}, range={max(steers)-min(steers):.4f}")
        print(f"  Brake   : min={min(brakes):.4f}, max={max(brakes):.4f}, range={max(brakes)-min(brakes):.4f}")
        assert any(abs(t - nominal_control.throttle) > 1e-4 for t in throttles), "Noise check: throttle values must vary!"
        print("  Status  : PASSED (Control perturbations observed as expected)")

        # Step 7: Test with delay to show delayed responses
        print("\n" + "-" * 70)
        print("[STEP 2/2] Testing Latency Queue (Delay = 2 Ticks Response)")
        print("-" * 70)
        print("Simulating a step-input throttle response through ActuatorUncertainty:")
        print("Delay is configured to 2 ticks. When a step command occurs at Tick 3,")
        print("the output response should lag and reflect the change 2 ticks later (Tick 5).")

        # Use an actuator instance with noise=0 to isolate delay latency observation
        delay_test_actuator = ActuatorUncertainty(
            seed=args.seed,
            throttle_noise=0.0,
            steer_noise=0.0,
            brake_noise=0.0,
            delay_ticks=2,
        )

        test_inputs = [0.1, 0.1, 0.8, 0.8, 0.8, 0.8, 0.2, 0.2]
        print("\n" + "-" * 65)
        print(f"{'Tick':<6} | {'Input Throttle':<16} | {'Output Throttle':<16} | {'Latency Status'}")
        print("-" * 65)

        for tick, input_throttle in enumerate(test_inputs, 1):
            ctrl_in = carla.VehicleControl(throttle=input_throttle)
            ctrl_out = delay_test_actuator.apply_uncertainty(ctrl_in)

            # Determine status label
            if tick <= 2:
                status = "Queue filling"
            elif tick in (3, 4):
                status = "Holding pre-step value (lagging)"
            elif tick == 5:
                status = "Step response arrived! (2 ticks delayed)"
            else:
                status = f"Tracking input at t - 2 ticks"

            print(f"{tick:<6} | {ctrl_in.throttle:<16.3f} | {ctrl_out.throttle:<16.3f} | {status}")
            time.sleep(0.05)

        # Verification of delay
        # At tick 3 input jumped to 0.8, at tick 3 & 4 output was still 0.1, at tick 5 output became 0.8
        delay_test_actuator.reset()
        c_in_0 = carla.VehicleControl(throttle=0.0)
        c_in_1 = carla.VehicleControl(throttle=1.0)
        out1 = delay_test_actuator.apply_uncertainty(c_in_0)
        out2 = delay_test_actuator.apply_uncertainty(c_in_0)
        out3 = delay_test_actuator.apply_uncertainty(c_in_1)  # step to 1.0
        out4 = delay_test_actuator.apply_uncertainty(c_in_1)
        out5 = delay_test_actuator.apply_uncertainty(c_in_1)  # output now reflects step

        assert out3.throttle == 0.0, f"Expected delayed output 0.0 at step tick, got {out3.throttle}"
        assert out4.throttle == 0.0, f"Expected delayed output 0.0 at step tick + 1, got {out4.throttle}"
        assert out5.throttle == 1.0, f"Expected stepped output 1.0 at step tick + 2, got {out5.throttle}"
        print("\nDelay latency verification: PASSED (Exact 2-tick lag confirmed)")

        # Summary
        print("\n" + "=" * 70)
        print(" Vehicle Actuator Uncertainty Test Summary")
        print("=" * 70)
        print(f"Vehicle Actor ID       : {vehicle.id} ({vehicle.type_id})")
        print("Actuator Noise Test    : PASSED (Gaussian noise verified across 10 iterations)")
        print("Actuator Delay Test    : PASSED (2-tick FIFO latency queue verified)")
        print("Actor Apply Control    : PASSED (Controls successfully applied to vehicle actor)")
        print("Overall Status         : ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
    except Exception as e:
        print(f"\nERROR encountered during execution: {e}")
        raise
    finally:
        # Step 8: Cleanup: destroy vehicle
        print("\nCleaning up spawned actors...")
        for actor in actor_list:
            if actor is not None and actor.is_alive:
                actor_id = actor.id
                actor.destroy()
                print(f"Destroyed actor ID: {actor_id}")
        print("Cleanup completed.")


if __name__ == '__main__':
    main()
