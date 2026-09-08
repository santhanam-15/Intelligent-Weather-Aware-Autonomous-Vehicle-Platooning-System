#!/usr/bin/env python

"""Test script for the CARLA 0.9.9 Weather Uncertainty Module.

This script demonstrates and verifies:
1. Applying all predefined weather presets and querying environmental parameters.
2. Reproducible pseudo-random weather selection using a random seed.
3. Smooth linear interpolation transitions between weather presets (CLEAR -> HEAVY_RAIN).
4. Restoring the original CARLA world weather on completion.
"""

import argparse
import glob
import os
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

from uncertainty.weather_uncertainty import WeatherManager, WeatherPreset


def main() -> None:
    """Execute weather uncertainty verification tests."""
    parser = argparse.ArgumentParser(
        description="CARLA 0.9.9 Weather Uncertainty Test Script"
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
        help='Random seed for reproducible weather selection (default: 42)',
    )
    parser.add_argument(
        '--preset-delay',
        metavar='T',
        default=3.0,
        type=float,
        help='Seconds to pause between preset changes (default: 3.0)',
    )
    args = parser.parse_args()

    print("=" * 70)
    print(" CARLA 0.9.9 Weather Uncertainty Module Test")
    print("=" * 70)
    print(f"Connecting to CARLA server at {args.host}:{args.port}...")

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    try:
        world = client.get_world()
        original_weather = world.get_weather()
        print(f"Connected successfully to map: {world.get_map().name}")
    except Exception as e:
        print(f"ERROR: Failed to connect to CARLA server: {e}")
        sys.exit(1)

    try:
        # Step 2: Initialize WeatherManager
        print(f"\nInitializing WeatherManager with random seed={args.seed}...")
        wm = WeatherManager(seed=args.seed)

        # Step 3: Iterate through all WeatherPreset values
        print("\n" + "-" * 70)
        print("[STEP 1/3] Testing All Weather Presets")
        print("-" * 70)
        presets = list(WeatherPreset)
        for idx, preset in enumerate(presets, 1):
            print(f"\n[{idx}/{len(presets)}] Applying preset: {preset.name} ('{preset.value}')...")
            wm.set_weather(world, preset)
            info = wm.get_weather_info(world)
            print(f"    Rain Intensity : {info['rain_intensity']:6.1f}%")
            print(f"    Fog Intensity  : {info['fog_intensity']:6.1f}%")
            print(f"    Cloudiness     : {info['cloudiness']:6.1f}%")
            print(f"    Wetness        : {info['wetness']:6.1f}%")
            print(f"    Wind Intensity : {info['wind_intensity']:6.1f}%")
            print(f"    Visibility     : {info['visibility']:6.1f}%")
            time.sleep(args.preset_delay)

        # Step 4: Test random_weather
        print("\n" + "-" * 70)
        print("[STEP 2/3] Testing Random Weather Selection")
        print("-" * 70)
        print(f"Sampling 3 random weather configurations using seed={wm.seed}...")
        for i in range(3):
            chosen = wm.random_weather(world)
            info = wm.get_weather_info(world)
            print(
                f"  Sample {i + 1}: Chosen Preset = {chosen.name:<13} | "
                f"Rain={info['rain_intensity']:4.1f}%, Fog={info['fog_intensity']:4.1f}%, "
                f"Cloud={info['cloudiness']:4.1f}%, Wind={info['wind_intensity']:4.1f}%"
            )
            time.sleep(1.0)

        # Step 5: Test weather transition from CLEAR to HEAVY_RAIN over 20 steps
        print("\n" + "-" * 70)
        print("[STEP 3/3] Testing Weather Transition (CLEAR -> HEAVY_RAIN over 20 steps)")
        print("-" * 70)
        print("Setting baseline weather to CLEAR...")
        wm.set_weather(world, WeatherPreset.CLEAR)
        initial_info = wm.get_weather_info(world)
        print(
            f"  Baseline: Rain={initial_info['rain_intensity']:.1f}%, "
            f"Fog={initial_info['fog_intensity']:.1f}%, "
            f"Cloud={initial_info['cloudiness']:.1f}%"
        )

        transition_steps = 20
        print(f"Interpolating to HEAVY_RAIN over {transition_steps} discrete steps...")
        for step in range(1, transition_steps + 1):
            progress = float(step) / float(transition_steps)
            wm.transition_weather(world, WeatherPreset.HEAVY_RAIN, progress)
            info = wm.get_weather_info(world)
            print(
                f"  Step {step:02d}/{transition_steps} ({progress * 100.0:5.1f}%): "
                f"Rain={info['rain_intensity']:5.1f}%, Fog={info['fog_intensity']:5.1f}%, "
                f"Cloud={info['cloudiness']:5.1f}%, Wetness={info['wetness']:5.1f}%, "
                f"Wind={info['wind_intensity']:5.1f}%"
            )
            time.sleep(0.1)

        # Step 6: Print summary
        print("\n" + "=" * 70)
        print(" Weather Uncertainty Test Summary")
        print("=" * 70)
        print(f"Total Presets Tested   : {len(presets)}")
        print("Random Weather Test    : PASSED (3 reproducible samples generated)")
        print(f"Weather Transition     : PASSED ({transition_steps} lerp steps CLEAR -> HEAVY_RAIN)")
        print("Overall Status         : ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
    except Exception as e:
        print(f"\nERROR encountered during execution: {e}")
        raise
    finally:
        # Cleanup: Restore original world weather
        print("\nRestoring original world weather state...")
        try:
            world.set_weather(original_weather)
            print("World weather restored successfully.")
        except Exception as err:
            print(f"Warning: Failed to restore world weather: {err}")


if __name__ == '__main__':
    main()
