import glob
import os
import sys

# Setup CARLA egg path
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    egg_path = os.path.abspath('carla/dist/carla-0.9.9-py3.7-win-amd64.egg')
    sys.path.append(egg_path)

import carla
import random
import time

def main():
    print("Starting CARLA verification script...")
    client = None
    vehicle = None
    original_settings = None

    try:
        # Connect to CARLA
        print("Connecting to CARLA server at localhost:2000...")
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(10.0)

        world = client.get_world()
        
        server_version = client.get_server_version()
        map_name = world.get_map().name
        print(f"Connected! Server version: {server_version}")
        print(f"Loaded map: {map_name}")

        # Set synchronous mode
        print("Configuring synchronous mode...")
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        # Get blueprints and spawn points
        print("Getting blueprints and spawn points...")
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            print("FAILURE: No spawn points found on the map.")
            return
            
        spawn_point = random.choice(spawn_points)

        # Spawn vehicle
        print(f"Spawning vehicle at {spawn_point.location}...")
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print(f"Vehicle spawned successfully. Actor ID: {vehicle.id}")

        # Tick 10 frames
        print("Ticking 10 frames...")
        for i in range(10):
            world.tick()
            location = vehicle.get_location()
            print(f"  Frame {i+1}: Vehicle location: x={location.x:.2f}, y={location.y:.2f}, z={location.z:.2f}")
            
        print("\nSUCCESS: CARLA verification completed without errors!")

    except Exception as e:
        print(f"\nFAILURE: An error occurred during verification: {e}")
        
    finally:
        print("\nCleaning up...")
        if vehicle is not None:
            print("Destroying vehicle...")
            vehicle.destroy()
            
        if client and original_settings:
            print("Restoring asynchronous mode...")
            world = client.get_world()
            world.apply_settings(original_settings)
            
        print("Done.")

if __name__ == '__main__':
    main()
