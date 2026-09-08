import os
import sys
import glob
import time

# 1. Dynamically locate and import the CARLA .egg file
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

def main():
    actor_list = []
    try:
        # 2. Connect to the CARLA server
# 2. Connect to the CARLA server
client = carla.Client('localhost', 2000) 
client.set_timeout(10.0)
world = client.get_world()

# ... (synchronous mode setup) ...

# 4. Initialize Traffic Manager (Notice the corrected spelling and separate port)
traffic_manager = client.get_trafficmanager(8000)
traffic_manager.set_synchronous_mode(True)
        
        settings.fixed_delta_seconds = 0.05  # 20 FPS simulation steps
        world.apply_settings(settings)
        
        # 4. Initialize Traffic Manager
        traffic_manager = client.get_traffic_manager(8000)
        traffic_manager.set_synchronous_mode(True)
        
        # 5. Spawn the Audi E-Tron (from your original screenshot)
        blueprint_library = world.get_blueprint_library()
        bp = blueprint_library.find('vehicle.audi.etron')
        
        # Get a random valid spawn point on the map
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            print("No spawn points found!")
            return
        spawn_point = spawn_points[0]
        
        vehicle = world.spawn_actor(bp, spawn_point)
        actor_list.append(vehicle)
        print(f"Spawned {vehicle.type_id} successfully.")

        # 6. Configure Aggressive Driving & Overtaking behavior
        vehicle.set_autopilot(True, traffic_manager.get_port())
        
        # Negative percentage means exceed the speed limit (e.g., -30% means drive 30% faster than the limit)
        traffic_manager.global_percentage_speed_difference(-30.0) 
        
        # Allow the car to keep a tight 2.5-meter gap to force aggressive overtaking maneuvers
        traffic_manager.distance_to_leading_vehicle(vehicle, 2.5)
        
        # Enable autonomous lane changing for overtaking
        traffic_manager.auto_lane_change(vehicle, True)
        
        # 7. Simulation Loop
        print("Driving simulation started. Press Ctrl+C to exit.")
        while True:
            # Advance the simulator one tick
            world.tick()
            
            # (Optional) Print current speed to monitor performance
            v = vehicle.get_velocity()
            speed = 3.6 * (v.x**2 + v.y**2 + v.z**2)**0.5
            print(f"Current Speed: {speed:.2f} km/h", end="\r")
            
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
        
    finally:
        # 8. Clean up and restore original environment settings
        print("Cleaning up actors and settings...")
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        
        for actor in actor_list:
            actor.destroy()
        print("Done.")

if __name__ == '__main__':
    main()
