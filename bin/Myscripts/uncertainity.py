import glob
import os
import sys
import random
import time
import math
import logging

# ==============================================================================
# -- Find CARLA module robustly
# ==============================================================================
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    egg_paths = [
        os.path.join(script_dir, '../PythonAPI/carla/dist/carla-*%d.%d-%s.egg'),
        os.path.join(script_dir, '../../PythonAPI/carla/dist/carla-*%d.%d-%s.egg'),
        os.path.join(script_dir, '../carla/dist/carla-*%d.%d-%s.egg')
    ]
    for egg_pattern in egg_paths:
        matched = glob.glob(egg_pattern % (sys.version_info.major, sys.version_info.minor, 'win-amd64' if os.name == 'nt' else 'linux-x86_64'))
        if matched:
            sys.path.append(matched[0])
            break
except IndexError:
    pass

import carla

# ==============================================================================
# -- Localized Chaos Logic
# ==============================================================================
def get_ego_vehicle(world):
    """Finds the vehicle currently being driven by the manual_control script."""
    for actor in world.get_actors().filter('vehicle.*'):
        if actor.attributes.get('role_name') == 'hero':
            return actor
    return None

def apply_localized_chaos(world, tm, ego_vehicle, radius=70.0):
    """
    Finds all vehicles within a specific radius of the ego vehicle and 
    forces them to perform unpredictable, dangerous maneuvers.
    """
    ego_location = ego_vehicle.get_location()
    all_vehicles = world.get_actors().filter('vehicle.*')
    
    nearby_vehicles = []
    
    for v in all_vehicles:
        if v.id == ego_vehicle.id or not v.is_alive:
            continue
            
        # Calculate distance to ego vehicle
        v_loc = v.get_location()
        distance = math.sqrt((v_loc.x - ego_location.x)**2 + (v_loc.y - ego_location.y)**2 + (v_loc.z - ego_location.z)**2)
        
        if distance <= radius:
            nearby_vehicles.append((v, distance))
            
    if not nearby_vehicles:
        return

    logging.info(f"Detected {len(nearby_vehicles)} vehicles near you. Injecting chaos...")

    for v, dist in nearby_vehicles:
        behavior_roll = random.random()
        
        # 1. Sudden Hard Braking (20% chance for nearby cars)
        if behavior_roll < 0.20:
            logging.info(f"Vehicle {v.id} is BRAKING SUDDENLY!")
            # 80% slower than speed limit acts like a sudden brake
            tm.vehicle_percentage_speed_difference(v, 80.0) 
            
        # 2. Aggressive Swerve / Lane Change (30% chance)
        elif behavior_roll < 0.50:
            swerve_right = random.choice([True, False])
            logging.info(f"Vehicle {v.id} is SWERVING {'RIGHT' if swerve_right else 'LEFT'}!")
            tm.force_lane_change(v, swerve_right)
            # Speed up slightly during the swerve
            tm.vehicle_percentage_speed_difference(v, -20.0) 
            
        # 3. Extreme Tailgating (20% chance)
        elif behavior_roll < 0.70:
            tm.distance_to_leading_vehicle(v, 0.5) # Follow at 0.5 meters
            tm.vehicle_percentage_speed_difference(v, -30.0) # Speed up to catch the car ahead
            
        # 4. Return to Normal Behavior (30% chance to act as a baseline)
        else:
            tm.vehicle_percentage_speed_difference(v, 0.0)
            tm.distance_to_leading_vehicle(v, 2.5)

# ==============================================================================
# -- Main Loop
# ==============================================================================
def main():
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    
    try:
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(30.0)
        world = client.get_world()
        tm = client.get_trafficmanager(8000)
        
        # Global setting: Make cars outside the radius drive perfectly safe
        tm.global_percentage_speed_difference(0.0)
        tm.set_global_distance_to_leading_vehicle(2.5)
        
        logging.info("Waiting for manual_control.py vehicle (ego vehicle)...")
        ego_vehicle = None
        while ego_vehicle is None:
            ego_vehicle = get_ego_vehicle(world)
            time.sleep(1)
            
        logging.info(f"Found ego vehicle (ID: {ego_vehicle.id}). Starting local uncertainty loop.")
        
        while True:
            # Re-check ego vehicle in case it was destroyed and respawned
            if not ego_vehicle.is_alive:
                ego_vehicle = get_ego_vehicle(world)
                
            if ego_vehicle:
                # Apply chaos only to vehicles within 70 meters
                apply_localized_chaos(world, tm, ego_vehicle, radius=70.0)
            
            # Pulse the chaos every 3 seconds so you have time to react
            time.sleep(3) 

    except KeyboardInterrupt:
        logging.info("Localized uncertainty script stopped.")
    except Exception as e:
        logging.error(f"Error: {e}")

if __name__ == '__main__':
    main()