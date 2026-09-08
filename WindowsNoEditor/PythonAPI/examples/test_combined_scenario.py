#!/usr/bin/env python

"""
Test script for CARLA 0.9.9 Combined Uncertainty Scenario.

Comprehensive test that integrates:
  - UncertaintyManager top-level façade
  - Scenario levels (e.g. MEDIUM)
  - Weather variations (MODERATE_RAIN)
  - Tire/road friction changes (WET / reduced mu)
  - Actuator noise & delay
  - V2V communication channel (latency, packet loss, noise)
  - Traffic jams and accidents
  - Human driver behavioral profiles
  - Simulation loop ticking with world.wait_for_tick()
"""

import argparse
import glob
import logging
import os
import random
import sys
import time
from typing import Dict, List, Optional

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
from uncertainty.uncertainty_manager import UncertaintyManager
from uncertainty.scenario_manager import ScenarioLevel

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)
logger = logging.getLogger('test_combined_scenario')


def spawn_ego_vehicle(world: carla.World, spawn_point: carla.Transform) -> Optional[carla.Vehicle]:
    """Spawn the ego (player) vehicle."""
    bp_lib = world.get_blueprint_library()
    # Try preferred sedan models, fall back to any 4-wheel vehicle
    ego_bp = None
    for candidate in ['vehicle.tesla.model3', 'vehicle.audi.a2', 'vehicle.lincoln.mkz2017']:
        matches = list(bp_lib.filter(candidate))
        if matches:
            ego_bp = matches[0]
            break
    if ego_bp is None:
        vehicles_4w = [bp for bp in bp_lib.filter('vehicle.*') if int(bp.get_attribute('number_of_wheels')) >= 4]
        ego_bp = vehicles_4w[0] if vehicles_4w else bp_lib.filter('vehicle.*')[0]

    # Set distinct color (Red)
    if ego_bp.has_attribute('color'):
        ego_bp.set_attribute('color', '255,0,0')

    ego_vehicle = world.try_spawn_actor(ego_bp, spawn_point)
    return ego_vehicle


def spawn_npc_vehicles(world: carla.World, spawn_points: List[carla.Transform],
                       count: int = 10, seed: int = 42) -> List[carla.Vehicle]:
    """Spawn specified number of NPC vehicles at provided spawn points."""
    rng = random.Random(seed)
    bp_lib = world.get_blueprint_library()
    car_blueprints = [
        bp for bp in bp_lib.filter('vehicle.*')
        if int(bp.get_attribute('number_of_wheels')) >= 4
    ]
    if not car_blueprints:
        car_blueprints = list(bp_lib.filter('vehicle.*'))

    shuffled_sp = list(spawn_points)
    rng.shuffle(shuffled_sp)

    spawned: List[carla.Vehicle] = []
    for sp in shuffled_sp:
        if len(spawned) >= count:
            break
        bp = rng.choice(car_blueprints)
        if bp.has_attribute('color'):
            color = rng.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        actor = world.try_spawn_actor(bp, sp)
        if actor is not None:
            spawned.append(actor)
    return spawned


def print_state_summary(state: Dict, header: str = "Uncertainty State") -> None:
    """Format and print uncertainty state dictionary."""
    print("\n" + "=" * 65)
    print(" %s" % header)
    print("=" * 65)
    print(" Scenario Level   : %s" % state.get('scenario_level', 'N/A'))
    
    # Weather
    weather = state.get('weather')
    if weather:
        print(" Weather Preset   : %s (rain=%.1f, wetness=%.1f, fog=%.1f)" % (
            weather.get('preset', 'N/A'),
            weather.get('precipitation', 0.0),
            weather.get('wetness', 0.0),
            weather.get('fog_density', 0.0)
        ))

    # Friction
    if 'friction_mu' in state:
        print(" Ego Friction mu  : %.3f" % state['friction_mu'])

    # Actuator
    actuator = state.get('actuator')
    if actuator:
        print(" Actuator Noise   : throttle=%.3f, steer=%.3f, brake=%.3f, delay=%d ticks" % (
            actuator.get('throttle_noise', 0.0),
            actuator.get('steer_noise', 0.0),
            actuator.get('brake_noise', 0.0),
            actuator.get('delay_ticks', 0)
        ))

    # Communication
    comm = state.get('communication')
    if comm:
        print(" Comm Stats       : Sent=%d, Delivered=%d, Lost=%d (Loss Rate: %.1f%%)" % (
            comm.get('messages_sent', 0),
            comm.get('messages_delivered', 0),
            comm.get('messages_lost', 0),
            comm.get('loss_rate_percent', 0.0)
        ))

    # Active Jams & Accidents
    if 'active_jams' in state:
        print(" Active Traffic Jams: %d" % state['active_jams'])
    if 'active_accidents' in state:
        print(" Active Accidents   : %d" % state['active_accidents'])

    # Profiles
    profiles = state.get('behavior_profiles')
    if profiles:
        prof_counts = {}
        for p in profiles.values():
            p_name = p.name if hasattr(p, 'name') else str(p)
            prof_counts[p_name] = prof_counts.get(p_name, 0) + 1
        print(" Driver Profiles    : %s" % prof_counts)

    print("=" * 65 + "\n")


def main():
    argparser = argparse.ArgumentParser(description="Test Combined Uncertainty Scenario for CARLA 0.9.9")
    argparser.add_argument('--host', metavar='H', default='127.0.0.1',
                           help='IP of the CARLA host server (default: 127.0.0.1)')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int,
                           help='TCP port of CARLA host server (default: 2000)')
    argparser.add_argument('--tm-port', metavar='TM_P', default=8000, type=int,
                           help='Port to communicate with Traffic Manager (default: 8000)')
    argparser.add_argument('--level', default='medium', choices=['low', 'medium', 'high', 'extreme'],
                           help='Scenario level to test (default: medium)')
    argparser.add_argument('-d', '--duration', default=30.0, type=float,
                           help='Simulation duration in seconds (default: 30.0)')
    argparser.add_argument('--seed', metavar='S', default=42, type=int,
                           help='Random seed for reproducibility (default: 42)')
    args = argparser.parse_args()

    client: Optional[carla.Client] = None
    ego_vehicle: Optional[carla.Vehicle] = None
    npc_vehicles: List[carla.Vehicle] = []
    manager: Optional[UncertaintyManager] = None

    try:
        # Step 1: Connect to CARLA, get traffic manager
        print("=" * 65)
        print("CARLA 0.9.9 Combined Uncertainty Scenario Test")
        print("Connecting to CARLA at %s:%d..." % (args.host, args.port))
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)

        world = client.get_world()
        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        print("Connected! Map: %s, TM Port: %d" % (world.get_map().name, args.tm_port))

        # Step 2: Spawn ego vehicle + 10 NPC vehicles
        spawn_points = world.get_map().get_spawn_points()
        if len(spawn_points) < 2:
            print("ERROR: Map has insufficient spawn points (%d)." % len(spawn_points))
            return

        print("\nStep 2: Spawning ego vehicle and 10 NPC vehicles...")
        ego_spawn = spawn_points[0]
        ego_vehicle = spawn_ego_vehicle(world, ego_spawn)
        if ego_vehicle is None:
            print("ERROR: Failed to spawn ego vehicle.")
            return
        print("  [SUCCESS] Ego vehicle spawned (ID: %d, Type: %s)" % (ego_vehicle.id, ego_vehicle.type_id))

        npc_vehicles = spawn_npc_vehicles(world, spawn_points[1:], count=10, seed=args.seed)
        print("  [SUCCESS] Spawned %d NPC vehicles." % len(npc_vehicles))

        # Step 3: Set NPCs to autopilot
        print("\nStep 3: Enabling autopilot for all NPC vehicles...")
        tm_port = traffic_manager.get_port()
        for npc in npc_vehicles:
            npc.set_autopilot(True, tm_port)
        print("  Autopilot enabled for %d NPCs." % len(npc_vehicles))

        # Step 4: Create UncertaintyManager(seed=42)
        print("\nStep 4: Initializing UncertaintyManager(seed=%d)..." % args.seed)
        manager = UncertaintyManager(seed=args.seed)

        # Step 5: Load scenario level MEDIUM: manager.load_scenario('medium')
        print("\nStep 5: Loading scenario level '%s'..." % args.level.upper())
        manager.load_scenario(args.level)
        print("  Scenario '%s' loaded successfully." % args.level)

        # Step 6: Setup: manager.setup(world, ego_vehicle, npc_vehicles, traffic_manager)
        print("\nStep 6: Setting up UncertaintyManager with simulation references...")
        manager.setup(
            world=world,
            ego_vehicle=ego_vehicle,
            npc_vehicles=npc_vehicles,
            traffic_manager=traffic_manager
        )
        print("  Setup completed.")

        # Step 7: Print initial state: manager.get_state()
        print("\nStep 7: Initial Uncertainty State:")
        initial_state = manager.get_state()
        print_state_summary(initial_state, "INITIAL UNCERTAINTY STATE")

        # Step 8: Run simulation loop for 30 seconds (use world.wait_for_tick())
        #   - result = manager.tick(delta, current_time)
        #   - If any events triggered, print them
        #   - Every 5 seconds, print a status update
        print("Step 8: Running simulation loop for %.1f seconds..." % args.duration)
        start_sim_time = None
        start_wall_time = time.time()
        last_status_sim_time = 0.0
        total_ticks = 0

        while True:
            snapshot = world.wait_for_tick()
            delta = snapshot.timestamp.delta_seconds
            current_time = snapshot.timestamp.elapsed_seconds

            if start_sim_time is None:
                start_sim_time = current_time

            elapsed_sim = current_time - start_sim_time
            elapsed_wall = time.time() - start_wall_time
            total_ticks += 1

            # Tick uncertainty manager
            result = manager.tick(delta_seconds=delta, current_time=current_time)

            # Check if any events triggered
            triggered = result.get('triggered_events', [])
            if triggered:
                print("  [Sim t=%.1fs | Tick #%d] TRIGGERED EVENTS (%d):" % (
                    elapsed_sim, total_ticks, len(triggered)
                ))
                for ev in triggered:
                    ev_type = ev.event_type.name if hasattr(ev.event_type, 'name') else str(ev.event_type)
                    print("    -> Event: %s | Duration: %.1fs | Severity: %s" % (
                        ev_type, getattr(ev, 'duration', 0.0), getattr(ev, 'severity', 'N/A')
                    ))

            # Every 5 seconds, print a status update
            if (elapsed_sim - last_status_sim_time) >= 5.0:
                last_status_sim_time = elapsed_sim
                state_now = manager.get_state()
                active_events = state_now.get('event_stats', {}).get('active_events', 0)
                active_jams = state_now.get('active_jams', 0)
                active_accidents = state_now.get('active_accidents', 0)
                print("  [Status @ t=%.1fs (wall=%.1fs)] Active Events: %d | Active Jams: %d | Active Accidents: %d" % (
                    elapsed_sim, elapsed_wall, active_events, active_jams, active_accidents
                ))

            # Check termination condition
            if elapsed_sim >= args.duration or elapsed_wall >= (args.duration + 10.0):
                print("\n  Reached target duration (sim: %.1fs, wall: %.1fs, ticks: %d)." % (
                    elapsed_sim, elapsed_wall, total_ticks
                ))
                break

        # Step 9: Print final state
        print("\nStep 9: Final Uncertainty State:")
        final_state = manager.get_state()
        print_state_summary(final_state, "FINAL UNCERTAINTY STATE")

        # Step 10: Test V2V: send a message, receive it
        print("Step 10: Testing V2V Communication uncertainty channel...")
        if npc_vehicles:
            sender_id = ego_vehicle.id
            receiver_id = npc_vehicles[0].id
            v2v_data = {
                "sender_speed": 18.5,
                "lane_id": 1,
                "hazard_detected": True,
                "target_dist": 15.2
            }
            cur_sim_time = snapshot.timestamp.elapsed_seconds
            print("  Sending V2V message from Ego (ID: %d) to NPC (ID: %d)..." % (sender_id, receiver_id))
            print("  Original Payload: %s" % v2v_data)

            sent_ok = manager.send_v2v_message(
                sender_id=sender_id,
                receiver_id=receiver_id,
                data=v2v_data,
                current_time=cur_sim_time
            )
            print("  Message transmission attempt: %s" % ("ENQUEUED" if sent_ok else "DROPPED (packet loss / outage)"))

            # Advance time by 0.5s to account for latency and receive
            future_time = cur_sim_time + 0.5
            received_msgs = manager.receive_v2v_messages(receiver_id=receiver_id, current_time=future_time)
            print("  Attempting reception at t=%.2fs (+0.5s): received %d message(s)" % (
                future_time, len(received_msgs)
            ))
            for idx, msg in enumerate(received_msgs, 1):
                print("  Received Message [%d]: Payload=%s, Corrupted=%s, Delay=%.3fs" % (
                    idx, msg.data, msg.is_corrupted, (msg.receive_time - msg.send_time)
                ))

        # Step 11: Test actuator: apply control with uncertainty
        print("\nStep 11: Testing Actuator Uncertainty on Ego Vehicle...")
        nominal_control = carla.VehicleControl(throttle=0.70, steer=0.15, brake=0.0)
        print("  Nominal Command : throttle=%.3f, steer=%.3f, brake=%.3f" % (
            nominal_control.throttle, nominal_control.steer, nominal_control.brake
        ))

        applied_control = manager.apply_control_with_uncertainty(ego_vehicle, nominal_control)
        print("  Applied Command : throttle=%.3f, steer=%.3f, brake=%.3f (with noise/delay)" % (
            applied_control.throttle, applied_control.steer, applied_control.brake
        ))
        delta_throttle = applied_control.throttle - nominal_control.throttle
        delta_steer = applied_control.steer - nominal_control.steer
        print("  Actuator Deltas : delta_throttle=%+.4f, delta_steer=%+.4f" % (
            delta_throttle, delta_steer
        ))

        print("\nCombined scenario test finished successfully!")

    except Exception as exc:
        logger.error("Error occurred during combined scenario test: %s", exc, exc_info=True)

    finally:
        # Step 12: Cleanup: manager.cleanup(), destroy vehicles
        print("\nStep 12: Cleaning up simulation actors...")
        if manager is not None:
            manager.cleanup()
            print("  UncertaintyManager cleanup() finished.")

        actors_to_destroy = []
        if ego_vehicle is not None and getattr(ego_vehicle, 'is_alive', True):
            actors_to_destroy.append(ego_vehicle)
        for npc in npc_vehicles:
            if npc is not None and getattr(npc, 'is_alive', True):
                actors_to_destroy.append(npc)

        if actors_to_destroy and client is not None:
            client.apply_batch([carla.command.DestroyActor(a) for a in actors_to_destroy])
            print("  Destroyed %d vehicles (ego + NPCs)." % len(actors_to_destroy))

        print("Test script finished.")


if __name__ == '__main__':
    main()
