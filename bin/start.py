#!/usr/bin/env python
"""
CARLA 0.9.9 Uncertainty Toolkit — Interactive Launcher

Menu-driven script that:
  1. Launches the CARLA simulator
  2. Connects to CARLA
  3. Lets you spawn vehicles & walkers with custom counts
  4. Apply uncertainty scenarios (LOW / MEDIUM / HIGH / EXTREME)
  5. Toggle individual uncertainty modules
  6. Trigger specific events manually
  7. Run simulation with a live status dashboard

Usage:
    python start.py
"""

import glob
import os
import sys
import subprocess
import time
import signal
import random
import logging

# ============================================================
# PYTHON VERSION GUARD
# CARLA 0.9.9 egg requires Python 3.7.
# If we're running under a different Python, re-launch with
# the venv Python 3.7 automatically.
# ============================================================
if sys.version_info[:2] != (3, 7):
    _this_file = os.path.abspath(__file__)
    _venv_python = os.path.join(
        os.path.dirname(_this_file), '..', 'Scripts', 'python.exe')
    _venv_python = os.path.normpath(_venv_python)
    if os.path.exists(_venv_python):
        print(f"  [INFO] Current Python is {sys.version_info.major}.{sys.version_info.minor}.")
        print(f"  [INFO] CARLA needs Python 3.7 — switching to venv Python...")
        print(f"  [INFO] {_venv_python}\n")
        # Use subprocess (not os.execv) to handle Windows paths with spaces
        ret = subprocess.call([_venv_python, _this_file] + sys.argv[1:])
        sys.exit(ret)
    else:
        print(f"  [WARNING] Python {sys.version_info.major}.{sys.version_info.minor} detected.")
        print(f"  [WARNING] CARLA 0.9.9 needs Python 3.7 for the .egg package.")
        print(f"  [WARNING] Venv not found at: {_venv_python}")
        print(f"  [WARNING] Continuing anyway — carla import may fail.\n")


# --- CARLA egg path setup ---
try:
    sys.path.append(glob.glob(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'carla', 'dist', 'carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64')))[0])
except IndexError:
    pass

import carla

# Add parent directory for uncertainty package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uncertainty.weather_uncertainty import WeatherManager, WeatherPreset
from uncertainty.friction_uncertainty import FrictionManager, FrictionCondition
from uncertainty.vehicle_uncertainty import ActuatorUncertainty
from uncertainty.communication_uncertainty import V2VChannel, CommPreset
from uncertainty.traffic_uncertainty import TrafficJamManager, JamSeverity
from uncertainty.accident_manager import AccidentManager, AccidentType
from uncertainty.human_behavior import HumanBehaviorManager, DriverProfile
from uncertainty.event_manager import EventManager, EventType
from uncertainty.scenario_manager import ScenarioManager, ScenarioLevel
from uncertainty.uncertainty_manager import UncertaintyManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# GLOBAL STATE
# ============================================================

class SimState:
    """Holds the current simulation state."""
    def __init__(self):
        self.client = None
        self.world = None
        self.traffic_manager = None
        self.carla_process = None

        self.ego_vehicle = None
        self.npc_vehicles = []
        self.walkers = []
        self.walker_controllers = []
        self.all_actors = []  # everything we spawned

        self.uncertainty_manager = None
        self.scenario_level = None
        self.is_running = False
        self.seed = 42

state = SimState()

# ============================================================
# UTILITIES
# ============================================================

BANNER = r"""
  ╔══════════════════════════════════════════════════════════════╗
  ║    CARLA 0.9.9  —  Uncertainty Toolkit  v1.0                ║
  ║    Intelligent Weather-Aware Autonomous Vehicle Platooning  ║
  ╚══════════════════════════════════════════════════════════════╝
"""

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    print(BANNER)

def get_int_input(prompt, default=None):
    """Get an integer from the user with optional default."""
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"  {prompt}{suffix}: ").strip()
        if raw == "" and default is not None:
            return default
        try:
            return int(raw)
        except ValueError:
            print("  ⚠ Please enter a valid number.")

def get_choice(prompt, options):
    """Present numbered options and get user choice."""
    for i, opt in enumerate(options, 1):
        print(f"    {i}. {opt}")
    while True:
        raw = input(f"\n  {prompt}: ").strip()
        try:
            idx = int(raw)
            if 1 <= idx <= len(options):
                return idx
            print(f"  ⚠ Choose between 1 and {len(options)}.")
        except ValueError:
            print("  ⚠ Enter a number.")

def pause():
    input("\n  Press Enter to continue...")

# ============================================================
# 1. LAUNCH CARLA
# ============================================================

def launch_carla():
    """Launch the CARLA simulator executable."""
    print("\n  ═══ Launch CARLA Simulator ═══\n")

    carla_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..')
    carla_exe = os.path.join(carla_root, 'CarlaUE4.exe')

    if not os.path.exists(carla_exe):
        print(f"  ⚠ CarlaUE4.exe not found at: {carla_exe}")
        print("  ⚠ Please start CARLA manually and then connect.")
        pause()
        return

    print(f"  Starting CARLA from: {carla_exe}")
    print("  (This may take 30-60 seconds to load...)\n")

    try:
        state.carla_process = subprocess.Popen(
            [carla_exe],
            cwd=carla_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("  ✓ CARLA process started (PID: %d)" % state.carla_process.pid)
        print("  ⏳ Waiting 15 seconds for CARLA to initialize...")
        time.sleep(15)
        print("  ✓ CARLA should be ready now.")
    except Exception as e:
        print(f"  ✗ Failed to start CARLA: {e}")
        print("  ⚠ Please start CARLA manually.")

    pause()

# ============================================================
# 2. CONNECT TO CARLA
# ============================================================

def connect_to_carla():
    """Connect to a running CARLA server."""
    print("\n  ═══ Connect to CARLA ═══\n")

    host = input("  Host [127.0.0.1]: ").strip() or "127.0.0.1"
    port = get_int_input("Port", 2000)

    print(f"\n  Connecting to {host}:{port}...")

    try:
        state.client = carla.Client(host, port)
        state.client.set_timeout(10.0)
        state.world = state.client.get_world()
        state.traffic_manager = state.client.get_trafficmanager(8000)
        state.traffic_manager.set_global_distance_to_leading_vehicle(2.5)

        map_name = state.world.get_map().name
        weather = state.world.get_weather()

        print(f"\n  ✓ Connected successfully!")
        print(f"    Map: {map_name}")
        print(f"    Weather: cloudiness={weather.cloudiness:.0f}, "
              f"rain={weather.precipitation:.0f}")

    except Exception as e:
        print(f"\n  ✗ Connection failed: {e}")
        print("  ⚠ Make sure CARLA is running.")
        state.client = None
        state.world = None

    pause()

# ============================================================
# 3. SPAWN VEHICLES
# ============================================================

def spawn_vehicles():
    """Spawn NPC vehicles with user-specified count."""
    if not state.world:
        print("\n  ⚠ Not connected to CARLA. Connect first!")
        pause()
        return

    print("\n  ═══ Spawn Vehicles ═══\n")
    print(f"  Currently spawned: {len(state.npc_vehicles)} NPC vehicles"
          f" + {'1 ego' if state.ego_vehicle else '0 ego'}")
    print()

    num = get_int_input("How many NPC vehicles to spawn?", 30)
    spawn_ego = input("  Spawn ego (player) vehicle? [y/N]: ").strip().lower()

    spawn_points = state.world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    bp_lib = state.world.get_blueprint_library()
    vehicle_bps = bp_lib.filter('vehicle.*')

    # Filter to 4-wheeled vehicles for realism
    four_wheel_bps = [bp for bp in vehicle_bps
                      if int(bp.get_attribute('number_of_wheels').as_int()) == 4]
    if not four_wheel_bps:
        four_wheel_bps = list(vehicle_bps)

    print(f"\n  Spawning {num} NPC vehicles...")

    spawned = 0
    for i in range(min(num, len(spawn_points) - 1)):
        bp = random.choice(four_wheel_bps)
        if bp.has_attribute('color'):
            color = random.choice(
                bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        sp = spawn_points[i + (1 if spawn_ego == 'y' else 0)]
        vehicle = state.world.try_spawn_actor(bp, sp)
        if vehicle:
            vehicle.set_autopilot(True, state.traffic_manager.get_port())
            state.npc_vehicles.append(vehicle)
            state.all_actors.append(vehicle)
            spawned += 1

    print(f"  ✓ Spawned {spawned}/{num} NPC vehicles")

    # Spawn ego vehicle
    if spawn_ego == 'y' and not state.ego_vehicle:
        bp = random.choice(four_wheel_bps)
        if bp.has_attribute('color'):
            bp.set_attribute('color', '0,255,0')  # green ego
        ego = state.world.try_spawn_actor(bp, spawn_points[0])
        if ego:
            state.ego_vehicle = ego
            state.all_actors.append(ego)
            print(f"  ✓ Ego vehicle spawned (ID: {ego.id})")
            print("    Note: Ego is NOT on autopilot (controlled by you)")
        else:
            print("  ✗ Failed to spawn ego vehicle")

    pause()

# ============================================================
# 4. SPAWN WALKERS
# ============================================================

def spawn_walkers():
    """Spawn pedestrian walkers with user-specified count."""
    if not state.world:
        print("\n  ⚠ Not connected to CARLA. Connect first!")
        pause()
        return

    print("\n  ═══ Spawn Walkers ═══\n")
    print(f"  Currently spawned: {len(state.walkers)} walkers")
    print()

    num = get_int_input("How many walkers to spawn?", 50)

    bp_lib = state.world.get_blueprint_library()
    walker_bps = bp_lib.filter('walker.pedestrian.*')
    controller_bp = bp_lib.find('controller.ai.walker')

    print(f"\n  Spawning {num} walkers...")

    spawned = 0
    for i in range(num):
        # Get random spawn point on sidewalk
        spawn_point = carla.Transform()
        loc = state.world.get_random_location_from_navigation()
        if loc is None:
            continue
        spawn_point.location = loc

        bp = random.choice(walker_bps)
        if bp.has_attribute('is_invincible'):
            bp.set_attribute('is_invincible', 'false')

        walker = state.world.try_spawn_actor(bp, spawn_point)
        if walker:
            state.walkers.append(walker)
            state.all_actors.append(walker)

            # Spawn AI controller
            controller = state.world.try_spawn_actor(
                controller_bp, carla.Transform(), walker)
            if controller:
                state.walker_controllers.append(controller)
                state.all_actors.append(controller)
                controller.start()
                controller.go_to_location(
                    state.world.get_random_location_from_navigation())
                controller.set_max_speed(
                    1.0 + random.random() * 1.5)  # 1.0 - 2.5 m/s
            spawned += 1

    print(f"  ✓ Spawned {spawned}/{num} walkers")
    pause()

# ============================================================
# 5. APPLY UNCERTAINTY SCENARIO
# ============================================================

def apply_scenario():
    """Apply a predefined uncertainty scenario level."""
    if not state.world:
        print("\n  ⚠ Not connected to CARLA. Connect first!")
        pause()
        return

    print("\n  ═══ Apply Uncertainty Scenario ═══\n")

    if state.uncertainty_manager:
        print("  ⚠ Uncertainty is already active. Cleaning up first...")
        state.uncertainty_manager.cleanup()

    # Choose seed
    state.seed = get_int_input("Random seed for reproducibility", 42)

    # Choose scenario level
    print()
    print("  Choose scenario level:")
    levels = [
        "LOW     — Mild weather, normal traffic, minimal uncertainty",
        "MEDIUM  — Rain, friction variation, comm delay, behavior variation",
        "HIGH    — Heavy rain, low friction, traffic jams, aggressive drivers",
        "EXTREME — Rain+fog, very low friction, accidents, erratic drivers",
        "CUSTOM  — Configure each module individually",
    ]
    choice = get_choice("Select level", levels)

    state.uncertainty_manager = UncertaintyManager(seed=state.seed)

    if choice <= 4:
        level_map = {1: 'low', 2: 'medium', 3: 'high', 4: 'extreme'}
        level = level_map[choice]
        state.uncertainty_manager.load_scenario(level)
        state.scenario_level = level.upper()
        print(f"\n  ✓ Scenario loaded: {level.upper()}")
    else:
        # Custom configuration
        configure_custom_scenario()

    # Setup with current vehicles
    state.uncertainty_manager.setup(
        state.world,
        state.ego_vehicle,
        state.npc_vehicles,
        state.traffic_manager
    )

    print("\n  ✓ Uncertainty applied to the simulation!")
    print(f"    Seed: {state.seed}")
    print(f"    NPC vehicles affected: {len(state.npc_vehicles)}")
    if state.ego_vehicle:
        print(f"    Ego vehicle: ID {state.ego_vehicle.id}")

    # Print current state summary
    print_uncertainty_state()

    pause()

def configure_custom_scenario():
    """Let user toggle each module and set parameters."""
    print("\n  ═══ Custom Scenario Configuration ═══\n")

    mgr = state.uncertainty_manager

    # Weather
    enable = input("  Enable weather uncertainty? [Y/n]: ").strip().lower()
    if enable != 'n':
        print("  Weather presets:")
        presets = [p.name for p in WeatherPreset]
        for i, p in enumerate(presets, 1):
            print(f"    {i}. {p}")
        idx = get_int_input("Choose preset", 4) - 1
        preset_name = presets[min(idx, len(presets) - 1)]
        mgr._config['weather'] = {'enabled': True, 'preset': preset_name}
    else:
        mgr.enable_module('weather', False)

    # Friction
    enable = input("\n  Enable friction uncertainty? [Y/n]: ").strip().lower()
    if enable != 'n':
        print("  Conditions: 1=DRY  2=WET  3=SLIPPERY  4=VERY_SLIPPERY")
        idx = get_int_input("Choose condition", 2) - 1
        conditions = ['DRY', 'WET', 'SLIPPERY', 'VERY_SLIPPERY']
        mgr._config['friction'] = {
            'enabled': True,
            'condition': conditions[min(idx, 3)]
        }
    else:
        mgr.enable_module('friction', False)

    # Actuator
    enable = input("\n  Enable actuator uncertainty? [Y/n]: ").strip().lower()
    if enable != 'n':
        noise = float(input("    Throttle/brake noise std [0.05]: ").strip()
                       or "0.05")
        steer = float(input("    Steering noise std [0.02]: ").strip()
                       or "0.02")
        delay = get_int_input("  Actuator delay ticks", 1)
        mgr.actuator.set_noise(throttle_noise=noise, steer_noise=steer,
                                brake_noise=noise)
        mgr.actuator.set_delay(delay)
        mgr._config['vehicle_uncertainty'] = {'enabled': True}
    else:
        mgr.enable_module('vehicle_uncertainty', False)

    # Communication
    enable = input("\n  Enable V2V communication uncertainty? [Y/n]: "
                   ).strip().lower()
    if enable != 'n':
        print("  Presets: 1=NORMAL  2=MEDIUM  3=BAD")
        idx = get_int_input("Choose preset", 2)
        preset_map = {1: CommPreset.NORMAL, 2: CommPreset.MEDIUM,
                      3: CommPreset.BAD}
        mgr.communication.configure_from_preset(
            preset_map.get(idx, CommPreset.MEDIUM))
        mgr._config['communication'] = {'enabled': True}
    else:
        mgr.enable_module('communication', False)

    # Traffic jams
    enable = input("\n  Enable traffic jams? [Y/n]: ").strip().lower()
    if enable != 'n':
        prob = float(input("    Jam probability per tick [0.1]: ").strip()
                     or "0.1")
        mgr._config['traffic_jam'] = {
            'enabled': True, 'probability': prob,
            'severity': 'MEDIUM', 'duration': 30.0, 'num_vehicles': 5
        }
    else:
        mgr.enable_module('traffic_jam', False)

    # Accidents
    enable = input("\n  Enable accidents/obstacles? [Y/n]: ").strip().lower()
    if enable != 'n':
        prob = float(input("    Accident probability per tick [0.05]: "
                           ).strip() or "0.05")
        mgr._config['accident'] = {
            'enabled': True, 'probability': prob, 'duration': 60.0
        }
    else:
        mgr.enable_module('accident', False)

    # Human behavior
    enable = input("\n  Enable human driver behavior? [Y/n]: ").strip().lower()
    if enable != 'n':
        mgr._config['human_behavior'] = {
            'enabled': True,
            'profile_weights': {
                'NORMAL': 0.40, 'AGGRESSIVE': 0.15,
                'CONSERVATIVE': 0.15, 'DISTRACTED': 0.10,
                'ERRATIC': 0.10, 'SLOW': 0.10
            }
        }
    else:
        mgr.enable_module('human_behavior', False)

    # Events
    enable = input("\n  Enable sudden events? [Y/n]: ").strip().lower()
    if enable != 'n':
        # Register default events
        mgr.events.register_event(EventType.SUDDEN_BRAKING,
                                  probability=0.05, cooldown=20)
        mgr.events.register_event(EventType.SUDDEN_ACCELERATION,
                                  probability=0.03, cooldown=25)
        mgr.events.register_event(EventType.LANE_CUT_IN,
                                  probability=0.05, cooldown=15)
        mgr.events.register_event(EventType.SUDDEN_RAIN,
                                  probability=0.02, cooldown=60)
        mgr.events.register_event(EventType.SUDDEN_FOG,
                                  probability=0.02, cooldown=60)
        mgr.events.register_event(EventType.FRICTION_REDUCTION,
                                  probability=0.02, cooldown=60)
        mgr._config['events'] = {'enabled': True}
    else:
        mgr.enable_module('events', False)

    state.scenario_level = "CUSTOM"
    print("\n  ✓ Custom scenario configured!")

# ============================================================
# 6. TRIGGER MANUAL EVENT
# ============================================================

def trigger_manual_event():
    """Manually trigger a specific uncertain event."""
    if not state.uncertainty_manager:
        print("\n  ⚠ No uncertainty active. Apply a scenario first!")
        pause()
        return

    print("\n  ═══ Trigger Manual Event ═══\n")

    events = [
        ("Sudden Braking (random NPC)", EventType.SUDDEN_BRAKING),
        ("Sudden Acceleration (random NPC)", EventType.SUDDEN_ACCELERATION),
        ("Lane Cut-In", EventType.LANE_CUT_IN),
        ("Traffic Jam (spawn slow vehicles)", EventType.TRAFFIC_JAM),
        ("Accident (spawn stopped vehicles)", EventType.ACCIDENT),
        ("Stopped Vehicle", EventType.STOPPED_VEHICLE),
        ("Communication Failure (outage)", EventType.COMM_FAILURE),
        ("Sudden Rain", EventType.SUDDEN_RAIN),
        ("Sudden Fog", EventType.SUDDEN_FOG),
        ("Friction Reduction", EventType.FRICTION_REDUCTION),
        ("Change Weather Preset", None),
    ]

    options = [e[0] for e in events]
    choice = get_choice("Select event to trigger", options)

    if choice == 11:
        # Change weather manually
        print("\n  Weather presets:")
        presets = list(WeatherPreset)
        for i, p in enumerate(presets, 1):
            print(f"    {i}. {p.name}")
        idx = get_int_input("Choose preset", 1) - 1
        preset = presets[min(idx, len(presets) - 1)]
        state.uncertainty_manager.weather.set_weather(
            state.world, preset)
        print(f"\n  ✓ Weather changed to: {preset.name}")
    else:
        event_type = events[choice - 1][1]
        current_time = time.time()
        event = state.uncertainty_manager.events.force_trigger(
            event_type, current_time, duration=15.0)
        state.uncertainty_manager._execute_event(event, current_time)
        print(f"\n  ✓ Event triggered: {event_type.value}")
        print(f"    Duration: 15 seconds")

    pause()

# ============================================================
# 7. RUN SIMULATION WITH LIVE STATUS
# ============================================================

def run_simulation():
    """Run the simulation with live status updates."""
    if not state.world:
        print("\n  ⚠ Not connected to CARLA. Connect first!")
        pause()
        return

    if not state.uncertainty_manager:
        print("\n  ⚠ No uncertainty active. Apply a scenario first!")
        pause()
        return

    print("\n  ═══ Run Simulation ═══\n")

    duration = get_int_input("Run for how many seconds?", 60)
    update_interval = get_int_input("Status update interval (seconds)", 5)

    print(f"\n  Running simulation for {duration}s...")
    print(f"  Status updates every {update_interval}s")
    print("  Press Ctrl+C to stop early\n")
    print("  " + "═" * 56)

    state.is_running = True
    start_time = time.time()
    last_status = start_time
    total_events = 0

    try:
        while state.is_running:
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break

            # Wait for CARLA tick
            try:
                snapshot = state.world.wait_for_tick(seconds=5.0)
            except Exception:
                time.sleep(0.1)
                continue

            delta = snapshot.timestamp.delta_seconds
            sim_time = snapshot.timestamp.elapsed_seconds

            # Update uncertainty
            result = state.uncertainty_manager.tick(delta, sim_time)

            # Count triggered events
            triggered = result.get("triggered_events", [])
            if triggered:
                total_events += len(triggered)
                for event in triggered:
                    print(f"\n  ⚡ EVENT: {event.event_type.value} "
                          f"(severity={event.severity}, "
                          f"dur={event.duration:.0f}s)")

            # Print status at intervals
            if time.time() - last_status >= update_interval:
                last_status = time.time()
                print_live_status(elapsed, sim_time, total_events, result)

    except KeyboardInterrupt:
        print("\n\n  ⚠ Simulation stopped by user (Ctrl+C)")

    state.is_running = False
    elapsed = time.time() - start_time
    print(f"\n  ═══ Simulation ended after {elapsed:.1f}s ═══")
    print(f"  Total events triggered: {total_events}")
    print_uncertainty_state()
    pause()

def print_live_status(elapsed, sim_time, total_events, result):
    """Print a compact live status line."""
    weather = result.get("weather_info", {})
    comm = result.get("communication_stats", {})
    active = result.get("active_events", [])

    rain = weather.get("rain_intensity", 0) if weather else 0
    fog = weather.get("fog_intensity", 0) if weather else 0
    vis = weather.get("visibility", 100) if weather else 100

    active_types = [e.event_type.value for e in active] if active else ["none"]

    print(f"\n  ┌── Status @ {elapsed:.0f}s (sim: {sim_time:.1f}s) "
          f"──────────────────────")
    print(f"  │  Weather:  rain={rain:.0f}%  fog={fog:.0f}%  "
          f"vis={vis:.0f}%")
    print(f"  │  Events:   {total_events} total, "
          f"{len(active)} active: {', '.join(active_types[:3])}")
    if comm:
        print(f"  │  Comms:    sent={comm.get('messages_sent', 0)}  "
              f"lost={comm.get('messages_lost', 0)}  "
              f"outage={'YES' if comm.get('current_outage') else 'no'}")
    print(f"  │  Vehicles: {len(state.npc_vehicles)} NPCs  "
          f"{len(state.walkers)} walkers")

    # Show active jams & accidents
    if state.uncertainty_manager:
        jams = len(state.uncertainty_manager.traffic.get_active_jams())
        accidents = len(
            state.uncertainty_manager.accident.get_active_events())
        if jams > 0 or accidents > 0:
            print(f"  │  Hazards:  {jams} traffic jams, "
                  f"{accidents} accidents")

    print(f"  └──────────────────────────────────────────────────")

# ============================================================
# 8. VIEW CURRENT STATE
# ============================================================

def print_uncertainty_state():
    """Print the current uncertainty state summary."""
    if not state.uncertainty_manager:
        print("\n  No uncertainty manager active.")
        return

    print(f"\n  ┌── Uncertainty State ──────────────────────────────")
    print(f"  │  Scenario: {state.scenario_level or 'None'}  "
          f"Seed: {state.seed}")

    st = state.uncertainty_manager.get_state()

    # Enabled modules
    enabled = st.get('enabled_modules', {})
    on_modules = [k for k, v in enabled.items() if v]
    off_modules = [k for k, v in enabled.items() if not v]
    print(f"  │  Modules ON:  {', '.join(on_modules) or 'none'}")
    if off_modules:
        print(f"  │  Modules OFF: {', '.join(off_modules)}")

    # Weather
    weather = st.get('weather')
    if weather:
        print(f"  │  Weather:  rain={weather.get('rain_intensity', 0):.0f}% "
              f" fog={weather.get('fog_intensity', 0):.0f}%  "
              f"vis={weather.get('visibility', 100):.0f}%  "
              f"cloud={weather.get('cloudiness', 0):.0f}%")

    # Friction
    mu = st.get('friction_mu')
    if mu is not None:
        print(f"  │  Friction:  μ = {mu:.2f}")

    # Actuator
    act = st.get('actuator')
    if act:
        print(f"  │  Actuator:  throttle_noise={act.get('throttle_noise')} "
              f" steer={act.get('steer_noise')} "
              f" delay={act.get('delay_ticks')} ticks")

    # Communication
    comm = st.get('communication')
    if comm:
        print(f"  │  Comms:  sent={comm.get('messages_sent', 0)} "
              f" lost={comm.get('messages_lost', 0)} "
              f" outage={'YES' if comm.get('current_outage') else 'no'}")

    # Behavior profiles
    profiles = st.get('behavior_profiles', {})
    if profiles:
        profile_counts = {}
        for vid, profile in profiles.items():
            name = profile.name if hasattr(profile, 'name') else str(profile)
            profile_counts[name] = profile_counts.get(name, 0) + 1
        summary = ", ".join(f"{k}={v}" for k, v in profile_counts.items())
        print(f"  │  Drivers:  {summary}")

    # Traffic / Accidents
    jams = st.get('active_jams', 0)
    acc = st.get('active_accidents', 0)
    print(f"  │  Hazards:  {jams} jams, {acc} accidents")

    # Events
    evs = st.get('event_stats', {})
    print(f"  │  Events:  {evs.get('total_events_triggered', 0)} triggered, "
          f"{evs.get('active_events', 0)} active")

    print(f"  └──────────────────────────────────────────────────")

def view_state():
    """View current state from menu."""
    print("\n  ═══ Current State ═══")

    if state.world:
        print(f"  Connected: YES")
        print(f"  Map: {state.world.get_map().name}")
    else:
        print(f"  Connected: NO")

    print(f"  Ego Vehicle: {'ID ' + str(state.ego_vehicle.id) if state.ego_vehicle else 'None'}")
    print(f"  NPC Vehicles: {len(state.npc_vehicles)}")
    print(f"  Walkers: {len(state.walkers)}")

    print_uncertainty_state()
    pause()

# ============================================================
# 9. CLEANUP
# ============================================================

def cleanup():
    """Destroy all spawned actors and reset."""
    print("\n  ═══ Cleanup ═══\n")
    print("  Cleaning up all spawned actors...")

    # Cleanup uncertainty manager
    if state.uncertainty_manager:
        try:
            state.uncertainty_manager.cleanup()
        except Exception as e:
            logger.warning("Uncertainty cleanup error: %s", e)

    # Stop walker controllers
    for controller in state.walker_controllers:
        try:
            controller.stop()
        except Exception:
            pass

    # Destroy all actors
    destroyed = 0
    for actor in reversed(state.all_actors):
        try:
            actor.destroy()
            destroyed += 1
        except Exception:
            pass

    state.ego_vehicle = None
    state.npc_vehicles.clear()
    state.walkers.clear()
    state.walker_controllers.clear()
    state.all_actors.clear()
    state.uncertainty_manager = None
    state.scenario_level = None

    print(f"  ✓ Destroyed {destroyed} actors")
    print("  ✓ Cleanup complete")
    pause()

# ============================================================
# MAIN MENU
# ============================================================

def main_menu():
    """Display the main menu and handle user input."""

    while True:
        clear_screen()
        print_banner()

        # Status line
        connected = "✓ Connected" if state.world else "✗ Not connected"
        vehicles = f"{len(state.npc_vehicles)} NPCs"
        walkers = f"{len(state.walkers)} walkers"
        scenario = state.scenario_level or "None"

        print(f"  Status: {connected} | {vehicles} | {walkers} | "
              f"Scenario: {scenario}")
        print()
        print("  ┌────────────────────────────────────────────────┐")
        print("  │                   MAIN MENU                    │")
        print("  ├────────────────────────────────────────────────┤")
        print("  │  1. Launch CARLA Simulator                     │")
        print("  │  2. Connect to CARLA                           │")
        print("  │  3. Spawn Vehicles (set count)                 │")
        print("  │  4. Spawn Walkers (set count)                  │")
        print("  │  5. Apply Uncertainty Scenario                 │")
        print("  │  6. Trigger Manual Event                       │")
        print("  │  7. Run Simulation (live status)               │")
        print("  │  8. View Current State                         │")
        print("  │  9. Cleanup (destroy all actors)               │")
        print("  │  0. Exit                                       │")
        print("  └────────────────────────────────────────────────┘")
        print()

        choice = input("  Enter choice: ").strip()

        if choice == '1':
            launch_carla()
        elif choice == '2':
            connect_to_carla()
        elif choice == '3':
            spawn_vehicles()
        elif choice == '4':
            spawn_walkers()
        elif choice == '5':
            apply_scenario()
        elif choice == '6':
            trigger_manual_event()
        elif choice == '7':
            run_simulation()
        elif choice == '8':
            view_state()
        elif choice == '9':
            cleanup()
        elif choice == '0':
            print("\n  Cleaning up before exit...")
            cleanup_quiet()
            print("  Goodbye! 👋")
            break
        else:
            print("  ⚠ Invalid choice. Try again.")
            time.sleep(1)

def cleanup_quiet():
    """Quiet cleanup on exit."""
    if state.uncertainty_manager:
        try:
            state.uncertainty_manager.cleanup()
        except Exception:
            pass
    for controller in state.walker_controllers:
        try:
            controller.stop()
        except Exception:
            pass
    for actor in reversed(state.all_actors):
        try:
            actor.destroy()
        except Exception:
            pass
    if state.carla_process:
        print("  Note: CARLA server is still running. Close it manually.")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n  Interrupted! Cleaning up...")
        cleanup_quiet()
        print("  Goodbye!")
    except Exception as e:
        print(f"\n  ✗ Fatal error: {e}")
        cleanup_quiet()
        raise

