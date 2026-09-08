#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Real-World Uncertainty Runner for CARLA 0.9.9 & ScenarioRunner
-------------------------------------------------------------
Allows selecting and launching safety-critical, real-world uncertainty
driving scenarios (emergency stops, aggressive cut-ins, darting pedestrians,
slippery roads, red light violators, and oncoming overtakers) with optional
AI autopilot or human driving HUD.
"""

import os
import sys
import time
import subprocess

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CARLA_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, '..'))
VENV_PYTHON = os.path.join(CARLA_ROOT, 'Scripts', 'python.exe')
if not os.path.exists(VENV_PYTHON):
    VENV_PYTHON = sys.executable

# Ensure PYTHONPATH has CARLA egg and libraries
egg_path = os.path.join(CARLA_ROOT, 'PythonAPI', 'carla', 'dist', 'carla-0.9.9-py3.7-win-amd64.egg')
carla_api = os.path.join(CARLA_ROOT, 'PythonAPI', 'carla')
carla_agents = os.path.join(CARLA_ROOT, 'PythonAPI', 'carla', 'agents')

for p in [egg_path, carla_agents, carla_api, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import carla
except ImportError:
    pass

SCENARIOS = {
    "1": {
        "title": "Emergency Braking Hazard",
        "desc": "Leading vehicle suddenly brakes hard down to a complete stop.",
        "category": "Sudden Deceleration / Perception Uncertainty",
        "scenario": "FollowLeadingVehicle_1",
        "town": "Town01",
        "ego_goal": "Follow lead car at safe distance and brake to avoid collision."
    },
    "2": {
        "title": "Hidden Road Obstacle / Swerving Lead Car",
        "desc": "Leading vehicle abruptly dodges an unexpected stationary obstacle in the lane.",
        "category": "Perception Uncertainty / Sudden Occlusion",
        "scenario": "FollowLeadingVehicleWithObstacle_1",
        "town": "Town01",
        "ego_goal": "React to newly revealed obstacle and brake or maneuver safely."
    },
    "3": {
        "title": "Aggressive Highway Cut-In (Left Lane)",
        "desc": "A reckless vehicle suddenly cuts directly in front of ego vehicle from left lane.",
        "category": "Behavioral Uncertainty / Erratic Drivers",
        "scenario": "CutInFrom_left_Lane",
        "town": "Town04",
        "ego_goal": "Detect sudden intrusion and adjust speed to avoid sideswipe collision."
    },
    "4": {
        "title": "Aggressive Highway Cut-In (Right Lane)",
        "desc": "A vehicle swerves into ego vehicle's lane from the right lane unexpectedly.",
        "category": "Behavioral Uncertainty / Blindspot Intrusions",
        "scenario": "CutInFrom_right_Lane",
        "town": "Town04",
        "ego_goal": "Detect right cut-in and decelerate promptly."
    },
    "5": {
        "title": "Darting Pedestrian / Cyclist Hazard",
        "desc": "A pedestrian or cyclist suddenly crosses into the ego vehicle's direct path.",
        "category": "Vulnerable Road User Uncertainty / Unsignalized Crossing",
        "scenario": "DynamicObjectCrossing_1",
        "town": "Town01",
        "ego_goal": "Perform emergency stop until pedestrian completely clears the road."
    },
    "6": {
        "title": "Stationary Obstacle in Lane",
        "desc": "A stationary cyclist or road debris blocks the driving lane.",
        "category": "Road Hazard / Static Obstacle",
        "scenario": "StationaryObjectCrossing_1",
        "town": "Town01",
        "ego_goal": "Detect stationary obstacle ahead and come to a safe stop."
    },
    "7": {
        "title": "Slippery Road / Hydroplaning / Traction Loss",
        "desc": "Severe loss of friction and dynamic tire slip causing steering loss.",
        "category": "Environmental & Friction Uncertainty",
        "scenario": "ControlLoss_1",
        "town": "Town01",
        "ego_goal": "Counter-steer, stabilize vehicle trajectory, and regain control."
    },
    "8": {
        "title": "Severe Wet Surface Traction Loss (Junction)",
        "desc": "Loss of control on slippery wet pavement near a street corner.",
        "category": "Environmental & Friction Uncertainty",
        "scenario": "ControlLoss_4",
        "town": "Town02",
        "ego_goal": "Maintain vehicle stability and avoid crashing into curbs/structures."
    },
    "9": {
        "title": "Red Light Violator at Intersection",
        "desc": "An oncoming vehicle illegally runs a red traffic light across the junction.",
        "category": "Traffic Violation / Intersection Uncertainty",
        "scenario": "OppositeVehicleRunningRedLight011",
        "town": "Town01",
        "ego_goal": "Yield despite green light to avoid a fatal T-bone collision."
    },
    "10": {
        "title": "Reckless Oncoming Overtaker (Encroachment)",
        "desc": "An oncoming vehicle pulls into your lane to pass another car in rural road.",
        "category": "Extreme Behavioral Uncertainty / Head-On Hazard",
        "scenario": "ManeuverOppositeDirection_1",
        "town": "Town01",
        "ego_goal": "Slow down and hug lane edge to prevent head-on collision."
    },
    "11": {
        "title": "Unsignalized Blind Intersection",
        "desc": "Navigating an intersection without traffic signals with crossing traffic.",
        "category": "Negotiation Uncertainty / Occlusion",
        "scenario": "NoSignalJunctionCrossing",
        "town": "Town03",
        "ego_goal": "Negotiate right-of-way and safely pass through without collision."
    },
    "12": {
        "title": "Signalized Left Turn Crossing Traffic",
        "desc": "Hero vehicle turns left while opposing traffic crosses straight.",
        "category": "Intersection Crossing / Gap Acceptance",
        "scenario": "SignalizedJunctionLeftTurn_1",
        "town": "Town01",
        "ego_goal": "Wait for safe gap in traffic before executing left turn."
    }
}


def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_banner():
    banner = r"""
================================================================================
       CARLA 0.9.9  -  REAL-WORLD UNCERTAINTY TESTBENCH (ScenarioRunner)
                 Intelligent Testing & Verification Framework
================================================================================
"""
    print(banner)


def check_carla_running(host='127.0.0.1', port=2000):
    """Check if CARLA server is reachable."""
    try:
        client = carla.Client(host, port)
        client.set_timeout(2.0)
        _ = client.get_world()
        return True
    except Exception:
        return False


def launch_carla_server():
    """Launch CARLA UE4 with low-end GPU settings."""
    carla_exe = os.path.join(CARLA_ROOT, 'CarlaUE4.exe')
    if not os.path.exists(carla_exe):
        print(f"  [ERROR] CarlaUE4.exe not found at {carla_exe}")
        return None

    print("\n  [INFO] Launching CARLA 0.9.9 (Optimized Low-End: 800x600, Low Quality)...")
    cmd = [
        carla_exe,
        "-quality-level=Low",
        "-windowed",
        "-ResX=800",
        "-ResY=600",
        "-carla-rpc-port=2000"
    ]
    proc = subprocess.Popen(cmd, cwd=CARLA_ROOT)
    print("  [INFO] Waiting 15 seconds for CARLA to initialize...")
    for remaining in range(15, 0, -1):
        print(f"\r  Starting CARLA... {remaining}s remaining   ", end="", flush=True)
        time.sleep(1)
    print("\n  [INFO] CARLA server launch process completed.")
    return proc


def print_menu():
    print("\n  --- REAL-WORLD UNCERTAINTY SCENARIOS ---")
    for key, item in SCENARIOS.items():
        print(f"  [{key:>2}] {item['title']:<40} [{item['town']}]")
        print(f"       -> {item['desc']}")
    print()
    print("  --- ADVANCED / BATCH RUNNERS ---")
    print("  [R ] RANDOMIZED RUN   : Select any scenario with randomized parameters")
    print("  [G ] GROUP MARATHON   : Run all scenarios of a class (e.g. group:ControlLoss)")
    print("  [C ] CUSTOM SCENARIO  : Enter any scenario name manually")
    print("  [V ] VISUALIZER ONLY  : Launch manual driving HUD (manual_control.py)")
    print("  [T ] 2D MAP ONLY      : Launch lightweight top-down view (no_rendering_mode.py)")
    print("  [0 ] EXIT")
    print("=" * 80)


def choose_control_mode():
    print("\n  --- SELECT EGO VEHICLE CONTROL MODE ---")
    print("  [1] AI Autopilot (Recommended) - Autonomous driver via Traffic Manager")
    print("      * Displays 3D Cockpit HUD with live speed, steering, & sensors")
    print("      * Press 'P' in Pygame window to toggle manual control anytime!")
    print("  [2] Human Driver - Drive manually using WASD / Arrow Keys in Pygame HUD")
    print("  [3] Headless / Background - Terminal output only (no extra Pygame window)")
    print()
    while True:
        choice = input("  Select control mode [1]: ").strip() or "1"
        if choice in ["1", "2", "3"]:
            return choice
        print("  Please enter 1, 2, or 3.")


def wait_for_hero_vehicle(host='127.0.0.1', port=2000, timeout=40):
    """Wait until ScenarioRunner has loaded the map and spawned the hero vehicle."""
    print("  [WAIT] Waiting for ScenarioRunner to load town and spawn hero vehicle...", end="", flush=True)
    start_t = time.time()
    while time.time() - start_t < timeout:
        try:
            client = carla.Client(host, port)
            client.set_timeout(3.0)
            world = client.get_world()
            actors = world.get_actors().filter('vehicle.*')
            for a in actors:
                if a.attributes.get('role_name') == 'hero':
                    print(" [READY]")
                    return True
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(1.0)
    print(" [TIMEOUT]")
    return False


def run_scenario(scenario_name, randomize=False, control_mode="1", host="127.0.0.1", port=2000):
    """Execute the scenario runner and optional manual_control visualizer."""
    scenario_runner_py = os.path.join(SCRIPT_DIR, 'scenario_runner.py')
    manual_control_py = os.path.join(SCRIPT_DIR, 'manual_control.py')

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{carla_api};{carla_agents};{egg_path};{SCRIPT_DIR}"

    sr_cmd = [
        VENV_PYTHON,
        scenario_runner_py,
        "--host", host,
        "--port", str(port),
        "--scenario", scenario_name,
        "--reloadWorld"
    ]
    if randomize:
        sr_cmd.append("--randomize")

    print(f"\n  [INFO] Launching ScenarioRunner: {scenario_name} (Randomize={randomize})")
    print(f"  [INFO] Command: {' '.join(sr_cmd)}\n")
    print("  " + "-" * 76)

    sr_process = None
    hud_process = None

    try:
        # Start ScenarioRunner in background process
        sr_process = subprocess.Popen(sr_cmd, env=env, cwd=SCRIPT_DIR)

        # If HUD requested, wait for hero vehicle to spawn before starting Pygame
        if control_mode in ["1", "2"]:
            hero_ready = wait_for_hero_vehicle(host, port, timeout=40)
            if hero_ready:
                hud_cmd = [
                    VENV_PYTHON,
                    manual_control_py,
                    "--host", host,
                    "--port", str(port),
                    "--res", "1024x576"
                ]
                if control_mode == "1":
                    hud_cmd.append("--autopilot")

                print("  [INFO] Spawning Driving HUD Window (Pygame)...")
                hud_process = subprocess.Popen(hud_cmd, env=env, cwd=SCRIPT_DIR)
            else:
                print("  [WARNING] Hero vehicle was not detected within timeout. Continuing in console mode.")

        # Wait for ScenarioRunner to complete
        ret = sr_process.wait()
        print("\n  " + "-" * 76)
        if ret == 0:
            print("  [SUCCESS] Scenario completed successfully!")
        else:
            print(f"  [INFO] Scenario finished with code: {ret}")

    except KeyboardInterrupt:
        print("\n  [ABORT] Scenario interrupted by user.")
    finally:
        if sr_process and sr_process.poll() is None:
            try:
                sr_process.terminate()
            except Exception:
                pass
        if hud_process and hud_process.poll() is None:
            print("  [INFO] Closing Driving HUD...")
            try:
                hud_process.terminate()
            except Exception:
                pass


def main():
    clear_console()
    print_banner()

    # Verify CARLA server
    print("  [CHECK] Checking CARLA simulator connection...")
    if not check_carla_running():
        print("  [WARNING] CARLA simulator is NOT running on localhost:2000.")
        ans = input("  Would you like to start CARLA with low-end GPU settings now? [Y/n]: ").strip().lower()
        if ans != 'n':
            launch_carla_server()
            if not check_carla_running():
                print("  [WAIT] Giving CARLA an extra 5 seconds...")
                time.sleep(5)
        else:
            print("  [NOTE] Please ensure CARLA is running before starting a scenario.")
    else:
        print("  [OK] CARLA simulator is online and responding!")

    while True:
        print_menu()
        choice = input("  Enter option number [0 to exit]: ").strip().upper()

        if choice == "0":
            print("\n  Exiting Real-World Uncertainty Testbench. Safe travels!\n")
            break

        if choice in SCENARIOS:
            sc_info = SCENARIOS[choice]
            print(f"\n  Selected: {sc_info['title']} ({sc_info['scenario']})")
            print(f"  Category: {sc_info['category']}")
            print(f"  Map     : {sc_info['town']}")
            print(f"  Goal    : {sc_info['ego_goal']}")
            ctrl_mode = choose_control_mode()
            run_scenario(sc_info['scenario'], randomize=False, control_mode=ctrl_mode)

        elif choice == "R":
            print("\n  --- RANDOMIZED UNCERTAINTY RUN ---")
            print("  Select scenario to run with randomized distances/speeds:")
            for k, v in SCENARIOS.items():
                print(f"    [{k:>2}] {v['title']}")
            sub = input("\n  Select number: ").strip()
            if sub in SCENARIOS:
                sc_info = SCENARIOS[sub]
                ctrl_mode = choose_control_mode()
                run_scenario(sc_info['scenario'], randomize=True, control_mode=ctrl_mode)
            else:
                print("  [ERROR] Invalid scenario choice.")

        elif choice == "G":
            print("\n  --- GROUP MARATHON ---")
            print("  [1] group:FollowLeadingVehicle  (Lead braking & obstacle series)")
            print("  [2] group:ControlLoss           (All friction / hydroplane tests)")
            print("  [3] group:OtherLeadingVehicle   (Deceleration & lane shifts)")
            grp_choice = input("  Select group [1]: ").strip() or "1"
            grp_map = {
                "1": "group:FollowLeadingVehicle",
                "2": "group:ControlLoss",
                "3": "group:OtherLeadingVehicle"
            }
            grp_name = grp_map.get(grp_choice, "group:FollowLeadingVehicle")
            ctrl_mode = choose_control_mode()
            run_scenario(grp_name, randomize=False, control_mode=ctrl_mode)

        elif choice == "C":
            custom_name = input("\n  Enter Scenario Name (e.g. DynamicObjectCrossing_2): ").strip()
            if custom_name:
                ctrl_mode = choose_control_mode()
                run_scenario(custom_name, randomize=False, control_mode=ctrl_mode)

        elif choice == "V":
            print("\n  [INFO] Launching standalone Driving HUD (manual_control.py)...")
            manual_control_py = os.path.join(SCRIPT_DIR, 'manual_control.py')
            ap = input("  Enable Autopilot? [Y/n]: ").strip().lower()
            cmd = [VENV_PYTHON, manual_control_py, "--res", "1024x576"]
            if ap != 'n':
                cmd.append("--autopilot")
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{carla_api};{carla_agents};{egg_path};{SCRIPT_DIR}"
            subprocess.call(cmd, env=env, cwd=SCRIPT_DIR)

        elif choice == "T":
            print("\n  [INFO] Launching Lightweight 2D Top-Down View (no_rendering_mode.py)...")
            no_render_py = os.path.join(SCRIPT_DIR, 'no_rendering_mode.py')
            cmd = [VENV_PYTHON, no_render_py, "--res", "1024x576"]
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{carla_api};{carla_agents};{egg_path};{SCRIPT_DIR}"
            subprocess.call(cmd, env=env, cwd=SCRIPT_DIR)

        else:
            print("  [WARNING] Invalid selection. Please try again.")

        input("\n  Press Enter to continue...")


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\n\n  [INFO] Exiting session. Goodbye!\n")
