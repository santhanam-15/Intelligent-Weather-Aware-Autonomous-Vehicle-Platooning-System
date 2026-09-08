# Module 1: Unified Uncertain Platoon Simulation

This document details the architecture, integrations, and logic implementations completed to unify the Autonomous Platoon System with the Living City Uncertainty Environment into a single, cohesive CARLA simulation.

## 1. Single Entry Point Architecture
Two completely separate execution loops (`autopilot_platoon_window.py` and `real_world_simulation.py`) have been unified into a single authoritative launcher sequence.
- **`run_uncertain_platoon.bat`**: The new root batch launcher. It automatically provisions a separate live-log PowerShell window and forwards all terminal arguments down to the Python backend.
- **`uncertain_platoon_main.py`**: The unified Python entry point. It manages the connection to the CARLA server, loads the requested map, handles all asynchronous actor spawning (Platoon, NPCs, Pedestrians), and establishes a single `while running:` Pygame tick loop to orchestrate world events.

## 2. World and State Synchronization
To prevent simulation conflicts, the unification logic strictly adheres to sharing states rather than duplicating them:
- **Shared CARLA World**: Connects to the CARLA client exactly once. The same `world` and `trafficmanager` instances are synchronously passed into both the Platoon systems and the Uncertainty Manager.
- **Single Simulation Tick**: There are no competing `world.tick()` commands. All event states (Ghost timeline ticks, Platoon path updates, Weather timers) are processed sequentially, followed by a single authoritative `world.tick()` (or `advance_world()`).
- **Protected Platoon Roster**: The Ghost Driver system natively monitors the simulation to possess NPC vehicles. To prevent it from accidentally hijacking platoon members, the IDs of the spawned platoon are registered as `platoon_vehicles` into the `UncertaintyTimelineDirector`, explicitly marking them as immune.

## 3. Advanced Ghost Driver Features
The Ghost Driver system has been updated with detailed tuning and new visual tracking implementations:
- **Possession & Lurking Timings**: Ghost behavior intervals have been fine-tuned. The ghost possesses a target vehicle for a strict duration of **15 to 20 seconds**. Afterward, it "lurks" in the background for a cooldown of **5 to 20 seconds** before striking again.
- **3D Debug Bounding Boxes**: Integrated CARLA's `world.debug.draw_box()` to project a real-time red wireframe cuboid around whichever vehicle is currently possessed. This provides immediate, physical visual tracking within the 3D camera viewport.
- **Telemetry Extraction**: The Ghost Driver's internal state is constantly parsed via `ghost_driver.get_status()`, allowing the Pygame loop to monitor distance, possession states, and disruption actions without directly coupling logic.

## 4. UI/HUD Unification
The existing Pygame visualization dashboard (`MiniHUD`) has been subclassed into a custom `UnifiedHUD`. 
- **Dynamic HUD**: In addition to the standard platoon tracking metrics (throttle, speed, distance), it now renders a conditional "GHOST DRIVER SYSTEM" panel.
- **Ghost Data**: When active, the panel flashes red and alerts the user of the possessed vehicle's ID, the specific adversarial action being taken (e.g., *Leader Cut-In*), and a live countdown timer until the ghost relinquishes control.

## 5. Dynamic Environmental Controls and CLI Flags
The command-line arguments have been extensively expanded to give researchers complete control over the environment without hardcoding values:
- **`--platoon-size [INT]`**: Dynamically adjusts the number of vehicles generated in the platoon (e.g., 3 creates 1 Leader + 2 Followers).
- **`--weather [FLOAT]`**: Establishes a dynamic weather shift interval defined in minutes. Once the simulation time crosses the defined interval, the `WeatherManager` is invoked to randomly transition atmospheric conditions (Clear, Cloudy, Foggy, Rain).
- **Pygame Resolution Arguments**: Implemented `--windowed`, `--width [INT]`, and `--height [INT]` arguments. These override the hardcoded screen dimensions and dynamically scale the camera sensors, UI elements, and Pygame surface to fit the requested resolution.

## 6. Execution Example
The fully integrated module can be executed with a rich set of environment parameters:
```bat
.\run_uncertain_platoon.bat --town Town05 --vehicles 32 --pedestrians 20 --platoon-size 3 --windowed --width 1280 --height 720 --launch-server --weather 10
```

