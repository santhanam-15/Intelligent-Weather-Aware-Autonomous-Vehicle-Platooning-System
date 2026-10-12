# Module 2 Current System Audit (Baseline for Module 3)

**Document Status:** Source-Verified Audit  
**Date of Audit:** October 11, 2026  
**Environment:** CARLA 0.9.9 | Python 3.7.9 64-bit | PyTorch 1.13.1+cpu | Windows 10/11 x64  
**Project:** Intelligent Weather-Aware Autonomous Vehicle Platooning System using MAPPO in CARLA  
**Audit Purpose:** Comprehensive, source-grounded baseline description for designing Module 3 (Twin-World Safety Projection System).

---

## 1. Project Overview & Module Boundaries

The project implements an autonomous vehicle platooning framework operating in the high-fidelity CARLA 0.9.9 simulator under dynamic environmental, traffic, and adversarial uncertainties.

```mermaid
graph LR
    M1[Module 1: World & Weather Uncertainty Engine] --> M2[Module 2: Tactical MAPPO Platooning Policy]
    M2 --> M3[Module 3: Twin-World Safety Projection System<br/>(PROPOSED / IN PREPARATION)]
    M3 --> M4[Module 4: Low-Level Vehicle Control<br/>(Execution & Actuation)]
    M4 --> CARLA[CARLA 0.9.9 Physics Engine & Vehicle Actors]
```

### Module Breakdown
- **Module 1 (Uncertainty & World Management):**
  - Weather simulation (Clear, Cloudy, Light Rain, Heavy Rain, Fog, Heavy Fog).
  - Dynamic surface tire friction ($\mu \in [0.2, 1.2]$).
  - Background NPC traffic with behavioral driver profiles (Normal, Aggressive, Conservative, Distracted, Erratic).
  - Ghost Driver adversarial vehicle possession (in-gap intrusion, cut-ins, tailgating, adjacent blocking).
- **Module 2 (Tactical Coordination & Multi-Agent RL - Current Baseline):**
  - Cooperative Multi-Agent Proximal Policy Optimization (MAPPO) with centralized training and decentralized execution (CTDE).
  - 32-dimensional ego-centric observation vector per Connected and Autonomous Vehicle (CAV).
  - 5-discrete tactical action space (Cruise, Accelerate, Decelerate, Change Left, Change Right).
  - Action hysteresis filtering (2-step persistence threshold) and follower action masking.
  - Comprehensive telemetry logging (`mappo_decisions.log`, `uncertain_platoon.log`, `mappo_training.log`).
- **Module 3 (Twin-World Safety Projection System - To Be Designed):**
  - High-speed, isolated predictive digital twin.
  - Forward-projects candidate tactical actions across a near-future temporal horizon without affecting the real CARLA simulation.
  - Approves, rejects, or substitutes actions based on predictive collision risk, headway degradation, signal non-compliance, and road-friction boundaries.
- **Module 4 (Low-Level Vehicle Control & Actuation):**
  - Lateral path tracking (Pure Pursuit steering with rate limiting).
  - Longitudinal speed tracking (Proportional throttle/brake control, standstill breakout logic, emergency braking clamp).
  - Direct actuation via `carla.VehicleControl(throttle, brake, steer)`.

---

## 2. Verified Environment & Dependencies

| Component | Verified Specification | Verification Method |
| :--- | :--- | :--- |
| **Operating System** | Windows x64 (NT) | Native PowerShell execution environment |
| **Simulator** | CARLA 0.9.9 (`CarlaUE4.exe`) | Server handshake on TCP port 2000 |
| **Python API** | Python 3.7.9 64-bit | `py -3.7 --version` confirmed 3.7.9 |
| **CARLA Egg** | `carla-0.9.9-py3.7-win-amd64.egg` | `sys.path` imported successfully |
| **Deep Learning Framework** | PyTorch 1.13.1+cpu | `torch.__version__` verified in `MAPPOTrainer` |
| **Linear Algebra** | NumPy 1.21.6 | Observation arrays and buffer operations |
| **Graphics / Display** | Pygame 2.1.2 | Standalone HUD window in `run_module_2.py` |

---

## 3. Directory Structure & Key Files

```
MiniProj/
├── checkpoints/
│   └── mappo_checkpoint.pt                   # Authoritative 32-D MAPPO model checkpoint
├── project_docs/
│   └── module3_preparation/                  # Documentation for Module 3 design
├── run_module_2.bat                          # Master unified launcher (Sim + Telemetry + Monitor)
├── run_realtime_log_terminal.bat             # Dedicated live terminal streaming simulation telemetry
├── run_mappo_training_live.bat               # Dedicated live terminal monitoring MAPPO neural updates
├── WindowsNoEditor/
│   ├── CarlaUE4.exe                          # CARLA 0.9.9 native executable
│   ├── Module2/
│   │   ├── __init__.py
│   │   ├── config.py                         # Hyperparameters, rates, and thresholds
│   │   ├── module2_manager.py                # Platoon coordinator & 15 Hz MAPPO step logic
│   │   ├── mission_controller.py             # Route planning, safety constraints, low-level control
│   │   ├── action/
│   │   │   └── action_manager.py             # Action space, hysteresis filtering, masking
│   │   ├── observation/
│   │   │   └── observation_builder.py        # 32-D local observation & 128-D global state builder
│   │   ├── reward/
│   │   │   └── reward_manager.py             # Multi-objective cooperative reward calculation
│   │   ├── mappo/
│   │   │   ├── actor.py                      # SharedActor (32 -> 64 -> 64 -> 5)
│   │   │   ├── critic.py                     # CentralizedCritic (128 -> 128 -> 64 -> 1)
│   │   │   ├── buffer.py                     # RolloutBuffer (rollout length = 128)
│   │   │   └── trainer.py                    # MAPPO PPO update loop & atomic checkpointing
│   │   └── tests/
│   │       └── test_module2_logic.py         # 31 unit & regression tests (all passing)
│   └── PythonAPI/
│       ├── run_module_2.py                   # Master simulation entry point
│       ├── live_log_monitor.py               # Real-time ANSI-color log parser and streaming engine
│       ├── live_training_terminal.py         # Real-time training dashboard & weight norm visualizer
│       ├── real_world_simulation.py          # Timeline director, weather presets, traffic orchestration
│       ├── uncertainty/
│       │   └── ghost_driver.py               # Deterministic adversarial intrusion generator
│       └── logs/
│           ├── uncertain_platoon.log         # General simulation output stream
│           ├── mappo_decisions.log           # Structured JSONL decision audit records
│           ├── mappo_training.log            # Snapshot of latest neural network loss & layer weights
│           └── mappo_training_stream.log     # Incremental training step update stream
```

---

## 4. Entry Points, Startup & Shutdown Sequence

### Primary Entry Points
1. `run_module_2.bat`: Master batch script that spawns three concurrent windows:
   - **Simulation Real-Time Live Logs Terminal** (`run_realtime_log_terminal.bat` calling `live_log_monitor.py`).
   - **MAPPO Live Training Terminal** (`run_mappo_training_live.bat` calling `live_training_terminal.py`).
   - **CARLA Module 2 Simulation Window** (`py -3.7 run_module_2.py`).
2. `WindowsNoEditor/PythonAPI/run_module_2.py`: The Python application entry point.

### Execution Architecture
- **Main Thread (`UnifiedHUD` / Pygame):** Handles Pygame window rendering at 30–60 FPS, user keyboard inputs (camera toggle, pause, quit), hardware monitoring display, and HUD drawing.
- **Worker Thread (`CarlaSimulationThread`):** Sole owner of the CARLA client socket, synchronous physics ticks, MAPPO inference, and low-level control.

### Startup Sequence (`CarlaSimulationThread.run()`, lines 518–700)
1. Connects `carla.Client(args.host, args.port)` with a 30.0s timeout.
2. Configures synchronous world mode: `settings.synchronous_mode = True`, `settings.fixed_delta_seconds = 1.0 / 30.0` (33.3 ms per tick).
3. Configures Traffic Manager port 8000: `tm.set_synchronous_mode(True)`.
4. Spawns platoon vehicles (1 Leader + $N-1$ Followers) spaced safely at spawn points.
5. **Autopilot Enforcement:** Calls `veh.set_autopilot(False)` on every platoon vehicle (lines 567–573).
6. Spawns background NPC vehicles and pedestrians outside the platoon's 32-meter bubble.
7. Initializes `UncertaintyManager` and sets initial road friction based on weather preset.
8. Instantiates `MissionPlatoonController` (generating a 300m route from the leader's spawn point).
9. Instantiates `Module2Manager` (loads authoritative checkpoint, builds observation spaces).
10. Signals `ready_event` to release the UI thread and enters the synchronous simulation loop.

### Authoritative Step Loop (`CarlaSimulationThread.run()`, lines 701–810)
For each simulation tick:
1. Dequeue user commands from `command_queue`.
2. Inspect simulation timestamp: `elapsed_sim`, `delta_sec`.
3. Check and trigger weather shifts (configurable interval, default every 5 minutes).
4. Update `uncertainty_manager.tick()` and `timeline_director.tick()`.
5. Update Ghost Driver state machine (`timeline_director.ghost_driver.tick()`).
6. **Tactical MAPPO Decision (15 Hz):** If interval $\ge 0.0667\text{s}$, call `module2_manager.update(platoon, elapsed_sim)`.
7. Pass tactical actions to `mission_controller.set_tactical_actions(tactical_actions)`.
8. Check if rollout buffer is full; if so, trigger `module2_manager.train_if_ready()`.
9. **Low-Level Control Tick (30 Hz):** Execute `mission_controller.tick(elapsed_sim)`.
   - Evaluates route geometry, obstacle clearance, traffic light distance, and follower gap.
   - Computes target speed, steering, throttle, and brake.
   - Executes `vehicle.apply_control(carla.VehicleControl(...))` for each vehicle.
10. **Synchronous Physics Step:** Call `world.tick()`.
11. Update HUD telemetry snapshot via `snapshot_holder.set(snap)`.

### Shutdown Sequence (`run_module_2.py`, lines 910–990)
1. Sets `stop_event`.
2. Brakes all platoon vehicles to an emergency halt (`throttle=0.0, brake=1.0`).
3. Destroys camera sensor.
4. Cleans up all spawned NPC vehicles, walkers, walker controllers, and platoon vehicles.
5. Reverts CARLA world settings to asynchronous mode (`synchronous_mode = False`).
6. Terminates cleanly and flushes final logs.

---

## 5. MAPPO Architecture & Tensor Dimensions

MAPPO implements Centralized Training with Decentralized Execution (CTDE) using PyTorch 1.13.1.

```mermaid
graph TD
    subgraph Decentralized Execution (Actor)
        O_i["Local Observation o_i (32 dims)"] --> L1["Linear(32, 64) + Tanh"]
        L1 --> L2["Linear(64, 64) + Tanh"]
        L2 --> L3["Linear(64, 5)"]
        L3 --> Mask["Action Masking<br/>(Followers: mask left/right)"]
        Mask --> Cat["Categorical Distribution"]
        Cat --> A_i["Discrete Action a_i ∈ {0,1,2,3,4}"]
    end

    subgraph Centralized Training (Critic)
        S["Global State s (128 dims = 32 × 4)"] --> CL1["Linear(128, 128) + Tanh"]
        CL1 --> CL2["Linear(128, 64) + Tanh"]
        CL2 --> CL3["Linear(64, 1)"]
        CL3 --> V["State Value Estimate V(s)"]
    end
```

### Shared Actor Network (`WindowsNoEditor/Module2/mappo/actor.py`)
- **Type:** Parameter-shared Multi-Layer Perceptron across all CAVs in the platoon.
- **Input Dimension:** 32 (local observation $o_i$).
- **Hidden Layers:**
  - `Linear(32, 64)`, activation: `Tanh()`
  - `Linear(64, 64)`, activation: `Tanh()`
- **Output Layer:** `Linear(64, 5)` producing unnormalized action logits.
- **Action Masking:** Follower vehicles receive boolean mask `[True, True, True, False, False]`. Masked logits are clamped to $-10^9$ before softmax.
- **Inference Mode:**
  - Training: `Categorical(logits).sample()`
  - Evaluation: `torch.argmax(logits, dim=-1)` (fully deterministic)

### Centralized Critic Network (`WindowsNoEditor/Module2/mappo/critic.py`)
- **Type:** Centralized State-Value Estimator.
- **Input Dimension:** 128 (concatenation of up to 4 local observations: $32 \times 4$).
- **Hidden Layers:**
  - `Linear(128, 128)`, activation: `Tanh()`
  - `Linear(128, 64)`, activation: `Tanh()`
- **Output Layer:** `Linear(64, 1)` yielding baseline scalar state-value $V(s)$.

### Rollout Buffer & Training Hyperparameters (`WindowsNoEditor/Module2/config.py`)
- `ROLLOUT_LENGTH = 128` tactical decisions before PPO update.
- `MINI_BATCH_SIZE = 64`.
- `K_EPOCHS = 4` PPO update epochs per rollout.
- `LEARNING_RATE_ACTOR = 3e-4` (Adam optimizer).
- `LEARNING_RATE_CRITIC = 1e-3` (Adam optimizer).
- `GAMMA = 0.99`, `GAE_LAMBDA = 0.95`.
- `PPO_CLIP_RANGE = 0.2`.
- `MAX_GRAD_NORM = 0.5`.
- `ENTROPY_COEF = 0.01`, `VALUE_LOSS_COEF = 0.5`.

---

## 6. Observation Vector Specification (32 Elements)

The observation vector is constructed by `ObservationBuilder.get_local_observation()` (`WindowsNoEditor/Module2/observation/observation_builder.py`, lines 200–236). All values are bounded and normalized to $[-1.0, 1.0]$ or $[0.0, 1.0]$.

| Index | Feature Name | Raw Value Range | Normalization Formula | Semantic Description |
| :---: | :--- | :--- | :--- | :--- |
| `0` | `speed_norm` | $[0.0, 50.0]$ m/s | $\text{speed} / 50.0$ | Ego scalar speed |
| `1` | `vel_x_norm` | $[-30.0, 30.0]$ m/s | $v_x / 30.0$ | Ego longitudinal velocity (world frame) |
| `2` | `vel_y_norm` | $[-30.0, 30.0]$ m/s | $v_y / 30.0$ | Ego lateral velocity (world frame) |
| `3` | `is_leader` | $\{0.0, 1.0\}$ | $1.0$ if Leader, else $0.0$ | Role indicator |
| `4` | `agent_slot` | $[0.0, 1.0]$ | $\text{index} / \max(1, N_{\max} - 1)$ | Normalized platoon index slot |
| `5` | `size_norm` | $[0.0, 1.0]$ | $N_{\text{platoon}} / N_{\max}$ | Active platoon size fraction |
| `6` | `rel_x_ahead` | $[-100.0, 100.0]$ m | $\Delta x_{\text{ahead}} / 100.0$ | Longitudinal distance to predecessor |
| `7` | `rel_y_ahead` | $[-100.0, 100.0]$ m | $\Delta y_{\text{ahead}} / 100.0$ | Lateral offset to predecessor |
| `8` | `rel_vx_ahead` | $[-30.0, 30.0]$ m/s | $\Delta v_{x,\text{ahead}} / 30.0$ | Relative longitudinal velocity to predecessor |
| `9` | `rel_vy_ahead` | $[-30.0, 30.0]$ m/s | $\Delta v_{y,\text{ahead}} / 30.0$ | Relative lateral velocity to predecessor |
| `10` | `clearance_norm` | $[0.0, 100.0]$ m | $\max(0, \text{gap} - 4.7) / 100.0$ | Bumper-to-bumper bumper clearance |
| `11` | `rel_speed_ahead` | $[-30.0, 30.0]$ m/s | $(v_{\text{ahead}} - v_{\text{ego}}) / 30.0$ | Scalar closing velocity to predecessor |
| `12` | `dist_leader` | $[0.0, 100.0]$ m | $\max(0, d_{\text{leader}} - 4.7) / 100.0$ | Bumper clearance to platoon leader |
| `13` | `rel_speed_leader` | $[-30.0, 30.0]$ m/s | $(v_{\text{leader}} - v_{\text{ego}}) / 30.0$ | Relative speed to platoon leader |
| `14` | `tl_val` | $\{0.0, 0.5, 1.0\}$ | Red=1.0, Yellow=0.5, Green/None=0.0 | Relevant traffic signal state |
| `15` | `is_at_tl` | $\{0.0, 1.0\}$ | $1.0$ if at stop line box | Native CARLA stop-line proximity |
| `16` | `tl_dist` | $[0.0, 100.0]$ m | $d_{\text{tl}} / 100.0$ | Distance to relevant traffic light pole/trigger |
| `17` | `friction` | $[0.2, 1.2]$ | $\mu$ directly | Active road-tire friction multiplier |
| `18` | `route_progress` | $[0.0, 1.0]$ | $\text{cursor} / (N_{\text{route}} - 1)$ | Normalized mission route progress |
| `19` | `dist_to_waypoint` | $[0.0, 100.0]$ m | $\text{clamp}(d_{\text{wp}} / 100.0, 0, 1)$ | Distance to current lookahead waypoint |
| `20` | `lateral_error` | $[-6.0, 6.0]$ m | $\text{clamp}(e_{\text{lat}} / 6.0, -1, 1)$ | Cross-track error from planned route |
| `21` | `heading_error` | $[-\pi, \pi]$ rad | $\text{clamp}(e_{\text{yaw}} / \pi, -1, 1)$ | Yaw heading error relative to waypoint |
| `22` | `lane_id_norm` | $[-1.0, 1.0]$ | $\text{lane\_id} / 10.0$ | OpenDRIVE signed lane ID |
| `23` | `road_id_norm` | $[0.0, 10.0]$ | $\text{road\_id} / 100.0$ | OpenDRIVE road segment ID |
| `24` | `left_lane_ok` | $\{0.0, 1.0\}$ | $1.0$ if left driving lane exists | Left lane change feasibility flag |
| `25` | `right_lane_ok` | $\{0.0, 1.0\}$ | $1.0$ if right driving lane exists | Right lane change feasibility flag |
| `26` | `nearby_x` | $[-80.0, 80.0]$ m | $\Delta x_{\text{nearest}} / 80.0$ | Longitudinal distance to nearest NPC/obstacle |
| `27` | `nearby_y` | $[-12.0, 12.0]$ m | $\Delta y_{\text{nearest}} / 12.0$ | Lateral offset to nearest NPC/obstacle |
| `28` | `nearby_vx` | $[-30.0, 30.0]$ m/s | $\Delta v_{x,\text{nearest}} / 30.0$ | Relative speed of nearest NPC |
| `29` | `nearby_is_platoon` | $\{0.0, 1.0\}$ | $1.0$ if nearest is in platoon | Platoon membership of nearest actor |
| `30` | `intrusion_flag` | $\{0.0, 1.0\}$ | $1.0$ if intrusion active | Active ghost driver intrusion indicator |
| `31` | `platoon_state_val` | $[0.0, 1.0]$ | Mapped discrete value | Platoon state machine enumeration |

*Platoon state value mapping:* `NORMAL_PLATOON` = 0.0, `INTRUSION_DETECTED` = 0.25, `PLATOON_SPLIT` = 0.5, `SAFE_FOLLOWING` = 0.65, `REJOIN_ELIGIBLE` = 0.8, `REJOINING` = 1.0.

---

## 7. Action Space & Selection Behavior

### Action Definitions (`WindowsNoEditor/Module2/action/action_manager.py`)
```python
ACTION_MAP = {
    0: "CRUISE",        # Maintain current tactical target speed
    1: "ACCELERATE",    # Increase target speed by +1.1 m/s (+3.96 km/h)
    2: "DECELERATE",    # Decrease target speed by -1.8 m/s (-6.48 km/h)
    3: "CHANGE_LEFT",   # Initiate left lane change (Leader only)
    4: "CHANGE_RIGHT"   # Initiate right lane change (Leader only)
}
```

### Hysteresis & Filtering Pipeline
1. **Raw Action Normalization:** Follower actions $3$ and $4$ are remapped to $0$ (`CRUISE`).
2. **Two-Step Hysteresis Filter:**
   - A proposed tactical action must persist for at least 2 consecutive decisions (`ACTION_HYSTERESIS_STEPS = 2`) before it becomes the effective tactical action.
   - Prevents high-frequency switching or chatter (e.g., alternating between Accelerate and Decelerate).
3. **Log-Probability Correction:**
   - If hysteresis overrides a candidate raw action, PPO training recalculates `log_probs` using the *effective* action actually passed to the controller, preventing policy gradient distortion.

---

## 8. Checkpoint Format & Loading/Saving Architecture

- **Authoritative Checkpoint File:** `checkpoints/mappo_checkpoint.pt`
- **Secondary Mirror File:** `WindowsNoEditor/PythonAPI/checkpoints/mappo_checkpoint.pt`
- **Current State:**
  - Checkpoint update step: **121** (active, fresh weights).
  - SHA-256 fingerprint: Verified and logged on every load.
  - File size: $\approx 130$ KB (contains actor, critic, optimizers, and update step).

### Checkpoint Dictionary Schema
```python
checkpoint = {
    "actor": self.actor.state_dict(),           # 32 -> 64 -> 64 -> 5 weights & biases
    "critic": self.critic.state_dict(),         # 128 -> 128 -> 64 -> 1 weights & biases
    "actor_opt": self.actor_optimizer.state_dict(),
    "critic_opt": self.critic_optimizer.state_dict(),
    "update_step": self.update_step,            # Integer counter
    "obs_dim": 32,
    "global_state_dim": 128,
    "action_dim": 5,
    "file_hash": "..."                          # SHA-256 hash string
}
```

### Safety Features in Checkpoint Management
1. **Atomic Saving:** Saves to `*.tmp` first, then executes atomic replacement via `os.replace()` to prevent partial corruption during unexpected termination.
2. **Dimension Compatibility Validation:** Rejects legacy checkpoints (e.g., historical 14-D or 21-D models) with explicit errors instead of silent weight truncations.
3. **Strict Evaluation Mode:** If running in `--mode evaluate`, failure to load the checkpoint raises an immediate `RuntimeError`. Falling back to random weights is prohibited.

---

## 9. Route Planning & Tracking Implementation

Implemented in `RoutePlanner` (`WindowsNoEditor/Module2/mission_controller.py`, lines 38–205).

1. **Pre-Planned Global Route:**
   - Generates an ordered sequence of waypoints at 2.0m spacing from the platoon origin along driving lanes (`distance_m = 300.0`).
   - Route waypoints are stored in `self.route`.
2. **Monotonic Sequential Tracking:**
   - Uses an internal progress `cursor` index per vehicle.
   - The cursor can only advance forward within a strict search window $[cursor, cursor + 20]$.
   - Prevents the vehicle from rewinding or jumping to adjacent, parallel, or looping road sections.
3. **Dynamic Horizon Extension:**
   - When the leader approaches within 45 waypoints of the route destination, `extend_route(distance_m=150.0)` dynamically appends connected successor waypoints.
   - Prevents unexpected destination stops while preserving continuous platooning.

---

## 10. Low-Level Control Pipeline

Implemented in `MissionPlatoonController` (`WindowsNoEditor/Module2/mission_controller.py`, lines 216–786).

### 1. Lateral Control (Pure Pursuit)
- Lookahead distance: $L_a = \text{clamp}(5.0 + 0.7 \times v, 6.0, 16.0)$ meters.
- Local angular error $\alpha = \text{atan2}(y_{\text{local}}, x_{\text{local}})$.
- Raw steering angle: $\delta_{\text{raw}} = \text{atan2}(2.7 \times \sin(\alpha), L_a) \times 1.8$.
- Steering rate limiter: $\delta_t = \text{clamp}(0.78 \times \delta_{t-1} + 0.22 \times \delta_{\text{raw}}, \delta_{t-1} - 0.08, \delta_{t-1} + 0.08)$.
- Final steering bounded to $[-0.65, 0.65]$.

### 2. Longitudinal Target Speed Calculation
- **Leader:** Modulates baseline cruise speed ($v_{\text{cruise}} = 28\text{ km/h} \approx 7.78\text{ m/s}$):
  $$v_{\text{target, leader}} = (v_{\text{cruise}} + \Delta v_{\text{action}}) \times (0.65 + 0.35 \times \mu)$$
  where $\Delta v_{\text{action}} = +1.1\text{ m/s}$ (Accelerate), $-1.8\text{ m/s}$ (Decelerate), or $0.0$ (Cruise).
- **Followers:** Follows predecessor speed $v_{\text{pred}}$ with headway error correction:
  $$v_{\text{target, follower}} = (v_{\text{pred}} + \Delta v_{\text{action}} + \text{bias}_{\text{headway}}) \times (0.65 + 0.35 \times \mu)$$
  where $\text{bias}_{\text{headway}} = \text{clamp}((\text{gap} - d_{\text{desired}}) \times 0.20, -2.5, 2.0)$.
- **Obstacle Bound:**
  $$v_{\text{target}} \le \sqrt{\max(0, 2 \times 3.0 \times \mu \times (\text{clearance} - 2.0))}$$

### 3. Actuation Generation (Throttle & Brake)
- Velocity tracking error $e_v = v_{\text{target}} - v_{\text{current}}$.
- If $v_{\text{target}} \le 0.05\text{ m/s}$ (Stop requested):
  $$\text{throttle} = 0.0, \quad \text{brake} = 1.0 \text{ (or } 0.5 \text{ stationary hold)}$$
- If driving ($v_{\text{target}} > 0.05\text{ m/s}$):
  $$\text{throttle} = \text{clamp}(0.23 \times e_v, 0.0, 0.65)$$
  - Standstill breakout boost: if $v < 0.5\text{ m/s}$ and $e_v > 1.0\text{ m/s}$, $\text{throttle} \ge 0.38$.
  $$\text{brake} = \text{clamp}(-0.42 \times e_v, 0.0, 1.0)$$
  - Follower emergency gap clamp: if $\text{gap} < 5.0\text{m}$, $\text{brake} \ge 0.65$; if $\text{gap} < 3.8\text{m}$, $\text{brake} = 1.0$.
  - Mutual exclusion: if $\text{brake} > 0.02$, $\text{throttle} = 0.0$.

---

## 11. Feature Verification & Audit Summary Table

| Feature / Subsystem | Source Location | Implementation Status | Evidence / Verification Test | Known Limitations & Gaps |
| :--- | :--- | :--- | :--- | :--- |
| **Synchronous Simulation** | `run_module_2.py`: 549, 805 | **VERIFIED IMPLEMENTED** | `sync=True`, fixed $\Delta t = 1/30\text{s}$ | Requires CARLA server running in sync mode |
| **Autopilot Disabled** | `run_module_2.py`: 568–572 | **VERIFIED IMPLEMENTED** | `veh.set_autopilot(False)` enforced | Must be verified every spawn cycle |
| **MAPPO Inference (32-D)** | `actor.py`: 33; `module2_manager.py`: 181 | **VERIFIED IMPLEMENTED** | Deterministic argmax in eval; tensor shape `(N, 32)` | Shared actor assumes uniform vehicle dynamics |
| **Action Hysteresis** | `action_manager.py`: 52–75 | **VERIFIED IMPLEMENTED** | Unit test `test_action_persistence_blocks_one_tick_chatter` | Fixed 2-step threshold; not adaptive to speed |
| **Follower Action Masking** | `action_manager.py`: 81–84 | **VERIFIED IMPLEMENTED** | Unit test `test_follower_lane_actions_are_masked` | Followers cannot execute tactical lane splits |
| **Centralized Critic (128-D)** | `critic.py`: 5–20; `trainer.py`: 245 | **VERIFIED IMPLEMENTED** | Tensor shape `(B, 128)`, GAE $\lambda=0.95$ | Fixed maximum platoon size $N_{\max}=4$ |
| **Atomic Checkpointing** | `trainer.py`: 385–435 | **VERIFIED IMPLEMENTED** | Unit test `test_checkpoint_atomic_save_and_hash_advancement` | Saves locally; mirrors to project root |
| **Traffic Light Stopping** | `mission_controller.py`: 386–462 | **VERIFIED IMPLEMENTED** | Unit test `test_traffic_light_red_stops_vehicle_at_various_speeds` | Uses bounding box trigger; geometry projection |
| **Amber Dilemma Zone** | `mission_controller.py`: 444–458 | **VERIFIED IMPLEMENTED** | Unit test `test_traffic_light_amber_dilemma_zone_physics` | Simple kinematic braking distance formula |
| **Obstacle Corridor Check** | `mission_controller.py`: 293–314 | **VERIFIED IMPLEMENTED** | Unit test `test_obstacle_corridor_ignores_adjacent_lane_traffic` | Hard-coded corridor width (1.65m near / 1.95m far) |
| **Ghost Driver Maneuvers** | `ghost_driver.py`: 320–570 | **VERIFIED IMPLEMENTED** | Unit test `test_ghost_driver_scripted_scenarios_and_state_transitions` | Possesses existing NPCs; requires traffic density |
| **Dynamic Weather Shifts** | `run_module_2.py`: 741–750 | **VERIFIED IMPLEMENTED** | Periodically shifts presets and recalculates friction | Sudden discrete preset change; no continuous transition |
| **Structured Decision Log** | `mission_controller.py`: 742–777 | **VERIFIED IMPLEMENTED** | JSONL records in `logs/mappo_decisions.log` | Flushes in batches of 20 or every 1.0 second |
| **Predictive Digital Twin** | *(Module 3 Scope)* | **NOT IMPLEMENTED** | No forward trajectory prediction exists in Module 2 | Safety filters are purely reactive rule-based |

