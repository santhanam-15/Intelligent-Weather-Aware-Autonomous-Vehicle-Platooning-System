# Control Flow & Interface Map: From MAPPO Decisions to CARLA Actuation

**Document Status:** Source-Verified Architecture Map  
**Date:** October 11, 2026  
**Target Module:** Module 3 (Twin-World Safety Projection System) Integration Design  
**Project:** Intelligent Weather-Aware Autonomous Vehicle Platooning System using MAPPO in CARLA  

---

## 1. End-to-End Control Flow Architecture

The simulation operates a two-rate synchronous pipeline:
- **30 Hz Outer Loop:** CARLA physics engine, low-level trajectory tracking, and safety overrides (`CONTROL_DT = 1.0 / 30.0` s).
- **15 Hz Tactical Loop:** MAPPO observation construction, neural policy evaluation, action hysteresis filtering, and multi-agent coordination (`STATE_EXTRACTION_INTERVAL = 1.0 / 15.0` s).

```mermaid
flowchart TD
    subgraph CARLA_30Hz [CARLA Synchronous Simulation (30 Hz Tick)]
        Tick[world.tick & get_snapshot]
        EnvUpdate[Weather & Friction Update]
        TrafficUpdate[Traffic Manager & Ghost Driver Tick]
        Tick --> EnvUpdate --> TrafficUpdate
    end

    subgraph MAPPO_15Hz [Tactical MAPPO Decision Loop (15 Hz Interval)]
        ObsBuild[ObservationBuilder.get_local_observation<br/>Shape: N × 32]
        CriticState[ObservationBuilder.get_global_state<br/>Shape: 1 × 128]
        ActorForward[SharedActor.forward / sample_action<br/>Categorical distribution over 5 discrete actions]
        RawActions[Raw Proposed Actions: a_raw ∈ {0,1,2,3,4}]
        Hysteresis[ActionManager.process_actions<br/>2-Step Persistence Filter & Follower Masking]
        TacticalAction[Effective Tactical Actions<br/>CRUISE, ACCELERATE, DECELERATE, ...]
        
        ObsBuild --> ActorForward --> RawActions --> Hysteresis --> TacticalAction
        ObsBuild --> CriticState
    end

    subgraph Module3_Integration [Recommended Module 3 Integration Boundary]
        M3_In["Input: Proposed Tactical Action + Real World State Snapshot"]
        M3_Virtual["Twin-World Forward Rollout (Virtual Shadow Projection)"]
        M3_Risk["Multi-Objective Predictive Safety Evaluator"]
        M3_Out["Approved / Filtered Tactical Action a*"]
        M3_In --> M3_Virtual --> M3_Risk --> M3_Out
    end

    subgraph LowLevel_30Hz [MissionPlatoonController (30 Hz Execution)]
        RouteGeo[RoutePlanner._route_geometry<br/>Pure Pursuit Lookahead & Steer Calculation]
        TrafficCheck[_traffic_light_constraint<br/>Geometric Projection & Dilemma Zone Check]
        ObstacleCheck[_closest_forward_obstacle<br/>In-Lane Forward Corridor Filtering]
        TargetSpeedCalc[_target_speed<br/>Tactical Delta + Headway Bias + Friction Bound]
        SafetyOverride[Reactive Safety Override Engine<br/>RED_LIGHT, AMBER_STOP, COLLISION_RISK, INSUFFICIENT_HEADWAY]
        PID_Control[Longitudinal P-Controller & Brake Logic]
        Actuate[vehicle.apply_control<br/>carla.VehicleControl throttle, brake, steer]

        RouteGeo --> TargetSpeedCalc
        TrafficCheck --> SafetyOverride
        ObstacleCheck --> SafetyOverride
        TargetSpeedCalc --> SafetyOverride --> PID_Control --> Actuate
    end

    TrafficUpdate -->|Every 2nd tick| ObsBuild
    TacticalAction -.->|Proposed Path Without M3| TargetSpeedCalc
    TacticalAction -->|Proposed Tactical Action| M3_In
    M3_Out -->|Approved Action a*| TargetSpeedCalc
    Actuate --> Tick
```

---

## 2. Detailed Step-by-Step Execution Sequence

### Phase 1: Simulation Time & Environment Tick (30 Hz)
- **Source:** `WindowsNoEditor/PythonAPI/run_module_2.py`, lines 732–760.
- `snapshot = world.get_snapshot()` reads the authoritative simulation frame and elapsed time.
- `delta_sec = snapshot.timestamp.delta_seconds` (fixed at 0.0333s).
- `uncertainty_manager.tick()` updates weather, precipitation, and tire-road friction multipliers $\mu \in [0.2, 1.2]$.
- `timeline_director.tick()` advances background NPC behaviors and steps the Ghost Driver possession state machine.

### Phase 2: MAPPO Tactical Decision (15 Hz)
- **Source:** `run_module_2.py`: 774; `module2_manager.py`: 155–230.
- Guard condition: `if current_time - self.last_decision_time < config.STATE_EXTRACTION_INTERVAL: return cached_actions`.
- **Step 2.1: Observation Gathering (`ObservationBuilder`)**
  - Collects local 32-dimensional observation vector $o_i$ for each vehicle index $i \in \{0, \dots, N-1\}$.
  - Validates that values are finite (`np.isfinite(o_i).all()`).
  - Concatenates local observations into 128-dimensional global state $s = [o_0, o_1, o_2, o_3]$ (zero-padded if $N < 4$).
- **Step 2.2: Neural Policy Inference (`SharedActor`)**
  - Evaluates logits: $z_i = \text{Linear}_{64 \to 5}(\text{Tanh}(\text{Linear}_{64 \to 64}(\text{Tanh}(\text{Linear}_{32 \to 64}(o_i)))))$.
  - Applies action mask: followers receive mask $[1, 1, 1, 0, 0]$, masking out lane change actions $3$ and $4$.
  - In evaluation mode: selects greedy action $a_i^{\text{raw}} = \text{argmax}(z_i)$.
  - In training mode: samples $a_i^{\text{raw}} \sim \text{Categorical}(\text{Softmax}(z_i))$.
- **Step 2.3: Action Hysteresis Filtering (`ActionManager`)**
  - Inspects proposed action $a_i^{\text{raw}}$.
  - If $a_i^{\text{raw}} \ne a_{i, \text{prev}}^{\text{raw}}$, increments candidate counter. Only commits the change if the proposed action persists for at least 2 consecutive decisions (`ACTION_HYSTERESIS_STEPS = 2`).
  - Followers have actions $3$ and $4$ forcibly translated to `CRUISE`.
  - Produces effective tactical action strings: e.g., `["ACCELERATE", "CRUISE", "CRUISE"]`.
- **Step 2.4: Tactical Telemetry Record**
  - Generates a unique `decision_id` (`DEC-{frame}-{veh_id}-{timestamp_ms}`) binding raw logits, action probabilities, sampled action, and tactical action string.

### Phase 3: Tactical Delegation to Controller
- **Source:** `run_module_2.py`, line 780.
- `mission_controller.set_tactical_actions(tactical_actions)` stores the active action per vehicle.

### Phase 4: Low-Level Controller Execution & Safety Overrides (30 Hz)
- **Source:** `WindowsNoEditor/Module2/mission_controller.py`, lines 546–770 (`MissionPlatoonController.tick()`).
- Executed on every 30 Hz tick for every active platoon vehicle:

1. **Lateral Tracking:**
   - Queries `RoutePlanner` for target waypoint at lookahead distance $L_a = \text{clamp}(5.0 + 0.7 \times v, 6.0, 16.0)$m.
   - Computes local coordinate error $(x_{\text{local}}, y_{\text{local}})$ and angular error $\alpha = \text{atan2}(y_{\text{local}}, x_{\text{local}})$.
   - Computes Pure Pursuit steering angle $\delta_{\text{raw}} = \text{atan2}(2.7 \times \sin(\alpha), L_a) \times 1.8$.
   - Applies exponential smoothing filter and rate clamping: $\delta_t = \text{clamp}(0.78 \times \delta_{t-1} + 0.22 \times \delta_{\text{raw}}, \delta_{t-1} - 0.08, \delta_{t-1} + 0.08)$.
2. **Forward Hazard Checks:**
   - `_closest_forward_obstacle(vehicle)`: scans for vehicles and walkers within in-lane corridor (width $\pm 1.65$m near, $\pm 1.95$m far).
   - `_traffic_light_constraint(vehicle)`: evaluates signal state, stop line trigger volume, junction presence, and dilemma zone stopping distance.
   - `_vehicle_clearance(predecessor, vehicle)`: calculates physical gap to leading platoon member.
3. **Tactical Target Speed Calculation:**
   - Modulates base speed using tactical action: $\Delta v = +1.1\text{ m/s}$ (`ACCELERATE`), $-1.8\text{ m/s}$ (`DECELERATE`), or $0.0$ (`CRUISE`).
   - Scales target speed by road surface friction $\mu$: factor $(0.65 + 0.35 \times \mu)$.
   - For followers: calculates desired headway $d_{\text{desired}} = \text{clamp}(5.0 + 0.9 v + 0.25 v_{\text{close}} + (1 - \mu) \times 10.0, 6.0, 30.0)$ and adds headway bias $\text{clamp}((\text{gap} - d_{\text{desired}}) \times 0.20, -2.5, 2.0)$.
4. **Safety Overrides & Emergency Intervention:**
   - **Destination Complete:** If within 6m and $\le 3$ waypoints of route end: $v_{\text{target}} = 0$, override = `destination_complete`.
   - **Red Traffic Light:** If $d_{\text{light}} \le 35$m or at stop line: $v_{\text{target}} \to 0$, override = `RED_LIGHT_STOP`.
   - **Amber Traffic Light:** If stopping is feasible without severe wheel slip: $v_{\text{target}} \to 0$, override = `AMBER_STOP`.
   - **Obstacle Proximity:** If obstacle clearance $< 3.0$m: $v_{\text{target}} = 0$, override = `COLLISION_RISK`.
   - **Predecessor Headway:** If follower bumper gap $< 3.8$m: $v_{\text{target}} = 0$, override = `INSUFFICIENT_HEADWAY`.
   - **Unsafe Lane Change:** If leader requests lane change but adjacent lane is occupied: override = `LANE_CHANGE_UNSAFE`.
5. **Actuation Generation & Direct CARLA Application:**
   - Proportional speed tracking error: $e_v = v_{\text{target}} - v_{\text{current}}$.
   - If stopping ($v_{\text{target}} \le 0.05$ m/s): $\text{throttle} = 0.0, \text{brake} = 1.0$ (or $0.5$ standstill hold).
   - If driving: $\text{throttle} = \text{clamp}(0.23 \times e_v, 0.0, 0.65)$, $\text{brake} = \text{clamp}(-0.42 \times e_v, 0.0, 1.0)$.
   - Follower emergency hard-braking clamp: if $\text{gap} < 5.0$m, $\text{brake} \ge 0.65$; if $\text{gap} < 3.8$m, $\text{brake} = 1.0$.
   - Applies command: `vehicle.apply_control(carla.VehicleControl(throttle, brake, steer))`.

---

## 3. Component Interface & Data Specification Table

| Producer | Consumer | Data Passed | Data Type & Shape | Frequency | Exact Source Location |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CARLA Simulator** | `run_module_2.py` | World Snapshot | `carla.WorldSnapshot` | 30 Hz | `run_module_2.py`: line 732 |
| `UncertaintyManager` | `ObservationBuilder` | Weather & Friction | `float` ($\mu \in [0.2, 1.2]$) | 15 Hz | `observation_builder.py`: line 167 |
| `MissionPlatoonController` | `ObservationBuilder` | Route/Safety Context | `dict` (progress, errors, TL state) | 15 Hz | `mission_controller.py`: lines 714–739 |
| `ObservationBuilder` | `SharedActor` | Local Observations | `np.ndarray (N, 32)` float32 | 15 Hz | `module2_manager.py`: line 181 |
| `ObservationBuilder` | `CentralizedCritic` | Global State | `np.ndarray (1, 128)` float32 | 15 Hz | `module2_manager.py`: line 229 |
| `SharedActor` | `ActionManager` | Raw Actions & Logits | `torch.Tensor (N,)` int64, logits (N, 5) | 15 Hz | `actor.py`: lines 33–44 |
| `ActionManager` | `Module2Manager` | Filtered Tactical Actions | `List[str]` length $N$ (e.g. `['CRUISE', ...]`) | 15 Hz | `action_manager.py`: lines 85–90 |
| `Module2Manager` | `MissionPlatoonController` | Tactical Actions | `List[str]` length $N$ | 15 Hz | `run_module_2.py`: line 780 |
| `Module2Manager` | `MissionPlatoonController` | Decision Audit Records | `dict` mapping `veh_id -> record` | 15 Hz | `module2_manager.py`: lines 201–219 |
| `MissionPlatoonController` | `carla.Vehicle` | Low-Level Controls | `carla.VehicleControl(thr, brk, str)` | 30 Hz | `mission_controller.py`: line 706 |
| `MissionPlatoonController` | File System | Decision Log Lines | JSONL serialized strings | Batch / 1s | `mission_controller.py`: lines 771–774 |
| `CarlaSimulationThread` | UI Thread (`UnifiedHUD`) | Telemetry Snapshot | `TelemetrySnapshot` object | 30 Hz | `run_module_2.py`: line 885 |

---

## 4. Verification of Critical Control Points

### Point 1: Exact Location Where MAPPO Proposes an Action
- **File:** `WindowsNoEditor/Module2/mappo/actor.py`, lines 33–44 (`SharedActor.sample_action`).
- **File:** `WindowsNoEditor/Module2/module2_manager.py`, lines 181–183:
  ```python
  raw_actions, log_probs, raw_logits, action_probs = self.trainer.select_action(
      local_obs_list, deterministic=(self.mode == "evaluate"), return_details=True
  )
  ```

### Point 2: Exact Location Where the Final Tactical Action is Chosen
- **File:** `WindowsNoEditor/Module2/action/action_manager.py`, lines 70–86 (`ActionManager.process_actions`):
  ```python
  if self._candidate_counts[i] >= config.ACTION_HYSTERESIS_STEPS:
      effective = proposed
  ...
  action_str = ACTION_MAP.get(effective, "CRUISE")
  ```

### Point 3: Exact Location Where Existing Safety Overrides Occur
- **File:** `WindowsNoEditor/Module2/mission_controller.py`, lines 575–625:
  - Lane change rejection: lines 575–581 (`LANE_CHANGE_UNSAFE`, `route_lane_lock`).
  - Destination complete stop: lines 596–600 (`destination_complete`).
  - Red/Amber traffic light override: lines 601–611 (`RED_LIGHT_STOP`, `AMBER_STOP`).
  - Collision risk stop: lines 612–616 (`COLLISION_RISK`).
  - Insufficient headway stop: lines 617–621 (`INSUFFICIENT_HEADWAY`).

### Point 4: Exact Location Where Throttle, Brake, and Steering are Applied
- **File:** `WindowsNoEditor/Module2/mission_controller.py`, lines 702–706:
  ```python
  control = carla.VehicleControl(
      throttle=float(throttle), brake=float(brake), steer=float(_clamp(steer, -0.65, 0.65)),
      hand_brake=False, reverse=False,
  )
  vehicle.apply_control(control)
  ```

---

## 5. Dual Control Path Analysis: Active vs. Stale Paths

Source inspection identified two separate control path implementations in the repository:

1. **Path A (ACTIVE): `MissionPlatoonController` (in `mission_controller.py`)**
   - **Invocation:** Directly initialized and called in `run_module_2.py` (lines 600, 770, 797).
   - **Status:** **Active Authoritative Controller**. Sole owner of the vehicle route, headway tracking, traffic light compliance, and actuation.
2. **Path B (STALE / INACTIVE): `Module1Adapter` (in `integration/module1_adapter.py`)**
   - **Invocation:** Referenced only in `module2_manager.py:apply_actions()`, which is never called anywhere in `run_module_2.py`.
   - **Status:** **Legacy / Inactive Adapter**. Designed for an older architecture where `RoadLeaderController` and `PlatoonPathController` were separate classes.
   - **Impact on Module 3:** Module 3 must interface with **Path A** (`MissionPlatoonController`), NOT with `Module1Adapter`.

---

## 6. Recommended Integration Boundary for Module 3

The safest and most modular integration point for Module 3 is located between **Phase 2 (MAPPO Tactical Decision)** and **Phase 4 (MissionPlatoonController Target Speed Calculation)**:

```
[Module 2 MAPPO Policy]
          │
          ▼  Proposed Tactical Action a_proposed ∈ {CRUISE, ACCELERATE, DECELERATE, CHANGE_L, CHANGE_R}
┌────────────────────────────────────────────────────────┐
│  RECOMMENDED INTEGRATION POINT: MODULE 3 INTERCEPTOR   │
│                                                        │
│  Input:                                                │
│    - Proposed tactical action a_proposed               │
│    - Current real-world snapshot (poses, velocities,   │
│      weather friction, traffic light, obstacle states) │
│                                                        │
│  Output:                                               │
│    - Approved / Replaced tactical action a*            │
│    - Rejection reason code (if overridden)             │
│    - Predictive risk metrics (TTC, headway margin)     │
└────────────────────────────────────────────────────────┘
          │
          ▼  Approved Action a*
[MissionPlatoonController: Low-Level Speed & Lateral Execution]
```

### Why This is the Safest Boundary:
1. **Preserves Module 2 Integrity:** Does not alter the 32-D observation vector, rollout buffer, or PPO training pipeline.
2. **Preserves Module 4 Actuation:** Leaves lateral Pure Pursuit tracking and low-level throttle/brake generation untouched in `MissionPlatoonController`.
3. **True Predictive Tactical Filtering:** Operates at the tactical level (modulating speeds and approving lane changes) before mechanical throttle/brake commands are formed, preventing abrupt controller fighting or actuator jitter.

