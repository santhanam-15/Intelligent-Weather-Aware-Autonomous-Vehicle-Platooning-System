# Existing Safety Mechanisms & Failure Audit

**Document Status:** Source-Verified Safety Audit  
**Date:** October 11, 2026  
**Target:** Baseline Safety Analysis for Module 3 (Twin-World Safety Projection System)  
**Project:** Intelligent Weather-Aware Autonomous Vehicle Platooning System using MAPPO in CARLA  

---

## 1. Executive Summary & Audit Methodology

This document systematically audits every safety mechanism, failure-handling routine, override filter, and boundary check currently implemented in Module 2 and the CARLA 0.9.9 runtime.

### Critical Safety Finding: Purely Reactive vs. Predictive
> [!IMPORTANT]
> **Every existing safety mechanism in Module 2 is 100% REACTIVE and RULE-BASED.**
> There is currently **zero forward-in-time state projection**, zero digital twin prediction, and zero speculative kinematic rollout in the codebase.
> Safety interventions are triggered strictly when an instantaneous sensor measurement (e.g., current clearance $< 3.8$m, or current traffic light within 35m) violates a static threshold.
> This reactive design creates significant vulnerability to dynamic hazards: high-speed closing velocity, abrupt lead-vehicle deceleration, low-friction skidding, and blind cut-ins. This confirms the critical necessity of **Module 3: Twin-World Safety Projection System**.

---

## 2. Comprehensive Inventory of Existing Safety Mechanisms

```mermaid
graph TD
    MAPPO[MAPPO Proposed Tactical Action] --> Hysteresis[Action Hysteresis & Follower Masking]
    Hysteresis --> CtrlTick[Controller Tick: MissionPlatoonController]
    
    subgraph Reactive_Safety_Stack [Existing Reactive Safety Filters (Executed sequentially)]
        RouteLock[1. Route Lock Filter]
        LaneSafety[2. Static Gap Lane-Change Check]
        ObstacleCorr[3. In-Lane Forward Corridor Obstacle Filter]
        SignalGeo[4. Traffic Light Geometric Projection & Amber Check]
        HeadwayRule[5. Predecessor Clearance & Headway Clamp]
        DestCheck[6. Route Horizon & Destination Check]
    end
    
    CtrlTick --> RouteLock --> LaneSafety --> ObstacleCorr --> SignalGeo --> HeadwayRule --> DestCheck
    DestCheck --> OverrideExec[Override Execution: Zero Target Speed]
    OverrideExec --> ActuatorClamp[Actuator Emergency Hard Brake: brake=1.0]
    ActuatorClamp --> CARLA_Physics[CARLA Vehicle Actuation]
```

### Safety Mechanism 1: Predecessor Headway & Emergency Gap Clamp
- **Location:** `WindowsNoEditor/Module2/mission_controller.py`: lines 284–292, 509–531, 617–621, 639–645.
- **Trigger Condition:**
  1. Follower gap error: $\text{gap} < d_{\text{desired}} \times 0.8$ modulates target speed downward.
  2. Urgent following proximity: bumper clearance $< 5.0$m forces $\text{brake} \ge 0.65$.
  3. Emergency violation: bumper clearance $< 3.8$m forces target speed to $0.0$, sets override `INSUFFICIENT_HEADWAY`, and clamps $\text{brake} = 1.0$.
- **Inputs:** `predecessor.get_location()`, `predecessor.get_velocity()`, `vehicle.get_location()`, `vehicle.get_velocity()`, friction $\mu$.
- **Outputs:** Target speed modulation, `override = "INSUFFICIENT_HEADWAY"`, `brake = 1.0`.
- **Order of Execution:** Runs **AFTER** MAPPO action selection, inside `MissionPlatoonController.tick()` (30 Hz).
- **Control Override Capability:** **YES.** Completely overrides any MAPPO action (including `ACCELERATE`).
- **Available Unit Tests:** `test_controller_headway_is_speed_and_friction_aware`, `test_follower_tactical_action_modulates_speed`.
- **Identified Limitations:**
  - Evaluates instantaneous Euclidean distance only; does not forward-simulate closing rate $\Delta v$ or time-to-collision (TTC) over the next 2.0 seconds.
  - Under rain ($\mu = 0.4$), clamping brake at 3.8m is physically insufficient to prevent a rear-end collision if the leader brakes aggressively from 30 km/h.

---

### Safety Mechanism 2: Traffic Light Compliance & Amber Dilemma Zone
- **Location:** `WindowsNoEditor/Module2/mission_controller.py`: lines 386–462, 601–611.
- **Trigger Condition:**
  1. **Red Signal:** Vehicle is within $35.0$m forward distance of stop line box or `vehicle.is_at_traffic_light() == True`.
  2. **Amber Signal:** Computes kinematic stopping distance $d_{\text{needed}} = \frac{v^2}{2 a_{\text{brake}}} + 0.35 v$ (where $a_{\text{brake}} = 3.5 \mu$). If vehicle is within stopping range without skidding, triggers stop; if within dilemma zone ($d_{\text{stop}} < 0.65 d_{\text{needed}}$), allows proceeding.
- **Inputs:** `vehicle.get_traffic_light()`, `vehicle.get_traffic_light_state()`, forward projection vector, friction $\mu$, vehicle speed $v$.
- **Outputs:** `hard_stop = True`, target speed $v_{\text{target}} = \min(v_{\text{target}}, \sqrt{2 \times 2.8 \mu \times d_{\text{stop}}})$, override `RED_LIGHT_STOP` or `AMBER_STOP`.
- **Order of Execution:** Runs **AFTER** MAPPO action selection.
- **Control Override Capability:** **YES.** Forces vehicle deceleration to 0 km/h even if MAPPO requests `ACCELERATE`.
- **Available Unit Tests:**
  - `test_traffic_light_red_stops_vehicle_at_various_speeds`
  - `test_traffic_light_override_suppresses_mappo_accelerate`
  - `test_traffic_light_green_proceeds_after_waiting`
  - `test_traffic_light_amber_dilemma_zone_physics`
  - `test_traffic_light_ignores_light_behind_vehicle`
  - `test_traffic_light_ignores_light_inside_junction`
- **Identified Limitations:**
  - If a signal turns yellow when multiple platoon vehicles are close, the leader might proceed while the follower decides to stop, causing platoon fracture.
  - Relies on CARLA native light binding `vehicle.get_traffic_light()`, which can occasionally fail to bind on complex multi-branch junctions in custom maps.

---

### Safety Mechanism 3: In-Lane Forward Corridor Obstacle Avoidance
- **Location:** `WindowsNoEditor/Module2/mission_controller.py`: lines 293–314, 533–535, 612–616.
- **Trigger Condition:**
  - Scans all CARLA vehicles and pedestrians.
  - Filters out platoon members.
  - Checks if obstacle is inside forward corridor: longitudinal $x \in (0, 80)$m and lateral offset $|y| < 1.65$m (near) or $< 1.95$m (far).
  - If obstacle detected with clearance $< 3.0$m: target speed set to $0.0$, override `COLLISION_RISK`.
- **Inputs:** World actor transforms and bounding boxes.
- **Outputs:** Closest obstacle actor, effective clearance distance, speed ceiling $v_{\text{target}} \le \sqrt{2 \times 3.0 \mu \times d_{\text{clear}}}$, override `COLLISION_RISK`.
- **Order of Execution:** Runs **AFTER** MAPPO action selection.
- **Control Override Capability:** **YES.** Immediately suppresses throttle and applies braking.
- **Available Unit Tests:** `test_obstacle_corridor_ignores_adjacent_lane_traffic`.
- **Identified Limitations:**
  - Static rectangular corridor check cannot anticipate curved trajectories (e.g., an oncoming car turning across the platoon's path).
  - Does not account for obstacle acceleration or velocity vector; treats all obstacles as static hurdles.

---

### Safety Mechanism 4: Action Hysteresis & Follower Action Masking
- **Location:** `WindowsNoEditor/Module2/action/action_manager.py`: lines 32–86.
- **Trigger Condition:**
  - High-frequency action flipping: requires 2 consecutive decisions (`ACTION_HYSTERESIS_STEPS = 2`) before an action switch is committed.
  - Follower lane change request: follower actions $3$ (`CHANGE_LEFT`) and $4$ (`CHANGE_RIGHT`) are unconditionally remapped to $0$ (`CRUISE`).
- **Inputs:** Proposed integer actions from neural policy.
- **Outputs:** Effective tactical action list.
- **Order of Execution:** Runs **DURING** MAPPO action post-processing.
- **Control Override Capability:** **YES.** Filters out high-frequency neural policy chatter.
- **Available Unit Tests:** `test_action_persistence_blocks_one_tick_chatter`, `test_follower_lane_actions_are_masked`.
- **Identified Limitations:**
  - Introduces a minimum delay of $\approx 0.133$s ($2 \times 0.0667$s) before a valid tactical acceleration or deceleration takes effect.

---

### Safety Mechanism 5: Lane Change Feasibility & Route-Lock Enforcement
- **Location:** `WindowsNoEditor/Module2/mission_controller.py`: lines 480–498, 575–582.
- **Trigger Condition:**
  - Leader proposes `CHANGE_LEFT` or `CHANGE_RIGHT`.
  - Rejects if vehicle is near a junction ($\le 15$m).
  - Rejects if target lane does not exist or is not `Driving`.
  - Rejects if any vehicle is within 15.0m in the target lane (`LANE_CHANGE_UNSAFE`).
  - Rejects if planned route enforces single-lane progression (`route_lane_lock`).
- **Inputs:** OpenDRIVE waypoint connectivity, neighbor actor locations.
- **Outputs:** Boolean approval; override string `LANE_CHANGE_UNSAFE` or `route_lane_lock`.
- **Order of Execution:** Runs **AFTER** MAPPO action selection.
- **Control Override Capability:** **YES.** Suppresses lateral maneuver; vehicle remains in current lane.
- **Identified Limitations:**
  - Fixed 15m distance check does not consider target-lane vehicle speed (e.g., a car approaching from behind at 80 km/h will enter the gap in $< 0.7$s).

---

### Safety Mechanism 6: Road Friction & Weather Coupling
- **Location:** `WindowsNoEditor/Module2/mission_controller.py`: lines 265–270, 505, 517; `observation_builder.py`: lines 32–58.
- **Trigger Condition:** Weather preset changes (e.g., Clear $\to$ Heavy Rain). Friction drops from $\mu = 1.0 \to 0.4$.
- **Inputs:** `UncertaintyManager.friction.get_friction()`.
- **Outputs:**
  - Target speeds scaled by $(0.65 + 0.35 \mu)$.
  - Desired headway expanded by $(1.0 - \mu) \times 10.0$ meters.
  - Deceleration bounds clamped by $\mu$.
- **Order of Execution:** Runs **BEFORE** and **AFTER** MAPPO.
- **Available Unit Tests:** `test_weather_headway_increases_when_friction_drops`.
- **Identified Limitations:**
  - Instantaneous friction step change; tire dynamics model in CARLA 0.9.9 does not model longitudinal slip curves or hydroplaning dynamics.

---

### Safety Mechanism 7: Ghost Driver Intrusion Handling & Platoon Splitting
- **Location:** `WindowsNoEditor/Module2/mission_controller.py`: lines 315–375, 536–539.
- **Trigger Condition:** An external non-platoon vehicle penetrates the longitudinal corridor between platoon members ($1.5\text{m} < ox < \text{gap} - 1.5\text{m}$ and $|oy| < 3.2\text{m}$).
- **State Machine Transitions:**
  - Detection: State transitions `NORMAL` $\to$ `INTRUSION`.
  - If intrusion lasts $> 0.5$s: State transitions to `SPLIT`.
  - Rear sub-platoon automatically restricts maximum target speed to $65\%$ of cruise speed ($v_{\text{target}} \le 0.65 v_{\text{cruise}}$) to widen the safety buffer.
  - Clearance: When intruder vacates gap, transitions to `SAFE_FOLLOWING`.
  - Rejoin Eligibility: When bumper gap $\ge 1.25 d_{\text{desired}}$ and vehicle is clear of junctions for $\ge 2.0$s, transitions `REJOIN_ELIGIBLE` $\to$ `REJOINING`.
  - Normalization: After 8.0s of stable rejoining, transitions back to `NORMAL`.
- **Available Unit Tests:** `test_ghost_driver_scripted_scenarios_and_state_transitions`.
- **Identified Limitations:**
  - Lateral positioning during intrusion is not actively controlled; vehicles remain locked to lane centerlines.

---

## 3. Analysis of Critical Safety Gaps & Failure Modes

| Failure Mode / Safety Gap | Current Behavior | Severity | Root Cause in Current Architecture | Module 3 Mitigation |
| :--- | :--- | :---: | :--- | :--- |
| **High Closing-Speed Rear-End Collision** | Follower only brakes hard when physical gap $< 5.0$m. If leader emergency-stops from 30 km/h, follower cannot stop in 5m. | **CRITICAL** | Zero Time-To-Collision (TTC) predictive horizon; relies solely on static distance. | Module 3 forward-rollout calculates future TTC at $t + 1.5$s and rejects `ACCELERATE`. |
| **Wet Road Skidding / Overshoot** | When $\mu=0.4$, braking distance quadruples. Reactive stop line detection at 35m causes red light overshoot. | **CRITICAL** | Friction is used as an instantaneous gain scaler, not a physical trajectory integrator. | Module 3 integrates tire-force limits over the prediction horizon to enforce early braking. |
| **Cut-In Trap During Acceleration** | MAPPO requests `ACCELERATE` while an aggressive NPC is cutting into the platoon gap from an adjacent lane. | **HIGH** | In-lane corridor check only sees obstacles *after* they cross $|y| < 1.65$m. | Twin-world models NPC lateral velocity vector $\dot{y}$ and predicts impending gap penetration. |
| **Platoon Fracture at Amber Signals** | Leader passes stop line under amber dilemma rule; follower abruptly emergency stops. | **HIGH** | Independent, decentralized stop decisions at traffic signals without forward coordination. | Module 3 verifies joint feasibility: if follower cannot clear signal, entire platoon is instructed to stop. |
| **Controller Fighting / Jitter** | MAPPO requests `ACCELERATE` while reactive safety layer requests `RED_LIGHT_STOP`. | **MEDIUM** | MAPPO policy lacks explicit constraint boundary awareness. | Module 3 acts as tactical filter, substituting `DECELERATE` before low-level control. |
| **Observation Stalling / Freezing** | If CARLA actor sensor returns stale/stuck transforms, controller repeats previous command. | **MEDIUM** | No watchdog timer or temporal variance check on incoming state observations. | Module 3 state-snapshot layer validates state timestamp and variance before projection. |

---

## 4. Conflict & Dual Authority Audit

### 1. Autopilot vs. Manual Controller
- **Audit Result:** **RESOLVED & SAFE.**
- **Evidence:** `run_module_2.py` explicitly enforces `veh.set_autopilot(False)` for every platoon member at startup (lines 568–572). No CARLA autopilot thread competes with `MissionPlatoonController`.

### 2. Low-Level Control Authority
- **Audit Result:** **SINGLE AUTHORITATIVE OWNER.**
- **Evidence:** `MissionPlatoonController` is the sole entity calling `vehicle.apply_control()`. The legacy `Module1Adapter` is completely bypassed in the active execution path.

### 3. Route vs. Tactical Lane Changes
- **Audit Result:** **SAFE ROUTE LOCK.**
- **Evidence:** `MissionPlatoonController` actively rejects lane-change requests (`CHANGE_LEFT`, `CHANGE_RIGHT`) when following a fixed mission route (`route_lane_lock`). This prevents vehicles from diverging from pre-computed path coordinates.

