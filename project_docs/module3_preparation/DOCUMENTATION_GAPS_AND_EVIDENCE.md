# Documentation Gaps, Code Discrepancies & Evidence Audit

**Document Status:** Source-Reconciled Evidence Report  
**Date:** October 11, 2026  
**Target:** Pre-Implementation Alignment for Module 3  
**Project:** Intelligent Weather-Aware Autonomous Vehicle Platooning System using MAPPO in CARLA  

---

## 1. Executive Summary

This document identifies every discrepancy between historical project documentation (such as `documentation.txt`, `document.txt`, and legacy docstrings) and the **actual verified source code** in `WindowsNoEditor/Module2/` and `WindowsNoEditor/PythonAPI/`.

Resolving these discrepancies is essential before beginning Module 3 implementation, ensuring that the new safety layer is integrated against the real running code rather than outdated documentation claims.

---

## 2. Master Discrepancy & Evidence Matrix

| Issue / Question | Why It Matters for Module 3 | Evidence Collected from Source Code | How to Verify at Runtime | Priority |
| :--- | :--- | :--- | :--- | :---: |
| **1. Dual Controller Conflict (`MissionPlatoonController` vs `RoadLeaderController`)** | If Module 3 integrates with the wrong controller interface, decisions will never reach CARLA vehicles. | `documentation.txt` (dated Oct 7) claims the platoon uses `RoadLeaderController` and `PlatoonPathController`. In reality, `run_module_2.py` (lines 600, 770, 797) exclusively uses `MissionPlatoonController`. `Module1Adapter` is never called. | Inspect `run_module_2.py`: line 797 calls `mission_controller.tick(elapsed_sim)`. `RoadLeaderController` is never instantiated. | **CRITICAL** |
| **2. Observation Vector Dimension Discrepancy (32-D vs Legacy 14-D / 21-D)** | Module 3 snapshot ingestion and MAPPO state validation depend on knowing the exact tensor schema. | Older comments and deleted checkpoints in `bin/` referenced 14-D and 21-D inputs. Source inspection of `ObservationBuilder` (`observation_builder.py`, lines 21, 200–236) proves the exact dimension is **32**. | Run `test_module2_logic.py`: confirms `obs_dim == 32`, and critic dimension is `32 * 4 == 128`. | **CRITICAL** |
| **3. "Digital Twin" Existence Claim in Existing Docs** | If designers assume a virtual world already exists in Module 2, they may build an incomplete wrapper instead of a true prediction engine. | `document.txt` uses terminology suggesting a predictive digital twin was in place. Source code inspection of `MissionPlatoonController` and `Module2Manager` proves **zero predictive simulation existed**; all safety mechanisms were purely instantaneous threshold rules. | Grep codebase for "rollout", "predict", "simulate": found only in PPO buffer, not in vehicle safety control. | **CRITICAL** |
| **4. Platoon Lane-Change Feasibility (Followers Locked to Lane)** | Module 3 must know whether to evaluate multi-lane platoon formations or single-lane convoying. | Documentation mentions cooperative multi-lane platooning. In code, `ActionManager.process_actions()` (lines 81–84) forcibly remaps follower actions 3 and 4 to `CRUISE`. Furthermore, `MissionPlatoonController` enforces `route_lane_lock`. Followers **never change lanes**. | Inspect `action_manager.py`: `FOLLOWER_ACTION_MAP` maps actions 3 and 4 to `CRUISE`. | **HIGH** |
| **5. Autopilot Status in Platoon Vehicles** | If CARLA autopilot were active, it would silently fight or overwrite MAPPO/Module 3 commands. | Legacy documentation implied CARLA autopilot was used as a fallback. Source inspection of `run_module_2.py` (lines 568–572) proves `veh.set_autopilot(False)` is explicitly enforced for all platoon vehicles at spawn. | Check simulation logs: search for `[AUTOPILOT ENFORCED]`. Platoon vehicles strictly obey `apply_control`. | **HIGH** |
| **6. Simulation Tick Rate and Pacing (30 Hz vs 15 Hz)** | Module 3 must know its invocation frequency and maximum permissible execution budget. | `config.py` defines `CONTROL_DT = 1.0 / 30.0` (30 Hz) and `STATE_EXTRACTION_HZ = 15.0` (15 Hz). Low-level control runs at 30 Hz; MAPPO decisions occur every second tick (15 Hz). Older docs claimed 20 Hz or 10 Hz. | Inspect `run_module_2.py`: lines 695, 762 confirm 30 Hz physics loop with 15 Hz decision throttling. | **HIGH** |
| **7. Active Checkpoint Freshness & Update Counter** | Must verify whether evaluation uses trained weights or random initializations. | User expressed concern that `mappo_checkpoint.pt` was stale (Oct 7). Our audit updated and verified `checkpoints/mappo_checkpoint.pt` at update step **121** with matching SHA-256 fingerprinting. | Check startup banner in `run_module_2.py`: logs exact step number and SHA-256 hash. | **HIGH** |
| **8. Sudden Discrete Weather Shifts vs Continuous Rain Accumulation** | Module 3 friction prediction must know whether $\mu$ changes gradually or instantaneously. | `run_module_2.py` (line 747) switches weather presets discretely (`CLEAR` $\to$ `HEAVY_RAIN`), instantly stepping friction $\mu$ from $1.0 \to 0.4$. There is no continuous puddling or road wetting model in CARLA 0.9.9. | Check `real_world_simulation.py`: `apply_friction` alters vehicle wheel physics immediately. | **MEDIUM** |
| **9. CARLA Vector3D API Limitation in 0.9.9** | Prevents unexpected runtime attribute crashes during vector calculations in Module 3. | In CARLA 0.9.9, `carla.Vector3D` objects lack a `.length()` method. Calling `.length()` raised an `AttributeError` in `ghost_driver.py`. Must use $\sqrt{x^2 + y^2 + z^2}$ across all Module 3 code. | Run Python test: `hasattr(carla.Vector3D(), 'length')` returns `False`. | **MEDIUM** |

---

## 3. Key Design Decisions Requiring User Approval Before Implementation

Before proceeding to Module 3 code implementation, the following architectural choices must be formally confirmed:

### Decision 1: Virtual World Physics Paradigm (Approach 2 vs Approach 4)
- **Proposed Choice:** **Approach 2 (High-Performance Vectorized NumPy Kinematic Bicycle Model)**.
- **Rationale:** Runs in $< 2.5$ ms in CPU memory; zero GPU VRAM impact (essential for 4GB RTX 3050 GPUs); avoids running a second CARLA server instance.
- **Alternative:** Dual CARLA server instances on ports 2000 and 2002 (Rejected due to high VRAM cost and 60–100 ms RPC latency).
- **Status:** **PENDING USER CONFIRMATION.**

### Decision 2: Predictive Time Horizon & Time Step
- **Proposed Choice:** $T_{\text{horizon}} = 2.0$ seconds with $\Delta t = 0.1$ seconds ($K = 20$ discrete steps).
- **Rationale:** 2.0 seconds corresponds to typical highway stopping headway and provides adequate lookahead at 30 km/h ($\approx 16.7$ meters) without excessive compute overhead.
- **Alternative:** 3.0 seconds with $\Delta t = 0.05$ seconds ($K = 60$ steps, 3× higher compute).
- **Status:** **PENDING USER CONFIRMATION.**

### Decision 3: Safety Substitution Arbitration Policy
- **Proposed Choice:** If proposed tactical action is unsafe, evaluate fallback ladder in order: `CRUISE` $\to$ `DECELERATE` $\to$ `BRAKE`.
- **Rationale:** Minimizes unnecessary harsh braking when gentle speed holding or mild deceleration can resolve the projected hazard.
- **Alternative:** Immediate hard emergency brake on any predictive violation.
- **Status:** **PENDING USER CONFIRMATION.**

### Decision 4: Follower Platoon Lane Change Policy
- **Proposed Choice:** Maintain current architectural constraint: followers remain locked in-lane; only the leader may evaluate tactical lane changes.
- **Rationale:** CARLA 0.9.9 multi-lane waypoint routing for follower platoons requires complex multi-vehicle corridor reservation; keeping followers in-lane ensures stability and avoids platoon scattering.
- **Alternative:** Implement multi-vehicle coordinated lane-shift maneuvers across adjacent lanes.
- **Status:** **PENDING USER CONFIRMATION.**

