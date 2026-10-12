# Module 3: Requirements & Interface Contract (Twin-World Safety Projection System)

**Document Status:** Specification & Interface Contract  
**Design Phase:** Pre-Implementation Specification  
**Notice:** All interfaces specified in this document are **PROPOSED — NOT YET IMPLEMENTED**.  
**Project:** Intelligent Weather-Aware Autonomous Vehicle Platooning System using MAPPO in CARLA  

---

## 1. System Mission & Core Architectural Principles

Module 3 is the **Twin-World Safety Projection System**. Its fundamental purpose is:
> **Evaluate the safety of MAPPO's proposed tactical action using a virtual prediction of near-future vehicle states, then approve the action or substitute a safer alternative when required.**

### Four-Layer Architectural Separation
```mermaid
graph TD
    subgraph Layer1 [1. Real World (CARLA 0.9.9 Physical Simulation)]
        RealCAVs[Real Platoon Vehicles]
        RealNPCs[Real Traffic & Pedestrians]
        RealEnv[Real Road Surface, Weather & Signals]
    end

    subgraph Layer2 [2. Virtual World (Isolated Predictive Digital Twin)]
        Snapshot[Real-World State Snapshot Generator]
        TwinEngine[Forward Kinematic Projection Engine<br/>Horizon: T_proj = 2.0s, Δt = 0.1s]
        SpeculativeActors[Shadow CAVs & Shadow Surrounding Actors]
        Snapshot --> TwinEngine --> SpeculativeActors
    end

    subgraph Layer3 [3. Safety Decision Layer (Predictive Risk Evaluator)]
        RiskEval[Multi-Criterion Risk Evaluator<br/>TTC, Minimum Gap, Stop-Line Clearance, Friction Limits]
        DecisionGate[Safety Gate & Action Selector<br/>Approve, Reject, or Substitute Safer Action]
        RiskEval --> DecisionGate
    end

    subgraph Layer4 [4. Real-World Execution (Module 4 Low-Level Control)]
        Execute[MissionPlatoonController.tick<br/>Executes Approved Action a* Only]
        Actuate[carla.VehicleControl<br/>Throttle, Brake, Steering]
        Execute --> Actuate
    end

    Layer1 -->|Read-Only Snapshot| Snapshot
    SpeculativeActors --> RiskEval
    DecisionGate -->|Approved Action a*| Execute
    Actuate --> Layer1
```

### Mandatory Non-Interference Constraints
1. **Speculative Isolation:** The virtual projection engine must **NEVER** apply candidate or speculative control commands to real CARLA vehicle actors.
2. **Read-Only Ingestion:** The virtual world receives a read-only copy of state data.
3. **Deterministic Rollback:** Virtual state rollouts take place in CPU memory space without altering CARLA server state.
4. **Preservation of MAPPO:** Module 3 does not train or modify the MAPPO policy weights; it acts as an online predictive safety filter.

---

## 2. Functional Requirements for Module 3

### R1. State Snapshot Ingestion
- **R1.1:** Must capture an instantaneous, synchronized state snapshot of all platoon vehicles ($x, y, z, v_x, v_y, \text{yaw}, \text{steering angle}$).
- **R1.2:** Must capture all non-platoon obstacle vehicles and pedestrians within an active safety radius ($R_{\text{safety}} = 60.0$m) around the platoon centroid.
- **R1.3:** Must ingest current environmental parameters: road friction multiplier $\mu$, active traffic light state, stop line distance, and route waypoint coordinates.

### R2. Predictive Forward Rollout Engine
- **R2.1 (Temporal Horizon):** Must project forward across a lookahead horizon of $T_{\text{horizon}} = 2.0$ seconds.
- **R2.2 (Discretization):** Must evaluate state trajectories using discrete steps of $\Delta t_{\text{sim}} = 0.1$ seconds ($20$ projection steps per rollout).
- **R2.3 (Vehicle Kinematics):** Must simulate CAV trajectories using a bicycle kinematic model incorporating tire-road friction limits:
  $$a_{\text{max, accel}} = 2.5 \times \mu \quad (\text{m/s}^2), \qquad a_{\text{max, brake}} = 5.0 \times \mu \quad (\text{m/s}^2)$$
- **R2.4 (Surrounding Actor Prediction):** Surrounding NPCs must be projected forward using constant-velocity constant-yaw-rate extrapolation ($\text{CV-CYR}$).

### R3. Multi-Objective Risk Evaluation
The virtual rollout must compute four explicit risk metrics across the trajectory:
1. **Time-To-Collision (TTC):**
   $$\text{TTC}(t) = \frac{d_{\text{clearance}}(t)}{\max(0.01, v_{\text{rear}}(t) - v_{\text{front}}(t))}$$
   - Safety threshold: $\text{TTC}_{\text{min}} \ge 1.8$ seconds.
2. **Inter-Vehicle Headway Margin:**
   $$d_{\text{gap}}(t) \ge d_{\text{safe, min}} = \max(4.0, 1.0 \times v(t) + (1.0 - \mu) \times 8.0) \quad \text{meters}$$
3. **Traffic Signal Stop-Line Margin:**
   - For red/stopping yellow signals: predicted vehicle location at stop line must satisfy $x_{\text{proj}}(t) \le x_{\text{stopline}} + 0.5$m.
4. **Lateral Stability & Lane Boundary Margin:**
   - Predicted cross-track error must remain within active lane bounds: $|e_{\text{lat}}(t)| \le 1.4$m.

### R4. Candidate Action Arbitration & Substitution Policy
- **R4.1 (Approval):** If MAPPO's proposed action $a_{\text{prop}}$ satisfies all risk criteria across $t \in [0, T_{\text{horizon}}]$, approve $a^* = a_{\text{prop}}$.
- **R4.2 (Substitution):** If $a_{\text{prop}}$ violates any safety constraint, evaluate candidate alternative actions in ranked order:
  1. `CRUISE` (Hold steady speed)
  2. `DECELERATE` (Gentle tactical braking, $-1.8$ m/s)
  3. `BRAKE` (Full safe deceleration, $-4.0 \mu$ m/s$^2$)
- **R4.3 (Emergency Fallback):** If no candidate action satisfies the minimum safety threshold, select `EMERGENCY_BRAKE` and signal Module 4 to execute maximal deceleration.

### R5. Fail-Safe & Timeout Handling
- **R5.1 (Computation Budget):** Prediction and risk evaluation must complete within **12.0 milliseconds** per decision cycle.
- **R5.2 (Timeout Fallback):** If projection exceeds the timeout budget, the system aborts the rollout, logs `M3_TIMEOUT_EXCEEDED`, and executes deterministic fail-safe following (`CRUISE` or `DECELERATE`).
- **R5.3 (Invalid Data Fallback):** If the snapshot contains NaN, infinite, or stale timestamps ($> 0.1$s old), logs `M3_INVALID_SNAPSHOT` and falls back to reactive headway maintenance.

---

## 3. Proposed Interface Contracts (PROPOSED — NOT YET IMPLEMENTED)

All data structures below define the exact Python/NumPy schemas for Module 3.

### Interface 1: Real-World State Snapshot (`TwinWorldSnapshot`)
```python
# PROPOSED — NOT YET IMPLEMENTED
class TwinWorldSnapshot:
    """Read-only instantaneous snapshot of real CARLA environment."""
    def __init__(self):
        self.frame: int = 0
        self.timestamp: float = 0.0          # Authoritative simulation time in seconds
        self.friction_mu: float = 1.0        # Road friction [0.2, 1.2]
        self.weather_name: str = "CLEAR"
        
        # Platoon CAV States: List of Dicts (length N)
        # Each entry: {
        #   "id": int, "role": str,
        #   "pos": Tuple[float, float, float],    # (x, y, z) in meters
        #   "vel": Tuple[float, float, float],    # (vx, vy, vz) in m/s
        #   "yaw": float,                         # Heading in degrees
        #   "steer": float,                       # Current steering [-1.0, 1.0]
        #   "speed": float                        # Scalar speed in m/s
        # }
        self.platoon_states: List[Dict[str, Any]] = []

        # Surrounding Obstacles: List of Dicts within 60m radius
        # Each entry: {
        #   "id": int, "type": str,               # 'vehicle' or 'walker'
        #   "pos": Tuple[float, float, float],
        #   "vel": Tuple[float, float, float],
        #   "yaw": float,
        #   "extent": Tuple[float, float, float]  # Bounding box extents (x, y, z)
        # }
        self.surrounding_actors: List[Dict[str, Any]] = []

        # Route & Infrastructure Context
        self.traffic_light_state: str = "NONE"    # 'RED', 'YELLOW', 'GREEN', 'NONE'
        self.traffic_light_dist: float = 80.0
        self.stop_line_location: Optional[Tuple[float, float]] = None
        self.current_lane_id: int = 0
        self.next_waypoints: List[Tuple[float, float, float]] = [] # Next 10 wps
```

### Interface 2: Tactical Action Request (`CandidateActionRequest`)
```python
# PROPOSED — NOT YET IMPLEMENTED
class CandidateActionRequest:
    """Input query to Module 3 Twin-World Evaluator."""
    def __init__(self):
        self.decision_id: str = ""                # e.g., 'DEC-1204-236-4520'
        self.vehicle_id: int = 0
        self.role: str = "Leader"                 # 'Leader' or 'Follower 1'
        self.proposed_action: str = "CRUISE"      # 'CRUISE', 'ACCELERATE', 'DECELERATE', etc.
        self.raw_action_id: int = 0
        self.current_speed_mps: float = 0.0
```

### Interface 3: Virtual Projected Trajectory (`ProjectedTrajectoryResult`)
```python
# PROPOSED — NOT YET IMPLEMENTED
class ProjectedTrajectoryResult:
    """Output of virtual forward rollout across horizon."""
    def __init__(self):
        self.candidate_action: str = ""
        self.horizon_sec: float = 2.0
        self.num_steps: int = 20                  # 20 steps at dt=0.1s
        self.time_array: np.ndarray = np.zeros(20, dtype=np.float32)
        
        # Trajectories: shape (20,)
        self.projected_speeds: np.ndarray = np.zeros(20, dtype=np.float32)
        self.projected_x: np.ndarray = np.zeros(20, dtype=np.float32)
        self.projected_y: np.ndarray = np.zeros(20, dtype=np.float32)
        
        # Platoon Interactions: shape (20,)
        self.clearance_to_predecessor: np.ndarray = np.zeros(20, dtype=np.float32)
        self.estimated_ttc: np.ndarray = np.zeros(20, dtype=np.float32)
        
        # Boundary & Obstacle Margins
        self.min_obstacle_distance: float = 80.0
        self.stop_line_overshoot: float = 0.0     # > 0 indicates crossed stop line on RED
        self.is_kinematically_feasible: bool = True
```

### Interface 4: Predictive Risk Metrics & Decision (`SafetyDecisionReport`)
```python
# PROPOSED — NOT YET IMPLEMENTED
class SafetyDecisionReport:
    """Authoritative decision returned by Module 3 to the controller."""
    def __init__(self):
        self.decision_id: str = ""
        self.vehicle_id: int = 0
        self.original_proposed_action: str = ""
        self.approved_action: str = ""            # The tactical action to execute
        self.is_overridden: bool = False          # True if proposed action was changed
        self.override_reason: str = "NONE"        # One of REASON CODES below
        
        # Computed Safety Metrics
        self.min_projected_ttc: float = 99.0      # Minimum TTC across horizon
        self.min_projected_clearance: float = 99.0# Minimum gap across horizon
        self.projected_stop_line_dist: float = 80.0
        self.risk_severity_score: float = 0.0     # [0.0 (Safe) to 1.0 (Critical Hazard)]
        
        # Diagnostic Telemetry
        self.projection_compute_time_ms: float = 0.0
        self.evaluated_candidates: List[str] = []
        self.status: str = "SUCCESS"              # 'SUCCESS', 'FALLBACK_TIMEOUT', 'FALLBACK_DATA_ERROR'
```

---

## 4. Standardized Rejection & Override Reason Codes

Every safety override emitted by Module 3 must utilize one of these standardized reason codes:

| Reason Code | Trigger Condition in Virtual World | Typical Action Replacement |
| :--- | :--- | :--- |
| `M3_PREDICTIVE_COLLISION_RISK` | Virtual forward rollout predicts $\text{TTC} < 1.8\text{s}$ with forward obstacle. | `DECELERATE` or `BRAKE` |
| `M3_INSUFFICIENT_FUTURE_HEADWAY` | Virtual forward rollout predicts predecessor clearance $< d_{\text{safe, min}}(t)$. | `DECELERATE` |
| `M3_PREDICTIVE_RED_LIGHT_OVERSHOOT` | Virtual projection indicates vehicle will cross stop line while signal is Red. | `BRAKE` |
| `M3_AMBER_DILEMMA_UNSAFE_PROCEED` | Virtual projection indicates stopping is possible, but accelerating is hazardous. | `DECELERATE` |
| `M3_LOW_FRICTION_STOP_DISTANCE_DEFICIT` | Projected braking distance under current $\mu$ exceeds available gap. | `DECELERATE` |
| `M3_INTRUDING_GHOST_CUT_IN_PROJECTED` | Lateral velocity of possessed vehicle projects into platoon gap within $1.5$s. | `DECELERATE` |
| `M3_LANE_CHANGE_COLLISION_PROJECTED` | Candidate lane change trajectory intercepts adjacent-lane vehicle envelope. | Revert to `CRUISE` |
| `M3_TIMEOUT_EXCEEDED` | Projection computation exceeded 12 ms budget limit. | Safe fallback (`CRUISE`) |
| `M3_INVALID_SNAPSHOT_DATA` | State snapshot contained non-finite values or stale timestamps. | Safe fallback (`CRUISE`) |

---

## 5. Fallback Decision Matrix

When candidate actions are evaluated, Module 3 follows this deterministic arbitration ladder:

```
Proposed: ACCELERATE
   │
   ├─► Virtual Rollout Safe? ──► YES ──► Approve: ACCELERATE (Override: NONE)
   │
   └─► NO (Collision / Headway / Signal Risk Detected)
         │
         ├─► Evaluate CRUISE Rollout Safe? ──► YES ──► Approve: CRUISE (Override: M3_PREDICTIVE_...)
         │
         └─► NO (Gap still shrinking or hazard approaching)
               │
               ├─► Evaluate DECELERATE Rollout Safe? ──► YES ──► Approve: DECELERATE (Override: M3_PREDICTIVE_...)
               │
               └─► NO (Critical impending collision)
                     │
                     └─► Approve: BRAKE / EMERGENCY_HALT (Override: M3_CRITICAL_EMERGENCY)
```

