# WindowsNoEditor/Module3/config.py
"""Configuration parameters and feature flags for Module 3 Twin-World Safety Projection System."""

from __future__ import annotations

import os

# ── FEATURE FLAGS & OPERATING MODE ──────────────────────────────────────────
# Operating modes: "DISABLED", "SHADOW", "ACTIVE"
MODULE3_ENABLED = True
MODULE3_MODE = os.environ.get("MODULE3_MODE", "SHADOW").upper()

# ── PREDICTION HORIZON (WORLD B) ─────────────────────────────────────────────
PREDICTION_HORIZON_SEC = 2.0     # Total forward projection horizon (seconds)
PREDICTION_DT_SEC = 0.1          # Discrete step size (seconds) -> 20 prediction steps

# ── LATENCY & EXECUTION BUDGET ──────────────────────────────────────────────
MAX_EXECUTION_BUDGET_MS = 12.0   # Target execution time budget per decision cycle

# ── KINEMATICS & DYNAMICS CONSTRAINTS ───────────────────────────────────────
# Action-induced acceleration targets (m/s^2)
ACCEL_RATE_MPS2 = 1.2            # Forward acceleration under ACCELERATE
DECEL_RATE_MPS2 = -2.2           # Moderate deceleration under DECELERATE
CRUISE_RATE_MPS2 = 0.0           # Nominal acceleration under CRUISE

# Emergency stopping capability
MAX_EMERGENCY_DECEL_MPS2 = 6.5   # Maximum nominal dry-road deceleration capacity (m/s^2)
DRIVER_REACTION_TIME_SEC = 0.35  # Conservative sensor+processing+actuation reaction delay
MIN_BUFFER_DISTANCE_M = 2.0      # Absolute minimum bumper-to-bumper buffer (m)

# ── RISK & THRESHOLDS ───────────────────────────────────────────────────────
CRITICAL_TTC_THRESHOLD_SEC = 1.8 # TTC below this triggers critical risk / emergency
WARNING_TTC_THRESHOLD_SEC = 3.2  # TTC below this triggers caution / action replacement
MIN_COLLISION_SEPARATION_M = 1.2 # Predicted bounding separation below this is critical collision

# ── LANE CHANGE VALIDATION ──────────────────────────────────────────────────
LANE_CHANGE_TARGET_CLEAR_M = 18.0   # Required longitudinal clearance in target lane
LANE_CHANGE_LEAD_TTC_SEC = 3.0      # Minimum TTC to any vehicle in target lane ahead
LANE_CHANGE_LAG_TTC_SEC = 3.0       # Minimum TTC from any vehicle approaching from target lane behind

# ── WEATHER & ENVIRONMENT ───────────────────────────────────────────────────
NOMINAL_ROAD_FRICTION = 1.0         # Baseline dry asphalt friction coefficient
NOMINAL_VISIBILITY_M = 120.0        # Baseline clear weather sight distance (meters)
FOG_MIN_VISIBILITY_M = 25.0         # Heavy fog sight distance (meters)
RAIN_LOW_FRICTION = 0.65            # Wet asphalt friction estimate
WET_LOWEST_FRICTION = 0.35          # Standing water / low friction floor

# ── REAR-IMPACT PREDICTION & DYNAMIC EVASION ────────────────────────────────
ENABLE_REAR_EVASION = True          # Master switch for rear-threat prediction & evasion
REAR_PREDICTION_HORIZON_SEC = 4.0   # Lookahead horizon for approaching rear traffic (seconds)
REAR_TTC_CRITICAL_SEC = 1.8         # TTC below this triggers CRITICAL rear threat
REAR_TTC_HIGH_RISK_SEC = 3.0        # TTC below this triggers HIGH_RISK rear threat
REAR_TTC_CAUTION_SEC = 4.5          # TTC below this triggers CAUTION rear threat
REAR_DISTANCE_THRESHOLD_M = 45.0    # Maximum range to track rear vehicles (meters)
NPC_CONSERVATIVE_BRAKE_MPS2 = 4.5   # Assumed maximum braking capacity for approaching NPCs (m/s^2)
NPC_REACTION_TIME_SEC = 0.5         # Conservative reaction delay for approaching NPCs (seconds)
REAR_HYSTERESIS_FRAMES = 2          # Consecutive frames required to escalate rear threat state
REAR_SAFE_CLEARANCE_FRAMES = 3      # Consecutive frames required to confirm threat has passed
EVASION_LANE_CHANGE_CLEARANCE_M = 12.0 # Minimum longitudinal gap required in escape lane (meters)
EVASION_LATERAL_OFFSET_M = 3.5      # Lateral offset for adjacent lane evasion (meters)
REJOIN_SAFE_GAP_M = 10.0            # Minimum gap in platoon lane required to initiate rejoin (meters)

# ── DEDICATED LIVE DECISION TERMINAL & LOGGING ─────────────────────────────
MODULE3_LOG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "PythonAPI", "logs", "module3_decisions.log"
)
MODULE3_LOG_HEARTBEAT_SEC = 2.0     # Interval to emit NORMAL heartbeat records (seconds)


