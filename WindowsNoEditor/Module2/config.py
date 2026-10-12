# Module 2 Configuration File
import os

# Module 1/2 state and decision extraction rate.
# CARLA advances at 30 Hz, while the learning transition stream is sampled at
# approximately 15 Hz as required by the project architecture.
STATE_EXTRACTION_HZ = 15.0
STATE_EXTRACTION_INTERVAL = 1.0 / STATE_EXTRACTION_HZ
TACTICAL_DECISION_INTERVAL = STATE_EXTRACTION_INTERVAL
ACTION_HYSTERESIS_STEPS = 2
CONTROL_DT = 1.0 / 30.0
MIN_LANE_CHANGE_GAP = 15.0
LANE_CHANGE_TIMEOUT = 4.0

# MAPPO Training Configuration
ROLLOUT_LENGTH = 128 # tactical decisions
LEARNING_RATE_ACTOR = 3e-4
LEARNING_RATE_CRITIC = 1e-3
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_CLIP_RANGE = 0.2
ENTROPY_COEF = 0.01
VALUE_LOSS_COEF = 0.5
MAX_GRAD_NORM = 0.5
K_EPOCHS = 4
MINI_BATCH_SIZE = 64

# Platoon configuration
MAX_PLATOON_SIZE = 4 # Leader + 3 Followers

# Weather & Headway Parameters
DEFAULT_DESIRED_HEADWAY = 7.5 # meters
HEADWAY_DEADBAND = 2.0 # meters for hysteresis
DESIRED_HEADWAY_TIME_SEC = 1.2  # seconds time headway
MIN_STANDSTILL_GAP_M = 4.5      # minimum standstill distance (meters)
CATCHUP_GAP_THRESHOLD_RATIO = 1.15 # activate catch-up when gap > 1.15 * desired_gap
CATCHUP_MAX_SPEED_INCREMENT_MPS = 3.5 # max extra speed follower can use to catch up (m/s)
CATCHUP_KP_GAIN = 0.35          # proportional gain for gap error closing
LEADER_FAILURE_TIMEOUT_SEC = 8.0 # seconds immobilized before declaring leader failure
LEADER_COLLISION_TIMEOUT_SEC = 5.0 # seconds immobilized post-collision before promoting follower
LEADER_SUSPECT_TIMEOUT_SEC = 3.0 # seconds immobilized before entering SUSPECTED_FAILURE state
LEADER_RECOVERY_ATTEMPT_TIMEOUT_SEC = 2.0 # duration to attempt breakout recovery before declaring FAILED
LEADER_STOPPED_SPEED_THRESHOLD_MPS = 0.2 # vehicle speed threshold below which vehicle is considered stopped
LEADER_RECOVERY_SPEED_THRESHOLD_MPS = 0.35 # vehicle speed above which recovered motion is confirmed

# Reward Weights
COLLECTIVE_REWARD_WEIGHT = 0.5
COLLISION_PENALTY = -50.0
RED_LIGHT_PENALTY = -20.0
UNSAFE_HEADWAY_PENALTY = -2.0
GOOD_HEADWAY_REWARD = 0.5
EFFICIENCY_REWARD = 0.1

# Lane Change Parameters
LANE_CHANGE_DURATION = 3.0 # simulated time required for lane change
LANE_CHANGE_COOLDOWN = 2.0 # cooldown after a lane change before accepting another

# Debug
DEBUG_MODE = True
DEBUG_INTERVAL = 2.0 # print debug every X seconds

# Checkpointing
_MODULE2_DIR = os.path.dirname(os.path.abspath(__file__))
_CARLA_ROOT = os.path.normpath(os.path.join(_MODULE2_DIR, ".."))
_PROJ_ROOT = os.path.normpath(os.path.join(_CARLA_ROOT, ".."))
_DEFAULT_CKPT_DIR = os.path.join(_CARLA_ROOT, "PythonAPI", "checkpoints")
_ROOT_CKPT_DIR = os.path.join(_PROJ_ROOT, "checkpoints")

CHECKPOINT_DIR = os.environ.get("MAPPO_CHECKPOINT_DIR", _DEFAULT_CKPT_DIR)
CHECKPOINT_ROOT_DIR = _ROOT_CKPT_DIR
CHECKPOINT_FILENAME = "mappo_checkpoint.pt"
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILENAME)
CHECKPOINT_INTERVAL_UPDATES = 10
LOG_DIR = os.environ.get("MAPPO_LOG_DIR", os.path.join(_CARLA_ROOT, "PythonAPI", "logs"))
DECISION_LOG_FILE = os.environ.get("MAPPO_DECISION_LOG", os.path.join(LOG_DIR, "mappo_decisions.log"))


