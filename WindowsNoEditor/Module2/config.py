# Module 2 Configuration File

# Tactical Action Interval
TACTICAL_DECISION_INTERVAL = 1.0 # seconds

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

# Platoon configuration
MAX_PLATOON_SIZE = 4 # Leader + 3 Followers

# Weather & Headway Parameters
DEFAULT_DESIRED_HEADWAY = 7.5 # meters
HEADWAY_DEADBAND = 2.0 # meters for hysteresis

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
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_INTERVAL_UPDATES = 10

