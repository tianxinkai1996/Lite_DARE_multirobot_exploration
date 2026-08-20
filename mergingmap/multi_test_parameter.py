"""Parameters for the isolated merging-map + dynamic-region experiment."""
from __future__ import annotations

import os
from pathlib import Path

from parameter import MAX_EPISODE_STEP, NODE_RESOLUTION


# ----------------------------------------------------------------------
# Frozen DARE policy
# ----------------------------------------------------------------------
DEFAULT_DARE_CHECKPOINT_PATH = (
    "/root/lite dare/DARE/lite_dare/runs/"
    "DiscreteLiteDARE_NodeEncSA_L2_H4_D256_DecL1_H4/"
    "seed_42_20260730_095849/checkpoints/"
    "epoch=0150-val_loss=0.058.ckpt"
)
# 中文目的：让原始 DARE 与 MergingMap 消融通过同一个环境变量加载同一检查点。
# English purpose: make original-DARE and MergingMap runs share one checkpoint override.
DARE_CHECKPOINT_PATH = os.environ.get(
    "DARE_CHECKPOINT_PATH",
    DEFAULT_DARE_CHECKPOINT_PATH,
)

TEAM_SIZES = (1, 2, 4, 6, 8)
BASE_SEED = 42
MANUAL_TESTS_PER_MAP = 3
MAP_TEST_REPEATS = int(
    os.environ.get("MERGINGMAP_MAP_TEST_REPEATS", MANUAL_TESTS_PER_MAP)
)
if MAP_TEST_REPEATS <= 0:
    raise ValueError("MAP_TEST_REPEATS must be positive")

# Compatibility aliases used by older scripts.
NUM_EPISODES_PER_SETTING = MAP_TEST_REPEATS
NUM_TEST_PER_MAP = MAP_TEST_REPEATS
COMMUNICATION_MODES = ("compressed",)


# ----------------------------------------------------------------------
# Contact model
# ----------------------------------------------------------------------
CONTACT_HOPS = 3
REQUIRE_LINE_OF_SIGHT = True


# ----------------------------------------------------------------------
# Initial positions
# ----------------------------------------------------------------------
RANDOM_DISTINCT_STARTS = True
SAME_START_LOCATION = False
MIN_START_SEPARATION = 3 * NODE_RESOLUTION
START_SAMPLE_MAX_ATTEMPTS = 200
START_CLEARANCE_RADIUS_CELLS = 0
ALLOW_SHARED_DEPOT_AT_STEP_ZERO = False


# ----------------------------------------------------------------------
# Balanced random initial cardinal direction
# ----------------------------------------------------------------------
# At episode start, each robot is assigned a fixed primary direction from
# north/east/south/west. Assignments are shuffled in balanced groups of four.
ENABLE_INITIAL_CARDINAL_DIRECTION = True
INITIAL_DIRECTION_BIAS_STEPS = 12
INITIAL_DIRECTION_MAX_BIAS_WEIGHT = 0.80
INITIAL_DIRECTION_DECAY = True
INITIAL_DIRECTION_DEBUG = True


# ----------------------------------------------------------------------
# Correctness-first map merging
# ----------------------------------------------------------------------
GHOST_MODE = False
MAP_MAX_CELLS_PER_PACKET = 0
MAP_CONFLICT_POLICY = "occupied_wins"
# Reliability experiments default to the original synchronous/no-loss setting.
# 可靠性实验默认保持同步无丢包；论文附加实验可通过环境变量覆盖。
MAP_PACKET_HEADER_BYTES = 64
RESET_OBS_HISTORY_ON_MAP_MERGE = True
MAP_DEBUG = True
MAP_DEBUG_INTERVAL = 10

# Legacy single-run defaults and evaluation logging.
# 核心论文消融由 ablation_profiles.py 与 run_paper_ablation.py 控制。
# Core paper ablations are controlled by ablation_profiles.py and run_paper_ablation.py.
# 碰撞/死锁消融与实验指标记录。
# GHOST_MODE=True: no coordination / 不进行动作协调。
# GHOST_MODE=False and ENABLE_COLLISION_DEADLOCK_AVOIDANCE=False:
# collision-only / 仅当前步碰撞消解。
# Both flags enable full collision + persistent deadlock handling.
# 两项均启用时执行碰撞消解与持久化死锁处理。
ENABLE_COLLISION_DEADLOCK_AVOIDANCE = True
DEADLOCK_WAIT_THRESHOLD = 3
DEADLOCK_STALL_THRESHOLD = 3
DEADLOCK_SOFT_RELAX_THRESHOLD = 6
DEADLOCK_LEASE_RELEASE_THRESHOLD = 9
DEADLOCK_GRAPH_BACKTRACK_THRESHOLD = 12
DEADLOCK_WAIT_WEIGHT = 1.0
DEADLOCK_STALL_WEIGHT = 1.0
DEADLOCK_MAX_BACKTRACKING_NODES = 2000
OSCILLATION_BASE_PENALTY = 2.0
OSCILLATION_REPEAT_PENALTY = 1.0
MOTION_RESERVATION_HORIZON = 2
MOTION_CACHE_TTL_STEPS = 2
MOTION_PACKET_HEADER_BYTES = 40
ENABLE_METRIC_RECORDING = True
METRIC_COVERAGE_THRESHOLDS = (0.90, 0.95, 0.99)
EXPERIMENT_PROFILE = os.environ.get(
    "LITEDARE_EXPERIMENT_PROFILE", "fast_exploration"
).strip().lower()
if EXPERIMENT_PROFILE not in {"fast_exploration", "efficiency_benchmark"}:
    raise ValueError(
        "LITEDARE_EXPERIMENT_PROFILE must be 'fast_exploration' or 'efficiency_benchmark'"
    )

FAST_EXPLORATION_MODE = EXPERIMENT_PROFILE == "fast_exploration"
ENABLE_DETAILED_RUNTIME_METRICS = not FAST_EXPLORATION_MODE
CUDA_SYNCHRONIZE_FOR_TIMING = not FAST_EXPLORATION_MODE
TRACK_PYTHON_MEMORY = not FAST_EXPLORATION_MODE
TRACK_HARDWARE_ENERGY = not FAST_EXPLORATION_MODE
PROFILE_POLICY_FLOPS_ONCE = not FAST_EXPLORATION_MODE


# ----------------------------------------------------------------------
# Coverage-preserving dynamic frontier-region assignment
# ----------------------------------------------------------------------
ENABLE_DYNAMIC_REGION_ASSIGNMENT = True
REGION_MIN_FRONTIER_CELLS = 3
REGION_MAX_FRONTIER_CELLS = 80
REGION_ID_QUANTIZATION_CELLS = 4
REGION_MATCH_IOU_THRESHOLD = 0.05
REGION_MATCH_CENTROID_CELLS = 12.0
REGION_CONFLICT_DISTANCE = 2.0 * NODE_RESOLUTION
REGION_LEASE_STEPS = 30
REGION_CLAIM_TTL_STEPS = 30
REGION_MIN_COMMITMENT_STEPS = 12
REGION_NO_PROGRESS_RELEASE_STEPS = 25
REGION_FORCE_PROGRESS_AFTER_STEPS = 8
REGION_PROGRESS_KNOWN_CELLS = 1
REGION_ARRIVAL_DISTANCE = 1.5 * NODE_RESOLUTION
REGION_DISTANCE_SLACK = 0.5 * NODE_RESOLUTION
REGION_DISTANCE_WEIGHT = 1.0
REGION_UTILITY_WEIGHT = 2.0
REGION_AGE_WEIGHT = 0.05
REGION_LEASE_PENALTY_WEIGHT = 1000.0
REGION_STALL_PENALTY_WEIGHT = 1.0
REGION_TRACKING_IOU_WEIGHT = 1.0
REGION_TRACKING_CENTROID_WEIGHT = 1.0
REGION_MESSAGE_HEADER_BYTES = 32
REGION_DEBUG = True
REGION_DEBUG_INTERVAL = 10


# ----------------------------------------------------------------------
# Frozen DARE execution
# ----------------------------------------------------------------------
ACTION_HORIZON_OVERRIDE = 1
DIRECTION_LOOKAHEAD = 3
MAX_MULTI_ROBOT_STEPS = MAX_EPISODE_STEP
TEAM_DONE_TOLERANCE_CELLS = 250
SAFE_DISTANCE = 0.75 * NODE_RESOLUTION


# ----------------------------------------------------------------------
# Visualisation and outputs
# ----------------------------------------------------------------------
SAVE_VISUALISATION = False
VISUAL_FRAME_STRIDE = 1
GIF_FRAME_DURATION = 0.25
VISUAL_OUTPUT_ROOT = os.environ.get(
    "MERGINGMAP_RUN_DIR",
    str(Path(__file__).resolve().parent / "test_outputs"),
)
VISUAL_BACKGROUND_MODE = "ground_truth"


