"""Test parameters for the map-only baseline experiments."""

import os
from pathlib import Path

from parameter import MAX_EPISODE_STEP, NODE_RESOLUTION

# ----------------------------------------------------------------------
# Frozen DARE policy
# ----------------------------------------------------------------------
DARE_CHECKPOINT_PATH = (
    "/root/lite dare/DARE/runs/2026.06.25/"
    "12.50.20_train_diffusion_transformer_node_exploration_node/"
    "checkpoints/latest.ckpt"
)

TEAM_SIZES = (2, 4, 6, 8)

# ----------------------------------------------------------------------
# Manual number of tests for every selected map
# ----------------------------------------------------------------------
# Edit this value when launching without a command-line override.
#
# Scope:
#   each selected map
#   x each TEAM_SIZES entry
#   x each selected communication mode
# is repeated this many times with independently derived seeds.
MANUAL_TESTS_PER_MAP = 3

# ``run_mergingmap.py --runs-per-map N`` sets the environment override only for
# that launch and does not modify this file.
MAP_TEST_REPEATS = int(
    os.environ.get("MERGINGMAP_MAP_TEST_REPEATS", MANUAL_TESTS_PER_MAP)
)
if MAP_TEST_REPEATS <= 0:
    raise ValueError(
        "MAP_TEST_REPEATS must be greater than zero, "
        f"got {MAP_TEST_REPEATS}"
    )

# Compatibility name consumed by the unchanged project multi_test_driver.py.
NUM_EPISODES_PER_SETTING = MAP_TEST_REPEATS

BASE_SEED = 20260730

# Existing driver mode names are retained:
#   none       -> no knowledge sharing baseline
#   raw        -> send every known cell at every contact
#   compressed -> send only cells not previously sent to that peer
COMMUNICATION_MODES = ("compressed",)

# ----------------------------------------------------------------------
# Contact model
# ----------------------------------------------------------------------
CONTACT_HOPS = 3
REQUIRE_LINE_OF_SIGHT = True

# ----------------------------------------------------------------------
# Initial positions
# ----------------------------------------------------------------------
# Every worker run samples a fresh set of random starts through MultiRobotEnv.
# The environment is reconstructed with a deterministic retry seed until all
# starts are in ground-truth FREE cells and no two robots share a position.
RANDOM_DISTINCT_STARTS = True
SAME_START_LOCATION = False

# Robots must begin at least this far apart in world coordinates.
MIN_START_SEPARATION = 3 * NODE_RESOLUTION

# Maximum number of environment reconstructions when a sampled start is invalid.
START_SAMPLE_MAX_ATTEMPTS = 200

# 0 checks only the start cell itself. Set to 1 to require the surrounding
# 3x3 ground-truth cells to be FREE as well.
START_CLEARANCE_RADIUS_CELLS = 0

# Retained for compatibility with the optional non-ghost motion resolver.
ALLOW_SHARED_DEPOT_AT_STEP_ZERO = False

# ----------------------------------------------------------------------
# Teacher-requested correctness-first experiment
# ----------------------------------------------------------------------
# True: robots ignore one another dynamically and may overlap like "ghosts".
# Static map collision checking still occurs through graph edges and env.step_all.
# This isolates map/graph fusion from multi-robot motion coordination.
GHOST_MODE = True

# Send every currently missing known cell when contact occurs. This verifies
# correct merging first. After correctness is established, change to e.g. 4096
# or 1024 to study bandwidth-limited submap exchange.
MAP_MAX_CELLS_PER_PACKET = 0
MAP_CONFLICT_POLICY = "occupied_wins"
RESET_OBS_HISTORY_ON_MAP_MERGE = True

MAP_DEBUG = True
MAP_DEBUG_INTERVAL = 10

# Evaluation-only settings; these do not change the map-only policy.
ENABLE_METRIC_RECORDING = True
METRIC_COVERAGE_THRESHOLDS = (0.90, 0.95, 0.99)
DEADLOCK_WAIT_THRESHOLD = 3

# ----------------------------------------------------------------------
# Frozen DARE execution
# ----------------------------------------------------------------------
ACTION_HORIZON_OVERRIDE = 1
DIRECTION_LOOKAHEAD = 3
MAX_MULTI_ROBOT_STEPS = MAX_EPISODE_STEP
TEAM_DONE_TOLERANCE_CELLS = 250

# Used only when GHOST_MODE=False.
SAFE_DISTANCE = 0.75 * NODE_RESOLUTION

# ----------------------------------------------------------------------
# Visualisation
# ----------------------------------------------------------------------
SAVE_VISUALISATION = True
VISUAL_FRAME_STRIDE = 1
GIF_FRAME_DURATION = 0.25
VISUAL_OUTPUT_ROOT = os.environ.get(
    "MERGINGMAP_RUN_DIR",
    str(Path(__file__).resolve().parent / "test_outputs"),
)
VISUAL_BACKGROUND_MODE = "ground_truth"
