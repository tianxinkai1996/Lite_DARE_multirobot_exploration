"""Shared parameters for multi-robot DARE/LiteDARE evaluation experiments."""

import os

from parameter import MAX_EPISODE_STEP, NODE_RESOLUTION

# DARE checkpoint

# Path to the trained single-robot DARE checkpoint.
# This checkpoint will be loaded once and shared by all robots as a frozen
# local planner.
DARE_CHECKPOINT_PATH = "/root/lite dare/DARE/runs/2026.06.25/12.50.20_train_diffusion_transformer_node_exploration_node/checkpoints/latest.ckpt"

# Evaluate the same frozen checkpoint with each team size.
# Evaluate 2, 4, 6, and 8 robots with the same frozen checkpoint.
TEAM_SIZES = (1, 2, 4, 6, 8)

# Each map is tested four times. In each trial all robots share one depot, while
# the four depot locations are distinct from one another on that map.
START_SAMPLES_PER_MAP = 4

# Map selection. "all" tests every map; "specified" tests only the indices in
# SPECIFIED_MAP_INDICES. The command-line driver can override these values.
MAP_TEST_MODE = "all"  # "all" or "specified"
SPECIFIED_MAP_INDICES = (0,)

# Set this to the number of maps in the selected DARE dataset. If None, the
# driver attempts to discover map files from common dataset directories.
TOTAL_MAP_COUNT = None

# Kept as a compatibility alias for older scripts.
NUM_EPISODES_PER_SETTING = START_SAMPLES_PER_MAP
BASE_SEED = 20260706

# "none" is the no-communication baseline.
# "compressed" exchanges only compressed recent trails and short plans.

COMMUNICATION_MODES = ("compressed", )

# Local encounter definition: shortest hidden traversable graph distance
# <= CONTACT_HOPS, and line of sight must be unobstructed.
CONTACT_HOPS = 3 #2,4
REQUIRE_LINE_OF_SIGHT = True


# Early departure diversity
ENABLE_DEPARTURE_DIVERSITY = True

# Apply role-based direction bias during early steps.
DEPARTURE_DIVERSITY_STEPS = 10

# Larger value means robots are more strongly encouraged to leave the depot
# in different directions.
ROLE_DIVERSITY_WEIGHT = 0.7

# Independent local maps are used. These values are intentionally small 
# communication remains local rather than becoming a broadcast mechanism.
RESERVATION_HORIZON = 3 
CACHE_TTL_STEPS = 3
TRAIL_SHARE_STEPS = 8 #24
PLAN_SHARE_STEPS = 3
PACKET_BUDGET_BYTES = 128

# ----------------------------------------------------------------------
# Start-location setting
# ----------------------------------------------------------------------
# True:
#   All robots start from exactly the same valid free-space node.
#   This is useful when the multi-robot setting must be directly comparable
#   with the original single-robot DARE baseline.
# False:
#   Robots are sampled from separated free-space nodes.
SAME_START_LOCATION = True

# If robots start from the same depot, the initial overlap at t=0 is allowed.
# Safe-distance constraints are applied after robots choose their first moves.
ALLOW_SHARED_DEPOT_AT_STEP_ZERO = True

# Used only when SAME_START_LOCATION is False. Start positions are sampled from
# hidden free-space nodes for scenario setup. This is not shown to DARE policies.
MIN_START_SEPARATION = 3 * NODE_RESOLUTION

# Minimum wall clearance for every sampled/shared depot. The complete circular
# footprint, not only its centre cell, must be FREE. With NODE_RESOLUTION=4 m,
# this default requires 2 m clearance from occupied cells and map boundaries.
START_CLEARANCE = 0.5 * NODE_RESOLUTION

# Dynamic safety margins.
# Adjacent DARE nodes are 4 m apart in the original parameter.py.
SAFE_DISTANCE = 0.75 * NODE_RESOLUTION
TRAIL_AVOID_RADIUS = 1.0 * NODE_RESOLUTION
TRAIL_PENALTY_WEIGHT = 0.5

# Execute exactly one graph move per DARE prediction, then re-observe/re-plan.
ACTION_HORIZON_OVERRIDE = 1
DIRECTION_LOOKAHEAD = 3

# Team completion rule: same tolerance as the original Env.check_done().
TEAM_DONE_TOLERANCE_CELLS = 250
MAX_MULTI_ROBOT_STEPS = MAX_EPISODE_STEP

# Visualisation output.
SAVE_VISUALISATION = True
VISUAL_FRAME_STRIDE = 1
GIF_FRAME_DURATION = 0.25
VISUAL_OUTPUT_ROOT = "multi_robot_outputs"
VISUAL_BACKGROUND_MODE = "ground_truth"  # "ground_truth" or "team_belief"

# ----------------------------------------------------------------------
# Sparse coverage and goal-claim exchange
# ----------------------------------------------------------------------
# This is the part that reduces repeated exploration. It still does not share
# full maps; it shares sparse dynamic tile IDs in a common mission coordinate
# frame. Unknown map size is allowed because tile IDs are unbounded integers.
ENABLE_COVERAGE_EXCHANGE = True

# Use DARE's node resolution so each coverage tile corresponds roughly to one
# graph cell. With NODE_RESOLUTION=4.0, each tile is 4m x 4m.
COVERAGE_TILE_SIZE = NODE_RESOLUTION

# If update_from_local_belief() is used, a tile is marked explored when this
# fraction of its cells are known in that robot's local belief map.
COVERAGE_THRESHOLD = 0.5

# Maximum number of new coverage tiles sent to one peer in one contact packet.
MAX_COVERAGE_DELTA_TILES = 16 #64

# Soft penalties used when selecting among DARE-ranked candidate neighbours.
# Reservation conflicts remain hard constraints; these only reduce repetition.
COVERAGE_PENALTY_WEIGHT = 3.0
GOAL_CLAIM_PENALTY_WEIGHT = 8.0
GOAL_CLAIM_RADIUS_TILES = 1
GOAL_CLAIM_TTL_STEPS = 4

# ----------------------------------------------------------------------
# Deadlock prevention
# ----------------------------------------------------------------------
# A robot enters escape mode after this many consecutive executed waits.
# Escape mode keeps physical safety constraints but temporarily suspends
# peer-plan filtering and coverage/goal/trail soft penalties for one robot.
DEADLOCK_WAIT_THRESHOLD = 3

# Upper bound for the priority-ordered joint-action backtracking search.
DEADLOCK_MAX_BACKTRACKING_NODES = 2000

# Per-step and per-episode paper metrics.
ENABLE_METRIC_RECORDING = True
METRIC_COVERAGE_THRESHOLDS = (0.90, 0.95, 0.99)

# ----------------------------------------------------------------------
# Detailed computation/resource instrumentation
# ----------------------------------------------------------------------
EXPERIMENT_PROFILE = os.environ.get(
    "LITEDARE_EXPERIMENT_PROFILE", "efficiency_benchmark"
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
