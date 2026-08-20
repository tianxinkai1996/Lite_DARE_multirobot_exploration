"""Single-seed Chapter 4 experiment configuration.

只需要修改这个 Python 文件即可配置模型、地图、重复次数和输出目录。
No JSON model registry is used. All three depth variants share one training/base
random seed; evaluation trial seeds are deterministic offsets from this base seed
so repeated trials are reproducible without becoming identical copies.
"""
from __future__ import annotations

from pathlib import Path

from paper_experiments.common import ModelSpec

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# One random seed for the controlled comparison
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Checkpoints: edit these three paths on the target machine.
# All models must have been trained with RANDOM_SEED above.
# ---------------------------------------------------------------------------
DARE_L6_CHECKPOINT = Path(
    "/root/lite dare/DARE/runs/2026.06.25/"
    "12.50.20_train_diffusion_transformer_node_exploration_node/"
    "checkpoints/epoch=0180-val_loss=0.071.ckpt"
)
LITE_L4_CHECKPOINT = Path(
    "/root/lite dare/DARE/lite_dare/runs/"
    "DiscreteLiteDARE_NodeEncSA_L4_H4_D256_DecL1_H4/"
    "seed_42_20260726_004608/checkpoints/epoch=0150-val_loss=0.058.ckpt"
)
LITE_L2_CHECKPOINT = Path(
    "/root/lite dare/DARE/lite_dare/runs/"
    "DiscreteLiteDARE_NodeEncSA_L2_H4_D256_DecL1_H4/"
    "seed_42_20260730_095849/checkpoints/epoch=0150-val_loss=0.058.ckpt"
)

MODELS = {
    "DARE-L6": ModelSpec(
        key="DARE-L6",
        display_name="DARE-L6",
        checkpoint=DARE_L6_CHECKPOINT,
        encoder_layers=6,
        training_seed=RANDOM_SEED,
        family="DARE",
    ),
    "LiteDARE-L4": ModelSpec(
        key="LiteDARE-L4",
        display_name="LiteDARE-L4",
        checkpoint=LITE_L4_CHECKPOINT,
        encoder_layers=4,
        training_seed=RANDOM_SEED,
        family="LiteDARE",
    ),
    "LiteDARE-L2": ModelSpec(
        key="LiteDARE-L2",
        display_name="LiteDARE-L2",
        checkpoint=LITE_L2_CHECKPOINT,
        encoder_layers=2,
        training_seed=RANDOM_SEED,
        family="LiteDARE",
    ),
}

# Stage 1 always compares the three self-attention depths.
ATTENTION_COMPARISON_MODELS = ("DARE-L6", "LiteDARE-L4", "LiteDARE-L2")

# ---------------------------------------------------------------------------
# Maps and repeats
# ---------------------------------------------------------------------------
# Examples: "7", "0,3,7-10", or "all".
MAPS = "all"
MAP_COUNT = 100
RUNS_PER_MAP = 4
# Optional per-map override. Example: {0: 10, 7: 20}.
MAP_REPEATS: dict[int, int] = {}

# Downstream multi-robot settings used only after automatic model selection.
TEAM_SIZES = (2, 4, 6, 8)
COMMUNICATION_TEAM_SIZES = TEAM_SIZES
PARALLEL_WORKERS = 3
# Independent downstream episodes per ablation process. Start with 4 on one GPU.
EPISODE_WORKERS = 4
SAVE_VISUALISATION = True
CONTINUE_ON_ERROR = False
RUN_ORIGINAL_DARE_REFERENCE = True

# Runtime profile for Chapter 4 execution.
# "fast_exploration": preserve exploration/coordination metrics while disabling
# expensive per-step hardware profiling and CUDA synchronisation.
# "efficiency_benchmark": enable detailed timing/memory/energy instrumentation;
# use PARALLEL_WORKERS=1 and EPISODE_WORKERS=1 for final latency/efficiency numbers.
EXPERIMENT_PROFILE = "fast_exploration"

# ---------------------------------------------------------------------------
# Automatic depth-selection rule
# ---------------------------------------------------------------------------
# LiteDARE-L4 and LiteDARE-L2 are both compared directly with DARE-L6 on the
# exact same single-robot map/trial pairs. The downstream LiteDARE model is the
# candidate whose exploration behaviour is closest to L6. Parameter count and
# inference latency are reported as efficiency outcomes, but they do NOT enter
# the similarity score.
#
# The score normalises the mean paired differences in success rate, final
# coverage and Coverage AUC by practical comparison scales and takes their
# Euclidean norm. Smaller is closer to L6. Non-inferiority is still reported as
# supporting evidence, but it is not the selection gate. If the two similarity
# scores are effectively tied, the shallower L2 model is preferred.
DELTA_FINAL_COVERAGE = 0.01
DELTA_SUCCESS_RATE = 0.02
DELTA_COVERAGE_AUC = 0.02
SELECTION_TIE_TOLERANCE = 0.05
BOOTSTRAP_SAMPLES = 5000

# Figure 4-5 default team size and output root.
ABLATION_TEAM_SIZE = 4
DEFAULT_OUTPUT = PROJECT_ROOT / "paper_outputs" / "chapter4_single_seed"
