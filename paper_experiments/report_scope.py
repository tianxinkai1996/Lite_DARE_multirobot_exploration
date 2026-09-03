"""Primary dissertation reporting scope without changing experiment execution.

The experiments keep full run results; the main dissertation text reads only the
single-robot depth comparison, Map-only vs Full under compressed communication,
and the none/raw/compressed comparison under Full coordination, all stratified by
N=2/4/6/8.
"""
from __future__ import annotations

# Main-text single-robot depth comparison.
SINGLE_ROBOT_DEPTHS = (6, 4, 2)
SINGLE_ROBOT_MODEL_ORDER = ("DARE-L6", "LiteDARE-L4", "LiteDARE-L2")

# Team-size sweep used in every main-text multi-robot table/figure.
PRIMARY_TEAM_SIZES = (2, 4, 6, 8)

# Main-text multi-robot system comparison.
PRIMARY_MULTI_ROLES = ("map_only", "full")
PRIMARY_MULTI_MODE = "compressed"
PRIMARY_MULTI_LABELS = {
    "map_only": "Map-only",
    "full": "Full",
}

# Main-text occupancy-map communication comparison.
PRIMARY_COMM_ROLE = "full"
PRIMARY_COMM_MODES = ("none", "raw", "compressed")

# Extra runs are retained, never deleted. They may be exported for an appendix,
# supervisor discussion, or robustness checks, but are not used for the main claim.
SUPPLEMENTARY_ROLES = (
    "policy_only",
    "map_region",
    "map_reservation",
    "original_dare",
)

ROLE_LABELS = {
    "policy_only": "LiteDARE-only",
    "map_only": "Map-only",
    "map_region": "Map+Region",
    "map_reservation": "Map+Reservation",
    "full": "Full",
    "original_dare": "Original DARE",
}

# Accept both the existing full-run directory names and the previously proposed
# reduced-run names. Existing results therefore do not need to be moved or rerun.
MULTI_GROUP_NAMES = {"multi_ablation", "multi_maponly_full"}
COMM_GROUP_NAMES = {"communication", "communication_reduced"}
SINGLE_GROUP_NAME = "e1_policy_depth"

