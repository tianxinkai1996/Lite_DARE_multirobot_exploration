"""Data models for dynamic MergingMap frontier assignment.

MergingMap 
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, Optional, Set, Tuple

Cell = Tuple[int, int]
WorldPoint = Tuple[float, float]

@dataclass(frozen=True)
class DynamicRegionConfig:
    """Parameters for coverage-preserving dynamic frontier assignment.

    The method never uses the ground-truth map. Regions are extracted only from
    each robot's current merged occupancy belief.

    Ownership is temporary:
      * a frontier region is claimed with a lease;
      * new/changed frontiers are re-extracted every planning step;
      * a disappearing region is released as completed;
      * an expired or non-progressing lease is recycled;
      * unclaimed frontiers remain in the local task pool.

    Under limited communication, strict global exclusivity is impossible. This
    manager therefore guarantees exclusivity only inside the current contact
    component and for peer claims learned through earlier contacts.
    """

    unknown_value: int
    free_value: int
    node_resolution: float

    min_frontier_cells: int = 3
    max_frontier_cells_per_region: int = 80
    region_id_quantization_cells: int = 4

    region_match_iou_threshold: float = 0.05
    region_match_centroid_cells: float = 12.0
    region_conflict_distance: float = 8.0

    lease_steps: int = 30
    claim_ttl_steps: int = 30
    min_commitment_steps: int = 12
    no_progress_release_steps: int = 25
    force_progress_after_steps: int = 8
    progress_known_cells: int = 1

    arrival_distance: float = 6.0
    distance_slack: float = 2.0

    distance_weight: float = 1.0
    utility_weight: float = 2.0
    age_weight: float = 0.05
    lease_penalty_weight: float = 1000.0
    stall_penalty_weight: float = 1.0
    tracking_iou_weight: float = 1.0
    tracking_centroid_weight: float = 1.0
    region_message_header_bytes: int = 32

    debug: bool = True
    debug_interval: int = 10

    # English purpose: Validate ranges and positive-value constraints in region configuration.
    def __post_init__(self):
        positive_ints = {
            "min_frontier_cells": self.min_frontier_cells,
            "max_frontier_cells_per_region": self.max_frontier_cells_per_region,
            "region_id_quantization_cells": self.region_id_quantization_cells,
            "lease_steps": self.lease_steps,
            "claim_ttl_steps": self.claim_ttl_steps,
            "min_commitment_steps": self.min_commitment_steps,
            "no_progress_release_steps": self.no_progress_release_steps,
            "force_progress_after_steps": self.force_progress_after_steps,
            "progress_known_cells": self.progress_known_cells,
        }
        for name, value in positive_ints.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        if self.node_resolution <= 0:
            raise ValueError("node_resolution must be positive")
        if self.arrival_distance < 0 or self.distance_slack < 0:
            raise ValueError("distance thresholds cannot be negative")
        if not 0 <= self.region_match_iou_threshold <= 1:
            raise ValueError("region_match_iou_threshold must be in [0, 1]")
        if self.lease_penalty_weight < 0 or self.stall_penalty_weight < 0:
            raise ValueError("lease/stall penalty weights cannot be negative")
        if self.tracking_iou_weight < 0 or self.tracking_centroid_weight < 0:
            raise ValueError("tracking weights cannot be negative")
        if self.region_message_header_bytes < 0:
            raise ValueError("region_message_header_bytes cannot be negative")

@dataclass(frozen=True)
class FrontierRegion:
    """One temporary frontier task extracted from a partial merged belief."""

    region_id: str
    frontier_cells: frozenset[Cell]
    centroid_cell: Tuple[float, float]
    centroid_world: WorldPoint
    target_cell: Cell
    target_world: WorldPoint
    bbox_cell: Tuple[int, int, int, int]
    utility: float

    @property
    # English purpose: Return the number of frontier cells in a region.
    def frontier_count(self):
        return len(self.frontier_cells)

@dataclass
class RegionLease:
    """Temporary ownership of one evolving frontier region."""

    owner_robot_id: int
    region: FrontierRegion
    claimed_step: int
    expiry_step: int
    last_progress_step: int
    last_target_distance: float = math.inf
    last_frontier_count: int = 0
    last_progress_reason: str = "assigned"
    generation: int = 0

    @property
    # English purpose: Expose the current region identifier from a lease.
    def region_id(self):
        return self.region.region_id

@dataclass(frozen=True)
class PeerClaim:
    owner_robot_id: int
    region_id: str
    centroid_world: WorldPoint
    bbox_cell: Tuple[int, int, int, int]
    expiry_step: int

@dataclass
class RobotRegionState:
    regions: Dict[str, FrontierRegion] = field(default_factory=dict)
    first_seen_step: Dict[str, int] = field(default_factory=dict)
    lease: Optional[RegionLease] = None
    known_peer_claims: Dict[str, PeerClaim] = field(default_factory=dict)
    unreachable_steps: int = 0
    graph_cache_key: Optional[tuple] = None
    graph_distance_cache: Dict[Tuple[float, float], float] = field(
        default_factory=dict
    )
    visited_nodes: Set[Tuple[float, float]] = field(default_factory=set)
    stall_steps: int = 0
    last_effective_progress_step: int = 0
    region_failure_counts: Dict[str, int] = field(default_factory=dict)
    assignment_count: int = 0
    reassignment_count: int = 0
    wait_steps: int = 0

