"""Data models shared by collision and deadlock coordination.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

@dataclass(frozen=True)
class CollisionResolution:
    """Store one synchronous collision-resolution result.

    English implementation: contains selected positions, blocked robot IDs,
    bounded-search effort, and whether serial departure fallback was used.
    """

    positions: Tuple[np.ndarray, ...]
    blocked_robot_ids: Tuple[int, ...]
    backtracking_nodes: int
    used_serial_fallback: bool = False

@dataclass(frozen=True)
class ResolutionInfo:
    """Expose combined collision/deadlock diagnostics to experiment logging.

    English implementation: records priority order, selected escape robots,
    blocked robots, backtracking nodes, and serial fallback usage.
    """

    priority_order: Tuple[int, ...]
    escape_robot_ids: Tuple[int, ...]
    blocked_robot_ids: Tuple[int, ...]
    backtracking_nodes: int
    used_serial_fallback: bool = False
    preferred_action_selection_ms: float = 0.0
    deadlock_priority_ms: float = 0.0
    deadlock_escape_ms: float = 0.0
    collision_resolution_ms: float = 0.0
    coordination_total_ms: float = 0.0
    recovery_stages: Tuple[int, ...] = ()
    soft_penalty_relaxed_robot_ids: Tuple[int, ...] = ()
    lease_release_robot_ids: Tuple[int, ...] = ()
    graph_backtrack_robot_ids: Tuple[int, ...] = ()
    oscillation_robot_ids: Tuple[int, ...] = ()
    motion_exchange_ms: float = 0.0
    motion_filter_ms: float = 0.0
    graph_backtrack_ms: float = 0.0
