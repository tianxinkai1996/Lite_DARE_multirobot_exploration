"""Candidate-list preparation and post-resolution diagnostics.

"""
from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

from collision.geometry import as_position, is_wait

def normalise_candidates(current_positions, candidate_lists):
    """Deduplicate candidates and append one waiting fallback per robot.

    English implementation: rounds coordinates for stable deduplication while
    preserving the planner's original candidate order.
    """

    current = [as_position(position) for position in current_positions]
    if len(current) != len(candidate_lists):
        raise ValueError("current_positions and candidate_lists must have equal length")

    normalised: List[List[np.ndarray]] = []
    for robot_id, robot_candidates in enumerate(candidate_lists):
        deduplicated: List[np.ndarray] = []
        seen: set[tuple[float, ...]] = set()
        for value in robot_candidates:
            candidate = as_position(value)
            key = tuple(np.round(candidate, 5))
            if key not in seen:
                deduplicated.append(candidate)
                seen.add(key)

        wait_key = tuple(np.round(current[robot_id], 5))
        if wait_key not in seen:
            deduplicated.append(current[robot_id].copy())
        normalised.append(deduplicated)

    return current, normalised

def blocked_robot_ids(current, selected, candidates):
    """Identify robots forced to wait despite having a moving candidate.

    English implementation: marks a robot only when its selected action is wait
    and at least one candidate in its list represented a movement.
    """

    return [
        robot_id
        for robot_id in range(len(current))
        if is_wait(current[robot_id], selected[robot_id])
        and any(
            not is_wait(current[robot_id], candidate)
            for candidate in candidates[robot_id]
        )
    ]

def preferred_positions(current_positions, candidate_lists):
    """Return each robot's first planner candidate, falling back to wait.

    English implementation: selects index zero from each non-empty candidate list
    and otherwise copies the current position.
    """

    current = [as_position(position) for position in current_positions]
    return [
        as_position(candidates[0]).copy() if candidates else current[robot_id].copy()
        for robot_id, candidates in enumerate(candidate_lists)
    ]
