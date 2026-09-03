"""Geometric predicates for one-step multi-robot collision checks.

"""
from __future__ import annotations

from typing import Sequence

import numpy as np

def as_position(value):
    """Convert a coordinate to the resolver's float32 representation.

    float64 
    English implementation: converts any two-dimensional sequence to a float32
    NumPy array used by all collision predicates.
    """

    return np.asarray(value, dtype=np.float32)

def is_wait(current, candidate):
    """Return whether a candidate keeps the robot at its current position.

    English implementation: uses NumPy's tolerance-aware equality check.
    """

    return bool(np.allclose(current, candidate))

def pair_conflict(current_i, next_i, current_j, next_j, safe_distance, allow_shared_start_wait):
    """Detect vertex/distance and edge-swap conflicts for one robot pair.

    English implementation: compares current and next pairwise distances and
    rejects synchronous A→B/B→A swaps. Existing shared-depot overlap is retained
    only when both robots wait and the caller explicitly permits it.
    """

    i_waits = is_wait(current_i, next_i)
    j_waits = is_wait(current_j, next_j)
    current_distance = float(np.linalg.norm(current_i - current_j))
    next_distance = float(np.linalg.norm(next_i - next_j))

    preserve_shared_wait = (
        allow_shared_start_wait
        and current_distance < safe_distance
        and i_waits
        and j_waits
    )
    if next_distance < safe_distance and not preserve_shared_wait:
        return True

    edge_swap = (
        not i_waits
        and not j_waits
        and np.linalg.norm(current_i - next_j) < safe_distance
        and np.linalg.norm(current_j - next_i) < safe_distance
    )
    return bool(edge_swap)

def shared_overlap_exists(positions, safe_distance):
    """Return whether any robots currently occupy one safe-distance cluster.

    English implementation: scans every unordered pair for a distance below the
    configured safety threshold.
    """

    return any(
        np.linalg.norm(positions[i] - positions[j]) < safe_distance
        for i in range(len(positions))
        for j in range(i + 1, len(positions))
    )
