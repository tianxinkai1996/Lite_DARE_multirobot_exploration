"""Geometric predicates for one-step multi-robot collision checks.

多机器人单步运动碰撞判定的几何工具。
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def as_position(value: Sequence[float]) -> np.ndarray:
    """Convert a coordinate to the resolver's float32 representation.

    中文目的：统一坐标数据类型，避免列表、float64 与张量转换造成比较偏差。
    English implementation: converts any two-dimensional sequence to a float32
    NumPy array used by all collision predicates.
    """

    return np.asarray(value, dtype=np.float32)


def is_wait(current: np.ndarray, candidate: np.ndarray) -> bool:
    """Return whether a candidate keeps the robot at its current position.

    中文目的：识别等待动作，用于区分停留、移动和边交换冲突。
    English implementation: uses NumPy's tolerance-aware equality check.
    """

    return bool(np.allclose(current, candidate))


def pair_conflict(
    current_i: np.ndarray,
    next_i: np.ndarray,
    current_j: np.ndarray,
    next_j: np.ndarray,
    *,
    safe_distance: float,
    allow_shared_start_wait: bool,
) -> bool:
    """Detect vertex/distance and edge-swap conflicts for one robot pair.

    中文目的：判定两个机器人下一步是否发生同点、过近或位置互换冲突。
    中文实现：允许已经处于共享起点的机器人继续共同等待，但不允许移动机器人
    新进入该重叠区域，也不允许在其他位置产生新的重叠。

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


def shared_overlap_exists(
    positions: Sequence[np.ndarray],
    *,
    safe_distance: float,
) -> bool:
    """Return whether any robots currently occupy one safe-distance cluster.

    中文目的：检测共享起点或现有重叠，以决定是否启用串行离开兜底策略。
    English implementation: scans every unordered pair for a distance below the
    configured safety threshold.
    """

    return any(
        np.linalg.norm(positions[i] - positions[j]) < safe_distance
        for i in range(len(positions))
        for j in range(i + 1, len(positions))
    )
