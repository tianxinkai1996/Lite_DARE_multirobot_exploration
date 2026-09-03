"""Graph-safe recovery toward the nearest branching node.

"""
from __future__ import annotations

from collections import deque
from typing import Iterable, Sequence

import numpy as np

def _key(position):
    value = np.asarray(position, dtype=float).reshape(2)
    return round(float(value[0]), 4), round(float(value[1]), 4)

def nearest_branch_backtrack_step(robot, current_position, safe_candidates, min_branch_degree=3, max_nodes=2000):
    """Return the first graph step toward the nearest reachable branch.

    Implements the final graph-recovery stage without inventing off-graph motion:
    performs bounded BFS to the nearest node with sufficient degree and returns
    only a first step already present in the supplied hard-safe candidate set.
    """

    records = list(robot.node_manager.nodes_dict.__iter__())
    if not records:
        return None
    nodes: dict[tuple[float, float], object] = {
        _key(record.data.coords): record.data for record in records
    }
    start = _key(current_position)
    if start not in nodes:
        start = min(
            nodes,
            key=lambda node: float(
                np.linalg.norm(np.asarray(node) - np.asarray(current_position))
            ),
        )
    allowed = {
        _key(candidate): np.asarray(candidate, dtype=np.float32).copy()
        for candidate in safe_candidates
        if not np.allclose(candidate, current_position)
    }
    if not allowed:
        return None

    queue = deque([start])
    parent: dict[tuple[float, float], tuple[float, float] | None] = {start: None}
    visited = 0
    target = None
    while queue and visited < int(max_nodes):
        node_key = queue.popleft()
        visited += 1
        node = nodes[node_key]
        neighbours = [
            _key(value)
            for value in getattr(node, "neighbor_list", [])
            if _key(value) in nodes and _key(value) != node_key
        ]
        if node_key != start and len(set(neighbours)) >= int(min_branch_degree):
            target = node_key
            break
        for neighbour in neighbours:
            if neighbour not in parent:
                parent[neighbour] = node_key
                queue.append(neighbour)
    if target is None:
        return None

    cursor = target
    while parent.get(cursor) is not None and parent[cursor] != start:
        cursor = parent[cursor]
    return allowed.get(cursor)
