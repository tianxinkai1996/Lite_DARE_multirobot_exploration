"""Pure starvation and no-progress priority functions.

"""
from __future__ import annotations

from typing import Iterable, Sequence

def priority_score(robot_id, wait_steps, stall_steps=None, wait_weight=1.0, stall_weight=1.0):
    """Return the dynamic priority value used by short motion messages.

    English implementation: computes lambda_wait*n_wait + lambda_stall*n_stall;
    deterministic token/ID tie-breaking is applied separately by ``priority_key``.
    """

    stall = 0 if stall_steps is None else int(stall_steps[int(robot_id)])
    return float(
        float(wait_weight) * int(wait_steps[int(robot_id)])
        + float(stall_weight) * stall
    )

def priority_key(robot_id, wait_steps, priority_token, n_agents, stall_steps=None, wait_weight=1.0, stall_weight=1.0):
    """Build a deterministic starvation/no-progress-aware priority key.

    English implementation: sorts by descending weighted dynamic priority,
    rotating-token distance, and robot ID as the final deterministic tie-breaker.
    """

    token_distance = (int(robot_id) - int(priority_token)) % int(n_agents)
    score = priority_score(
        robot_id,
        wait_steps,
        stall_steps,
        wait_weight=wait_weight,
        stall_weight=stall_weight,
    )
    return (-score, token_distance, int(robot_id))

def priority_order(wait_steps, priority_token, stall_steps=None, wait_weight=1.0, stall_weight=1.0):
    """Return all robot IDs in dynamic-priority execution order."""

    n_agents = len(wait_steps)
    if stall_steps is not None and len(stall_steps) != n_agents:
        raise ValueError("stall_steps must match wait_steps")
    return sorted(
        range(n_agents),
        key=lambda robot_id: priority_key(
            robot_id,
            wait_steps,
            priority_token,
            n_agents,
            stall_steps,
            wait_weight=wait_weight,
            stall_weight=stall_weight,
        ),
    )

def select_escape_robot(wait_steps, priority_token, threshold, candidates=None, stall_steps=None, stall_threshold=None, wait_weight=1.0, stall_weight=1.0):
    """Select at most one starving or stalled robot for recovery priority.

    English implementation: filters threshold-qualified robots
    and chooses the best dynamic-priority key.
    """

    n_agents = len(wait_steps)
    if stall_steps is not None and len(stall_steps) != n_agents:
        raise ValueError("stall_steps must match wait_steps")
    stall_limit = int(threshold if stall_threshold is None else stall_threshold)
    pool = range(n_agents) if candidates is None else candidates
    qualified = []
    for robot_id in pool:
        robot_id = int(robot_id)
        wait_hit = int(wait_steps[robot_id]) >= int(threshold)
        stall_hit = (
            stall_steps is not None
            and int(stall_steps[robot_id]) >= stall_limit
        )
        if wait_hit or stall_hit:
            qualified.append(robot_id)
    if not qualified:
        return None
    return min(
        qualified,
        key=lambda robot_id: priority_key(
            robot_id,
            wait_steps,
            priority_token,
            n_agents,
            stall_steps,
            wait_weight=wait_weight,
            stall_weight=stall_weight,
        ),
    )

