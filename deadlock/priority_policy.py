"""Pure starvation and no-progress priority functions.

连续等待与无进展状态使用的无状态优先级函数。
"""
from __future__ import annotations

from typing import Iterable, Sequence


def priority_score(
    robot_id: int,
    wait_steps: Sequence[int],
    stall_steps: Sequence[int] | None = None,
    *,
    wait_weight: float = 1.0,
    stall_weight: float = 1.0,
) -> float:
    """Return the dynamic priority value used by short motion messages.

    中文目的：把连续等待和区域无进展合成为论文中的动态运动优先级。
    English implementation: computes lambda_wait*n_wait + lambda_stall*n_stall;
    deterministic token/ID tie-breaking is applied separately by ``priority_key``.
    """

    stall = 0 if stall_steps is None else int(stall_steps[int(robot_id)])
    return float(
        float(wait_weight) * int(wait_steps[int(robot_id)])
        + float(stall_weight) * stall
    )


def priority_key(
    robot_id: int,
    wait_steps: Sequence[int],
    priority_token: int,
    n_agents: int,
    stall_steps: Sequence[int] | None = None,
    *,
    wait_weight: float = 1.0,
    stall_weight: float = 1.0,
) -> tuple[float, int, int]:
    """Build a deterministic starvation/no-progress-aware priority key.

    中文目的：让等待或长期无进展的机器人优先，同时利用轮转令牌和机器人编号
    给出稳定且长期公平的平局处理。
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


def priority_order(
    wait_steps: Sequence[int],
    priority_token: int,
    stall_steps: Sequence[int] | None = None,
    *,
    wait_weight: float = 1.0,
    stall_weight: float = 1.0,
) -> list[int]:
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


def select_escape_robot(
    wait_steps: Sequence[int],
    priority_token: int,
    threshold: int,
    candidates: Iterable[int] | None = None,
    stall_steps: Sequence[int] | None = None,
    *,
    stall_threshold: int | None = None,
    wait_weight: float = 1.0,
    stall_weight: float = 1.0,
) -> int | None:
    """Select at most one starving or stalled robot for recovery priority.

    中文目的：避免多个机器人同时抢占恢复优先级；连续等待或无进展达到各自阈值
    均可触发候选资格。English implementation: filters threshold-qualified robots
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

