"""Bounded synchronous collision resolver with deterministic fallback.

具有确定性兜底策略的有界同步碰撞消解器。
"""
from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

from collision.candidates import blocked_robot_ids, normalise_candidates
from collision.geometry import is_wait, pair_conflict, shared_overlap_exists
from collision.models import CollisionResolution


class JointCollisionResolver:
    """Resolve a joint one-step action without vertex or edge-swap conflicts.

    中文目的：在保留各机器人候选动作顺序的前提下，搜索一组无冲突同步动作。
    中文实现：按照上层给定的优先级执行有界回溯；若共享起点下搜索受限，
    则允许一台高优先级机器人串行离开，其余机器人等待。

    English implementation: performs priority-ordered bounded backtracking. When
    a permitted shared-depot overlap exists and the search budget is exhausted,
    it attempts a one-robot serial-departure fallback.
    """

    def __init__(
        self,
        n_agents: int,
        *,
        safe_distance: float,
        max_backtracking_nodes: int = 20000,
    ) -> None:
        """Initialise one bounded collision resolver.

        中文目的：为指定机器人数量配置安全距离和联合搜索预算。
        中文实现：验证参数后保存配置，并初始化共享起点串行兜底计数。
        English purpose: configure collision resolution for a fixed robot team.
        English implementation: validates limits, stores search parameters, and
        initialises the serial-departure fallback counter.
        """

        if n_agents <= 0:
            raise ValueError("n_agents must be positive")
        if safe_distance <= 0:
            raise ValueError("safe_distance must be positive")
        if max_backtracking_nodes < 1:
            raise ValueError("max_backtracking_nodes must be at least 1")
        self.n_agents = int(n_agents)
        self.safe_distance = float(safe_distance)
        self.max_backtracking_nodes = int(max_backtracking_nodes)
        self.serial_fallback_events = 0

    def _feasible_against_assignment(
        self,
        robot_id: int,
        candidate: np.ndarray,
        current: Sequence[np.ndarray],
        assignment: Sequence[np.ndarray | None],
        *,
        allow_shared_start_wait: bool,
    ) -> bool:
        """Check one candidate against already assigned higher-priority robots.

        中文目的：为回溯搜索提供增量冲突检查，避免重复验证未分配机器人。
        English implementation: applies pairwise conflict predicates only to
        entries that already have a selected next position.
        """

        for other_id, other_next in enumerate(assignment):
            if other_next is None:
                continue
            if pair_conflict(
                current[robot_id],
                candidate,
                current[other_id],
                other_next,
                safe_distance=self.safe_distance,
                allow_shared_start_wait=allow_shared_start_wait,
            ):
                return False
        return True

    def _serial_departure(
        self,
        current: List[np.ndarray],
        candidates: List[List[np.ndarray]],
        priority_order: Sequence[int],
        *,
        allow_shared_start_wait: bool,
    ) -> List[np.ndarray]:
        """Move at most one priority robot while all peers remain waiting.

        中文目的：共享起点场景中避免“全体永久等待”，提供保守的单机器人离开方案。
        English implementation: scans priority order and returns the first moving
        candidate compatible with every other robot's waiting action.
        """

        selected = [position.copy() for position in current]
        for robot_id in priority_order:
            for candidate in candidates[robot_id]:
                if is_wait(current[robot_id], candidate):
                    continue
                if all(
                    other_id == robot_id
                    or not pair_conflict(
                        current[robot_id],
                        candidate,
                        current[other_id],
                        current[other_id],
                        safe_distance=self.safe_distance,
                        allow_shared_start_wait=allow_shared_start_wait,
                    )
                    for other_id in range(self.n_agents)
                ):
                    selected[robot_id] = candidate.copy()
                    self.serial_fallback_events += 1
                    return selected
        return selected

    def resolve(
        self,
        current_positions: Sequence[Sequence[float]],
        candidate_lists: Sequence[Iterable[Sequence[float]]],
        *,
        priority_order: Sequence[int] | None = None,
        time_step: int = 0,
        allow_shared_start_step0: bool = False,
    ) -> CollisionResolution:
        """Choose one collision-free next position for every robot.

        中文目的：输出可直接交给环境同步执行的无冲突联合动作。
        中文实现：标准化候选列表、按优先级有界回溯，并在允许共享起点时使用
        串行离开兜底。`time_step` 用于保持逐步实验调用与日志语义一致。

        English implementation: normalises candidates, runs bounded recursive
        assignment, and returns conservative waits when no legal assignment is
        found outside a permitted shared-depot cluster.
        """

        del time_step
        current, candidates = normalise_candidates(current_positions, candidate_lists)
        if len(current) != self.n_agents:
            raise ValueError("position arrays must match n_agents")

        order = list(range(self.n_agents)) if priority_order is None else list(priority_order)
        if sorted(order) != list(range(self.n_agents)):
            raise ValueError("priority_order must contain every robot ID exactly once")

        assignment: List[np.ndarray | None] = [None] * self.n_agents
        search_nodes = 0

        def search(depth: int) -> bool:
            nonlocal search_nodes
            if depth >= len(order):
                return True
            if search_nodes >= self.max_backtracking_nodes:
                return False

            robot_id = order[depth]
            for candidate in candidates[robot_id]:
                if search_nodes >= self.max_backtracking_nodes:
                    return False
                search_nodes += 1
                if not self._feasible_against_assignment(
                    robot_id,
                    candidate,
                    current,
                    assignment,
                    allow_shared_start_wait=allow_shared_start_step0,
                ):
                    continue
                assignment[robot_id] = candidate
                if search(depth + 1):
                    return True
                assignment[robot_id] = None
            return False

        solved = search(0)
        used_serial_fallback = False
        if solved:
            selected = [
                current[robot_id].copy()
                if assignment[robot_id] is None
                else assignment[robot_id].copy()
                for robot_id in range(self.n_agents)
            ]
        elif allow_shared_start_step0 and shared_overlap_exists(
            current, safe_distance=self.safe_distance
        ):
            selected = self._serial_departure(
                current,
                candidates,
                order,
                allow_shared_start_wait=True,
            )
            used_serial_fallback = any(
                not is_wait(current[robot_id], selected[robot_id])
                for robot_id in range(self.n_agents)
            )
        else:
            selected = [position.copy() for position in current]

        blocked = blocked_robot_ids(current, selected, candidates)
        return CollisionResolution(
            positions=tuple(position.copy() for position in selected),
            blocked_robot_ids=tuple(blocked),
            backtracking_nodes=int(search_nodes),
            used_serial_fallback=bool(used_serial_fallback),
        )


def resolve_synchronous_moves(
    current_positions: Sequence[Sequence[float]],
    candidate_lists: Sequence[Iterable[Sequence[float]]],
    *,
    safe_distance: float,
    time_step: int,
    allow_shared_start_step0: bool = False,
) -> tuple[List[np.ndarray], List[int]]:
    """Run stateless collision-only resolution for ablation experiments.

    中文目的：提供不保存等待历史的碰撞消融接口，用于 MergingMap 的仅碰撞消融实验。
    English implementation: creates one temporary resolver using robot-ID order
    and returns selected positions plus dynamically blocked robot IDs.
    """

    resolver = JointCollisionResolver(
        len(current_positions),
        safe_distance=safe_distance,
    )
    result = resolver.resolve(
        current_positions,
        candidate_lists,
        priority_order=range(len(current_positions)),
        time_step=time_step,
        allow_shared_start_step0=allow_shared_start_step0,
    )
    return [position.copy() for position in result.positions], list(result.blocked_robot_ids)
