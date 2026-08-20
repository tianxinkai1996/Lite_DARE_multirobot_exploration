"""A-B-A oscillation detection and soft candidate penalties.

A-B-A 往返振荡检测与候选动作软惩罚。
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable, Sequence

import numpy as np


class OscillationTracker:
    """Track recent executed positions and discourage immediate reversals.

    中文目的：检测机器人在同一无向边上反复执行 A->B->A，并在不违反硬安全
    约束的前提下，降低立即返回前一节点的候选优先级。
    中文实现：每台机器人保存短位置历史；环境执行后识别 A-B-A，按无向边累计
    次数；候选排序时只增加软惩罚，因此死胡同中的必要回退仍然可行。

    English purpose: detect repeated A-B-A reversals and softly demote an
    immediate return to the previous node. English implementation: stores short
    executed histories, counts reversals per undirected edge, and adds a finite
    rank penalty without ever turning a legal backtrack into a hard constraint.
    """

    def __init__(
        self,
        n_agents: int,
        *,
        base_penalty: float = 2.0,
        repeat_penalty: float = 1.0,
        history_length: int = 8,
    ) -> None:
        if n_agents <= 0:
            raise ValueError("n_agents must be positive")
        if history_length < 3:
            raise ValueError("history_length must be at least three")
        if base_penalty < 0 or repeat_penalty < 0:
            raise ValueError("oscillation penalties cannot be negative")
        self.n_agents = int(n_agents)
        self.base_penalty = float(base_penalty)
        self.repeat_penalty = float(repeat_penalty)
        self.histories = [deque(maxlen=int(history_length)) for _ in range(n_agents)]
        self.edge_reversal_counts = [defaultdict(int) for _ in range(n_agents)]
        self.oscillation_events = 0
        self.candidate_demotions = 0
        self.max_edge_reversal_count = 0

    @staticmethod
    def _key(position: Sequence[float]) -> tuple[float, float]:
        value = np.asarray(position, dtype=float).reshape(2)
        return round(float(value[0]), 4), round(float(value[1]), 4)

    @classmethod
    def _edge_key(
        cls,
        first: Sequence[float],
        second: Sequence[float],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        endpoints = sorted((cls._key(first), cls._key(second)))
        return endpoints[0], endpoints[1]

    def initialise(self, positions: Sequence[Sequence[float]]) -> None:
        """Seed histories with the episode's actual initial positions.

        中文目的：保证第一次执行后即可正确识别位置历史，而不制造虚假振荡。
        English implementation: clears each history and inserts its real start node.
        """

        if len(positions) != self.n_agents:
            raise ValueError("positions must match n_agents")
        for history, position in zip(self.histories, positions):
            history.clear()
            history.append(self._key(position))

    def update_after_execution(
        self,
        actual_positions: Sequence[Sequence[float]],
    ) -> tuple[int, ...]:
        """Append executed positions and return robots that formed A-B-A.

        中文目的：只依据环境实际位置统计振荡，避免把被静态障碍拒绝的提议动作
        当作真实往返。English implementation: detects equality between the newest
        node and the node two executions earlier after appending the actual result.
        """

        if len(actual_positions) != self.n_agents:
            raise ValueError("actual_positions must match n_agents")
        detected: list[int] = []
        for robot_id, position in enumerate(actual_positions):
            history = self.histories[robot_id]
            key = self._key(position)
            if not history or history[-1] != key:
                history.append(key)
            else:
                # Waiting is tracked by the deadlock module, not as an oscillation.
                continue
            if len(history) >= 3 and history[-1] == history[-3]:
                edge = self._edge_key(history[-2], history[-1])
                count = self.edge_reversal_counts[robot_id][edge] + 1
                self.edge_reversal_counts[robot_id][edge] = count
                self.oscillation_events += 1
                self.max_edge_reversal_count = max(
                    self.max_edge_reversal_count, count
                )
                detected.append(robot_id)
        return tuple(detected)

    def penalty(
        self,
        robot_id: int,
        current_position: Sequence[float],
        candidate: Sequence[float],
    ) -> float:
        """Return a finite soft penalty for an immediate reverse candidate."""

        robot_id = int(robot_id)
        history = self.histories[robot_id]
        if len(history) < 2:
            return 0.0
        previous = history[-2]
        if self._key(candidate) != previous:
            return 0.0
        edge = self._edge_key(current_position, candidate)
        repeats = int(self.edge_reversal_counts[robot_id].get(edge, 0))
        return float(self.base_penalty + self.repeat_penalty * repeats)

    def order_candidates(
        self,
        *,
        robot_id: int,
        current_position: Sequence[float],
        ordered_candidates: Iterable[Sequence[float]],
        disabled: bool = False,
    ) -> list[np.ndarray]:
        """Rerank candidates by original rank plus oscillation penalty.

        中文目的：保持 DARE 原始顺序为主，仅在存在直接反向候选时施加可恢复的
        软代价；恢复阶段可通过 ``disabled`` 临时移除该代价。
        English implementation: stable-sorts by DARE rank plus the finite reversal
        penalty; disabling restores the supplied order exactly.
        """

        candidates = [np.asarray(value, dtype=np.float32).copy() for value in ordered_candidates]
        if disabled or len(candidates) <= 1:
            return candidates
        scored = [
            (
                float(rank)
                + self.penalty(robot_id, current_position, candidate),
                rank,
                candidate,
            )
            for rank, candidate in enumerate(candidates)
        ]
        scored.sort(key=lambda item: (item[0], item[1]))
        output = [item[2] for item in scored]
        if output and candidates and not np.allclose(output[0], candidates[0]):
            self.candidate_demotions += 1
        return output

    def metrics(self) -> dict[str, int | float]:
        """Return episode-level oscillation diagnostics."""

        return {
            "oscillation_events": int(self.oscillation_events),
            "oscillation_candidate_demotions": int(self.candidate_demotions),
            "oscillation_max_edge_reversal_count": int(
                self.max_edge_reversal_count
            ),
            "oscillation_base_penalty": float(self.base_penalty),
            "oscillation_repeat_penalty": float(self.repeat_penalty),
        }
