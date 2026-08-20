"""Local trajectory-reservation conflict filtering.

局部轨迹预留冲突过滤模块。
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np

from classes.multi_robot.trajectory_codec import decode_packet


class ReservationManager:
    """Filter local candidates with short-horizon peer reservations.

    中文目的：利用通信获得的同伴短期轨迹，提前过滤同点、可见机器人和边交换冲突。
    中文实现：缓存每个直接接触同伴的最新数据包；硬约束删除不安全候选，软代价
    根据近期轨迹以及可选 CoverageManager 对剩余候选重新排序。

    English purpose: prevent local motion conflicts using recently communicated
    teammate plans. English implementation: caches each peer's newest packet,
    removes candidates violating hard reservations, and reranks safe candidates
    with trail and optional coverage/goal costs.
    """

    # 中文目的：初始化本机器人缓存、预留时域与安全阈值。
    # English purpose: Initialise peer packet cache, reservation horizon, and safety thresholds.
    def __init__(
        self,
        robot_id: int,
        *,
        node_resolution: float,
        reservation_horizon: int,
        cache_ttl_steps: int,
        safe_distance: float,
        trail_avoid_radius: float,
        trail_penalty_weight: float,
    ) -> None:
        """Initialise one robot's local reservation cache.

        中文目的：配置本机器人使用的预留时域、缓存寿命、安全距离与轨迹软惩罚。
        中文实现：保存数值参数，并创建按同伴编号索引的最新数据包字典。
        English purpose: configure the local peer-reservation filter.
        English implementation: stores horizon and safety parameters and creates
        the newest-packet cache keyed by peer robot ID.
        """

        self.robot_id = int(robot_id)
        self.node_resolution = float(node_resolution)
        self.reservation_horizon = int(reservation_horizon)
        self.cache_ttl_steps = int(cache_ttl_steps)
        self.safe_distance = float(safe_distance)
        self.trail_avoid_radius = float(trail_avoid_radius)
        self.trail_penalty_weight = float(trail_penalty_weight)

        # Latest decoded packet from each directly contacted peer.
        self._packets: dict[int, dict] = {}

    # 中文目的：接收并保留同伴最新的短期轨迹数据包。
    # English purpose: Receive and retain the newest short-horizon packet from a peer.
    def receive_packet(self, packet: dict) -> None:
        """Store the newest valid packet from a directly contacted peer.

        中文目的：让后续候选过滤使用同伴最新的当前位置、短期计划和轨迹。
        中文实现：解码数据包，忽略自身消息，并仅在时间戳不旧于缓存时替换记录。
        English purpose: retain the newest peer state for later candidate filtering.
        English implementation: decodes the packet, ignores self-messages, and
        replaces the cache entry only when its step is not older.
        """

        if packet.get("type") == "short_motion_intent":
            decoded = {
                "sender_id": int(packet["sender_id"]),
                "step": int(packet["step"]),
                "current": np.asarray(packet["current"], dtype=np.float32).copy(),
                "plan": np.asarray(packet.get("plan", []), dtype=np.float32).reshape(-1, 2),
                "trail": np.empty((0, 2), dtype=np.float32),
                "priority": float(packet.get("priority", 0.0)),
                "byte_count": int(packet.get("byte_count", 0)),
            }
        else:
            decoded = decode_packet(packet, self.node_resolution)
        peer_id = decoded["sender_id"]
        if peer_id == self.robot_id:
            return
        previous = self._packets.get(peer_id)
        if previous is None or decoded["step"] >= previous["step"]:
            self._packets[peer_id] = decoded

    # 中文目的：删除超过有效期的同伴短期计划。
    # English purpose: Remove peer short-term plans that exceeded their cache TTL.
    def prune(self, current_step: int) -> None:
        """Remove short-term plans whose cache lifetime has expired.

        中文目的：避免继续依据已经失真的同伴计划阻塞候选动作。
        中文实现：比较当前步与数据包步号，删除超过 ``cache_ttl_steps`` 的记录；
        覆盖信息由独立 CoverageManager 管理，不在此处清理。
        English purpose: stop stale peer plans from blocking current candidates.
        English implementation: removes packets older than the configured TTL;
        long-lived coverage knowledge remains outside this module.
        """

        stale = [
            peer_id
            for peer_id, packet in self._packets.items()
            if int(current_step) - int(packet["step"]) > self.cache_ttl_steps
        ]
        for peer_id in stale:
            del self._packets[peer_id]

    # 中文目的：读取同伴在指定未来时间的上一节点和预留节点。
    # English purpose: Read a peer previous and reserved node at one future time.
    def _peer_plan_at(self, packet: dict, time_step: int):
        """Return a peer edge reservation for one future time step.

        中文目的：取得同伴在指定时间的上一节点和目标节点，用于同点与边交换检测。
        中文实现：把绝对时间转换为计划索引，越界时返回 ``None``，否则返回前后端点。
        English purpose: retrieve the peer edge occupied at a requested future step.
        English implementation: converts absolute time to a plan index and returns
        the previous/next nodes when the reservation is available.
        """

        # plan[0] is intended for packet_step + 1.
        plan_index = int(time_step - int(packet["step"]) - 1)
        plan = packet["plan"][: self.reservation_horizon]
        if plan_index < 0 or plan_index >= len(plan):
            return None
        previous = packet["current"] if plan_index == 0 else plan[plan_index - 1]
        return np.asarray(previous, dtype=np.float32), np.asarray(plan[plan_index], dtype=np.float32)

    @staticmethod
    # 中文目的：检测双方在同一时间步互换边端点。
    # English purpose: Detect two robots swapping edge endpoints in one time step.
    def _segment_swap(
        current: np.ndarray,
        candidate: np.ndarray,
        peer_previous: np.ndarray,
        peer_next: np.ndarray,
        tolerance: float,
    ) -> bool:
        """Detect a synchronous edge swap between two robots.

        中文目的：阻止本机器人与同伴在同一步互换边的两个端点。
        中文实现：分别比较本机器人起点与同伴终点、以及本机器人终点与同伴起点。
        English purpose: reject opposite traversal of the same edge in one step.
        English implementation: checks both crossed endpoint distances against
        the configured tolerance.
        """

        return (
            np.linalg.norm(current - peer_next) < tolerance
            and np.linalg.norm(candidate - peer_previous) < tolerance
        )

    # 中文目的：检测可见机器人、同点预留和边交换硬冲突。
    # English purpose: Detect hard conflicts from visible peers, vertex reservations, and edge swaps.
    def _hard_conflict(
        self,
        current: np.ndarray,
        candidate: np.ndarray,
        current_step: int,
        visible_peer_positions: Iterable[Sequence[float]],
        *,
        ignore_peer_reservations: bool = False,
        own_priority: float = 0.0,
    ) -> bool:
        """Return whether one candidate violates a local hard safety rule.

        中文目的：统一检测可见同伴占位、同一时刻目标预留和同步边交换。
        中文实现：先检查即时可见位置，再按下一时间步查询缓存计划；逃逸模式可跳过
        计划预留，但不会跳过可见机器人这一物理约束。
        English purpose: detect immediate and reserved hard conflicts for a move.
        English implementation: checks visible peers first, then cached vertex and
        edge reservations unless peer-plan filtering is explicitly disabled.
        """

        # A locally visible teammate is treated as a dynamic obstacle.
        for peer_pos in visible_peer_positions:
            if np.linalg.norm(candidate - np.asarray(peer_pos, dtype=np.float32)) < self.safe_distance:
                return True

        if ignore_peer_reservations:
            return False

        next_time = int(current_step) + 1
        for peer_id, packet in self._packets.items():
            peer_priority = float(packet.get("priority", 0.0))
            peer_wins = (
                peer_priority > float(own_priority)
                or (
                    abs(peer_priority - float(own_priority)) <= 1e-12
                    and int(peer_id) < self.robot_id
                )
            )
            if not peer_wins:
                continue
            time_reservation = self._peer_plan_at(packet, next_time)
            if time_reservation is None:
                continue
            peer_previous, peer_next = time_reservation
            # Same destination at the same time.
            if np.linalg.norm(candidate - peer_next) < self.safe_distance:
                return True
            # A -> B while peer reserves B -> A.
            if self._segment_swap(current, candidate, peer_previous, peer_next, self.safe_distance):
                return True
        return False

    # 中文目的：计算靠近同伴近期轨迹的软惩罚。
    # English purpose: Compute a soft penalty near recently communicated peer trails.
    def _trail_penalty(self, candidate: np.ndarray) -> float:
        """Compute a soft cost near recently communicated peer trails.

        中文目的：减少机器人重复经过同伴近期轨迹，同时不把该区域设为绝对禁区。
        中文实现：对落入避让半径的轨迹点累加线性距离惩罚。
        English purpose: discourage, rather than forbid, motion near peer trails.
        English implementation: accumulates a linear distance penalty for every
        trail point inside the avoidance radius.
        """

        penalty = 0.0
        for packet in self._packets.values():
            for point in packet["trail"]:
                distance = float(np.linalg.norm(candidate - point))
                if distance < self.trail_avoid_radius:
                    penalty += 1.0 - distance / self.trail_avoid_radius
        return penalty

    # 中文目的：过滤硬冲突并按协调代价重排 DARE 候选动作。
    # English purpose: Filter hard conflicts and rerank DARE candidates by coordination cost.
    def filter_candidates(
        self,
        *,
        current_position: Sequence[float],
        ordered_candidates: Iterable[Sequence[float]],
        current_step: int,
        visible_peer_positions: Iterable[Sequence[float]],
        coverage_manager: Optional[object] = None,
        ignore_peer_reservations: bool = False,
        ignore_soft_costs: bool = False,
        own_priority: float = 0.0,
    ) -> List[np.ndarray]:
        """Filter unsafe candidates and rerank the remaining DARE actions.

        中文目的：保持 DARE 候选偏好的同时，应用通信得到的碰撞约束与协作软代价。
        中文实现：先清理过期计划；移动候选通过硬冲突检查后，以原始排名为基础叠加
        轨迹和可选覆盖代价；等待动作保留为高成本兜底，全部移动不安全时返回等待。

        English purpose: preserve DARE preference while applying communicated
        collision constraints and optional cooperation costs. English implementation:
        prunes stale packets, removes hard conflicts, adds soft costs to the original
        rank, and retains waiting as a high-cost fallback.
        """

        self.prune(current_step)
        current = np.asarray(current_position, dtype=np.float32)
        scored = []

        for rank, candidate in enumerate(ordered_candidates):
            candidate = np.asarray(candidate, dtype=np.float32)
            is_wait = np.allclose(candidate, current)

            # Waiting is always retained as a final fallback. Moving actions
            # must pass hard physical/reservation checks.
            if not is_wait and self._hard_conflict(
                current,
                candidate,
                current_step,
                visible_peer_positions,
                ignore_peer_reservations=ignore_peer_reservations,
                own_priority=own_priority,
            ):
                continue

            # Base rank keeps DARE as the primary planner. Coordination terms
            # only reorder candidates around DARE's preference.
            score = float(rank)
            if not ignore_soft_costs:
                score += self.trail_penalty_weight * self._trail_penalty(candidate)

                if coverage_manager is not None:
                    score += float(coverage_manager.coordination_cost(candidate, current_step))

            # Waiting should be possible but not selected unless alternatives
            # are worse or unsafe.
            if is_wait:
                score += 1e3

            scored.append((score, candidate))

        if not scored:
            return [current.copy()]

        scored.sort(key=lambda item: item[0])
        return [candidate for _, candidate in scored]