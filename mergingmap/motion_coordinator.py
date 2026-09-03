"""MergingMap facade combining motion exchange, collision, and deadlock modules.

MergingMap 
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable, List, Literal, Sequence

import numpy as np

from collision.candidates import preferred_positions
from collision.joint_resolver import JointCollisionResolver
from collision.models import ResolutionInfo
from collision.motion_exchange import MotionExchangeConfig, MotionIntentExchange
from deadlock.backtracking import nearest_branch_backtrack_step
from deadlock.oscillation_tracker import OscillationTracker
from deadlock.state_tracker import DeadlockStateTracker

CoordinationMode = Literal["ghost", "collision", "collision_deadlock"]

@dataclass(frozen=True)
class MotionDecision:
    """Store one coordinated motion decision and recovery requests.

    English implementation: packages executable positions and
    cross-layer recovery requests without exposing lower-level module state.
    """

    preferred_positions: tuple[np.ndarray, ...]
    next_positions: tuple[np.ndarray, ...]
    blocked_robot_ids: tuple[int, ...]
    resolution_info: ResolutionInfo | None
    lease_release_robot_ids: tuple[int, ...] = ()

class MergingMapMotionCoordinator:
    """Coordinate graph-legal DARE moves through modular safety/recovery layers.

    English implementation: combines compact peer reservations, bounded joint
    collision search, wait/stall priority, soft A-B-A suppression, lease-release
    signalling, and graph-safe backtracking behind one MergingMap-facing API.
    """

    VALID_MODES = {"ghost", "collision", "collision_deadlock"}

    def __init__(self, n_agents, safe_distance, mode="collision_deadlock", deadlock_wait_threshold=3, deadlock_stall_threshold=None, deadlock_soft_relax_threshold=None, deadlock_lease_release_threshold=None, deadlock_backtrack_threshold=None, deadlock_wait_weight=1.0, deadlock_stall_weight=1.0, max_backtracking_nodes=20000, node_resolution=1.0, motion_reservation_horizon=2, motion_cache_ttl_steps=2, motion_packet_header_bytes=40, oscillation_base_penalty=2.0, oscillation_repeat_penalty=1.0):
        if mode not in self.VALID_MODES:
            raise ValueError(f"unsupported coordination mode: {mode}")
        self.n_agents = int(n_agents)
        self.mode: CoordinationMode = mode
        self.collision_resolver = JointCollisionResolver(
            self.n_agents,
            safe_distance=safe_distance,
            max_backtracking_nodes=max_backtracking_nodes,
        )
        self.deadlock_tracker = DeadlockStateTracker(
            self.n_agents,
            wait_threshold=deadlock_wait_threshold,
            stall_threshold=deadlock_stall_threshold,
            soft_relax_threshold=deadlock_soft_relax_threshold,
            lease_release_threshold=deadlock_lease_release_threshold,
            backtrack_threshold=deadlock_backtrack_threshold,
            wait_weight=deadlock_wait_weight,
            stall_weight=deadlock_stall_weight,
        )
        self.motion_exchange = MotionIntentExchange(
            self.n_agents,
            MotionExchangeConfig(
                node_resolution=float(node_resolution),
                safe_distance=float(safe_distance),
                reservation_horizon=int(motion_reservation_horizon),
                cache_ttl_steps=int(motion_cache_ttl_steps),
                packet_header_bytes=int(motion_packet_header_bytes),
            ),
        )
        self.oscillation_tracker = OscillationTracker(
            self.n_agents,
            base_penalty=oscillation_base_penalty,
            repeat_penalty=oscillation_repeat_penalty,
        )
        self._oscillation_initialised = False
        self.resolver_backtracking_nodes = 0
        self.deadlock_escape_activations = 0
        self.preferred_action_selection_ms = 0.0
        self.deadlock_priority_ms = 0.0
        self.deadlock_escape_ms = 0.0
        self.collision_resolution_ms = 0.0
        self.deadlock_state_update_ms = 0.0
        self.coordination_total_ms = 0.0
        self.graph_backtrack_ms = 0.0
        self.graph_backtrack_activations = 0
        self.lease_release_requests = 0
        self.soft_penalty_relaxations = 0

    @property
    def wait_steps(self):
        """Expose persistent consecutive-wait counters for tests/diagnostics."""

        return self.deadlock_tracker.wait_steps

    @property
    def priority_token(self):
        """Expose the rotating deterministic tie-break token."""

        return int(self.deadlock_tracker.priority_token)

    @property
    def deadlock_break_events(self):
        """Return successful recovery movements."""

        return int(self.deadlock_tracker.deadlock_break_events)

    @property
    def max_wait_steps(self):
        """Return the episode maximum consecutive wait count."""

        return int(self.deadlock_tracker.max_wait_steps)

    @property
    def serial_fallback_events(self):
        """Return shared-depot serial-departure fallback events."""

        return int(self.collision_resolver.serial_fallback_events)

    def priority_order(self):
        """Return wait/stall-aware robot order."""

        return self.deadlock_tracker.ordered_robot_ids()

    def escape_robot_ids(self):
        """Select at most one robot for staged recovery in full mode."""

        if self.mode != "collision_deadlock":
            return set()
        selected = self.deadlock_tracker.escape_robot_ids()
        self.deadlock_escape_activations += len(selected)
        return selected

    @staticmethod
    def _copy_candidate_lists(values):
        return [
            [np.asarray(candidate, dtype=np.float32).copy() for candidate in group]
            for group in values
        ]

    def _plans_or_preferences(self, current_positions, candidates, plans):
        """Normalise optional one/two-step plans for compact motion packets."""

        if plans is not None:
            if len(plans) != self.n_agents:
                raise ValueError("short_horizon_plans must match n_agents")
            return [
                [np.asarray(point, dtype=np.float32).copy() for point in plan[:2]]
                or [np.asarray(current_positions[i], dtype=np.float32).copy()]
                for i, plan in enumerate(plans)
            ]
        output = []
        for robot_id in range(self.n_agents):
            current = np.asarray(current_positions[robot_id], dtype=np.float32)
            first = candidates[robot_id][0] if candidates[robot_id] else current
            output.append([np.asarray(first, dtype=np.float32).copy()])
        return output

    def _apply_recovery_candidates(self, current_positions, candidates, recovery_candidate_lists, stages, robots):
        """Apply soft-relaxation, lease-release, and graph-backtrack stages."""

        recovery = (
            None
            if recovery_candidate_lists is None
            else self._copy_candidate_lists(recovery_candidate_lists)
        )
        soft_ids = tuple(i for i, stage in enumerate(stages) if stage >= 3)
        lease_ids = tuple(i for i, stage in enumerate(stages) if stage >= 4)
        backtrack_ids: list[int] = []
        for robot_id in range(self.n_agents):
            if robot_id in soft_ids and recovery is not None:
                candidates[robot_id] = recovery[robot_id]
            candidates[robot_id] = self.oscillation_tracker.order_candidates(
                robot_id=robot_id,
                current_position=current_positions[robot_id],
                ordered_candidates=candidates[robot_id],
                disabled=robot_id in soft_ids,
            )
            if (
                stages[robot_id] >= 5
                and robots is not None
                and robot_id < len(robots)
            ):
                started_at = time.perf_counter()
                backtrack = nearest_branch_backtrack_step(
                    robot=robots[robot_id],
                    current_position=current_positions[robot_id],
                    safe_candidates=candidates[robot_id],
                )
                self.graph_backtrack_ms += (time.perf_counter() - started_at) * 1000.0
                if backtrack is not None:
                    remaining = [
                        value for value in candidates[robot_id]
                        if not np.allclose(value, backtrack)
                    ]
                    candidates[robot_id] = [backtrack, *remaining]
                    backtrack_ids.append(robot_id)
        self.soft_penalty_relaxations += len(soft_ids)
        self.lease_release_requests += len(lease_ids)
        self.graph_backtrack_activations += len(backtrack_ids)
        return candidates, soft_ids, lease_ids, tuple(backtrack_ids)

    def resolve_step(self, current_positions, candidate_lists, time_step, allow_shared_start_step0=False, contact_pairs=(), short_horizon_plans=None, recovery_candidate_lists=None, stall_steps=None, robots=None):
        """Resolve one graph-legal multi-robot motion step.

        English implementation: keeps ghost
        mode unmodified, while coordinated modes use compact contact reservations and
        a final synchronous resolver; deadlock recovery never relaxes hard safety.
        """

        coordination_started_at = time.perf_counter()
        candidates = self._copy_candidate_lists(candidate_lists)
        preferred_started_at = time.perf_counter()
        preferred = preferred_positions(current_positions, candidates)
        preferred_ms = (time.perf_counter() - preferred_started_at) * 1000.0
        self.preferred_action_selection_ms += preferred_ms
        if not self._oscillation_initialised:
            self.oscillation_tracker.initialise(current_positions)
            self._oscillation_initialised = True
        if self.mode == "ghost":
            elapsed = (time.perf_counter() - coordination_started_at) * 1000.0
            self.coordination_total_ms += elapsed
            return MotionDecision(
                preferred_positions=tuple(value.copy() for value in preferred),
                next_positions=tuple(value.copy() for value in preferred),
                blocked_robot_ids=(),
                resolution_info=None,
            )

        priority_started_at = time.perf_counter()
        if self.mode == "collision_deadlock":
            self.deadlock_tracker.set_stall_steps(stall_steps)
            order = self.deadlock_tracker.ordered_robot_ids()
            priorities = self.deadlock_tracker.priorities()
            stages = self.deadlock_tracker.recovery_stages()
        else:
            order = list(range(self.n_agents))
            priorities = [0.0 for _ in range(self.n_agents)]
            stages = tuple(0 for _ in range(self.n_agents))
        priority_ms = (time.perf_counter() - priority_started_at) * 1000.0
        self.deadlock_priority_ms += priority_ms

        escape_started_at = time.perf_counter()
        escape_ids = (
            tuple(sorted(self.escape_robot_ids()))
            if self.mode == "collision_deadlock"
            else ()
        )
        escape_ms = (time.perf_counter() - escape_started_at) * 1000.0
        self.deadlock_escape_ms += escape_ms

        soft_ids: tuple[int, ...] = ()
        lease_ids: tuple[int, ...] = ()
        backtrack_ids: tuple[int, ...] = ()
        if self.mode == "collision_deadlock":
            candidates, soft_ids, lease_ids, backtrack_ids = self._apply_recovery_candidates(
                current_positions=current_positions,
                candidates=candidates,
                recovery_candidate_lists=recovery_candidate_lists,
                stages=stages,
                robots=robots,
            )

        plans = self._plans_or_preferences(current_positions, candidates, short_horizon_plans)
        exchange_before = self.motion_exchange.exchange_ms
        filter_before = self.motion_exchange.filter_ms
        if contact_pairs:
            self.motion_exchange.exchange(
                step=int(time_step),
                contact_pairs=contact_pairs,
                current_positions=current_positions,
                plans=plans,
                priorities=priorities,
            )
            candidates = self.motion_exchange.filter_candidates(
                step=int(time_step),
                current_positions=current_positions,
                candidate_lists=candidates,
                contact_pairs=contact_pairs,
                remove_soft_costs_for=soft_ids,
                priorities=priorities,
            )
        motion_exchange_ms = self.motion_exchange.exchange_ms - exchange_before
        motion_filter_ms = self.motion_exchange.filter_ms - filter_before

        collision_started_at = time.perf_counter()
        result = self.collision_resolver.resolve(
            current_positions,
            candidates,
            priority_order=order,
            time_step=time_step,
            allow_shared_start_step0=allow_shared_start_step0,
        )
        collision_ms = (time.perf_counter() - collision_started_at) * 1000.0
        stage_values = list(stages)
        forced_stationary = []
        if self.mode == "collision_deadlock":
            for robot_id, stage in enumerate(stage_values):
                if (
                    stage >= 5
                    and np.allclose(result.positions[robot_id], current_positions[robot_id])
                ):
                    stage_values[robot_id] = 6
                    forced_stationary.append(robot_id)
            self.deadlock_tracker.record_forced_stationary(forced_stationary)
        stages = tuple(stage_values)
        coordination_ms = (time.perf_counter() - coordination_started_at) * 1000.0
        self.collision_resolution_ms += collision_ms
        self.coordination_total_ms += coordination_ms
        self.resolver_backtracking_nodes += int(result.backtracking_nodes)
        info = ResolutionInfo(
            priority_order=tuple(order),
            escape_robot_ids=escape_ids,
            blocked_robot_ids=result.blocked_robot_ids,
            backtracking_nodes=int(result.backtracking_nodes),
            used_serial_fallback=bool(result.used_serial_fallback),
            preferred_action_selection_ms=float(preferred_ms),
            deadlock_priority_ms=float(priority_ms),
            deadlock_escape_ms=float(escape_ms),
            collision_resolution_ms=float(collision_ms),
            coordination_total_ms=float(coordination_ms),
            recovery_stages=tuple(int(value) for value in stages),
            soft_penalty_relaxed_robot_ids=soft_ids,
            lease_release_robot_ids=lease_ids,
            graph_backtrack_robot_ids=backtrack_ids,
            motion_exchange_ms=float(motion_exchange_ms),
            motion_filter_ms=float(motion_filter_ms),
            graph_backtrack_ms=float(self.graph_backtrack_ms),
        )
        return MotionDecision(
            preferred_positions=tuple(value.copy() for value in preferred),
            next_positions=tuple(value.copy() for value in result.positions),
            blocked_robot_ids=result.blocked_robot_ids,
            resolution_info=info,
            lease_release_robot_ids=lease_ids,
        )

    def update_after_execution(self, previous_positions, actual_positions):
        """Update persistent waiting and oscillation state from actual motion."""

        started_at = time.perf_counter()
        if self.mode == "collision_deadlock":
            self.deadlock_tracker.update_after_execution(previous_positions, actual_positions)
            self.oscillation_tracker.update_after_execution(actual_positions)
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        self.deadlock_state_update_ms += elapsed_ms
        return float(elapsed_ms)

    def metrics(self):
        """Return flattened collision, motion-message, oscillation, and recovery metrics."""

        return {
            "coordination_mode": self.mode,
            "deadlock_escape_activations": int(self.deadlock_escape_activations),
            "resolver_backtracking_nodes": int(self.resolver_backtracking_nodes),
            "serial_fallback_events": int(self.serial_fallback_events),
            "soft_penalty_relaxations": int(self.soft_penalty_relaxations),
            "lease_release_requests": int(self.lease_release_requests),
            "graph_backtrack_activations": int(self.graph_backtrack_activations),
            "graph_backtrack_total_ms": float(self.graph_backtrack_ms),
            "preferred_action_selection_total_ms": float(self.preferred_action_selection_ms),
            "deadlock_priority_total_ms": float(self.deadlock_priority_ms),
            "deadlock_escape_selection_total_ms": float(self.deadlock_escape_ms),
            "collision_resolution_total_ms": float(self.collision_resolution_ms),
            "deadlock_state_update_total_ms": float(self.deadlock_state_update_ms),
            "coordination_total_ms": float(self.coordination_total_ms),
            **self.deadlock_tracker.metrics(),
            **self.motion_exchange.metrics(),
            **self.oscillation_tracker.metrics(),
        }
