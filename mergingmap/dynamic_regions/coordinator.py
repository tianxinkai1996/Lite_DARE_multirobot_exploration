"""Lifecycle coordinator for contact-aware dynamic frontier leases.

面向接触通信的动态前沿租约生命周期协调器。
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .allocation import RegionAllocationMixin
from .diagnostics import RegionDiagnosticsMixin
from .extraction import (
    _centroid_cell_distance,
    _contact_components,
    _region_iou,
    _regions_equivalent,
    extract_frontier_regions,
    track_frontier_regions,
)
from .models import (
    DynamicRegionConfig,
    FrontierRegion,
    RegionLease,
    RobotRegionState,
)
from .routing import RegionRoutingMixin


class DynamicRegionCoordinator(
    RegionAllocationMixin,
    RegionRoutingMixin,
    RegionDiagnosticsMixin,
):
    """Maintain evolving frontier identities, leases, progress, and assignments."""

    def __init__(self, n_robots: int, config: DynamicRegionConfig) -> None:
        """Initialise per-robot region state, event queues, and paper metrics."""

        if n_robots <= 0:
            raise ValueError("n_robots must be positive")
        self.n_robots = int(n_robots)
        self.config = config
        self.states = [RobotRegionState() for _ in range(self.n_robots)]
        self._events: List[dict] = []
        for name in (
            "assignments_created",
            "reassignments_created",
            "leases_released_completed",
            "leases_released_expired",
            "leases_released_no_progress",
            "leases_released_conflict",
            "leases_released_unreachable",
            "leases_released_recovery",
            "claim_messages",
            "claim_conflicts_resolved",
            "region_candidate_overrides",
            "forced_progress_steps",
            "unreachable_candidate_steps",
            "max_unassigned_region_age",
            "region_message_packets",
            "region_message_bytes",
            "region_identity_matches",
            "effective_progress_events",
            "assignment_overlap_pair_steps",
            "assignment_overlap_steps",
            "assignment_overlap_evaluated_steps",
        ):
            setattr(self, name, 0)
        self._last_snapshot: dict = {}

    def _emit(self, event: str, **payload: object) -> None:
        """Append one structured region event to the in-memory queue."""

        self._events.append({"event": event, **payload})

    def drain_events(self) -> List[dict]:
        """Return and clear pending region events."""

        result = list(self._events)
        self._events.clear()
        return result

    def _mark_effective_progress(
        self,
        robot_id: int,
        *,
        step: int,
        reason: str,
        value: float | int | None = None,
    ) -> None:
        """Renew a lease when any methodology progress condition becomes true.

        中文目的：统一处理目标图距离下降、已知单元增加、首次到达图节点、前沿规模下降。
        English implementation: resets stall state, renews the lease, records the reason,
        and emits one event without coupling progress detection to a specific module.
        """

        state = self.states[int(robot_id)]
        lease = state.lease
        if lease is None:
            return
        lease.last_progress_step = int(step)
        lease.expiry_step = int(step + self.config.lease_steps)
        lease.last_progress_reason = str(reason)
        state.last_effective_progress_step = int(step)
        state.stall_steps = 0
        self.effective_progress_events += 1
        self._emit(
            "lease_progress",
            step=int(step),
            robot_id=int(robot_id),
            region_id=lease.region_id,
            reason=str(reason),
            value=value,
        )

    def _release(self, robot_id: int, *, step: int, reason: str) -> None:
        """Release one lease, update reason metrics, and retain failure memory."""

        state = self.states[robot_id]
        lease = state.lease
        if lease is None:
            return
        counter = {
            "completed": "leases_released_completed",
            "expired": "leases_released_expired",
            "no_progress": "leases_released_no_progress",
            "conflict": "leases_released_conflict",
            "unreachable": "leases_released_unreachable",
            "recovery": "leases_released_recovery",
        }.get(reason)
        if counter is not None:
            setattr(self, counter, int(getattr(self, counter)) + 1)
        if reason in {"no_progress", "unreachable", "recovery"}:
            state.region_failure_counts[lease.region_id] = (
                state.region_failure_counts.get(lease.region_id, 0) + 1
            )
        self._emit(
            "lease_released",
            step=int(step),
            robot_id=int(robot_id),
            region_id=lease.region_id,
            reason=reason,
        )
        state.lease = None
        state.unreachable_steps = 0
        state.graph_cache_key = None
        state.graph_distance_cache.clear()

    def request_recovery_release(self, robot_id: int, *, step: int) -> bool:
        """Release a stalled lease when staged deadlock recovery requests reassignment."""

        if self.states[int(robot_id)].lease is None:
            return False
        self._release(int(robot_id), step=int(step), reason="recovery")
        return True

    def _assign(
        self,
        robot_id: int,
        region: FrontierRegion,
        *,
        step: int,
    ) -> None:
        """Create one temporary lease and preserve assignment/reassignment counts."""

        state = self.states[robot_id]
        generation = state.assignment_count
        state.assignment_count += 1
        if generation > 0:
            state.reassignment_count += 1
            self.reassignments_created += 1
        state.lease = RegionLease(
            owner_robot_id=robot_id,
            region=region,
            claimed_step=int(step),
            expiry_step=int(step + self.config.lease_steps),
            last_progress_step=int(step),
            last_frontier_count=int(region.frontier_count),
            generation=generation,
        )
        state.last_effective_progress_step = int(step)
        state.stall_steps = 0
        state.unreachable_steps = 0
        state.graph_cache_key = None
        state.graph_distance_cache.clear()
        self.assignments_created += 1
        self._emit(
            "lease_assigned",
            step=int(step),
            robot_id=int(robot_id),
            region_id=region.region_id,
            frontier_count=region.frontier_count,
            target_world=list(region.target_world),
            generation=int(generation),
        )

    def _match_existing_lease(self, robot_id: int, *, step: int) -> None:
        """Attach an existing lease to its tracked frontier and detect progress."""

        state = self.states[robot_id]
        lease = state.lease
        if lease is None:
            return
        exact = state.regions.get(lease.region_id)
        if exact is None:
            best_region: Optional[FrontierRegion] = None
            best_score = -math.inf
            for region in state.regions.values():
                iou = _region_iou(lease.region, region)
                distance = _centroid_cell_distance(lease.region, region)
                if (
                    iou < self.config.region_match_iou_threshold
                    and distance > self.config.region_match_centroid_cells
                ):
                    continue
                score = 10.0 * iou - distance / max(
                    1.0, self.config.region_match_centroid_cells
                )
                if score > best_score:
                    best_score, best_region = score, region
            if best_region is None:
                self._release(robot_id, step=step, reason="completed")
                return
            exact = best_region

        old_count = int(lease.last_frontier_count or lease.region.frontier_count)
        lease.region = exact
        lease.last_frontier_count = int(exact.frontier_count)
        state.graph_cache_key = None
        state.graph_distance_cache.clear()
        if exact.frontier_count < old_count:
            self._mark_effective_progress(
                robot_id,
                step=step,
                reason="frontier_reduced",
                value=old_count - exact.frontier_count,
            )

        commitment_age = int(step) - int(lease.claimed_step)
        if step > lease.expiry_step:
            self._release(robot_id, step=step, reason="expired")
            return
        state.stall_steps = max(0, int(step) - int(lease.last_progress_step))
        if (
            commitment_age >= self.config.min_commitment_steps
            and state.stall_steps >= self.config.no_progress_release_steps
        ):
            self._release(robot_id, step=step, reason="no_progress")

    def _expire_known_claims(self, step: int) -> None:
        """Remove peer region claims whose communication TTL expired."""

        for state in self.states:
            expired = [
                region_id
                for region_id, claim in state.known_peer_claims.items()
                if claim.expiry_step < step
            ]
            for region_id in expired:
                del state.known_peer_claims[region_id]

    def update(
        self,
        *,
        step: int,
        robot_maps: Sequence[np.ndarray],
        robot_positions: Sequence[Sequence[float]],
        contact_pairs: Sequence[Tuple[int, int]],
        env: object,
        robots: Sequence[object] | None = None,
        wait_steps: Sequence[int] | None = None,
    ) -> dict:
        """Refresh tracked regions, leases, claims, and graph-aware assignments."""

        if len(robot_maps) != self.n_robots:
            raise ValueError("robot_maps length does not match n_robots")
        if len(robot_positions) != self.n_robots:
            raise ValueError("robot_positions length does not match n_robots")
        if robots is not None and len(robots) != self.n_robots:
            raise ValueError("robots length does not match n_robots")
        positions = [np.asarray(position, dtype=float) for position in robot_positions]
        wait_values = [0] * self.n_robots if wait_steps is None else list(wait_steps)
        self._expire_known_claims(int(step))

        for robot_id, occupancy in enumerate(robot_maps):
            state = self.states[robot_id]
            previous = list(state.regions.values())
            extracted = extract_frontier_regions(
                occupancy, env=env, config=self.config
            )
            tracked = track_frontier_regions(previous, extracted, self.config)
            self.region_identity_matches += sum(
                region.region_id in state.regions for region in tracked
            )
            state.regions = {region.region_id: region for region in tracked}
            state.wait_steps = int(wait_values[robot_id])
            for region in tracked:
                state.first_seen_step.setdefault(region.region_id, int(step))
            self._match_existing_lease(robot_id, step=int(step))

        components = _contact_components(self.n_robots, contact_pairs)
        for component in components:
            self._resolve_component_conflicts(
                component,
                positions,
                step=int(step),
                robots=robots,
                wait_steps=wait_values,
            )
            self._exchange_component_claims(component, step=int(step))
        for component in components:
            self._assign_component(
                component,
                positions,
                step=int(step),
                robots=robots,
            )

        # Paper-facing region overlap is measured after conflict resolution and
        # assignment. It counts simultaneous equivalent leases in the shared map
        # frame; this is independent of later robot trajectory overlap.
        overlap_pairs = 0
        for robot_i in range(self.n_robots):
            lease_i = self.states[robot_i].lease
            if lease_i is None:
                continue
            for robot_j in range(robot_i + 1, self.n_robots):
                lease_j = self.states[robot_j].lease
                if lease_j is None:
                    continue
                if _regions_equivalent(lease_i.region, lease_j.region, self.config):
                    overlap_pairs += 1
        self.assignment_overlap_evaluated_steps += 1
        self.assignment_overlap_pair_steps += int(overlap_pairs)
        if overlap_pairs > 0:
            self.assignment_overlap_steps += 1

        snapshot = self.snapshot(step=int(step))
        self._last_snapshot = snapshot
        return snapshot

    def stall_step_counts(self) -> list[int]:
        """Return per-robot no-progress counts for motion-priority integration."""

        return [int(state.stall_steps) for state in self.states]

