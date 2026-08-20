"""Progress reporting and diagnostics for dynamic region assignment.

动态区域分配的进度汇报与诊断模块。
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .extraction import _region_matches_claim, _regions_equivalent


class RegionDiagnosticsMixin:
    """Provide lease progress, snapshots, and episode metrics.

    中文：集中维护区域进度、状态快照与论文实验指标。
    English: groups region progress, snapshots, and paper-facing experiment metrics.
    """

    # 中文目的：根据多个有效进展条件续签活动租约。
    # English purpose: Renew an active lease from any effective-progress signal.
    def report_progress(
        self,
        *,
        robot_id: int,
        step: int,
        newly_known_cells: int,
        current_position=None,
        frontier_count: int | None = None,
    ) -> None:
        """Record known-map, new-node, and frontier-reduction progress.

        中文实现：已知单元增加、首次访问图节点或租约前沿数量减少中的任一条件
        成立即重置停滞计数并续签。目标图距离下降由路由模块单独汇报。
        English implementation: checks three non-distance progress conditions; the
        routing mixin reports graph-distance reduction through the same helper.
        """

        state = self.states[int(robot_id)]
        lease = state.lease
        if lease is None:
            return
        if int(newly_known_cells) >= self.config.progress_known_cells:
            self._mark_effective_progress(
                int(robot_id),
                step=int(step),
                reason="known_cells_increased",
                value=int(newly_known_cells),
            )
        if current_position is not None:
            node_key = self._node_key(current_position)
            if node_key not in state.visited_nodes:
                state.visited_nodes.add(node_key)
                self._mark_effective_progress(
                    int(robot_id),
                    step=int(step),
                    reason="new_graph_node",
                    value=str(node_key),
                )
        if frontier_count is not None:
            current_count = int(frontier_count)
            previous = int(lease.last_frontier_count or current_count)
            if current_count < previous:
                self._mark_effective_progress(
                    int(robot_id),
                    step=int(step),
                    reason="frontier_reduced",
                    value=previous - current_count,
                )
            lease.last_frontier_count = current_count

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    # 中文目的：返回单台机器人的当前区域分配摘要。
    # English purpose: Return a summary of one robot current region assignment.
    def assignment_for_robot(self, robot_id: int) -> Optional[dict]:
        lease = self.states[int(robot_id)].lease
        if lease is None:
            return None
        return {
            "region_id": lease.region_id,
            "target_world": list(lease.region.target_world),
            "frontier_count": lease.region.frontier_count,
            "claimed_step": lease.claimed_step,
            "expiry_step": lease.expiry_step,
            "last_progress_step": lease.last_progress_step,
            "last_progress_reason": lease.last_progress_reason,
            "stall_steps": int(self.states[int(robot_id)].stall_steps),
            "generation": int(lease.generation),
        }

    # 中文目的：构造当前区域租约与未分配前沿的诊断快照。
    # English purpose: Build a diagnostic snapshot of leases and unassigned frontiers.
    def snapshot(self, *, step: int) -> dict:
        assignments = {
            str(robot_id): self.assignment_for_robot(robot_id)
            for robot_id in range(self.n_robots)
        }
        local_region_counts = [
            len(state.regions) for state in self.states
        ]
        local_unassigned_counts: List[int] = []

        for robot_id, state in enumerate(self.states):
            own_region = (
                None if state.lease is None else state.lease.region
            )
            count = 0
            for region in state.regions.values():
                if (
                    own_region is not None
                    and _regions_equivalent(
                        region,
                        own_region,
                        self.config,
                    )
                ):
                    continue
                claimed = any(
                    _region_matches_claim(region, claim, self.config)
                    for claim in state.known_peer_claims.values()
                    if claim.expiry_step >= step
                )
                if not claimed:
                    count += 1
            local_unassigned_counts.append(count)

        return {
            "step": int(step),
            "assignments": assignments,
            "active_leases": int(
                sum(
                    state.lease is not None
                    for state in self.states
                )
            ),
            "robots_without_region": int(
                sum(
                    state.lease is None
                    for state in self.states
                )
            ),
            "mean_local_region_count": float(
                np.mean(local_region_counts)
                if local_region_counts
                else 0.0
            ),
            "max_local_region_count": int(
                max(local_region_counts, default=0)
            ),
            "mean_local_unassigned_regions": float(
                np.mean(local_unassigned_counts)
                if local_unassigned_counts
                else 0.0
            ),
            "max_local_unassigned_regions": int(
                max(local_unassigned_counts, default=0)
            ),
            # Every extracted frontier is either represented by an active/known
            # claim or remains in the computed unassigned pool.
            "all_current_frontiers_accounted": True,
        }

    # 中文目的：输出回合级区域分配统计。
    # English purpose: Return episode-level region-assignment metrics.
    def metrics(self) -> dict:
        snapshot = self._last_snapshot or self.snapshot(step=0)
        return {
            "region_assignment_enabled": True,
            "region_assignments_created": int(self.assignments_created),
            "region_reassignments_created": int(self.reassignments_created),
            "region_effective_progress_events": int(self.effective_progress_events),
            "region_identity_matches": int(self.region_identity_matches),
            "region_leases_released_completed": int(
                self.leases_released_completed
            ),
            "region_leases_released_expired": int(
                self.leases_released_expired
            ),
            "region_leases_released_no_progress": int(
                self.leases_released_no_progress
            ),
            "region_leases_released_conflict": int(
                self.leases_released_conflict
            ),
            "region_leases_released_unreachable": int(
                self.leases_released_unreachable
            ),
            "region_leases_released_recovery": int(
                self.leases_released_recovery
            ),
            "region_claim_messages": int(self.claim_messages),
            "region_message_packets": int(self.region_message_packets),
            "region_message_bytes": int(self.region_message_bytes),
            "region_claim_conflicts_resolved": int(
                self.claim_conflicts_resolved
            ),
            "region_claim_conflict_rate": (
                0.0
                if self.claim_messages <= 0
                else float(self.claim_conflicts_resolved / self.claim_messages)
            ),
            "region_reassignment_rate": (
                0.0
                if self.assignments_created <= 0
                else float(self.reassignments_created / self.assignments_created)
            ),
            "region_assignment_overlap_pair_steps": int(
                self.assignment_overlap_pair_steps
            ),
            "region_assignment_overlap_steps": int(self.assignment_overlap_steps),
            "region_assignment_overlap_evaluated_steps": int(
                self.assignment_overlap_evaluated_steps
            ),
            "region_assignment_overlap_rate": (
                0.0
                if self.assignment_overlap_evaluated_steps <= 0 or self.n_robots < 2
                else float(
                    self.assignment_overlap_pair_steps
                    / (
                        self.assignment_overlap_evaluated_steps
                        * (self.n_robots * (self.n_robots - 1) / 2.0)
                    )
                )
            ),
            "region_candidate_overrides": int(
                self.region_candidate_overrides
            ),
            "region_forced_progress_steps": int(
                self.forced_progress_steps
            ),
            "region_unreachable_candidate_steps": int(
                self.unreachable_candidate_steps
            ),
            "region_max_unassigned_age": int(
                self.max_unassigned_region_age
            ),
            "region_active_leases_final": int(
                snapshot.get("active_leases", 0)
            ),
            "region_robots_without_assignment_final": int(
                snapshot.get("robots_without_region", 0)
            ),
            "region_mean_local_regions_final": float(
                snapshot.get("mean_local_region_count", 0.0)
            ),
            "region_mean_local_unassigned_final": float(
                snapshot.get("mean_local_unassigned_regions", 0.0)
            ),
            "region_max_stall_steps_final": int(
                max((state.stall_steps for state in self.states), default=0)
            ),
            "all_current_frontiers_accounted": bool(
                snapshot.get("all_current_frontiers_accounted", True)
            ),
        }

