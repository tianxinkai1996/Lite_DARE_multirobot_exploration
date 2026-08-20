"""Map communication and diagnostics for the MergingMap-only baseline.

MergingMap-only 基线的地图通信与诊断模块。
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from classes.agent.agent import Agent
from mergingmap.baselines.merged_map_only.multi_test_parameter import MAP_DEBUG


class BaselineMapSyncMixin:
    """Handle contact map exchange and graph agreement metrics.

    中文：拆分接触地图交换和图一致性统计。
    English: separates contact map exchange and graph-agreement metrics.
    """

    # 中文目的：选择完整或增量地图数据包。
    # English purpose: Select full or delta map packets.
    def _map_packet_mode(self) -> str:
        # Preserve the existing driver's mode names:
        # raw -> complete known map at every contact
        # compressed -> sparse unsent delta
        if self.communication_mode == "raw":
            return "full"
        return "delta"

    # 中文目的：同步本地地图并交换接触数据包。
    # English purpose: Synchronise local maps and exchange contact packets.
    def _synchronise_maps(
        self,
        *,
        step: int,
        contact_pairs: Sequence[tuple[int, int]],
    ) -> set[int]:
        changed_robots: set[int] = set()

        # First incorporate each robot's newest direct sensor observation.
        for robot_id in range(self.n_agents):
            changed = self.map_merger.sync_local_map(
                robot_id,
                self.env.get_local_belief(robot_id),
            )
            if changed:
                # Local sensing changes do not reset the temporal history by
                # default; only peer-knowledge jumps are placed in this set.
                pass

        if self.communication_mode == "none":
            return changed_robots

        packet_mode = self._map_packet_mode()
        for robot_i, robot_j in contact_pairs:
            edge = tuple(sorted((int(robot_i), int(robot_j))))
            self.contact_history_edges.add(edge)

            changed_i, changed_j, packet_i, packet_j = (
                self.map_merger.exchange_pair(
                    robot_i,
                    robot_j,
                    step=step,
                    mode=packet_mode,
                )
            )
            if changed_i:
                changed_robots.add(int(robot_i))
            if changed_j:
                changed_robots.add(int(robot_j))

            if MAP_DEBUG:
                print(
                    f"[MAP-MERGE] step={step} pair={edge} "
                    f"i_to_j={packet_i['cell_count']} "
                    f"j_to_i={packet_j['cell_count']} "
                    f"changed_i={changed_i} changed_j={changed_j} "
                    f"agreement={self.map_merger.pairwise_agreement(robot_i, robot_j):.4f}",
                    flush=True,
                )

        return changed_robots

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    # 中文目的：提取量化图节点集合。
    # English purpose: Extract a quantised graph-node set.
    def _node_set(robot: Agent) -> set[tuple[float, float]]:
        result: set[tuple[float, float]] = set()
        for record in robot.node_manager.nodes_dict.__iter__():
            coords = np.asarray(record.data.coords, dtype=float)
            result.add(
                (
                    round(float(coords[0]), 3),
                    round(float(coords[1]), 3),
                )
            )
        return result

    # 中文目的：计算基线图一致性指标。
    # English purpose: Compute baseline graph-agreement metrics.
    def _graph_agreement_metrics(self) -> dict[str, float | int]:
        node_sets = [
            self._node_set(robot)
            for robot in self.robot_list
        ]
        pairwise_jaccard = []
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                union = node_sets[i] | node_sets[j]
                value = (
                    1.0
                    if not union
                    else len(node_sets[i] & node_sets[j]) / len(union)
                )
                pairwise_jaccard.append(float(value))

        counts = [len(nodes) for nodes in node_sets]
        return {
            "mean_graph_nodes": float(np.mean(counts)),
            "min_graph_nodes": int(min(counts, default=0)),
            "max_graph_nodes": int(max(counts, default=0)),
            "mean_pairwise_graph_jaccard": (
                1.0
                if not pairwise_jaccard
                else float(np.mean(pairwise_jaccard))
            ),
            "min_pairwise_graph_jaccard": (
                1.0
                if not pairwise_jaccard
                else float(np.min(pairwise_jaccard))
            ),
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------




