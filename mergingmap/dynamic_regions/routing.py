"""Graph-routing mixin for dynamic frontier assignments.

"""
from __future__ import annotations

from collections import defaultdict
import heapq
import math
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .extraction import _world_distance

class RegionRoutingMixin:
    """Provide graph-distance and candidate-ordering methods.

    English: separates graph routing and candidate ranking from lease lifecycle code.
    """

    @staticmethod
    # English purpose: Quantise graph-node coordinates for dictionary indexing.
    def _node_key(position):
        value = np.asarray(position, dtype=float).reshape(-1)
        return round(float(value[0]), 4), round(float(value[1]), 4)

    # English purpose: Compute graph shortest-path distances backward from a region target.
    def _graph_distance_map(self, robot_id, robot, target_world):
        state = self.states[robot_id]
        records = list(robot.node_manager.nodes_dict.__iter__())
        if not records:
            return {}

        coordinates: Dict[Tuple[float, float], np.ndarray] = {}
        adjacency: Dict[Tuple[float, float], Dict[Tuple[float, float], float]] = (
            defaultdict(dict)
        )

        for record in records:
            data = record.data
            coords = np.asarray(data.coords, dtype=float)
            key = self._node_key(coords)
            coordinates[key] = coords

        for record in records:
            data = record.data
            source = self._node_key(data.coords)
            for neighbour in getattr(data, "neighbor_list", []):
                target = self._node_key(neighbour)
                if target not in coordinates:
                    coordinates[target] = np.asarray(
                        neighbour,
                        dtype=float,
                    )
                weight = _world_distance(
                    coordinates[source],
                    coordinates[target],
                )
                if weight <= 1e-8:
                    continue
                adjacency[source][target] = min(
                    adjacency[source].get(target, math.inf),
                    weight,
                )
                adjacency[target][source] = min(
                    adjacency[target].get(source, math.inf),
                    weight,
                )

        target_key = min(
            coordinates,
            key=lambda key: _world_distance(
                coordinates[key],
                target_world,
            ),
        )
        edge_count = sum(len(neighbours) for neighbours in adjacency.values())
        lease = state.lease
        cache_key = (
            None if lease is None else lease.region_id,
            len(coordinates),
            edge_count,
            target_key,
        )
        if state.graph_cache_key == cache_key:
            return state.graph_distance_cache

        distances: Dict[Tuple[float, float], float] = {
            key: math.inf for key in coordinates
        }
        distances[target_key] = 0.0
        queue: List[Tuple[float, Tuple[float, float]]] = [(0.0, target_key)]

        while queue:
            current_distance, current = heapq.heappop(queue)
            if current_distance > distances[current] + 1e-12:
                continue
            for neighbour, weight in adjacency.get(current, {}).items():
                candidate = current_distance + weight
                if candidate + 1e-12 < distances.get(neighbour, math.inf):
                    distances[neighbour] = candidate
                    heapq.heappush(queue, (candidate, neighbour))

        state.graph_cache_key = cache_key
        state.graph_distance_cache = distances
        return distances

    # English purpose: Estimate nearest graph distance from a candidate to the region target.
    def _nearest_graph_distance(self, position, distances):
        if not distances:
            return math.inf
        key = self._node_key(position)
        if key in distances:
            return float(distances[key])

        finite_keys = [
            candidate
            for candidate, distance in distances.items()
            if math.isfinite(distance)
        ]
        if not finite_keys:
            return math.inf
        nearest = min(
            finite_keys,
            key=lambda candidate: _world_distance(candidate, position),
        )
        return float(distances[nearest])

    # English purpose: Reorder legal DARE neighbours using the region lease without inventing actions.
    def order_candidates(self, robot_id, robot, current_position, ordered_candidates, step):
        """Reorder DARE neighbours without changing the frozen policy.

        Normal transit:
            prefer candidates that reduce graph distance to the assigned region,
            then retain DARE rank.

        Inside/near the region:
            retain DARE rank unless a candidate moves clearly away.

        Coverage supervisor:
            after several no-progress steps, graph distance becomes the primary
            key until progress resumes.

        No candidate is permanently deleted; the original order is returned
        when no valid graph route to the temporary target exists.
        """
        candidates = [
            np.asarray(candidate, dtype=np.float32).copy()
            for candidate in ordered_candidates
        ]
        if not candidates:
            return [np.asarray(current_position, dtype=np.float32).copy()]

        state = self.states[int(robot_id)]
        lease = state.lease
        if lease is None:
            return candidates

        distances = self._graph_distance_map(
            int(robot_id),
            robot,
            lease.region.target_world,
        )
        current_distance = self._nearest_graph_distance(
            current_position,
            distances,
        )

        if not math.isfinite(current_distance):
            state.unreachable_steps += 1
            self.unreachable_candidate_steps += 1
            if (
                state.unreachable_steps
                >= self.config.no_progress_release_steps
            ):
                self._release(
                    int(robot_id),
                    step=int(step),
                    reason="unreachable",
                )
            return candidates

        state.unreachable_steps = 0
        if current_distance + 1e-6 < lease.last_target_distance:
            self._mark_effective_progress(
                int(robot_id),
                step=int(step),
                reason="graph_distance_decreased",
                value=(
                    None
                    if not math.isfinite(lease.last_target_distance)
                    else float(lease.last_target_distance - current_distance)
                ),
            )
        lease.last_target_distance = current_distance

        no_progress_steps = int(step - lease.last_progress_step)
        state.stall_steps = max(0, no_progress_steps)
        force_progress = (
            no_progress_steps >= self.config.force_progress_after_steps
        )
        arrived = current_distance <= self.config.arrival_distance

        scored = []
        for dare_rank, candidate in enumerate(candidates):
            candidate_distance = self._nearest_graph_distance(
                candidate,
                distances,
            )
            is_wait = bool(
                np.allclose(
                    candidate,
                    current_position,
                    atol=1e-4,
                )
            )

            if force_progress:
                key = (
                    1 if is_wait else 0,
                    candidate_distance,
                    dare_rank,
                )
            elif not arrived:
                makes_progress = (
                    candidate_distance + 1e-6 < current_distance
                )
                key = (
                    0 if makes_progress else 1,
                    candidate_distance,
                    dare_rank,
                    1 if is_wait else 0,
                )
            else:
                moves_too_far_away = (
                    candidate_distance
                    > current_distance + self.config.distance_slack
                )
                key = (
                    1 if moves_too_far_away else 0,
                    dare_rank,
                    candidate_distance,
                    1 if is_wait else 0,
                )

            scored.append((key, dare_rank, candidate))

        scored.sort(key=lambda item: (item[0], item[1]))
        reordered = [item[2] for item in scored]

        if force_progress:
            self.forced_progress_steps += 1
        if not np.allclose(reordered[0], candidates[0], atol=1e-4):
            self.region_candidate_overrides += 1

        return reordered

