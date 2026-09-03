"""Episode-level evaluation logging for multi-robot exploration.

The recorder is deliberately independent from the planner.  It observes the
positions and coverage produced by an episode, computes process metrics, and
writes per-episode CSV/JSON files.  Keeping the instrumentation outside DARE's
policy path avoids changing model inputs or learned actions.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


class EpisodeMetricsRecorder:
    """Collect step, trajectory, conflict, deadlock, and coverage metrics.

    Parameters are intentionally explicit so that metric definitions can be
    reported verbatim in a paper and reproduced across all ablations.
    """

    def __init__(self, output_root, episode, method, seed, team_size, communication_mode, initial_positions, initial_coverage, initial_known_free_cells, node_resolution, safe_distance, deadlock_wait_threshold=3, coverage_thresholds=(0.90, 0.95, 0.99)):
        if team_size <= 0:
            raise ValueError("team_size must be positive")
        if node_resolution <= 0:
            raise ValueError("node_resolution must be positive")
        if safe_distance <= 0:
            raise ValueError("safe_distance must be positive")
        if deadlock_wait_threshold < 1:
            raise ValueError("deadlock_wait_threshold must be at least 1")

        initial = np.asarray(initial_positions, dtype=np.float32)
        if initial.shape != (int(team_size), 2):
            raise ValueError(
                f"initial_positions must have shape {(int(team_size), 2)}, "
                f"received {initial.shape}"
            )

        thresholds = tuple(sorted({float(value) for value in coverage_thresholds}))
        if any(value <= 0.0 or value > 1.0 for value in thresholds):
            raise ValueError("coverage thresholds must lie in (0, 1]")

        self.episode = int(episode)
        self.method = str(method)
        self.seed = int(seed)
        self.team_size = int(team_size)
        self.communication_mode = str(communication_mode)
        self.node_resolution = float(node_resolution)
        self.safe_distance = float(safe_distance)
        self.deadlock_wait_threshold = int(deadlock_wait_threshold)
        self.coverage_thresholds = thresholds

        run_name = (
            f"episode_{self.episode:04d}_robots_{self.team_size}_"
            f"mode_{self.communication_mode}_seed_{self.seed}"
        )
        self.output_dir = Path(output_root).expanduser().resolve() / "evaluation_metrics" / run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.step_metrics_path = self.output_dir / "step_metrics.csv"
        self.trajectory_path = self.output_dir / "robot_trajectories.csv"
        self.event_log_path = self.output_dir / "event_log.jsonl"
        self.episode_metrics_path = self.output_dir / "episode_metrics.json"

        self.started_at = time.perf_counter()
        self.initial_positions = initial.copy()
        self.previous_positions = initial.copy()
        self.initial_coverage = float(initial_coverage)
        self.initial_known_free_cells = int(initial_known_free_cells)
        self.previous_coverage = self.initial_coverage
        self.previous_known_free_cells = self.initial_known_free_cells

        self.coverage_history: list[float] = [float(initial_coverage)]
        self.coverage_step_history: list[int] = [0]
        self.threshold_steps: dict[float, int | None] = {
            threshold: (0 if initial_coverage >= threshold else None)
            for threshold in thresholds
        }

        self.step_rows: list[dict] = []
        self.trajectory_rows: list[dict] = []
        self.event_rows: list[dict] = []

        self.path_lengths = np.zeros(self.team_size, dtype=np.float64)
        self.wait_steps = np.zeros(self.team_size, dtype=np.int64)
        self.consecutive_wait_steps = np.zeros(self.team_size, dtype=np.int64)
        self.max_consecutive_wait_steps = np.zeros(self.team_size, dtype=np.int64)
        self.deadlocked = np.zeros(self.team_size, dtype=bool)
        self.deadlock_count = 0
        self.deadlock_duration_robot_steps = 0
        self.deadlock_recovery_count = 0
        self.deadlock_started_at_step = np.full(self.team_size, -1, dtype=np.int64)
        self.deadlock_event_durations: list[int] = []

        self.preferred_vertex_conflicts = 0
        self.preferred_swap_conflicts = 0
        self.preferred_proximity_conflicts = 0
        self.actual_collision_pairs = 0
        self.preferred_conflict_steps = 0
        self.actual_collision_steps = 0
        self.conflict_avoided_steps = 0
        self.minimum_preferred_pairwise_distance = math.inf
        self.minimum_proposed_pairwise_distance = math.inf
        self.minimum_actual_pairwise_distance = math.inf
        self.dynamic_blocks = 0
        self.static_blocks = 0
        self.total_new_free_cells = 0
        self.total_contact_pairs = 0
        self.total_map_packets = 0
        self.total_map_bytes = 0
        self.total_runtime_ms = 0.0
        # Generic monotonic communication counters allow MergingMap and the
        # original DARE message path to log payload, coding, delivery, and loss
        # diagnostics with the same recorder interface.
        self.communication_cumulative: dict[str, float] = {}

        # Per-node visit counts provide both revisit and cross-robot overlap.
        self.node_visit_counts: dict[tuple[int, int], int] = {}
        self.node_visiting_robots: dict[tuple[int, int], set[int]] = {}
        for robot_id, position in enumerate(initial):
            self._register_visit(robot_id, position)
            self.trajectory_rows.append(
                self._trajectory_row(
                    step=0,
                    robot_id=robot_id,
                    previous=position,
                    preferred=position,
                    proposed=position,
                    actual=position,
                    path_increment=0.0,
                    wait_flag=True,
                    dynamic_blocked=False,
                    static_blocked=False,
                    consecutive_wait_steps=0,
                    deadlock_flag=False,
                    recovered_flag=False,
                    revisit_flag=False,
                    cross_robot_overlap_flag=False,
                )
            )

    @staticmethod
    def known_free_cell_count(env, free_value):
        """Return team-union observed free cells, restricted to true free space."""

        free_truth = np.asarray(env.ground_truth) == int(free_value)
        observed_free = np.zeros_like(free_truth, dtype=bool)
        for local_map in env.robot_beliefs:
            observed_free |= np.asarray(local_map) == int(free_value)
        return int(np.sum(observed_free & free_truth))

    @staticmethod
    def map_quality_metrics(beliefs, ground_truth, free_value, occupied_value, unknown_value):
        """Compute team-union and per-robot occupancy-map quality metrics.

        Forms a conservative team union (any occupied report wins, then free,
        otherwise unknown) and evaluates free and occupied classes against ground
        truth, reporting known accuracy, precision, recall, F1 and IoU, and then
        averages the same metrics across individual robot beliefs.
        """

        truth = np.asarray(ground_truth)
        maps = [np.asarray(belief) for belief in beliefs]
        if not maps:
            return {}
        if any(belief.shape != truth.shape for belief in maps):
            raise ValueError("belief maps and ground_truth must share one shape")

        def evaluate(prediction):
            known = prediction != int(unknown_value)
            truth_known = (truth == int(free_value)) | (truth == int(occupied_value))
            valid_known = known & truth_known
            known_count = int(np.count_nonzero(known))
            total = int(prediction.size)
            correct = int(np.count_nonzero((prediction == truth) & valid_known))
            result: dict[str, float] = {
                "known_ratio": 0.0 if total == 0 else float(known_count / total),
                "unknown_ratio": 0.0 if total == 0 else float(1.0 - known_count / total),
                "known_cell_accuracy": (
                    0.0
                    if int(np.count_nonzero(valid_known)) == 0
                    else float(correct / int(np.count_nonzero(valid_known)))
                ),
            }
            for label, value in (("free", free_value), ("occupied", occupied_value)):
                predicted_positive = prediction == int(value)
                true_positive_mask = truth == int(value)
                tp = int(np.count_nonzero(predicted_positive & true_positive_mask))
                fp = int(np.count_nonzero(predicted_positive & (~true_positive_mask)))
                fn = int(np.count_nonzero((~predicted_positive) & true_positive_mask))
                precision = 0.0 if tp + fp == 0 else float(tp / (tp + fp))
                recall = 0.0 if tp + fn == 0 else float(tp / (tp + fn))
                f1 = (
                    0.0
                    if precision + recall <= 1e-12
                    else float(2.0 * precision * recall / (precision + recall))
                )
                iou = 0.0 if tp + fp + fn == 0 else float(tp / (tp + fp + fn))
                result[f"{label}_precision"] = precision
                result[f"{label}_recall"] = recall
                result[f"{label}_f1"] = f1
                result[f"{label}_iou"] = iou
            return result

        stacked = np.stack(maps, axis=0)
        team_map = np.full(truth.shape, int(unknown_value), dtype=stacked.dtype)
        any_free = np.any(stacked == int(free_value), axis=0)
        any_occupied = np.any(stacked == int(occupied_value), axis=0)
        team_map[any_free] = int(free_value)
        team_map[any_occupied] = int(occupied_value)

        team = evaluate(team_map)
        per_robot = [evaluate(belief) for belief in maps]
        output: dict[str, float | int] = {"map_quality_robot_count": len(maps)}
        for key, value in team.items():
            output[f"team_map_{key}"] = value
        for key in per_robot[0]:
            values = [metrics[key] for metrics in per_robot]
            output[f"mean_robot_map_{key}"] = float(np.mean(values))
            output[f"min_robot_map_{key}"] = float(np.min(values))
        return output

    @staticmethod
    def map_structure_metrics(ground_truth, free_value, occupied_value):
        """Return static map descriptors used for difficulty stratification.

        Derives obstacle ratio and a topology-only narrow free-cell proxy from
        four-neighbour connectivity (free cells with degree at most two are used
        as a corridor proxy). Difficulty tiers are assigned later across the
        dataset, so no outcome metric leaks into the stratification.
        """

        truth = np.asarray(ground_truth)
        free = truth == int(free_value)
        occupied = truth == int(occupied_value)
        traversable = int(np.count_nonzero(free))
        occupied_count = int(np.count_nonzero(occupied))
        known_total = traversable + occupied_count
        if traversable == 0:
            return {
                "map_traversable_cells": 0,
                "map_occupied_cells": occupied_count,
                "map_obstacle_ratio": 0.0 if known_total == 0 else occupied_count / known_total,
                "map_narrow_free_cells": 0,
                "map_narrow_free_ratio": 0.0,
                "map_branch_free_ratio": 0.0,
                "map_structure_difficulty_score": 0.0,
            }

        degree = np.zeros(free.shape, dtype=np.uint8)
        degree[1:, :] += free[:-1, :]
        degree[:-1, :] += free[1:, :]
        degree[:, 1:] += free[:, :-1]
        degree[:, :-1] += free[:, 1:]
        narrow = free & (degree <= 2)
        branches = free & (degree >= 3)
        narrow_ratio = float(np.count_nonzero(narrow) / traversable)
        branch_ratio = float(np.count_nonzero(branches) / traversable)
        obstacle_ratio = 0.0 if known_total == 0 else float(occupied_count / known_total)
        # A transparent, bounded structural proxy; dataset tertiles are assigned later.
        score = float(0.5 * obstacle_ratio + 0.5 * narrow_ratio)
        return {
            "map_traversable_cells": traversable,
            "map_occupied_cells": occupied_count,
            "map_obstacle_ratio": obstacle_ratio,
            "map_narrow_free_cells": int(np.count_nonzero(narrow)),
            "map_narrow_free_ratio": narrow_ratio,
            "map_branch_free_ratio": branch_ratio,
            "map_structure_difficulty_score": score,
        }

    @staticmethod
    def _minimum_pairwise_distance(positions):
        """Return the minimum pairwise distance, or infinity for one robot."""

        if len(positions) < 2:
            return math.inf
        return min(
            float(np.linalg.norm(positions[i] - positions[j]))
            for i in range(len(positions))
            for j in range(i + 1, len(positions))
        )

    def reset_wall_clock(self):
        """Reset the episode wall-clock origin after one-time setup/profiling.

        Restarts only the wall timer, excluding environment initialisation and
        one-time FLOPs profiling so that episode wall times can be compared
        fairly, while preserving all metric definitions and initial state.
        """

        self.started_at = time.perf_counter()

    def _node_key(self, position):
        value = np.asarray(position, dtype=np.float64)
        quantised = np.rint(value / self.node_resolution).astype(np.int64)
        return int(quantised[0]), int(quantised[1])

    def _register_visit(self, robot_id, position):
        key = self._node_key(position)
        previous_count = self.node_visit_counts.get(key, 0)
        previous_robots = self.node_visiting_robots.setdefault(key, set())
        revisit = previous_count > 0
        cross_robot_overlap = bool(previous_robots and int(robot_id) not in previous_robots)
        self.node_visit_counts[key] = previous_count + 1
        previous_robots.add(int(robot_id))
        return revisit, cross_robot_overlap

    def _base_fields(self):
        return {
            "episode": self.episode,
            "method": self.method,
            "seed": self.seed,
            "team_size": self.team_size,
            "communication_mode": self.communication_mode,
        }

    def _trajectory_row(self, step, robot_id, previous, preferred, proposed, actual, path_increment, wait_flag, dynamic_blocked, static_blocked, consecutive_wait_steps, deadlock_flag, recovered_flag, revisit_flag, cross_robot_overlap_flag):
        previous = np.asarray(previous, dtype=float)
        preferred = np.asarray(preferred, dtype=float)
        proposed = np.asarray(proposed, dtype=float)
        actual = np.asarray(actual, dtype=float)
        return {
            **self._base_fields(),
            "step": int(step),
            "robot_id": int(robot_id),
            "previous_x": float(previous[0]),
            "previous_y": float(previous[1]),
            "preferred_x": float(preferred[0]),
            "preferred_y": float(preferred[1]),
            "proposed_x": float(proposed[0]),
            "proposed_y": float(proposed[1]),
            "actual_x": float(actual[0]),
            "actual_y": float(actual[1]),
            "path_increment": float(path_increment),
            "cumulative_robot_path": float(self.path_lengths[int(robot_id)]),
            "wait_flag": bool(wait_flag),
            "dynamic_blocked": bool(dynamic_blocked),
            "static_blocked": bool(static_blocked),
            "consecutive_wait_steps": int(consecutive_wait_steps),
            "deadlock_flag": bool(deadlock_flag),
            "recovered_flag": bool(recovered_flag),
            "revisit_flag": bool(revisit_flag),
            "cross_robot_overlap_flag": bool(cross_robot_overlap_flag),
        }

    def _preferred_conflicts(self, previous, preferred):
        vertex = 0
        swap = 0
        proximity = 0
        events: list[dict] = []
        for robot_i in range(self.team_size):
            for robot_j in range(robot_i + 1, self.team_size):
                next_distance = float(np.linalg.norm(preferred[robot_i] - preferred[robot_j]))
                same_destination = bool(
                    np.allclose(preferred[robot_i], preferred[robot_j], atol=1e-5)
                )
                edge_swap = bool(
                    np.allclose(previous[robot_i], preferred[robot_j], atol=1e-5)
                    and np.allclose(previous[robot_j], preferred[robot_i], atol=1e-5)
                    and not np.allclose(previous[robot_i], preferred[robot_i])
                    and not np.allclose(previous[robot_j], preferred[robot_j])
                )
                if same_destination:
                    vertex += 1
                    events.append(
                        {
                            "event_type": "preferred_vertex_conflict",
                            "robot_i": robot_i,
                            "robot_j": robot_j,
                            "distance": next_distance,
                        }
                    )
                if edge_swap:
                    swap += 1
                    events.append(
                        {
                            "event_type": "preferred_swap_conflict",
                            "robot_i": robot_i,
                            "robot_j": robot_j,
                        }
                    )
                if next_distance < self.safe_distance:
                    proximity += 1
        return vertex, swap, proximity, events

    def _actual_collisions(self, previous, actual):
        collisions = 0
        events: list[dict] = []
        for robot_i in range(self.team_size):
            for robot_j in range(robot_i + 1, self.team_size):
                distance = float(np.linalg.norm(actual[robot_i] - actual[robot_j]))
                if distance >= self.safe_distance:
                    continue

                # Do not label an authorised pre-existing shared-depot wait as a
                # newly caused collision.  A moving pair or a new overlap remains
                # an actual collision event.
                already_overlapping = (
                    np.linalg.norm(previous[robot_i] - previous[robot_j])
                    < self.safe_distance
                )
                both_wait = (
                    np.allclose(previous[robot_i], actual[robot_i])
                    and np.allclose(previous[robot_j], actual[robot_j])
                )
                if already_overlapping and both_wait:
                    continue

                collisions += 1
                events.append(
                    {
                        "event_type": "actual_collision_pair",
                        "robot_i": robot_i,
                        "robot_j": robot_j,
                        "distance": distance,
                    }
                )
        return collisions, events

    def _append_event(self, step, event):
        self.event_rows.append(
            {
                **self._base_fields(),
                "step": int(step),
                **dict(event),
            }
        )

    def record_step(self, step, preferred_positions, proposed_positions, actual_positions, coverage, known_free_cells, dynamic_blocked_robot_ids=(), static_blocked_robot_ids=(), contact_pairs=(), map_packets_cumulative=0, map_bytes_cumulative=0, communication_cumulative=None, resolver_info=None, step_runtime_ms=0.0, extra_step_metrics=None):
        """Record one executed synchronous team step.

        Combines core exploration fields with optional monotonic communication
        counters and detailed runtime/resource samples.
        """

        previous = self.previous_positions.copy()
        preferred = np.asarray(preferred_positions, dtype=np.float32)
        proposed = np.asarray(proposed_positions, dtype=np.float32)
        actual = np.asarray(actual_positions, dtype=np.float32)
        expected_shape = (self.team_size, 2)
        for name, value in (
            ("preferred_positions", preferred),
            ("proposed_positions", proposed),
            ("actual_positions", actual),
        ):
            if value.shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}, received {value.shape}")

        dynamic_blocked = {int(value) for value in dynamic_blocked_robot_ids}
        static_blocked = {int(value) for value in static_blocked_robot_ids}

        vertex, swap, proximity, preferred_events = self._preferred_conflicts(previous, preferred)
        actual_collision_count, collision_events = self._actual_collisions(previous, actual)
        preferred_min_distance = self._minimum_pairwise_distance(preferred)
        proposed_min_distance = self._minimum_pairwise_distance(proposed)
        actual_min_distance = self._minimum_pairwise_distance(actual)
        if math.isfinite(preferred_min_distance):
            self.minimum_preferred_pairwise_distance = min(
                self.minimum_preferred_pairwise_distance, preferred_min_distance
            )
        if math.isfinite(proposed_min_distance):
            self.minimum_proposed_pairwise_distance = min(
                self.minimum_proposed_pairwise_distance, proposed_min_distance
            )
        if math.isfinite(actual_min_distance):
            self.minimum_actual_pairwise_distance = min(
                self.minimum_actual_pairwise_distance, actual_min_distance
            )
        preferred_conflict_step = bool(vertex or swap or proximity)
        actual_collision_step = bool(actual_collision_count)
        conflict_avoided_step = bool(preferred_conflict_step and not actual_collision_step)
        self.preferred_conflict_steps += int(preferred_conflict_step)
        self.actual_collision_steps += int(actual_collision_step)
        self.conflict_avoided_steps += int(conflict_avoided_step)
        self.preferred_vertex_conflicts += vertex
        self.preferred_swap_conflicts += swap
        self.preferred_proximity_conflicts += proximity
        self.actual_collision_pairs += actual_collision_count
        self.dynamic_blocks += len(dynamic_blocked)
        self.static_blocks += len(static_blocked)
        self.total_contact_pairs += len(contact_pairs)

        packets_delta = max(0, int(map_packets_cumulative) - self.total_map_packets)
        bytes_delta = max(0, int(map_bytes_cumulative) - self.total_map_bytes)
        self.total_map_packets = max(self.total_map_packets, int(map_packets_cumulative))
        self.total_map_bytes = max(self.total_map_bytes, int(map_bytes_cumulative))
        self.total_runtime_ms += float(step_runtime_ms)

        communication_values = dict(communication_cumulative or {})
        communication_values.setdefault(
            "communication_packets", int(map_packets_cumulative)
        )
        communication_values.setdefault(
            "communication_payload_bytes", int(map_bytes_cumulative)
        )
        communication_deltas: dict[str, int | float] = {}
        for raw_key, raw_value in communication_values.items():
            if isinstance(raw_value, bool):
                continue
            try:
                current_value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(current_value):
                continue
            key = str(raw_key)
            previous_value = float(self.communication_cumulative.get(key, 0.0))
            delta_value = max(0.0, current_value - previous_value)
            self.communication_cumulative[key] = max(previous_value, current_value)
            if float(delta_value).is_integer():
                communication_deltas[f"{key}_delta"] = int(delta_value)
            else:
                communication_deltas[f"{key}_delta"] = float(delta_value)

        known_free_cells = int(known_free_cells)
        new_free_cells = max(0, known_free_cells - self.previous_known_free_cells)
        self.total_new_free_cells += new_free_cells
        coverage = float(coverage)
        coverage_delta = coverage - self.previous_coverage
        self.coverage_history.append(coverage)
        self.coverage_step_history.append(int(step))
        for threshold in self.coverage_thresholds:
            if self.threshold_steps[threshold] is None and coverage >= threshold:
                self.threshold_steps[threshold] = int(step)
                self._append_event(
                    step=step,
                    event={
                        "event_type": "coverage_threshold_reached",
                        "threshold": threshold,
                        "coverage": coverage,
                    },
                )

        for event in preferred_events:
            self._append_event(step=step, event=event)
        for event in collision_events:
            self._append_event(step=step, event=event)

        recovered_this_step = 0
        deadlocked_this_step = 0
        step_path_length = 0.0
        for robot_id in range(self.team_size):
            increment = float(np.linalg.norm(actual[robot_id] - previous[robot_id]))
            step_path_length += increment
            self.path_lengths[robot_id] += increment
            waits = bool(np.allclose(actual[robot_id], previous[robot_id]))
            was_deadlocked = bool(self.deadlocked[robot_id])

            if waits:
                self.wait_steps[robot_id] += 1
                self.consecutive_wait_steps[robot_id] += 1
            else:
                self.consecutive_wait_steps[robot_id] = 0

            self.max_consecutive_wait_steps[robot_id] = max(
                int(self.max_consecutive_wait_steps[robot_id]),
                int(self.consecutive_wait_steps[robot_id]),
            )

            entered_deadlock = bool(
                not was_deadlocked
                and self.consecutive_wait_steps[robot_id] >= self.deadlock_wait_threshold
            )
            recovered = bool(was_deadlocked and not waits)
            if entered_deadlock:
                self.deadlocked[robot_id] = True
                self.deadlock_started_at_step[robot_id] = int(step)
                self.deadlock_count += 1
                self._append_event(
                    step=step,
                    event={
                        "event_type": "deadlock_started",
                        "robot_id": robot_id,
                        "consecutive_wait_steps": int(self.consecutive_wait_steps[robot_id]),
                    },
                )
            elif recovered:
                self.deadlocked[robot_id] = False
                started = int(self.deadlock_started_at_step[robot_id])
                duration = max(1, int(step) - started) if started >= 0 else 1
                self.deadlock_event_durations.append(duration)
                self.deadlock_started_at_step[robot_id] = -1
                self.deadlock_recovery_count += 1
                recovered_this_step += 1
                self._append_event(
                    step=step,
                    event={
                        "event_type": "deadlock_recovered",
                        "robot_id": robot_id,
                        "deadlock_duration_steps": int(duration),
                    },
                )

            if self.deadlocked[robot_id]:
                deadlocked_this_step += 1
                self.deadlock_duration_robot_steps += 1

            revisit, cross_robot_overlap = self._register_visit(robot_id, actual[robot_id])
            self.trajectory_rows.append(
                self._trajectory_row(
                    step=step,
                    robot_id=robot_id,
                    previous=previous[robot_id],
                    preferred=preferred[robot_id],
                    proposed=proposed[robot_id],
                    actual=actual[robot_id],
                    path_increment=increment,
                    wait_flag=waits,
                    dynamic_blocked=robot_id in dynamic_blocked,
                    static_blocked=robot_id in static_blocked,
                    consecutive_wait_steps=int(self.consecutive_wait_steps[robot_id]),
                    deadlock_flag=bool(self.deadlocked[robot_id]),
                    recovered_flag=recovered,
                    revisit_flag=revisit,
                    cross_robot_overlap_flag=cross_robot_overlap,
                )
            )

        if dynamic_blocked:
            self._append_event(
                step=step,
                event={
                    "event_type": "dynamic_safety_block",
                    "robot_ids": sorted(dynamic_blocked),
                },
            )
        if static_blocked:
            self._append_event(
                step=step,
                event={
                    "event_type": "static_safety_block",
                    "robot_ids": sorted(static_blocked),
                },
            )

        resolver_backtracking_nodes = int(
            getattr(resolver_info, "backtracking_nodes", 0) if resolver_info is not None else 0
        )
        escape_robot_ids = tuple(
            getattr(resolver_info, "escape_robot_ids", ()) if resolver_info is not None else ()
        )
        used_serial_fallback = bool(
            getattr(resolver_info, "used_serial_fallback", False)
            if resolver_info is not None
            else False
        )

        step_row = {
                **self._base_fields(),
                "step": int(step),
                "coverage": coverage,
                "coverage_delta": float(coverage_delta),
                "known_free_cells": known_free_cells,
                "newly_explored_free_cells": new_free_cells,
                "step_path_length": float(step_path_length),
                "cumulative_team_path_length": float(np.sum(self.path_lengths)),
                "waiting_robots": int(np.sum(np.all(np.isclose(actual, previous), axis=1))),
                "deadlocked_robots": int(deadlocked_this_step),
                "deadlock_recoveries": int(recovered_this_step),
                "preferred_vertex_conflicts": int(vertex),
                "preferred_swap_conflicts": int(swap),
                "preferred_proximity_conflicts": int(proximity),
                "preferred_conflict_step": preferred_conflict_step,
                "actual_collision_pairs": int(actual_collision_count),
                "actual_collision_step": actual_collision_step,
                "conflict_avoided_step": conflict_avoided_step,
                "minimum_preferred_pairwise_distance": (
                    None if not math.isfinite(preferred_min_distance) else float(preferred_min_distance)
                ),
                "minimum_proposed_pairwise_distance": (
                    None if not math.isfinite(proposed_min_distance) else float(proposed_min_distance)
                ),
                "minimum_actual_pairwise_distance": (
                    None if not math.isfinite(actual_min_distance) else float(actual_min_distance)
                ),
                "dynamic_safety_blocks": int(len(dynamic_blocked)),
                "static_safety_blocks": int(len(static_blocked)),
                "contact_pairs": int(len(contact_pairs)),
                "communication_packets_delta": int(packets_delta),
                "communication_bytes_delta": int(bytes_delta),
                "map_packets_delta": int(packets_delta),
                "map_bytes_delta": int(bytes_delta),
                "resolver_backtracking_nodes": resolver_backtracking_nodes,
                "escape_robot_ids": json.dumps([int(value) for value in escape_robot_ids]),
                "used_serial_fallback": used_serial_fallback,
                "step_runtime_ms": float(step_runtime_ms),
                **communication_deltas,
            }
        if extra_step_metrics:
            for key, value in extra_step_metrics.items():
                if key not in step_row:
                    step_row[str(key)] = value
        self.step_rows.append(step_row)

        self.previous_positions = actual.copy()
        self.previous_coverage = coverage
        self.previous_known_free_cells = known_free_cells

    @staticmethod
    def _write_csv(path, rows):
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    def _coverage_auc(self):
        if not self.coverage_history:
            return 0.0
        if len(self.coverage_history) == 1:
            return float(self.coverage_history[0])
        # One equally spaced sample at t=0 and after every executed step.
        trapezoid = getattr(np, "trapezoid", np.trapz)
        return float(trapezoid(np.asarray(self.coverage_history, dtype=float))) / float(
            len(self.coverage_history) - 1
        )

    def finalise(self, success, extra_metrics=None):
        """Write files and return a flat episode-summary dictionary."""

        elapsed_ms = (time.perf_counter() - self.started_at) * 1000.0
        total_visits = int(sum(self.node_visit_counts.values()))
        unique_nodes = int(len(self.node_visit_counts))
        revisit_visits = int(sum(max(0, count - 1) for count in self.node_visit_counts.values()))
        overlap_nodes = int(
            sum(1 for robots in self.node_visiting_robots.values() if len(robots) >= 2)
        )
        cross_robot_duplicate_visits = int(
            sum(max(0, len(robots) - 1) for robots in self.node_visiting_robots.values())
        )

        mean_path = float(np.mean(self.path_lengths)) if self.team_size else 0.0
        path_std = float(np.std(self.path_lengths)) if self.team_size else 0.0
        path_balance_cv = 0.0 if mean_path <= 1e-12 else path_std / mean_path
        recovery_rate = (
            0.0
            if self.deadlock_count == 0
            else float(self.deadlock_recovery_count / self.deadlock_count)
        )
        steps = max(0, len(self.coverage_history) - 1)
        unresolved_deadlock_durations = [
            max(1, steps - int(started) + 1)
            for started, active in zip(self.deadlock_started_at_step, self.deadlocked)
            if bool(active) and int(started) >= 0
        ]
        all_deadlock_durations = [
            *self.deadlock_event_durations,
            *unresolved_deadlock_durations,
        ]
        preferred_conflict_total = (
            self.preferred_vertex_conflicts
            + self.preferred_swap_conflicts
            + self.preferred_proximity_conflicts
        )

        summary = {
            **self._base_fields(),
            "success": bool(success),
            "initial_coverage": float(self.initial_coverage),
            "initial_known_free_cells": int(self.initial_known_free_cells),
            "steps_recorded": int(steps),
            "final_coverage": float(self.coverage_history[-1]),
            "coverage_auc": self._coverage_auc(),
            "team_travel_distance_recorded": float(np.sum(self.path_lengths)),
            "max_robot_travel_distance_recorded": float(np.max(self.path_lengths)),
            "mean_robot_travel_distance_recorded": mean_path,
            "path_balance_cv": float(path_balance_cv),
            "waiting_robot_steps": int(np.sum(self.wait_steps)),
            "max_consecutive_wait_steps_recorded": int(
                np.max(self.max_consecutive_wait_steps)
            ),
            "deadlock_count": int(self.deadlock_count),
            "deadlock_episode": bool(self.deadlock_count > 0),
            "deadlock_duration_robot_steps": int(self.deadlock_duration_robot_steps),
            "deadlock_recovery_count": int(self.deadlock_recovery_count),
            "deadlock_recovery_rate": float(recovery_rate),
            "deadlock_unresolved_count": int(len(unresolved_deadlock_durations)),
            "deadlock_event_duration_mean_steps": (
                0.0 if not all_deadlock_durations else float(np.mean(all_deadlock_durations))
            ),
            "deadlock_event_duration_max_steps": int(max(all_deadlock_durations, default=0)),
            "stagnation_count": int(self.deadlock_count),
            "single_robot_stagnation_count": (
                int(self.deadlock_count) if self.team_size == 1 else 0
            ),
            "multi_robot_deadlock_count": (
                int(self.deadlock_count) if self.team_size > 1 else 0
            ),
            "preferred_vertex_conflicts": int(self.preferred_vertex_conflicts),
            "preferred_swap_conflicts": int(self.preferred_swap_conflicts),
            "preferred_proximity_conflicts": int(self.preferred_proximity_conflicts),
            "preferred_conflict_events_total": int(preferred_conflict_total),
            "preferred_conflict_steps": int(self.preferred_conflict_steps),
            "actual_collision_pairs": int(self.actual_collision_pairs),
            "actual_collision_steps": int(self.actual_collision_steps),
            "actual_collision_episode": bool(self.actual_collision_pairs > 0),
            "conflict_avoided_steps": int(self.conflict_avoided_steps),
            "conflict_step_avoidance_rate": (
                0.0
                if self.preferred_conflict_steps == 0
                else float(self.conflict_avoided_steps / self.preferred_conflict_steps)
            ),
            "minimum_preferred_pairwise_distance": (
                None
                if not math.isfinite(self.minimum_preferred_pairwise_distance)
                else float(self.minimum_preferred_pairwise_distance)
            ),
            "minimum_proposed_pairwise_distance": (
                None
                if not math.isfinite(self.minimum_proposed_pairwise_distance)
                else float(self.minimum_proposed_pairwise_distance)
            ),
            "minimum_actual_pairwise_distance": (
                None
                if not math.isfinite(self.minimum_actual_pairwise_distance)
                else float(self.minimum_actual_pairwise_distance)
            ),
            "dynamic_safety_blocks_recorded": int(self.dynamic_blocks),
            "static_safety_blocks_recorded": int(self.static_blocks),
            "contact_pairs_recorded": int(self.total_contact_pairs),
            "communication_packets_recorded": int(self.total_map_packets),
            "communication_bytes_recorded": int(self.total_map_bytes),
            "map_packets_recorded": int(self.total_map_packets),
            "map_bytes_recorded": int(self.total_map_bytes),
            "total_known_free_cells_gained": int(self.total_new_free_cells),
            "new_free_cells_per_team_step": (
                0.0 if steps == 0 else float(self.total_new_free_cells / steps)
            ),
            "new_free_cells_per_travel_distance": (
                0.0
                if float(np.sum(self.path_lengths)) <= 1e-12
                else float(self.total_new_free_cells / np.sum(self.path_lengths))
            ),
            "unique_visited_nodes": unique_nodes,
            "total_node_visits": total_visits,
            "revisit_visits": revisit_visits,
            "revisit_ratio": 0.0 if total_visits == 0 else float(revisit_visits / total_visits),
            "cross_robot_overlap_nodes": overlap_nodes,
            "overlap_node_ratio": 0.0 if unique_nodes == 0 else float(overlap_nodes / unique_nodes),
            # Paper-facing aliases make Figure 4-5 inputs explicit without changing
            # the original metric names used by earlier scripts.
            "trajectory_overlap_ratio": (
                0.0 if unique_nodes == 0 else float(overlap_nodes / unique_nodes)
            ),
            "exploration_revisit_ratio": (
                0.0 if total_visits == 0 else float(revisit_visits / total_visits)
            ),
            "cross_robot_duplicate_visits": cross_robot_duplicate_visits,
            "metric_runtime_ms": float(self.total_runtime_ms),
            "wall_clock_episode_ms": float(elapsed_ms),
            "step_metrics_file": str(self.step_metrics_path),
            "robot_trajectories_file": str(self.trajectory_path),
            "event_log_file": str(self.event_log_path),
            "episode_metrics_file": str(self.episode_metrics_path),
        }
        for key, value in sorted(self.communication_cumulative.items()):
            recorded_value: int | float = int(value) if float(value).is_integer() else float(value)
            summary[f"{key}_recorded"] = recorded_value

        communication_packets = float(
            self.communication_cumulative.get(
                "communication_packets", self.total_map_packets
            )
        )
        communication_bytes = float(
            self.communication_cumulative.get(
                "communication_payload_bytes", self.total_map_bytes
            )
        )
        summary.update(
            {
                "mean_communication_payload_bytes_per_packet": (
                    0.0
                    if communication_packets <= 0
                    else communication_bytes / communication_packets
                ),
                "communication_payload_bytes_per_step": (
                    0.0 if steps <= 0 else communication_bytes / steps
                ),
                "communication_payload_bytes_per_robot": (
                    0.0 if self.team_size <= 0 else communication_bytes / self.team_size
                ),
                "communication_payload_bytes_per_new_free_cell": (
                    0.0
                    if self.total_new_free_cells <= 0
                    else communication_bytes / self.total_new_free_cells
                ),
                "communication_packets_per_step": (
                    0.0 if steps <= 0 else communication_packets / steps
                ),
            }
        )
        raw_equivalent_bytes = float(
            self.communication_cumulative.get("communication_raw_equivalent_bytes", 0.0)
        )
        dense_reference_bytes = float(
            self.communication_cumulative.get(
                "communication_dense_grid_reference_bytes", 0.0
            )
        )
        delivery_events = float(
            self.communication_cumulative.get("communication_delivery_events", 0.0)
        )
        truncated_packets = float(
            self.communication_cumulative.get(
                "communication_budget_truncated_packets", 0.0
            )
        )
        encode_ms = float(
            self.communication_cumulative.get("communication_encode_ms", 0.0)
        )
        decode_ms = float(
            self.communication_cumulative.get("communication_decode_apply_ms", 0.0)
        )
        summary.update(
            {
                "communication_payload_to_raw_equivalent_ratio": (
                    None
                    if raw_equivalent_bytes <= 0
                    else float(communication_bytes / raw_equivalent_bytes)
                ),
                "communication_payload_to_dense_grid_ratio": (
                    None
                    if dense_reference_bytes <= 0
                    else float(communication_bytes / dense_reference_bytes)
                ),
                "communication_encode_ms_per_packet": (
                    0.0 if communication_packets <= 0 else encode_ms / communication_packets
                ),
                "communication_decode_apply_ms_per_delivery": (
                    0.0 if delivery_events <= 0 else decode_ms / delivery_events
                ),
                "communication_budget_truncation_rate": (
                    0.0
                    if communication_packets <= 0
                    else float(truncated_packets / communication_packets)
                ),
            }
        )
        for threshold in self.coverage_thresholds:
            label = int(round(threshold * 100))
            value = self.threshold_steps[threshold]
            summary[f"steps_to_{label}_coverage"] = -1 if value is None else int(value)
            summary[f"reached_{label}_coverage"] = bool(value is not None)

        if extra_metrics:
            for key, value in extra_metrics.items():
                if key not in summary:
                    summary[key] = value

        self._write_csv(self.step_metrics_path, self.step_rows)
        self._write_csv(self.trajectory_path, self.trajectory_rows)
        with self.event_log_path.open("w", encoding="utf-8") as handle:
            for row in self.event_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.episode_metrics_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return summary
