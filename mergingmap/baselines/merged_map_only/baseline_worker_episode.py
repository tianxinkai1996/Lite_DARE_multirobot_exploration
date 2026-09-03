"""Episode loop for the MergingMap-only baseline.

MergingMap-only 
"""
from __future__ import annotations

import json
import time
from typing import Dict

import numpy as np

from parameter import FREE
from classes.multi_robot.evaluation_recorder import EpisodeMetricsRecorder
from collision.joint_resolver import resolve_synchronous_moves
from mergingmap.baselines.merged_map_only.multi_test_parameter import (
    ALLOW_SHARED_DEPOT_AT_STEP_ZERO,
    GIF_FRAME_DURATION,
    GHOST_MODE,
    MAP_DEBUG,
    MAP_DEBUG_INTERVAL,
    MAX_MULTI_ROBOT_STEPS,
    SAFE_DISTANCE,
)

class BaselineEpisodeMixin:
    """Execute and summarise one map-only ablation episode.

    English: owns baseline step execution, recording, and final summary.
    """

    # English purpose: Execute one MergingMap-only baseline episode.
    def run_episode(self):
        last_step = 0

        for step in range(MAX_MULTI_ROBOT_STEPS):
            last_step = step
            step_started_at = time.perf_counter()
            previous_positions = [
                position.copy() for position in self.env.robot_locations
            ]

            contact_pairs = self.contact_model.get_contact_pairs(
                self.env
            )
            self.contact_events += len(contact_pairs)

            merged_changed = self._synchronise_maps(
                step=step,
                contact_pairs=contact_pairs,
            )
            self._refresh_observations(
                merged_changed_robots=merged_changed
            )

            action_predictions = [
                self._predict_action_sequence(robot_id)
                for robot_id in range(self.n_agents)
            ]
            candidate_lists = [
                self._ordered_dare_candidates(
                    robot_id,
                    action_predictions[robot_id],
                )
                for robot_id in range(self.n_agents)
            ]

            preferred_positions = [
                candidates[0].copy()
                if candidates
                else previous_positions[robot_id].copy()
                for robot_id, candidates in enumerate(candidate_lists)
            ]
            if GHOST_MODE:
                # Map-only ablation: no multi-robot motion coordination.
                next_positions = preferred_positions
                dynamically_blocked = []
            else:
                # Optional basic immediate safety only. No penalties, waiting
                # priorities, deadlock manager, or backtracking are used.
                next_positions, dynamically_blocked = (
                    resolve_synchronous_moves(
                        previous_positions,
                        candidate_lists,
                        safe_distance=SAFE_DISTANCE,
                        time_step=step,
                        allow_shared_start_step0=(
                            ALLOW_SHARED_DEPOT_AT_STEP_ZERO
                        ),
                    )
                )

            self.dynamic_blocks += len(dynamically_blocked)
            actual_positions, statically_blocked = self.env.step_all(
                next_positions
            )
            self.static_blocks += len(statically_blocked)

            for robot_id in range(self.n_agents):
                self.trajectories[robot_id].append(
                    actual_positions[robot_id].copy()
                )

            if self.metric_recorder is not None:
                current_map_metrics = self.map_merger.metrics()
                self.metric_recorder.record_step(
                    step=step + 1,
                    preferred_positions=preferred_positions,
                    proposed_positions=next_positions,
                    actual_positions=actual_positions,
                    coverage=float(self.env.explored_rate),
                    known_free_cells=EpisodeMetricsRecorder.known_free_cell_count(
                        self.env, FREE
                    ),
                    dynamic_blocked_robot_ids=dynamically_blocked,
                    static_blocked_robot_ids=statically_blocked,
                    contact_pairs=contact_pairs,
                    map_packets_cumulative=int(
                        current_map_metrics.get("map_packets_sent", 0)
                    ),
                    map_bytes_cumulative=int(
                        current_map_metrics.get("map_bytes_sent", 0)
                    ),
                    step_runtime_ms=(time.perf_counter() - step_started_at) * 1000.0,
                )

            if (
                MAP_DEBUG
                and MAP_DEBUG_INTERVAL > 0
                and (step + 1) % MAP_DEBUG_INTERVAL == 0
            ):
                map_metrics = self.map_merger.metrics()
                graph_metrics = self._graph_agreement_metrics()
                print(
                    f"[MERGED-DARE] step={step + 1} "
                    f"coverage={float(self.env.explored_rate):.4f} "
                    f"map_agree={map_metrics['mean_pairwise_map_agreement']:.4f} "
                    f"graph_jaccard={graph_metrics['mean_pairwise_graph_jaccard']:.4f} "
                    f"bytes={map_metrics['map_bytes_sent']}",
                    flush=True,
                )

            if self.visualizer is not None:
                self.visualizer.save_frame(
                    env=self.env,
                    trajectories=self.trajectories,
                    step=step + 1,
                    contact_pairs=(
                        contact_pairs
                        if self.communication_mode != "none"
                        else ()
                    ),
                    team_coverage=self.env.explored_rate,
                )

            if self.env.done:
                break

        # Ensure final local sensor data is included in map diagnostics.
        for robot_id in range(self.n_agents):
            self.map_merger.sync_local_map(
                robot_id,
                self.env.get_local_belief(robot_id),
            )

        gif_path = None
        if self.visualizer is not None:
            gif_path = self.visualizer.build_gif(
                GIF_FRAME_DURATION
            )

        per_robot_distance = [
            float(robot.travel_dist)
            for robot in self.robot_list
        ]

        map_metrics = self.map_merger.metrics()
        graph_metrics = self._graph_agreement_metrics()
        recorded_metrics = {}
        if self.metric_recorder is not None:
            recorded_metrics = self.metric_recorder.finalise(
                success=bool(self.env.done),
                extra_metrics={
                    "map_packets_sent": int(map_metrics.get("map_packets_sent", 0)),
                    "map_bytes_sent": int(map_metrics.get("map_bytes_sent", 0)),
                },
            )
            per_robot_distance = [
                float(value)
                for value in self.metric_recorder.path_lengths.tolist()
            ]

        self.metrics = {
            "episode": self.episode_index,
            "method": self.method_name,
            "seed": self.seed,
            "team_size": self.n_agents,
            "communication_mode": self.communication_mode,
            "ghost_mode": bool(GHOST_MODE),
            "random_distinct_starts": bool(RANDOM_DISTINCT_STARTS),
            "selected_start_seed": int(self.start_seed),
            "start_sampling_attempts": int(self.start_attempt_count),
            "minimum_start_distance": float(
                self.start_validation.minimum_pairwise_distance
            ),
            "start_positions": json.dumps(
                [
                    [float(position[0]), float(position[1])]
                    for position in self.initial_start_positions
                ]
            ),
            "start_cells": json.dumps(
                [
                    [int(cell[0]), int(cell[1])]
                    for cell in self.start_validation.cell_positions
                ]
            ),
            "start_position_file": self.start_position_file,
            "steps": last_step + 1,
            "success": bool(self.env.done),
            "team_coverage": float(self.env.explored_rate),
            "team_travel_distance": float(sum(per_robot_distance)),
            "max_robot_travel_distance": float(
                max(per_robot_distance, default=0.0)
            ),
            "mean_robot_travel_distance": float(
                np.mean(per_robot_distance)
                if per_robot_distance
                else 0.0
            ),
            "contact_events": int(self.contact_events),
            "unique_contact_edges": int(
                len(self.contact_history_edges)
            ),
            "dynamic_safety_blocks": int(self.dynamic_blocks),
            "static_safety_blocks": int(self.static_blocks),
            **map_metrics,
            **graph_metrics,
            **recorded_metrics,
            "gif_path": "" if gif_path is None else gif_path,
        }
        return self.metrics
