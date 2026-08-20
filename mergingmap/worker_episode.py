"""Episode execution loop for the MergingMap worker.

MergingMap 工作器的回合执行循环。
"""
from __future__ import annotations

import contextlib
import json
import time
from typing import Dict

import numpy as np

from parameter import FREE, OCCUPIED, UNKNOWN
from classes.multi_robot.evaluation_recorder import EpisodeMetricsRecorder
from multi_test_parameter import (
    ALLOW_SHARED_DEPOT_AT_STEP_ZERO,
    CUDA_SYNCHRONIZE_FOR_TIMING,
    GIF_FRAME_DURATION,
    MAP_DEBUG,
    MAP_DEBUG_INTERVAL,
    MAX_MULTI_ROBOT_STEPS,
    RANDOM_DISTINCT_STARTS,
)


class WorkerEpisodeMixin:
    """Execute and summarise one MergingMap evaluation episode.

    中文：负责逐步协调、环境执行、指标记录和最终结果汇总。
    English: owns step coordination, environment execution, metric recording, and final summary.
    """

    def _measure_runtime(self, name: str, *, cuda_sync: bool = False):
        """Return a real or no-op timing context.

        中文目的：让主循环在关闭详细指标时无需维护两套代码路径。
        English implementation: delegates to the runtime collector when enabled,
        otherwise returns ``nullcontext`` with zero behavioural impact.
        """

        if self.runtime_metrics is None:
            return contextlib.nullcontext()
        return self.runtime_metrics.measure(name, cuda_sync=cuda_sync)

    def _communication_cumulative(
        self,
        map_metrics: dict,
        region_metrics: dict | None = None,
        motion_metrics: dict | None = None,
    ) -> dict[str, int | float]:
        """Convert all logical payload layers to the common communication schema.

        中文目的：把地图、区域租约和短运动意图三类负载统一计入通信总量，同时保留
        各层独立指标。English implementation: sums monotonic packet/byte counters
        while retaining map-specific cell and encoding diagnostics.
        """

        region = region_metrics or {}
        motion = motion_metrics or {}
        map_packets = int(map_metrics.get("map_packets_sent", 0))
        region_packets = int(region.get("region_message_packets", 0))
        motion_packets = int(motion.get("motion_message_packets", 0))
        map_bytes = int(map_metrics.get("map_bytes_sent", 0))
        region_bytes = int(region.get("region_message_bytes", 0))
        motion_bytes = int(motion.get("motion_message_bytes", 0))
        return {
            "communication_packets": map_packets + region_packets + motion_packets,
            "communication_payload_bytes": map_bytes + region_bytes + motion_bytes,
            "communication_map_packets": map_packets,
            "communication_region_packets": region_packets,
            "communication_motion_packets": motion_packets,
            "communication_map_bytes": map_bytes,
            "communication_region_bytes": region_bytes,
            "communication_motion_bytes": motion_bytes,
            "communication_payload_cells": int(map_metrics.get("map_cells_sent", 0)),
            "communication_cells_received": int(map_metrics.get("map_cells_received", 0)),
            "communication_encode_ms": float(map_metrics.get("map_packet_encode_ms", 0.0)),
            "communication_decode_apply_ms": float(map_metrics.get("map_packet_apply_ms", 0.0)),
            "communication_exchange_wall_ms": float(map_metrics.get("map_exchange_wall_ms", 0.0))
            + float(motion.get("motion_exchange_total_ms", 0.0)),
            "communication_delivery_events": int(map_metrics.get("map_delivery_events", 0))
            + int(motion.get("motion_message_packets_delivered", 0)),
            "communication_conflicts": int(map_metrics.get("map_conflicts", 0)),
            "communication_local_map_sync_ms": float(map_metrics.get("map_local_sync_ms", 0.0)),
            "communication_header_bytes": int(map_metrics.get("map_header_bytes_sent", 0)),
            "communication_index_bytes": int(map_metrics.get("map_index_bytes_sent", 0)),
            "communication_value_bytes": int(map_metrics.get("map_value_bytes_sent", 0)),
            "communication_raw_equivalent_bytes": int(map_metrics.get("map_full_known_sparse_reference_bytes", 0)),
            "communication_dense_grid_reference_bytes": int(map_metrics.get("map_dense_grid_reference_bytes", 0)),
            "communication_deferred_cells": int(map_metrics.get("map_deferred_cells", 0)),
        }

    # 中文目的：执行一个完整的 MergingMap 多机器人测试回合。
    # English purpose: Execute one complete MergingMap multi-robot evaluation episode.
    def run_episode(self) -> Dict[str, float | int | str | bool]:
        last_step = 0

        for step in range(MAX_MULTI_ROBOT_STEPS):
            last_step = step
            fallback_step_started_at = time.perf_counter()
            if self.runtime_metrics is not None:
                self.runtime_metrics.start_step()

            previous_positions = [
                position.copy() for position in self.env.robot_locations
            ]

            with self._measure_runtime("contact_detection"):
                contact_pairs = self.contact_model.get_contact_pairs(self.env)
            self.contact_events += len(contact_pairs)

            with self._measure_runtime("map_local_sync_and_exchange"):
                merged_changed = self._synchronise_maps(
                    step=step,
                    contact_pairs=contact_pairs,
                )
            with self._measure_runtime("observation_graph_refresh"):
                self._refresh_observations(
                    merged_changed_robots=merged_changed
                )
            with self._measure_runtime("dynamic_region_update"):
                region_snapshot = self._update_dynamic_regions(
                    step=step,
                    contact_pairs=contact_pairs,
                )

            with self._measure_runtime(
                "policy_inference_team",
                cuda_sync=CUDA_SYNCHRONIZE_FOR_TIMING,
            ):
                action_predictions = [
                    self._predict_action_sequence(robot_id)
                    for robot_id in range(self.n_agents)
                ]
            with self._measure_runtime("candidate_generation"):
                base_candidate_lists = [
                    self._base_dare_candidates(robot_id, action_predictions[robot_id])
                    for robot_id in range(self.n_agents)
                ]
                candidate_lists = [
                    self._apply_candidate_supervisors(
                        robot_id,
                        base_candidate_lists[robot_id],
                        step=step,
                    )
                    for robot_id in range(self.n_agents)
                ]
                short_horizon_plans = [
                    self._short_horizon_plan(
                        robot_id,
                        action_predictions[robot_id],
                        candidate_lists[robot_id],
                    )
                    for robot_id in range(self.n_agents)
                ]

            # The facade exchanges only one/two-step intent, then applies hard
            # collision constraints and staged deadlock recovery.
            # 统一接口仅交换一到两步意图，再执行硬碰撞约束与分阶段死锁恢复。
            with self._measure_runtime("motion_coordination_facade"):
                motion_decision = self.motion_coordinator.resolve_step(
                    previous_positions,
                    candidate_lists,
                    time_step=step,
                    allow_shared_start_step0=ALLOW_SHARED_DEPOT_AT_STEP_ZERO,
                    contact_pairs=(contact_pairs if self.communication_mode != "none" else ()),
                    short_horizon_plans=short_horizon_plans,
                    recovery_candidate_lists=base_candidate_lists,
                    stall_steps=(
                        None
                        if self.region_coordinator is None
                        else self.region_coordinator.stall_step_counts()
                    ),
                    robots=self.robot_list,
                )
            if self.region_coordinator is not None:
                for robot_id in motion_decision.lease_release_robot_ids:
                    self.region_coordinator.request_recovery_release(
                        robot_id, step=step
                    )
            preferred_positions = list(motion_decision.preferred_positions)
            next_positions = list(motion_decision.next_positions)
            dynamically_blocked = list(motion_decision.blocked_robot_ids)
            resolution_info = motion_decision.resolution_info

            self.dynamic_blocks += len(dynamically_blocked)
            with self._measure_runtime("environment_step"):
                actual_positions, statically_blocked = self.env.step_all(
                    next_positions
                )
            with self._measure_runtime("deadlock_state_update"):
                deadlock_update_ms = self.motion_coordinator.update_after_execution(
                    previous_positions,
                    actual_positions,
                )
            self.static_blocks += len(statically_blocked)

            with self._measure_runtime("post_step_local_map_sync"):
                for robot_id in range(self.n_agents):
                    self.trajectories[robot_id].append(
                        actual_positions[robot_id].copy()
                    )
                    newly_known = self.map_merger.sync_local_map(
                        robot_id,
                        self.env.get_local_belief(robot_id),
                    )
                    if self.region_coordinator is not None:
                        assignment = self.region_coordinator.assignment_for_robot(robot_id)
                        self.region_coordinator.report_progress(
                            robot_id=robot_id,
                            step=step + 1,
                            newly_known_cells=int(newly_known),
                            current_position=actual_positions[robot_id],
                            frontier_count=(
                                None if assignment is None else int(assignment["frontier_count"])
                            ),
                        )
            with self._measure_runtime("region_event_io"):
                self._write_region_events()

            with self._measure_runtime("metric_state_snapshot"):
                current_map_metrics = self.map_merger.metrics()
                current_region_metrics = (
                    {} if self.region_coordinator is None else self.region_coordinator.metrics()
                )
                current_motion_metrics = self.motion_coordinator.metrics()
                current_coverage = float(self.env.explored_rate)
                current_known_free = EpisodeMetricsRecorder.known_free_cell_count(
                    self.env, FREE
                )

            if (
                MAP_DEBUG
                and MAP_DEBUG_INTERVAL > 0
                and (step + 1) % MAP_DEBUG_INTERVAL == 0
            ):
                with self._measure_runtime("debug_diagnostics"):
                    graph_metrics = self._graph_agreement_metrics()
                    print(
                        f"[MERGED-DARE] step={step + 1} "
                        f"coverage={current_coverage:.4f} "
                        f"map_agree={current_map_metrics['mean_pairwise_map_agreement']:.4f} "
                        f"graph_jaccard={graph_metrics['mean_pairwise_graph_jaccard']:.4f} "
                        f"bytes={current_map_metrics['map_bytes_sent']} "
                        f"regions={0 if region_snapshot is None else region_snapshot['active_leases']} "
                        f"unassigned={0 if region_snapshot is None else region_snapshot['mean_local_unassigned_regions']:.2f}",
                        flush=True,
                    )

            if self.visualizer is not None:
                with self._measure_runtime("visualization_frame"):
                    self.visualizer.save_frame(
                        env=self.env,
                        trajectories=self.trajectories,
                        step=step + 1,
                        contact_pairs=(
                            contact_pairs
                            if self.communication_mode != "none"
                            else ()
                        ),
                        team_coverage=current_coverage,
                    )

            step_profile: dict[str, object] = {}
            if self.runtime_metrics is not None:
                # Internal coordinator fields separate collision and deadlock cost
                # even though both are called through one MergingMap facade.
                if resolution_info is not None:
                    for field_name in (
                        "preferred_action_selection_ms",
                        "deadlock_priority_ms",
                        "deadlock_escape_ms",
                        "collision_resolution_ms",
                        "motion_exchange_ms",
                        "motion_filter_ms",
                        "graph_backtrack_ms",
                        "coordination_total_ms",
                    ):
                        self.runtime_metrics.add_step_value(
                            field_name,
                            getattr(resolution_info, field_name, 0.0),
                        )
                self.runtime_metrics.add_step_value(
                    "deadlock_state_update_internal_ms",
                    deadlock_update_ms,
                )
                step_profile = self.runtime_metrics.finish_step()
                team_inference = float(step_profile.get("policy_inference_team_ms", 0.0) or 0.0)
                step_profile["policy_inference_per_robot_ms"] = (
                    0.0 if self.n_agents <= 0 else team_inference / self.n_agents
                )

            step_runtime_ms = float(
                step_profile.get(
                    "profiled_step_wall_ms",
                    (time.perf_counter() - fallback_step_started_at) * 1000.0,
                )
            )
            if self.metric_recorder is not None:
                self.metric_recorder.record_step(
                    step=step + 1,
                    preferred_positions=preferred_positions,
                    proposed_positions=next_positions,
                    actual_positions=actual_positions,
                    coverage=current_coverage,
                    known_free_cells=current_known_free,
                    dynamic_blocked_robot_ids=dynamically_blocked,
                    static_blocked_robot_ids=statically_blocked,
                    contact_pairs=contact_pairs,
                    map_packets_cumulative=int(
                        current_map_metrics.get("map_packets_sent", 0)
                    ),
                    map_bytes_cumulative=int(
                        current_map_metrics.get("map_bytes_sent", 0)
                    ),
                    communication_cumulative=self._communication_cumulative(
                        current_map_metrics,
                        current_region_metrics,
                        current_motion_metrics,
                    ),
                    resolver_info=resolution_info,
                    step_runtime_ms=step_runtime_ms,
                    extra_step_metrics=step_profile,
                )

            if self.env.done:
                break

        # Ensure final local sensor data is included in map diagnostics.
        with self._measure_runtime("final_local_map_sync"):
            for robot_id in range(self.n_agents):
                self.map_merger.sync_local_map(
                    robot_id,
                    self.env.get_local_belief(robot_id),
                )

        gif_path = None
        if self.visualizer is not None:
            with self._measure_runtime("visualization_gif_build"):
                gif_path = self.visualizer.build_gif(GIF_FRAME_DURATION)

        per_robot_distance = [
            float(robot.travel_dist)
            for robot in self.robot_list
        ]

        with self._measure_runtime("final_diagnostics"):
            map_metrics = self.map_merger.metrics()
            graph_metrics = self._graph_agreement_metrics()
            direction_metrics = self.initial_direction_manager.metrics()
            region_metrics = (
                self.region_coordinator.metrics()
                if self.region_coordinator is not None
                else {"region_assignment_enabled": False}
            )
            motion_metrics = self.motion_coordinator.metrics()
            communication_metrics = self._communication_cumulative(
                map_metrics, region_metrics, motion_metrics
            )
            map_quality_metrics = EpisodeMetricsRecorder.map_quality_metrics(
                [self.map_merger.merged_map(robot_id) for robot_id in range(self.n_agents)],
                self.env.ground_truth,
                free_value=FREE,
                occupied_value=OCCUPIED,
                unknown_value=UNKNOWN,
            )
            map_structure_metrics = EpisodeMetricsRecorder.map_structure_metrics(
                self.env.ground_truth,
                free_value=FREE,
                occupied_value=OCCUPIED,
            )
            self._write_region_events()

        runtime_episode_metrics = (
            {} if self.runtime_metrics is None else self.runtime_metrics.finalise()
        )
        recorded_metrics = {}
        if self.metric_recorder is not None:
            recorded_metrics = self.metric_recorder.finalise(
                success=bool(self.env.done),
                extra_metrics={
                    "scenario_id": self.scenario_id,
                    "trial": self.trial,
                    "paper_method_key": self.ablation_profile.key,
                    "coordination_mode": self.ablation_profile.coordination_mode,
                    "enable_map_merging": bool(self.ablation_profile.enable_map_merging),
                    "initial_direction_enabled": bool(
                        self.ablation_profile.enable_initial_direction
                    ),
                    "dynamic_region_enabled": bool(
                        self.ablation_profile.enable_dynamic_regions
                    ),
                    "start_positions": json.dumps(
                        [
                            [float(position[0]), float(position[1])]
                            for position in self.initial_start_positions
                        ]
                    ),
                    "selected_start_seed": int(self.start_seed),
                    "map_packets_sent": int(map_metrics.get("map_packets_sent", 0)),
                    "map_bytes_sent": int(map_metrics.get("map_bytes_sent", 0)),
                    **communication_metrics,
                    **map_metrics,
                    **map_quality_metrics,
                    **map_structure_metrics,
                    **region_metrics,
                    **motion_metrics,
                    "communication_transport": "in_process_simulation",
                    "physical_network_latency_measured": False,
                    **runtime_episode_metrics,
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
            "scenario_id": self.scenario_id,
            "trial": self.trial,
            "paper_method_key": self.ablation_profile.key,
            "enable_map_merging": bool(self.ablation_profile.enable_map_merging),
            "initial_direction_enabled": bool(
                self.ablation_profile.enable_initial_direction
            ),
            "dynamic_region_enabled": bool(
                self.ablation_profile.enable_dynamic_regions
            ),
            "ghost_mode": self.ablation_profile.coordination_mode == "ghost",
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
            "initial_direction_file": self.initial_direction_file,
            "region_event_file": self.region_event_file,
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
            "collision_avoidance_enabled": (
                self.ablation_profile.coordination_mode != "ghost"
            ),
            "deadlock_avoidance_enabled": (
                self.ablation_profile.coordination_mode == "collision_deadlock"
            ),
            **motion_metrics,
            **map_metrics,
            **map_quality_metrics,
            **map_structure_metrics,
            **graph_metrics,
            **direction_metrics,
            **region_metrics,
            **communication_metrics,
            **runtime_episode_metrics,
            **recorded_metrics,
            "gif_path": "" if gif_path is None else gif_path,
        }
        return self.metrics
