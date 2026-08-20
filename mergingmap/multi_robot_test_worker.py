from __future__ import annotations

import collections
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from parameter import FREE, OCCUPIED, UNKNOWN, NODE_RESOLUTION
from test_parameter import DATA_TYPE, USE_DELTA_POSITION, USE_TEST_DATASET
from classes.agent.agent import Agent
from classes.agent.node_manager import NodeManager
from classes.env.multi_robot_env import MultiRobotEnv
from classes.multi_robot.contact_model import ContactModel
from merged_belief_manager import (
    MergedBeliefConfig,
    MergedBeliefManager,
)
from random_start_manager import (
    RandomStartConfig,
    create_environment_with_valid_random_starts,
)
from initial_direction_manager import (
    InitialDirectionConfig,
    InitialDirectionManager,
)
from mergingmap.dynamic_regions import (
    DynamicRegionConfig,
    DynamicRegionCoordinator,
)
from classes.multi_robot.evaluation_recorder import EpisodeMetricsRecorder
from classes.multi_robot.runtime_metrics import DetailedRuntimeMetrics
from motion_coordinator import MergingMapMotionCoordinator
from ablation_profiles import (
    DEFAULT_PROFILE_KEY,
    MergingMapAblationProfile,
    resolve_ablation_profile,
)
from worker_artifacts import WorkerArtifactMixin
from worker_dare_runtime import WorkerDareRuntimeMixin
from worker_episode import WorkerEpisodeMixin
from worker_map_sync import WorkerMapSyncMixin
from classes.multi_robot.visualizer import MultiRobotVisualizer
from multi_test_parameter import (
    ACTION_HORIZON_OVERRIDE,
    ALLOW_SHARED_DEPOT_AT_STEP_ZERO,
    CUDA_SYNCHRONIZE_FOR_TIMING,
    DARE_CHECKPOINT_PATH,
    CONTACT_HOPS,
    DIRECTION_LOOKAHEAD,
    DEADLOCK_MAX_BACKTRACKING_NODES,
    DEADLOCK_WAIT_THRESHOLD,
    DEADLOCK_STALL_THRESHOLD,
    DEADLOCK_SOFT_RELAX_THRESHOLD,
    DEADLOCK_LEASE_RELEASE_THRESHOLD,
    DEADLOCK_GRAPH_BACKTRACK_THRESHOLD,
    DEADLOCK_WAIT_WEIGHT,
    DEADLOCK_STALL_WEIGHT,
    ENABLE_METRIC_RECORDING,
    ENABLE_DETAILED_RUNTIME_METRICS,
    GIF_FRAME_DURATION,
    INITIAL_DIRECTION_BIAS_STEPS,
    INITIAL_DIRECTION_DECAY,
    INITIAL_DIRECTION_DEBUG,
    INITIAL_DIRECTION_MAX_BIAS_WEIGHT,
    MAP_CONFLICT_POLICY,
    MAP_PACKET_HEADER_BYTES,
    MAP_DEBUG,
    MAP_DEBUG_INTERVAL,
    MAP_MAX_CELLS_PER_PACKET,
    METRIC_COVERAGE_THRESHOLDS,
    MOTION_CACHE_TTL_STEPS,
    MOTION_PACKET_HEADER_BYTES,
    MOTION_RESERVATION_HORIZON,
    MAX_MULTI_ROBOT_STEPS,
    MIN_START_SEPARATION,
    REQUIRE_LINE_OF_SIGHT,
    RESET_OBS_HISTORY_ON_MAP_MERGE,
    RANDOM_DISTINCT_STARTS,
    REGION_AGE_WEIGHT,
    REGION_LEASE_PENALTY_WEIGHT,
    REGION_STALL_PENALTY_WEIGHT,
    REGION_TRACKING_IOU_WEIGHT,
    REGION_TRACKING_CENTROID_WEIGHT,
    REGION_MESSAGE_HEADER_BYTES,
    REGION_ARRIVAL_DISTANCE,
    REGION_CLAIM_TTL_STEPS,
    REGION_CONFLICT_DISTANCE,
    REGION_DEBUG,
    REGION_DEBUG_INTERVAL,
    REGION_DISTANCE_SLACK,
    REGION_DISTANCE_WEIGHT,
    REGION_FORCE_PROGRESS_AFTER_STEPS,
    REGION_ID_QUANTIZATION_CELLS,
    REGION_LEASE_STEPS,
    REGION_MATCH_CENTROID_CELLS,
    REGION_MATCH_IOU_THRESHOLD,
    REGION_MAX_FRONTIER_CELLS,
    REGION_MIN_COMMITMENT_STEPS,
    REGION_MIN_FRONTIER_CELLS,
    REGION_NO_PROGRESS_RELEASE_STEPS,
    REGION_PROGRESS_KNOWN_CELLS,
    REGION_UTILITY_WEIGHT,
    SAFE_DISTANCE,
    OSCILLATION_BASE_PENALTY,
    OSCILLATION_REPEAT_PENALTY,
    START_CLEARANCE_RADIUS_CELLS,
    START_SAMPLE_MAX_ATTEMPTS,
    SAVE_VISUALISATION,
    PROFILE_POLICY_FLOPS_ONCE,
    TRACK_HARDWARE_ENERGY,
    TRACK_PYTHON_MEMORY,
    TEAM_DONE_TOLERANCE_CELLS,
    VISUAL_BACKGROUND_MODE,
    VISUAL_FRAME_STRIDE,
    VISUAL_OUTPUT_ROOT,
)


# 中文目的：同步 Python、NumPy 与 Torch 随机种子以复现实验。
# English purpose: synchronise Python, NumPy, and Torch seeds for reproducibility.
def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class MultiRobotTestWorker(
    WorkerArtifactMixin,
    WorkerDareRuntimeMixin,
    WorkerMapSyncMixin,
    WorkerEpisodeMixin,
):
    """Full merging-map multi-robot DARE worker.

    DARE remains frozen. Each robot owns an independent Agent/NodeManager,
    builds its graph from a persistent contact-merged belief, receives a
    temporary frontier-region lease, and gets a balanced random cardinal
    departure role at episode start. Both supervisors only reorder legal DARE
    graph neighbours; neither changes the trained network or inserts an
    arbitrary off-graph action.
    """

    # 中文目的：组装环境、融合地图、策略运行时与实验记录组件。
    # English purpose: assemble the environment, merged belief, policy runtime, and recorders.
    def __init__(
        self,
        *,
        policy,
        episode_index: int,
        n_agents: int,
        device: torch.device | str,
        seed: int,
        communication_mode: str = "compressed",
        ablation_profile: str | MergingMapAblationProfile = DEFAULT_PROFILE_KEY,
        include_extra_supervisors: bool = False,
        output_root: str | Path | None = None,
        scenario_id: str | None = None,
        trial: int = 0,
        save_visualisation: bool | None = None,
    ) -> None:
        if DATA_TYPE != "node":
            raise NotImplementedError(
                "Merged-map multi-robot DARE currently supports DATA_TYPE='node'."
            )
        if communication_mode not in {"none", "raw", "compressed"}:
            raise ValueError(
                "communication_mode must be 'none', 'raw', or 'compressed'"
            )

        set_random_seed(seed)
        self.policy = policy
        self.policy.eval()
        self.episode_index = int(episode_index)
        self.n_agents = int(n_agents)
        self.device = torch.device(device)
        self.seed = int(seed)
        self.communication_mode = communication_mode
        self.ablation_profile = resolve_ablation_profile(
            ablation_profile,
            include_extra_supervisors=include_extra_supervisors,
        )
        if not self.ablation_profile.enable_map_merging:
            raise ValueError("MergingMap worker requires enable_map_merging=True")
        self.scenario_id = str(
            scenario_id
            or f"map_{self.episode_index:04d}_trial_{int(trial):02d}_"
            f"robots_{self.n_agents:02d}_seed_{self.seed}"
        )
        self.trial = int(trial)
        self.output_root = Path(
            output_root
            or os.environ.get("MERGINGMAP_RUN_DIR", VISUAL_OUTPUT_ROOT)
        ).expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.save_visualisation = (
            bool(SAVE_VISUALISATION)
            if save_visualisation is None
            else bool(save_visualisation)
        )
        self.obs_horizon = int(policy.n_obs_steps)
        self.action_horizon = int(ACTION_HORIZON_OVERRIDE)
        if self.action_horizon != 1:
            raise ValueError("Exactly one graph move must be executed per replan.")

        if not RANDOM_DISTINCT_STARTS:
            raise ValueError(
                "The isolated mergingmap experiment requires "
                "RANDOM_DISTINCT_STARTS=True."
            )

        def env_factory(start_seed: int) -> MultiRobotEnv:
            return MultiRobotEnv(
                episode_index=self.episode_index,
                n_agents=self.n_agents,
                test=USE_TEST_DATASET,
                seed=int(start_seed),
                min_start_separation=MIN_START_SEPARATION,
                done_tolerance_cells=TEAM_DONE_TOLERANCE_CELLS,
            )

        start_selection = create_environment_with_valid_random_starts(
            env_factory,
            base_seed=self.seed,
            n_agents=self.n_agents,
            config=RandomStartConfig(
                free_value=FREE,
                min_separation=MIN_START_SEPARATION,
                max_attempts=START_SAMPLE_MAX_ATTEMPTS,
                clearance_radius_cells=START_CLEARANCE_RADIUS_CELLS,
            ),
        )
        self.env = start_selection.env
        self.start_seed = int(start_selection.selected_seed)
        self.start_attempt_count = int(start_selection.attempt_count)
        self.start_validation = start_selection.validation
        self.initial_start_positions = [
            np.asarray(position, dtype=np.float32).copy()
            for position in self.env.robot_locations
        ]

        print(
            f"[RANDOM-START] episode={self.episode_index} "
            f"robots={self.n_agents} base_seed={self.seed} "
            f"selected_seed={self.start_seed} "
            f"attempts={self.start_attempt_count} "
            f"min_distance={self.start_validation.minimum_pairwise_distance:.3f} "
            f"positions={[(round(float(p[0]), 3), round(float(p[1]), 3)) for p in self.initial_start_positions]} "
            f"cells={list(self.start_validation.cell_positions)}",
            flush=True,
        )

        self.start_position_file = self._save_start_position_record()

        self.initial_direction_manager = InitialDirectionManager(
            self.n_agents,
            seed=self.seed,
            config=InitialDirectionConfig(
                enabled=self.ablation_profile.enable_initial_direction,
                bias_steps=INITIAL_DIRECTION_BIAS_STEPS,
                max_bias_weight=INITIAL_DIRECTION_MAX_BIAS_WEIGHT,
                decay=INITIAL_DIRECTION_DECAY,
            ),
        )
        self.initial_direction_file = self._save_initial_direction_record()
        if INITIAL_DIRECTION_DEBUG and self.ablation_profile.enable_initial_direction:
            print(
                f"[PRIMARY-DIRECTION] episode={self.episode_index} "
                f"robots={self.n_agents} seed={self.seed} "
                f"assignments={self.initial_direction_manager.assignment_payload()}",
                flush=True,
            )

        # Separate graph instances are deliberate. Their similarity is measured
        # after exchanging maps, rather than being forced by sharing one object.
        self.robot_list = [
            Agent(
                robot_id,
                NodeManager(plot=False),
                self.device,
                plot=False,
            )
            for robot_id in range(self.n_agents)
        ]

        initial_local_maps = [
            self.env.get_local_belief(robot_id).copy()
            for robot_id in range(self.n_agents)
        ]
        self.map_merger = MergedBeliefManager(
            initial_local_maps,
            MergedBeliefConfig(
                unknown_value=UNKNOWN,
                free_value=FREE,
                occupied_value=OCCUPIED,
                max_cells_per_packet=MAP_MAX_CELLS_PER_PACKET,
                conflict_policy=MAP_CONFLICT_POLICY,
                packet_header_bytes=MAP_PACKET_HEADER_BYTES,
            ),
        )

        self.region_coordinator = None
        if self.ablation_profile.enable_dynamic_regions:
            self.region_coordinator = DynamicRegionCoordinator(
                self.n_agents,
                DynamicRegionConfig(
                    unknown_value=UNKNOWN,
                    free_value=FREE,
                    node_resolution=NODE_RESOLUTION,
                    min_frontier_cells=REGION_MIN_FRONTIER_CELLS,
                    max_frontier_cells_per_region=REGION_MAX_FRONTIER_CELLS,
                    region_id_quantization_cells=REGION_ID_QUANTIZATION_CELLS,
                    region_match_iou_threshold=REGION_MATCH_IOU_THRESHOLD,
                    region_match_centroid_cells=REGION_MATCH_CENTROID_CELLS,
                    region_conflict_distance=REGION_CONFLICT_DISTANCE,
                    lease_steps=REGION_LEASE_STEPS,
                    claim_ttl_steps=REGION_CLAIM_TTL_STEPS,
                    min_commitment_steps=REGION_MIN_COMMITMENT_STEPS,
                    no_progress_release_steps=REGION_NO_PROGRESS_RELEASE_STEPS,
                    force_progress_after_steps=REGION_FORCE_PROGRESS_AFTER_STEPS,
                    progress_known_cells=REGION_PROGRESS_KNOWN_CELLS,
                    arrival_distance=REGION_ARRIVAL_DISTANCE,
                    distance_slack=REGION_DISTANCE_SLACK,
                    distance_weight=REGION_DISTANCE_WEIGHT,
                    utility_weight=REGION_UTILITY_WEIGHT,
                    age_weight=REGION_AGE_WEIGHT,
                    lease_penalty_weight=REGION_LEASE_PENALTY_WEIGHT,
                    stall_penalty_weight=REGION_STALL_PENALTY_WEIGHT,
                    tracking_iou_weight=REGION_TRACKING_IOU_WEIGHT,
                    tracking_centroid_weight=REGION_TRACKING_CENTROID_WEIGHT,
                    region_message_header_bytes=REGION_MESSAGE_HEADER_BYTES,
                    debug=REGION_DEBUG,
                    debug_interval=REGION_DEBUG_INTERVAL,
                ),
            )
        self.region_event_file = self._region_event_path()

        self.contact_model = ContactModel(
            CONTACT_HOPS,
            REQUIRE_LINE_OF_SIGHT,
        )
        self.trajectories: List[List[np.ndarray]] = [
            [self.env.robot_locations[robot_id].copy()]
            for robot_id in range(self.n_agents)
        ]
        self.obs_deques: List[collections.deque] = []
        self.metrics: Dict[str, float | int | str | bool] = {}
        self.contact_events = 0
        self.dynamic_blocks = 0
        self.static_blocks = 0
        self.contact_history_edges: set[tuple[int, int]] = set()
        self.motion_coordinator = MergingMapMotionCoordinator(
            self.n_agents,
            safe_distance=SAFE_DISTANCE,
            mode=self.ablation_profile.coordination_mode,
            deadlock_wait_threshold=DEADLOCK_WAIT_THRESHOLD,
            deadlock_stall_threshold=DEADLOCK_STALL_THRESHOLD,
            deadlock_soft_relax_threshold=DEADLOCK_SOFT_RELAX_THRESHOLD,
            deadlock_lease_release_threshold=DEADLOCK_LEASE_RELEASE_THRESHOLD,
            deadlock_backtrack_threshold=DEADLOCK_GRAPH_BACKTRACK_THRESHOLD,
            deadlock_wait_weight=DEADLOCK_WAIT_WEIGHT,
            deadlock_stall_weight=DEADLOCK_STALL_WEIGHT,
            max_backtracking_nodes=DEADLOCK_MAX_BACKTRACKING_NODES,
            node_resolution=NODE_RESOLUTION,
            motion_reservation_horizon=MOTION_RESERVATION_HORIZON,
            motion_cache_ttl_steps=MOTION_CACHE_TTL_STEPS,
            motion_packet_header_bytes=MOTION_PACKET_HEADER_BYTES,
            oscillation_base_penalty=OSCILLATION_BASE_PENALTY,
            oscillation_repeat_penalty=OSCILLATION_REPEAT_PENALTY,
        )
        self.method_name = self.ablation_profile.method_name

        self.metric_recorder = None
        if ENABLE_METRIC_RECORDING:
            self.metric_recorder = EpisodeMetricsRecorder(
                output_root=self.output_root,
                episode=self.episode_index,
                method=self.method_name,
                seed=self.seed,
                team_size=self.n_agents,
                communication_mode=self.communication_mode,
                initial_positions=self.env.robot_locations,
                initial_coverage=float(self.env.explored_rate),
                initial_known_free_cells=EpisodeMetricsRecorder.known_free_cell_count(
                    self.env, FREE
                ),
                node_resolution=NODE_RESOLUTION,
                safe_distance=SAFE_DISTANCE,
                deadlock_wait_threshold=DEADLOCK_WAIT_THRESHOLD,
                coverage_thresholds=METRIC_COVERAGE_THRESHOLDS,
            )

        self.visualizer = None
        if self.save_visualisation:
            self.visualizer = MultiRobotVisualizer(
                output_root=str(self.output_root),
                episode_index=self.episode_index,
                team_size=self.n_agents,
                seed=self.seed,
                frame_stride=VISUAL_FRAME_STRIDE,
                background_mode=VISUAL_BACKGROUND_MODE,
            )

        self._initialise_robot_states()
        self.runtime_metrics = None
        if ENABLE_DETAILED_RUNTIME_METRICS:
            self.runtime_metrics = DetailedRuntimeMetrics(
                device=self.device,
                policy=self.policy,
                team_size=self.n_agents,
                checkpoint_path=os.environ.get(
                    "DARE_CHECKPOINT_PATH", DARE_CHECKPOINT_PATH
                ),
                synchronize_cuda=CUDA_SYNCHRONIZE_FOR_TIMING,
                track_python_memory=TRACK_PYTHON_MEMORY,
                sample_energy=TRACK_HARDWARE_ENERGY,
            )
            self.runtime_metrics.profile_policy_once(
                lambda: self.policy.predict_action(self._obs_dict(0)),
                enabled=PROFILE_POLICY_FLOPS_ONCE,
            )
            self.runtime_metrics.start_episode()
            if self.metric_recorder is not None:
                self.metric_recorder.reset_wall_clock()
        if self.visualizer is not None:
            self.visualizer.save_frame(
                env=self.env,
                trajectories=self.trajectories,
                step=0,
                contact_pairs=(),
                team_coverage=self.env.explored_rate,
            )
