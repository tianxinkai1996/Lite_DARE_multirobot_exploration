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
from mergingmap.merged_belief_manager import (
    MergedBeliefConfig,
    MergedBeliefManager,
)
from mergingmap.random_start_manager import (
    RandomStartConfig,
    create_environment_with_valid_random_starts,
)
from classes.multi_robot.evaluation_recorder import EpisodeMetricsRecorder
from collision.joint_resolver import resolve_synchronous_moves
from classes.multi_robot.visualizer import MultiRobotVisualizer
from mergingmap.baselines.merged_map_only.baseline_worker_artifacts import BaselineArtifactMixin
from mergingmap.baselines.merged_map_only.baseline_worker_dare_runtime import BaselineDareRuntimeMixin
from mergingmap.baselines.merged_map_only.baseline_worker_episode import BaselineEpisodeMixin
from mergingmap.baselines.merged_map_only.baseline_worker_map_sync import BaselineMapSyncMixin
from mergingmap.baselines.merged_map_only.multi_test_parameter import (
    ACTION_HORIZON_OVERRIDE,
    ALLOW_SHARED_DEPOT_AT_STEP_ZERO,
    CONTACT_HOPS,
    DIRECTION_LOOKAHEAD,
    DEADLOCK_WAIT_THRESHOLD,
    ENABLE_METRIC_RECORDING,
    GIF_FRAME_DURATION,
    GHOST_MODE,
    MAP_CONFLICT_POLICY,
    MAP_DEBUG,
    MAP_DEBUG_INTERVAL,
    MAP_MAX_CELLS_PER_PACKET,
    METRIC_COVERAGE_THRESHOLDS,
    MAX_MULTI_ROBOT_STEPS,
    MIN_START_SEPARATION,
    REQUIRE_LINE_OF_SIGHT,
    RESET_OBS_HISTORY_ON_MAP_MERGE,
    RANDOM_DISTINCT_STARTS,
    SAFE_DISTANCE,
    START_CLEARANCE_RADIUS_CELLS,
    START_SAMPLE_MAX_ATTEMPTS,
    SAVE_VISUALISATION,
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
    BaselineArtifactMixin,
    BaselineDareRuntimeMixin,
    BaselineMapSyncMixin,
    BaselineEpisodeMixin,
):
    """Multi-robot DARE using contact-triggered merged occupancy beliefs.

    This worker intentionally ignores the previous route penalties, deadlock
    recovery, anti-oscillation logic, trail avoidance, goal claims, and joint
    backtracking. The experiment isolates the teacher-proposed question:

        Does correctly merging robot knowledge improve each robot's DARE graph
        and exploration performance?

    DARE remains frozen and unmodified. Each robot has its own Agent and
    NodeManager, but the map passed into Agent.update_graph() is its persistent
    merged belief rather than its private local belief.
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
            ),
        )

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
        self.method_name = "LiteDARE-MM"
        output_root = Path(
            os.environ.get("MERGINGMAP_RUN_DIR", VISUAL_OUTPUT_ROOT)
        )
        self.metric_recorder = None
        if ENABLE_METRIC_RECORDING:
            self.metric_recorder = EpisodeMetricsRecorder(
                output_root=output_root,
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
        if SAVE_VISUALISATION:
            self.visualizer = MultiRobotVisualizer(
                output_root=VISUAL_OUTPUT_ROOT,
                episode_index=self.episode_index,
                team_size=self.n_agents,
                seed=self.seed,
                frame_stride=VISUAL_FRAME_STRIDE,
                background_mode=VISUAL_BACKGROUND_MODE,
            )

        self._initialise_robot_states()
        if self.visualizer is not None:
            self.visualizer.save_frame(
                env=self.env,
                trajectories=self.trajectories,
                step=0,
                contact_pairs=(),
                team_coverage=self.env.explored_rate,
            )

