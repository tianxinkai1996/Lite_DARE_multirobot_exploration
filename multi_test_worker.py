"""Main multi-robot evaluation worker for DARE/LiteDARE team exploration."""

from __future__ import annotations

import collections
import json
import os
import random
import time
import contextlib
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
if torch.cuda.is_available():
    try:
        torch.cuda.set_per_process_memory_fraction(0.3, device=0)
    except (AssertionError, RuntimeError):
        pass

from parameter import FREE, OCCUPIED, UNKNOWN, NODE_RESOLUTION
from test_parameter import DATA_TYPE, USE_DELTA_POSITION, USE_TEST_DATASET
from classes.agent.agent import Agent
from classes.agent.node_manager import NodeManager
from classes.env.multi_robot_env import MultiRobotEnv
from classes.multi_robot.contact_model import ContactModel
from classes.multi_robot.coverage_manager import CoverageManager
from classes.multi_robot.evaluation_recorder import EpisodeMetricsRecorder
from classes.multi_robot.runtime_metrics import DetailedRuntimeMetrics
from classes.multi_robot.dare_test_profiles import (
    DareTestProfile,
    resolve_dare_test_profile,
)
from mergingmap.motion_coordinator import MergingMapMotionCoordinator
from collision.reservation_manager import ReservationManager
from classes.multi_robot.trajectory_codec import make_compressed_packet, make_raw_packet
from classes.multi_robot.visualizer import MultiRobotVisualizer
from multi_test_parameter import (
    ACTION_HORIZON_OVERRIDE,
    ALLOW_SHARED_DEPOT_AT_STEP_ZERO,
    CUDA_SYNCHRONIZE_FOR_TIMING,
    CACHE_TTL_STEPS,
    CONTACT_HOPS,
    DIRECTION_LOOKAHEAD,
    GIF_FRAME_DURATION,
    MAX_MULTI_ROBOT_STEPS,
    MIN_START_SEPARATION,
    START_CLEARANCE,
    PACKET_BUDGET_BYTES,
    PLAN_SHARE_STEPS,
    REQUIRE_LINE_OF_SIGHT,
    RESERVATION_HORIZON,
    SAFE_DISTANCE,
    SAVE_VISUALISATION,
    TEAM_DONE_TOLERANCE_CELLS,
    TRAIL_AVOID_RADIUS,
    TRAIL_PENALTY_WEIGHT,
    TRAIL_SHARE_STEPS,
    VISUAL_BACKGROUND_MODE,
    VISUAL_FRAME_STRIDE,
    VISUAL_OUTPUT_ROOT,
    ENABLE_COVERAGE_EXCHANGE,
    COVERAGE_TILE_SIZE,
    COVERAGE_THRESHOLD,
    MAX_COVERAGE_DELTA_TILES,
    COVERAGE_PENALTY_WEIGHT,
    GOAL_CLAIM_PENALTY_WEIGHT,
    GOAL_CLAIM_RADIUS_TILES,
    GOAL_CLAIM_TTL_STEPS,
    DEADLOCK_WAIT_THRESHOLD,
    DEADLOCK_MAX_BACKTRACKING_NODES,
    ENABLE_METRIC_RECORDING,
    ENABLE_DETAILED_RUNTIME_METRICS,
    METRIC_COVERAGE_THRESHOLDS,
    PROFILE_POLICY_FLOPS_ONCE,
    TRACK_HARDWARE_ENERGY,
    TRACK_PYTHON_MEMORY,
    DARE_CHECKPOINT_PATH,
)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class MultiRobotTestWorker:
    """Runs one multi-robot episode with a frozen, unmodified DARE policy."""

    def __init__(self, policy, episode_index, n_agents, device, seed, communication_mode="compressed", start_position=None, start_positions=None, start_sample=0, visual_output_root=None, test_profile="coordinated", scenario_id=None, trial=0, save_visualisation=None):
        if DATA_TYPE != "node":
            raise NotImplementedError(
                "This patch intentionally supports the official node-based DARE path. "
                "Set DATA_TYPE = 'node' in test_parameter.py."
            )
        if communication_mode not in {"none", "raw", "compressed"}:
            raise ValueError("communication_mode must be 'none', 'raw', or 'compressed'")

        set_random_seed(seed)
        self.policy = policy
        self.policy.eval()
        self.episode_index = int(episode_index)
        self.n_agents = int(n_agents)
        self.device = torch.device(device)
        self.seed = int(seed)
        self.communication_mode = communication_mode
        self.test_profile = resolve_dare_test_profile(test_profile)
        self.start_sample = int(start_sample)
        self.trial = int(trial)
        self.scenario_id = str(
            scenario_id
            or f"map_{self.episode_index:04d}_trial_{self.trial:02d}_"
            f"robots_{self.n_agents:02d}_seed_{self.seed}"
        )
        self.visual_output_root = str(
            Path(visual_output_root or VISUAL_OUTPUT_ROOT).expanduser().resolve()
        )
        self.save_visualisation = (
            bool(SAVE_VISUALISATION)
            if save_visualisation is None
            else bool(save_visualisation)
        )
        self.obs_horizon = int(policy.n_obs_steps)
        self.action_horizon = int(ACTION_HORIZON_OVERRIDE)
        if self.action_horizon != 1:
            raise ValueError("This multi-robot patch requires exactly one executed action per replan.")

        if start_position is not None and start_positions is not None:
            raise ValueError("supply either start_position or start_positions, not both")

        explicit_starts = None
        allow_shared_start = False
        if start_position is not None:
            depot = np.asarray(start_position, dtype=np.float32)
            if depot.shape != (2,):
                raise ValueError(f"start_position must have shape (2,), got {depot.shape}")
            explicit_starts = np.repeat(depot[None, :], self.n_agents, axis=0)
            allow_shared_start = True
        elif start_positions is not None:
            explicit_starts = np.asarray(start_positions, dtype=np.float32)
            if explicit_starts.shape != (self.n_agents, 2):
                raise ValueError(
                    f"start_positions must have shape {(self.n_agents, 2)}, "
                    f"got {explicit_starts.shape}"
                )
            allow_shared_start = (
                len({tuple(np.round(point, 5)) for point in explicit_starts}) == 1
            )

        effective_start_clearance = (
            0.0 if start_positions is not None else START_CLEARANCE
        )
        self.effective_start_clearance = float(effective_start_clearance)
        self.env = MultiRobotEnv(
            episode_index=self.episode_index,
            n_agents=self.n_agents,
            test=USE_TEST_DATASET,
            seed=self.seed,
            min_start_separation=MIN_START_SEPARATION,
            done_tolerance_cells=TEAM_DONE_TOLERANCE_CELLS,
            start_positions=explicit_starts,
            allow_shared_start=allow_shared_start,
            start_clearance=effective_start_clearance,
        )
        if self.n_agents not in {1, 2, 4, 6, 8}:
            raise ValueError("n_agents must be one of 1, 2, 4, 6, or 8")
        self.initial_positions = self.env.robot_locations.copy()
        if start_position is not None and len(
            {tuple(np.round(point, 5)) for point in self.initial_positions}
        ) != 1:
            raise RuntimeError("shared-depot start_position was not preserved")

        # Critical: one NodeManager per robot. The original TestWorker shares one.
        self.robot_list = [
            Agent(robot_id, NodeManager(plot=False), self.device, plot=False)
            for robot_id in range(self.n_agents)
        ]
        self.reservations = [
            ReservationManager(
                robot_id,
                node_resolution=NODE_RESOLUTION,
                reservation_horizon=RESERVATION_HORIZON,
                cache_ttl_steps=CACHE_TTL_STEPS,
                safe_distance=SAFE_DISTANCE,
                trail_avoid_radius=TRAIL_AVOID_RADIUS,
                trail_penalty_weight=TRAIL_PENALTY_WEIGHT,
            )
            for robot_id in range(self.n_agents)
        ]
        # Sparse coverage/goal memories reduce repeated exploration. They are
        # outside DARE and never modify the local belief map or observation.
        self.coverage_managers = [
            CoverageManager(
                robot_id,
                tile_size=COVERAGE_TILE_SIZE,
                coverage_threshold=COVERAGE_THRESHOLD,
                max_delta_tiles=MAX_COVERAGE_DELTA_TILES,
                coverage_penalty_weight=COVERAGE_PENALTY_WEIGHT,
                goal_penalty_weight=GOAL_CLAIM_PENALTY_WEIGHT,
                goal_claim_radius_tiles=GOAL_CLAIM_RADIUS_TILES,
                goal_claim_ttl_steps=GOAL_CLAIM_TTL_STEPS,
            )
            for robot_id in range(self.n_agents)
        ]
        if self.test_profile.enable_coverage_exchange:
            for robot_id in range(self.n_agents):
                self.coverage_managers[robot_id].update_from_local_belief(
                    local_belief=self.env.get_local_belief(robot_id),
                    env=self.env,
                    robot_position=self.env.robot_locations[robot_id],
                    sensor_range=self.env.sensor_range,
                )

        self.contact_model = ContactModel(CONTACT_HOPS, REQUIRE_LINE_OF_SIGHT)
        self.trajectories: List[List[np.ndarray]] = [
            [self.env.robot_locations[robot_id].copy()] for robot_id in range(self.n_agents)
        ]
        self.obs_deques: List[collections.deque] = []
        self.metrics: Dict[str, float | int | str | bool] = {}
        self.contact_events = 0
        self.packets_sent = 0
        self.total_bytes_sent = 0
        self.communication_encode_ms = 0.0
        self.communication_decode_apply_ms = 0.0
        self.communication_exchange_wall_ms = 0.0
        self.communication_delivery_events = 0
        self.communication_raw_equivalent_bytes = 0
        self.communication_metadata_bytes = 0
        self.communication_trail_payload_bytes = 0
        self.communication_plan_payload_bytes = 0
        self.communication_coverage_payload_bytes = 0
        self.communication_goal_payload_bytes = 0
        self.communication_trail_points = 0
        self.communication_plan_points = 0
        self.communication_coverage_tiles = 0
        self.communication_goal_claims = 0
        self.communication_budget_truncated_packets = 0
        self.communication_dropped_trail_points = 0
        self.communication_dropped_plan_points = 0
        self.communication_dropped_coverage_tiles = 0
        self.communication_dropped_goal_claims = 0
        self.dynamic_blocks = 0
        self.static_blocks = 0
        self.motion_coordinator = MergingMapMotionCoordinator(
            self.n_agents,
            safe_distance=SAFE_DISTANCE,
            deadlock_wait_threshold=DEADLOCK_WAIT_THRESHOLD,
            mode=self.test_profile.coordination_mode,
            max_backtracking_nodes=DEADLOCK_MAX_BACKTRACKING_NODES,
        )
        self.deadlock_escape_activations = 0
        self.resolver_backtracking_nodes = 0
        self.method_name = self.test_profile.method_name
        self.metric_recorder = None
        if ENABLE_METRIC_RECORDING:
            self.metric_recorder = EpisodeMetricsRecorder(
                output_root=self.visual_output_root,
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
                output_root=self.visual_output_root,
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

    def _measure_runtime(self, name, cuda_sync=False):
        """Return a detailed timer or a no-op context.

        English implementation: delegates to ``DetailedRuntimeMetrics`` when
        enabled and otherwise returns ``nullcontext`` without changing actions.
        """

        if self.runtime_metrics is None:
            return contextlib.nullcontext()
        return self.runtime_metrics.measure(name, cuda_sync=cuda_sync)

    def _communication_cumulative(self):
        """Return monotonic communication counters in the shared schema.

        English implementation: exposes cumulative counters for per-step deltas
        and episode-level communication summaries.
        """

        return {
            "communication_packets": int(self.packets_sent),
            "communication_payload_bytes": int(self.total_bytes_sent),
            "communication_encode_ms": float(self.communication_encode_ms),
            "communication_decode_apply_ms": float(
                self.communication_decode_apply_ms
            ),
            "communication_exchange_wall_ms": float(
                self.communication_exchange_wall_ms
            ),
            "communication_delivery_events": int(
                self.communication_delivery_events
            ),
            "communication_raw_equivalent_bytes": int(
                self.communication_raw_equivalent_bytes
            ),
            "communication_metadata_bytes": int(self.communication_metadata_bytes),
            "communication_trail_payload_bytes": int(
                self.communication_trail_payload_bytes
            ),
            "communication_plan_payload_bytes": int(
                self.communication_plan_payload_bytes
            ),
            "communication_coverage_payload_bytes": int(
                self.communication_coverage_payload_bytes
            ),
            "communication_goal_payload_bytes": int(
                self.communication_goal_payload_bytes
            ),
            "communication_trail_points": int(self.communication_trail_points),
            "communication_plan_points": int(self.communication_plan_points),
            "communication_coverage_tiles": int(
                self.communication_coverage_tiles
            ),
            "communication_goal_claims": int(self.communication_goal_claims),
            "communication_budget_truncated_packets": int(
                self.communication_budget_truncated_packets
            ),
            "communication_dropped_trail_points": int(
                self.communication_dropped_trail_points
            ),
            "communication_dropped_plan_points": int(
                self.communication_dropped_plan_points
            ),
            "communication_dropped_coverage_tiles": int(
                self.communication_dropped_coverage_tiles
            ),
            "communication_dropped_goal_claims": int(
                self.communication_dropped_goal_claims
            ),
        }

    # ------------------------------------------------------------------
    # Original-Dare observation/inference helpers
    # ------------------------------------------------------------------

    def _update_robot_graph(self, robot_id):
        robot = self.robot_list[robot_id]
        own_position = self.env.robot_locations[robot_id].copy()
        robot.update_graph(self.env.get_local_belief_info(robot_id), own_position)

        # Critical: pass only the robot's own location. Passing env.robot_locations
        # would encode remote teammates in the original node occupancy feature.
        robot.update_planning_state(np.asarray([own_position], dtype=np.float32))

    def _make_node_observation(self, robot_id):
        observation = self.robot_list[robot_id].get_observation()
        return {
            "node_inputs": observation[0].squeeze(0),
            "node_padding_mask": observation[1].squeeze(0),
            "edge_mask": observation[2].squeeze(0),
            "current_index": observation[3].squeeze(0),
            "current_edge": observation[4].squeeze(0),
            "edge_padding_mask": observation[5].squeeze(0),
        }

    def _initialise_robot_states(self):
        self.obs_deques = []
        for robot_id in range(self.n_agents):
            self._update_robot_graph(robot_id)
            obs = self._make_node_observation(robot_id)
            self.obs_deques.append(collections.deque([obs] * self.obs_horizon, maxlen=self.obs_horizon))

    def _obs_dict(self, robot_id):
        obs_deque = self.obs_deques[robot_id]
        node_inputs = torch.stack([x["node_inputs"] for x in obs_deque]).to(self.device, dtype=torch.float32)
        node_padding_mask = torch.stack([x["node_padding_mask"] for x in obs_deque]).to(self.device, dtype=torch.int16)
        edge_mask = torch.stack([x["edge_mask"] for x in obs_deque]).to(self.device, dtype=torch.int64)
        current_index = torch.stack([x["current_index"] for x in obs_deque]).to(self.device, dtype=torch.int64)
        current_edge = torch.stack([x["current_edge"] for x in obs_deque]).to(self.device, dtype=torch.int64)
        edge_padding_mask = torch.stack([x["edge_padding_mask"] for x in obs_deque]).to(self.device, dtype=torch.int16)
        return {
            "node_inputs": node_inputs.unsqueeze(0),
            "node_padding_mask": node_padding_mask.unsqueeze(0),
            "edge_mask": edge_mask.unsqueeze(0),
            "current_index": current_index.unsqueeze(0),
            "current_edge": current_edge.unsqueeze(0),
            "edge_padding_mask": edge_padding_mask.unsqueeze(0),
        }

    def _predict_action_sequence(self, robot_id):
        with torch.inference_mode():
            action_dict = self.policy.predict_action(self._obs_dict(robot_id))
        action_pred = action_dict["action_pred"].squeeze(0).detach().cpu().numpy()
        return np.round(action_pred / NODE_RESOLUTION) * NODE_RESOLUTION

    def _predicted_world_plan(self, robot_id, action_pred):
        start = self.obs_horizon - 1
        position = self.env.robot_locations[robot_id].copy()
        plan: List[np.ndarray] = []
        for action in action_pred[start:]:
            position = position + action if USE_DELTA_POSITION else action.copy()
            plan.append(np.asarray(position, dtype=np.float32).copy())
        return plan

    def _ordered_dare_candidates(self, robot_id, action_pred):
        """Rank legal local graph neighbours by agreement with DARE's short plan."""
        robot = self.robot_list[robot_id]
        current = self.env.robot_locations[robot_id].copy()
        node_record = robot.node_manager.nodes_dict.find(current.tolist())
        if node_record is None:
            return [current]
        current_node = node_record.data

        candidates: List[np.ndarray] = []
        seen = set()
        for neighbour in current_node.neighbor_list:
            candidate = np.asarray(neighbour, dtype=np.float32)
            key = tuple(np.round(candidate, 4))
            if key not in seen and not np.allclose(candidate, current):
                candidates.append(candidate)
                seen.add(key)

        start = self.obs_horizon - 1
        raw = action_pred[start : start + DIRECTION_LOOKAHEAD]
        if USE_DELTA_POSITION:
            direction_vectors = np.cumsum(raw, axis=0)
        else:
            direction_vectors = raw - current[None, :]

        def score(candidate):
            movement = candidate - current
            movement_norm = float(np.linalg.norm(movement))
            if movement_norm <= 1e-8:
                return 1e9
            angles = []
            weights = []
            for index, vector in enumerate(direction_vectors):
                vector_norm = float(np.linalg.norm(vector))
                if vector_norm <= 1e-8:
                    continue
                cosine = float(np.clip(np.dot(movement, vector) / (movement_norm * vector_norm), -1.0, 1.0))
                angles.append(float(np.arccos(cosine)))
                weights.append(len(direction_vectors) - index)
            return float(np.average(angles, weights=weights)) if angles else 1e8

        candidates.sort(key=score)
        # Waiting is the final safe fallback.
        candidates.append(current.copy())
        return candidates

    # ------------------------------------------------------------------
    # Contact messages and coordinated action selection
    # ------------------------------------------------------------------

    def _exchange_messages(self, step, contact_pairs, dare_plans):
        visible: Dict[int, List[np.ndarray]] = {robot_id: [] for robot_id in range(self.n_agents)}
        if not self.test_profile.enable_messages:
            return visible

        # Local visibility is available whenever the contact model fires, even in
        # the no-message baseline. This models a nearby robot as a local dynamic
        # obstacle without sharing maps or trajectories.
        for robot_i, robot_j in contact_pairs:
            visible[robot_i].append(self.env.robot_locations[robot_j].copy())
            visible[robot_j].append(self.env.robot_locations[robot_i].copy())

        if self.communication_mode == "none":
            return visible

        for robot_i, robot_j in contact_pairs:
            exchange_started_at = time.perf_counter()
            factory = make_compressed_packet if self.communication_mode == "compressed" else make_raw_packet
            common = {
                "step": step,
                "trail_steps": TRAIL_SHARE_STEPS,
                "plan_steps": PLAN_SHARE_STEPS,
            }
            if self.communication_mode == "compressed":
                common["node_resolution"] = NODE_RESOLUTION
                common["packet_budget_bytes"] = PACKET_BUDGET_BYTES

            # Coverage deltas and goal claims are sparse coordination summaries.
            # They are included only in contact messages and are not inserted into
            # DARE's observation encoder.
            if self.test_profile.enable_coverage_exchange:
                coverage_i = self.coverage_managers[robot_i].coverage_delta_for_peer(robot_j)
                coverage_j = self.coverage_managers[robot_j].coverage_delta_for_peer(robot_i)
                goal_i = self.coverage_managers[robot_i].goal_claim_from_plan(dare_plans[robot_i])
                goal_j = self.coverage_managers[robot_j].goal_claim_from_plan(dare_plans[robot_j])
            else:
                coverage_i, coverage_j = [], []
                goal_i, goal_j = None, None

            encode_started_at = time.perf_counter()
            packet_i = factory(
                sender_id=robot_i,
                current_position=self.env.robot_locations[robot_i],
                trail=self.trajectories[robot_i],
                plan=dare_plans[robot_i],
                coverage_tiles=coverage_i,
                goal_tile=goal_i,
                **common,
            )
            packet_j = factory(
                sender_id=robot_j,
                current_position=self.env.robot_locations[robot_j],
                trail=self.trajectories[robot_j],
                plan=dare_plans[robot_j],
                coverage_tiles=coverage_j,
                goal_tile=goal_j,
                **common,
            )
            self.communication_encode_ms += (
                time.perf_counter() - encode_started_at
            ) * 1000.0

            for packet in (packet_i, packet_j):
                self.communication_raw_equivalent_bytes += int(
                    packet.get("raw_equivalent_byte_count", packet.get("byte_count", 0))
                )
                self.communication_metadata_bytes += int(
                    packet.get("metadata_bytes", 0)
                )
                self.communication_trail_payload_bytes += int(
                    packet.get("trail_payload_bytes", 0)
                )
                self.communication_plan_payload_bytes += int(
                    packet.get("plan_payload_bytes", 0)
                )
                self.communication_coverage_payload_bytes += int(
                    packet.get("coverage_payload_bytes", 0)
                )
                self.communication_goal_payload_bytes += int(
                    packet.get("goal_payload_bytes", 0)
                )
                self.communication_trail_points += int(
                    packet.get("trail_point_count", 0)
                )
                self.communication_plan_points += int(
                    packet.get("plan_point_count", 0)
                )
                self.communication_coverage_tiles += int(
                    packet.get("coverage_tile_count", 0)
                )
                self.communication_goal_claims += int(
                    packet.get("goal_claim_count", 0)
                )
                self.communication_budget_truncated_packets += int(
                    bool(packet.get("packet_budget_truncated", False))
                )
                self.communication_dropped_trail_points += int(
                    packet.get("dropped_trail_points", 0)
                )
                self.communication_dropped_plan_points += int(
                    packet.get("dropped_plan_points", 0)
                )
                self.communication_dropped_coverage_tiles += int(
                    packet.get("dropped_coverage_tiles", 0)
                )
                self.communication_dropped_goal_claims += int(
                    packet.get("dropped_goal_claims", 0)
                )

            # Reservation managers use short plans for hard collision avoidance.
            receive_started_at = time.perf_counter()
            self.reservations[robot_j].receive_packet(packet_i)
            self.reservations[robot_i].receive_packet(packet_j)

            # Coverage managers use sparse explored tiles and goal claims for
            # reducing repeated exploration.
            if self.test_profile.enable_coverage_exchange:
                self.coverage_managers[robot_j].receive_packet(packet_i, step, node_resolution=NODE_RESOLUTION)
                self.coverage_managers[robot_i].receive_packet(packet_j, step, node_resolution=NODE_RESOLUTION)
                self.coverage_managers[robot_i].mark_tiles_sent(robot_j, packet_i.get("sent_coverage_tiles", []))
                self.coverage_managers[robot_j].mark_tiles_sent(robot_i, packet_j.get("sent_coverage_tiles", []))
            self.communication_decode_apply_ms += (
                time.perf_counter() - receive_started_at
            ) * 1000.0

            self.packets_sent += 2
            self.total_bytes_sent += int(packet_i["byte_count"]) + int(packet_j["byte_count"])
            self.communication_delivery_events += 2
            self.communication_exchange_wall_ms += (
                time.perf_counter() - exchange_started_at
            ) * 1000.0

        return visible

    def _update_next_observations(self):
        for robot_id in range(self.n_agents):
            self._update_robot_graph(robot_id)
            self.obs_deques[robot_id].append(self._make_node_observation(robot_id))

    # ------------------------------------------------------------------
    # Main episode loop
    # ------------------------------------------------------------------

    def run_episode(self):
        """Execute one original-DARE reference episode with full instrumentation.

        English implementation: times each major stage, records in-process message
        costs, samples resources, and preserves the original planning behaviour.
        """

        last_step = 0
        for step in range(MAX_MULTI_ROBOT_STEPS):
            last_step = step
            fallback_step_started_at = time.perf_counter()
            if self.runtime_metrics is not None:
                self.runtime_metrics.start_step()

            with self._measure_runtime(
                "policy_inference_team",
                cuda_sync=CUDA_SYNCHRONIZE_FOR_TIMING,
            ):
                action_predictions = [
                    self._predict_action_sequence(robot_id)
                    for robot_id in range(self.n_agents)
                ]
            with self._measure_runtime("predicted_plan_conversion"):
                dare_plans = [
                    self._predicted_world_plan(
                        robot_id,
                        action_predictions[robot_id],
                    )
                    for robot_id in range(self.n_agents)
                ]

            with self._measure_runtime("contact_detection"):
                contact_pairs = self.contact_model.get_contact_pairs(self.env)
            self.contact_events += len(contact_pairs)
            with self._measure_runtime("communication_exchange"):
                visible = self._exchange_messages(
                    step=step,
                    contact_pairs=contact_pairs,
                    dare_plans=dare_plans,
                )

            # A starving robot receives temporary escape priority. Physical
            # collision constraints remain active in the joint resolver.
            with self._measure_runtime("deadlock_escape_preselection"):
                escape_robot_ids = self.motion_coordinator.escape_robot_ids()
            self.deadlock_escape_activations += len(escape_robot_ids)

            candidate_lists = []
            preferred_positions = []
            with self._measure_runtime("candidate_generation_and_reservation_filter"):
                for robot_id in range(self.n_agents):
                    ordered = self._ordered_dare_candidates(
                        robot_id,
                        action_predictions[robot_id],
                    )
                    preferred_positions.append(
                        ordered[0].copy()
                        if ordered
                        else self.env.robot_locations[robot_id].copy()
                    )
                    escape_mode = robot_id in escape_robot_ids
                    if self.test_profile.enable_reservations:
                        safe_candidates = self.reservations[robot_id].filter_candidates(
                            current_position=self.env.robot_locations[robot_id],
                            ordered_candidates=ordered,
                            current_step=step,
                            visible_peer_positions=visible[robot_id],
                            coverage_manager=(
                                None
                                if escape_mode
                                or not self.test_profile.enable_coverage_exchange
                                else self.coverage_managers[robot_id]
                            ),
                            ignore_peer_reservations=escape_mode,
                            ignore_soft_costs=escape_mode,
                        )
                    else:
                        safe_candidates = [candidate.copy() for candidate in ordered]
                    candidate_lists.append(safe_candidates)

            previous_positions = [
                position.copy() for position in self.env.robot_locations
            ]
            with self._measure_runtime("motion_coordination_facade"):
                motion_decision = self.motion_coordinator.resolve_step(
                    previous_positions,
                    candidate_lists,
                    time_step=step,
                    allow_shared_start_step0=ALLOW_SHARED_DEPOT_AT_STEP_ZERO,
                )
            next_positions = [
                position.copy() for position in motion_decision.next_positions
            ]
            dynamically_blocked = list(motion_decision.blocked_robot_ids)
            resolution_info = motion_decision.resolution_info
            self.dynamic_blocks += len(dynamically_blocked)
            if resolution_info is not None:
                self.resolver_backtracking_nodes += resolution_info.backtracking_nodes

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
            with self._measure_runtime("trajectory_update"):
                for robot_id in range(self.n_agents):
                    self.trajectories[robot_id].append(
                        actual_positions[robot_id].copy()
                    )

            # Update each robot's own sparse coverage summary after local sensing.
            if self.test_profile.enable_coverage_exchange:
                with self._measure_runtime("coverage_summary_update"):
                    for robot_id in range(self.n_agents):
                        self.coverage_managers[robot_id].update_from_local_belief(
                            local_belief=self.env.get_local_belief(robot_id),
                            env=self.env,
                            robot_position=actual_positions[robot_id],
                            sensor_range=self.env.sensor_range,
                        )

            with self._measure_runtime("observation_graph_refresh"):
                self._update_next_observations()

            with self._measure_runtime("metric_state_snapshot"):
                current_coverage = float(self.env.explored_rate)
                current_known_free = EpisodeMetricsRecorder.known_free_cell_count(
                    self.env, FREE
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
                if resolution_info is not None:
                    for field_name in (
                        "preferred_action_selection_ms",
                        "deadlock_priority_ms",
                        "deadlock_escape_ms",
                        "collision_resolution_ms",
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
                team_inference = float(
                    step_profile.get("policy_inference_team_ms", 0.0) or 0.0
                )
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
                    map_packets_cumulative=int(self.packets_sent),
                    map_bytes_cumulative=int(self.total_bytes_sent),
                    communication_cumulative=self._communication_cumulative(),
                    resolver_info=resolution_info,
                    step_runtime_ms=step_runtime_ms,
                    extra_step_metrics=step_profile,
                )

            if self.env.done:
                break

        gif_path = None
        if self.visualizer is not None:
            with self._measure_runtime("visualization_gif_build"):
                gif_path = self.visualizer.build_gif(GIF_FRAME_DURATION)

        per_robot_distance = [
            float(robot.travel_dist) for robot in self.robot_list
        ]
        runtime_episode_metrics = (
            {} if self.runtime_metrics is None else self.runtime_metrics.finalise()
        )
        map_quality_metrics = EpisodeMetricsRecorder.map_quality_metrics(
            self.env.robot_beliefs,
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
        recorded_metrics = {}
        if self.metric_recorder is not None:
            recorded_metrics = self.metric_recorder.finalise(
                success=bool(self.env.done),
                extra_metrics={
                    "scenario_id": self.scenario_id,
                    "trial": self.trial,
                    "paper_method_key": self.test_profile.key,
                    "dare_test_profile": self.test_profile.as_dict(),
                    "start_positions": json.dumps(
                        [
                            [float(position[0]), float(position[1])]
                            for position in self.initial_positions
                        ]
                    ),
                    "packets_sent": int(self.packets_sent),
                    "total_bytes_sent": int(self.total_bytes_sent),
                    "communication_transport": "in_process_simulation",
                    "physical_network_latency_measured": False,
                    "communication_encode_total_ms": float(
                        self.communication_encode_ms
                    ),
                    "communication_decode_apply_total_ms": float(
                        self.communication_decode_apply_ms
                    ),
                    "communication_exchange_wall_total_ms": float(
                        self.communication_exchange_wall_ms
                    ),
                    **map_quality_metrics,
                    **map_structure_metrics,
                    **runtime_episode_metrics,
                },
            )
            per_robot_distance = [
                float(value)
                for value in self.metric_recorder.path_lengths.tolist()
            ]

        visited = {}
        total_visits = 0
        for robot_id, trajectory in enumerate(self.trajectories):
            for point in trajectory:
                key = tuple(
                    np.rint(np.asarray(point) / NODE_RESOLUTION).astype(int)
                )
                visited.setdefault(key, set()).add(robot_id)
                total_visits += 1
        duplicate_nodes = sum(
            max(0, len(robot_ids) - 1)
            for robot_ids in visited.values()
        )
        duplicate_ratio = (
            0.0 if total_visits == 0 else float(duplicate_nodes / total_visits)
        )

        self.metrics = {
            "episode": self.episode_index,
            "method": self.method_name,
            "seed": self.seed,
            "team_size": self.n_agents,
            "map_index": self.episode_index,
            "start_sample": self.start_sample,
            "shared_start": (
                f"({float(self.initial_positions[0, 0]):.3f},"
                f"{float(self.initial_positions[0, 1]):.3f})"
            ),
            "start_positions": json.dumps(
                [
                    [float(point[0]), float(point[1])]
                    for point in self.initial_positions
                ]
            ),
            "communication_mode": self.communication_mode,
            "effective_communication_mode": (
                self.communication_mode
                if self.test_profile.enable_messages
                else "none"
            ),
            "communication_transport": "in_process_simulation",
            "physical_network_latency_measured": False,
            "scenario_id": self.scenario_id,
            "trial": self.trial,
            "paper_method_key": self.test_profile.key,
            "dare_test_profile": json.dumps(
                self.test_profile.as_dict(), sort_keys=True
            ),
            "start_clearance": self.effective_start_clearance,
            "steps": last_step + 1,
            "success": bool(self.env.done),
            "team_coverage": float(self.env.explored_rate),
            "team_travel_distance": float(sum(per_robot_distance)),
            "max_robot_travel_distance": float(
                max(per_robot_distance, default=0.0)
            ),
            "mean_robot_travel_distance": float(
                np.mean(per_robot_distance) if per_robot_distance else 0.0
            ),
            "duplicate_node_ratio": duplicate_ratio,
            "mean_local_coverage_tiles": float(
                np.mean(
                    [
                        len(manager.local_covered_tiles)
                        for manager in self.coverage_managers
                    ]
                )
            ),
            "mean_peer_coverage_tiles": float(
                np.mean(
                    [
                        len(manager.peer_covered_union())
                        for manager in self.coverage_managers
                    ]
                )
            ),
            "contact_events": int(self.contact_events),
            "packets_sent": int(self.packets_sent),
            "total_bytes_sent": int(self.total_bytes_sent),
            "mean_packet_bytes": (
                0.0
                if self.packets_sent == 0
                else float(self.total_bytes_sent / self.packets_sent)
            ),
            "communication_encode_total_ms": float(
                self.communication_encode_ms
            ),
            "communication_decode_apply_total_ms": float(
                self.communication_decode_apply_ms
            ),
            "communication_exchange_wall_total_ms": float(
                self.communication_exchange_wall_ms
            ),
            "communication_delivery_events": int(
                self.communication_delivery_events
            ),
            "dynamic_safety_blocks": int(self.dynamic_blocks),
            "static_safety_blocks": int(self.static_blocks),
            "deadlock_escape_activations": int(
                self.deadlock_escape_activations
            ),
            "deadlock_break_events": int(
                self.motion_coordinator.deadlock_break_events
            ),
            "max_consecutive_wait_steps": int(
                self.motion_coordinator.max_wait_steps
            ),
            "resolver_backtracking_nodes": int(
                self.resolver_backtracking_nodes
            ),
            **self.motion_coordinator.metrics(),
            **map_quality_metrics,
            **map_structure_metrics,
            **runtime_episode_metrics,
            **recorded_metrics,
            "gif_path": "" if gif_path is None else gif_path,
        }
        return self.metrics
