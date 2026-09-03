"""Frozen-DARE graph and inference helpers for MergingMap.

MergingMap DARE 
"""
from __future__ import annotations

import collections
from typing import List

import numpy as np
import torch

from parameter import NODE_RESOLUTION
from test_parameter import USE_DELTA_POSITION
from mergingmap.multi_test_parameter import (
    DIRECTION_LOOKAHEAD,
    RESET_OBS_HISTORY_ON_MAP_MERGE,
)

class WorkerDareRuntimeMixin:
    """Build merged-map graphs and run the unchanged DARE policy.

    English: groups graph observation, temporal stacking, inference, and candidate ranking.
    """

    # English purpose: Adapt a robot merged belief to the map-info object expected by DARE.
    def _merged_map_info(self, robot_id):
        template = self.env.get_local_belief_info(robot_id)
        return self.map_merger.make_map_info(robot_id, template)

    # English purpose: Rebuild one robot exploration graph from its merged belief.
    def _update_robot_graph(self, robot_id):
        robot = self.robot_list[robot_id]
        own_position = self.env.robot_locations[robot_id].copy()

        # This is the central change: DARE builds its informative graph from the
        # merged map, not from only the robot's private sensor belief.
        robot.update_graph(
            self._merged_map_info(robot_id),
            own_position,
        )

        # Keep the original single-robot occupancy feature distribution. Other
        # robots are not injected into DARE's observation.
        robot.update_planning_state(
            np.asarray([own_position], dtype=np.float32)
        )

    # English purpose: Convert an Agent graph observation into policy input fields.
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

    # English purpose: Initialise graph state and observation history for every robot.
    def _initialise_robot_states(self):
        self.obs_deques = []
        for robot_id in range(self.n_agents):
            self._update_robot_graph(robot_id)
            obs = self._make_node_observation(robot_id)
            self.obs_deques.append(
                collections.deque(
                    [obs] * self.obs_horizon,
                    maxlen=self.obs_horizon,
                )
            )

    # English purpose: Refresh graph observations and optionally reset history after map fusion.
    def _refresh_observations(self, merged_changed_robots):
        for robot_id in range(self.n_agents):
            self._update_robot_graph(robot_id)
            obs = self._make_node_observation(robot_id)
            if (
                RESET_OBS_HISTORY_ON_MAP_MERGE
                and robot_id in merged_changed_robots
            ):
                self.obs_deques[robot_id] = collections.deque(
                    [obs] * self.obs_horizon,
                    maxlen=self.obs_horizon,
                )
            else:
                self.obs_deques[robot_id].append(obs)

    # English purpose: Stack temporal observations and move them to the inference device.
    def _obs_dict(self, robot_id):
        obs_deque = self.obs_deques[robot_id]
        node_inputs = torch.stack(
            [item["node_inputs"] for item in obs_deque]
        ).to(self.device, dtype=torch.float32)
        node_padding_mask = torch.stack(
            [item["node_padding_mask"] for item in obs_deque]
        ).to(self.device, dtype=torch.int16)
        edge_mask = torch.stack(
            [item["edge_mask"] for item in obs_deque]
        ).to(self.device, dtype=torch.int64)
        current_index = torch.stack(
            [item["current_index"] for item in obs_deque]
        ).to(self.device, dtype=torch.int64)
        current_edge = torch.stack(
            [item["current_edge"] for item in obs_deque]
        ).to(self.device, dtype=torch.int64)
        edge_padding_mask = torch.stack(
            [item["edge_padding_mask"] for item in obs_deque]
        ).to(self.device, dtype=torch.int16)
        return {
            "node_inputs": node_inputs.unsqueeze(0),
            "node_padding_mask": node_padding_mask.unsqueeze(0),
            "edge_mask": edge_mask.unsqueeze(0),
            "current_index": current_index.unsqueeze(0),
            "current_edge": current_edge.unsqueeze(0),
            "edge_padding_mask": edge_padding_mask.unsqueeze(0),
        }

    # English purpose: Run the frozen DARE policy to predict an action sequence.
    def _predict_action_sequence(self, robot_id):
        with torch.no_grad():
            action_dict = self.policy.predict_action(
                self._obs_dict(robot_id)
            )
        action_pred = (
            action_dict["action_pred"]
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )
        return (
            np.round(action_pred / NODE_RESOLUTION)
            * NODE_RESOLUTION
        )

    # English purpose: Rank graph-legal neighbours using only frozen DARE output.
    def _base_dare_candidates(self, robot_id, action_pred):
        """Return pure DARE neighbour ranking with waiting as final fallback.

        English implementation: extracts current-node neighbours, ranks them by
        angular agreement with the frozen policy horizon, and appends waiting.
        """

        robot = self.robot_list[robot_id]
        current = self.env.robot_locations[robot_id].copy()
        node_record = robot.node_manager.nodes_dict.find(current.tolist())
        if node_record is None:
            return [current]

        candidates: List[np.ndarray] = []
        seen = set()
        for neighbour in node_record.data.neighbor_list:
            candidate = np.asarray(neighbour, dtype=np.float32)
            key = tuple(np.round(candidate, 4))
            if key not in seen and not np.allclose(candidate, current):
                candidates.append(candidate)
                seen.add(key)

        start = self.obs_horizon - 1
        raw = action_pred[start : start + DIRECTION_LOOKAHEAD]
        direction_vectors = np.cumsum(raw, axis=0) if USE_DELTA_POSITION else raw - current[None, :]

        def direction_score(candidate):
            movement = candidate - current
            movement_norm = float(np.linalg.norm(movement))
            if movement_norm <= 1e-8:
                return 1e9
            angles, weights = [], []
            for index, vector in enumerate(direction_vectors):
                vector_norm = float(np.linalg.norm(vector))
                if vector_norm <= 1e-8:
                    continue
                cosine = float(np.clip(np.dot(movement, vector) / (movement_norm * vector_norm), -1.0, 1.0))
                angles.append(float(np.arccos(cosine)))
                weights.append(len(direction_vectors) - index)
            return float(np.average(angles, weights=weights)) if angles else 1e8

        candidates.sort(key=direction_score)
        candidates.append(current.copy())
        return candidates

    def _apply_candidate_supervisors(self, robot_id, candidates, step):
        """Apply optional direction and frontier-region soft re-ranking.

        English implementation: applies supervisors
        sequentially without deleting candidates or generating off-graph actions.
        """

        robot = self.robot_list[robot_id]
        current = self.env.robot_locations[robot_id].copy()
        output = self.initial_direction_manager.order_candidates(
            robot_id=robot_id,
            current_position=current,
            ordered_candidates=candidates,
            step=int(step),
        )
        if self.region_coordinator is not None:
            output = self.region_coordinator.order_candidates(
                robot_id=robot_id,
                robot=robot,
                current_position=current,
                ordered_candidates=output,
                step=int(step),
            )
        return output

    def _ordered_dare_candidates(self, robot_id, action_pred, step):
        """Compatibility wrapper returning supervised DARE candidates."""

        return self._apply_candidate_supervisors(
            robot_id,
            self._base_dare_candidates(robot_id, action_pred),
            step=step,
        )

    def _short_horizon_plan(self, robot_id, action_pred, ordered_candidates):
        """Construct a one/two-step graph plan for motion reservation only.

        English implementation: returns at most two graph nodes and never changes
        the action executed by the receding-horizon controller.
        """

        current = self.env.robot_locations[robot_id].copy()
        if not ordered_candidates:
            return [current]
        first = np.asarray(ordered_candidates[0], dtype=np.float32).copy()
        plan = [first]
        if np.allclose(first, current):
            return plan
        record = self.robot_list[robot_id].node_manager.nodes_dict.find(first.tolist())
        if record is None:
            return plan
        neighbours = [
            np.asarray(value, dtype=np.float32)
            for value in record.data.neighbor_list
            if not np.allclose(value, first)
        ]
        if not neighbours:
            return plan
        index = min(self.obs_horizon, max(0, len(action_pred) - 1))
        vector = np.asarray(action_pred[index], dtype=np.float32)
        if not USE_DELTA_POSITION:
            vector = vector - first
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8:
            return plan

        def score(candidate):
            movement = candidate - first
            movement_norm = float(np.linalg.norm(movement))
            if movement_norm <= 1e-8:
                return 1e9
            cosine = float(np.clip(np.dot(movement, vector) / (movement_norm * norm), -1.0, 1.0))
            return float(np.arccos(cosine))

        plan.append(min(neighbours, key=score).copy())
        return plan

    # ------------------------------------------------------------------
    # Contact-triggered map fusion
    # ------------------------------------------------------------------

   