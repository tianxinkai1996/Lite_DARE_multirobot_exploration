"""Frozen-DARE runtime helpers for the MergingMap-only baseline.

MergingMap-only 基线的冻结 DARE 运行辅助函数。
"""
from __future__ import annotations

import collections
from typing import List

import numpy as np
import torch

from parameter import NODE_RESOLUTION
from test_parameter import USE_DELTA_POSITION
from mergingmap.baselines.merged_map_only.multi_test_parameter import DIRECTION_LOOKAHEAD, RESET_OBS_HISTORY_ON_MAP_MERGE


class BaselineDareRuntimeMixin:
    """Group graph observation and frozen-policy inference methods.

    中文：集中图观测、时序输入和动作候选排序逻辑。
    English: groups graph observation, temporal input, and candidate ranking.
    """

    # 中文目的：构造基线机器人使用的融合地图信息。
    # English purpose: Build merged map information for one baseline robot.
    def _merged_map_info(self, robot_id: int):
        template = self.env.get_local_belief_info(robot_id)
        return self.map_merger.make_map_info(robot_id, template)

    # 中文目的：从融合地图更新单台机器人图。
    # English purpose: Update one robot graph from its merged belief.
    def _update_robot_graph(self, robot_id: int) -> None:
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

    # 中文目的：转换图观测为策略输入。
    # English purpose: Convert graph observations into policy inputs.
    def _make_node_observation(self, robot_id: int) -> dict:
        observation = self.robot_list[robot_id].get_observation()
        return {
            "node_inputs": observation[0].squeeze(0),
            "node_padding_mask": observation[1].squeeze(0),
            "edge_mask": observation[2].squeeze(0),
            "current_index": observation[3].squeeze(0),
            "current_edge": observation[4].squeeze(0),
            "edge_padding_mask": observation[5].squeeze(0),
        }

    # 中文目的：初始化机器人观测历史。
    # English purpose: Initialise robot observation histories.
    def _initialise_robot_states(self) -> None:
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

    # 中文目的：地图更新后刷新观测历史。
    # English purpose: Refresh observation histories after map updates.
    def _refresh_observations(
        self,
        *,
        merged_changed_robots: set[int],
    ) -> None:
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

    # 中文目的：堆叠策略时间窗口输入。
    # English purpose: Stack the policy temporal input window.
    def _obs_dict(self, robot_id: int) -> dict:
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

    # 中文目的：运行冻结策略预测动作序列。
    # English purpose: Run the frozen policy to predict an action sequence.
    def _predict_action_sequence(self, robot_id: int) -> np.ndarray:
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

    # 中文目的：按 DARE 方向排序合法邻居。
    # English purpose: Rank legal neighbours by the DARE direction.
    def _ordered_dare_candidates(
        self,
        robot_id: int,
        action_pred: np.ndarray,
    ) -> List[np.ndarray]:
        """Rank legal graph neighbours only by frozen DARE direction."""
        robot = self.robot_list[robot_id]
        current = self.env.robot_locations[robot_id].copy()
        node_record = robot.node_manager.nodes_dict.find(
            current.tolist()
        )
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
        raw = action_pred[
            start : start + DIRECTION_LOOKAHEAD
        ]
        if USE_DELTA_POSITION:
            direction_vectors = np.cumsum(raw, axis=0)
        else:
            direction_vectors = raw - current[None, :]

        def direction_score(candidate: np.ndarray) -> float:
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
                cosine = float(
                    np.clip(
                        np.dot(movement, vector)
                        / (movement_norm * vector_norm),
                        -1.0,
                        1.0,
                    )
                )
                angles.append(float(np.arccos(cosine)))
                weights.append(len(direction_vectors) - index)
            return (
                float(np.average(angles, weights=weights))
                if angles
                else 1e8
            )

        candidates.sort(key=direction_score)
        candidates.append(current.copy())
        return candidates

    # ------------------------------------------------------------------
    # Contact-triggered map fusion
    # ------------------------------------------------------------------

