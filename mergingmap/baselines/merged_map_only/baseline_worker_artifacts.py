"""Artifact helpers for the MergingMap-only baseline.

MergingMap-only 基线的输出记录辅助函数。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from mergingmap.baselines.merged_map_only.multi_test_parameter import (
    MIN_START_SEPARATION,
    START_CLEARANCE_RADIUS_CELLS,
    VISUAL_OUTPUT_ROOT,
)


class BaselineArtifactMixin:
    """Persist baseline experiment metadata outside the main worker.

    中文：把起点记录从主工作器中拆分，减少单文件长度。
    English: isolates start-position persistence from the main worker.
    """

    # 中文目的：保存 MergingMap-only 基线的随机起点。
    # English purpose: Persist random starts for the MergingMap-only baseline.
    def _save_start_position_record(self) -> str:
        """Write the validated random starts into this run's result folder."""
        output_root = Path(
            os.environ.get("MERGINGMAP_RUN_DIR", VISUAL_OUTPUT_ROOT)
        )
        start_root = output_root / "start_positions"
        start_root.mkdir(parents=True, exist_ok=True)

        filename = (
            f"episode_{self.episode_index:04d}_"
            f"robots_{self.n_agents}_"
            f"mode_{self.communication_mode}_"
            f"seed_{self.seed}.json"
        )
        path = start_root / filename

        payload = {
            "episode": self.episode_index,
            "team_size": self.n_agents,
            "communication_mode": self.communication_mode,
            "base_seed": self.seed,
            "selected_start_seed": self.start_seed,
            "sampling_attempts": self.start_attempt_count,
            "minimum_required_separation": float(MIN_START_SEPARATION),
            "minimum_actual_separation": float(
                self.start_validation.minimum_pairwise_distance
            ),
            "clearance_radius_cells": int(
                START_CLEARANCE_RADIUS_CELLS
            ),
            "world_positions": [
                [float(position[0]), float(position[1])]
                for position in self.initial_start_positions
            ],
            "grid_cells": [
                [int(cell[0]), int(cell[1])]
                for cell in self.start_validation.cell_positions
            ],
            "all_distinct": True,
            "all_ground_truth_free": True,
        }
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return str(path)

    # ------------------------------------------------------------------
    # Merged-map graph construction and frozen DARE inference
    # ------------------------------------------------------------------

