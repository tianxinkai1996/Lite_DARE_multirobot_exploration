"""Output and dynamic-region helpers for the MergingMap worker.

MergingMap 
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from mergingmap.multi_test_parameter import (
    INITIAL_DIRECTION_BIAS_STEPS,
    INITIAL_DIRECTION_DECAY,
    INITIAL_DIRECTION_MAX_BIAS_WEIGHT,
    MIN_START_SEPARATION,
    START_CLEARANCE_RADIUS_CELLS,
)

class WorkerArtifactMixin:
    """Provide small persistence and region-update methods.

    English: isolates artifact persistence and region-event handling from the main worker.
    """

    # English purpose: Persist validated random starts for reproducible paired experiments.
    def _save_start_position_record(self):
        """Write the validated random starts into this run's result folder."""
        output_root = Path(self.output_root)
        start_root = output_root / "start_positions"
        start_root.mkdir(parents=True, exist_ok=True)

        filename = (
            f"episode_{self.episode_index:04d}_"
            f"robots_{self.n_agents}_"
            f"profile_{self.ablation_profile.key}_"
            f"mode_{self.communication_mode}_"
            f"seed_{self.seed}.json"
        )
        path = start_root / filename

        payload = {
            "episode": self.episode_index,
            "team_size": self.n_agents,
            "communication_mode": self.communication_mode,
            "scenario_id": self.scenario_id,
            "trial": self.trial,
            "ablation_profile": self.ablation_profile.as_dict(),
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

    # English purpose: Persist initial cardinal-direction assignments and configuration.
    def _save_initial_direction_record(self):
        output_root = Path(self.output_root)
        direction_root = output_root / "initial_directions"
        direction_root.mkdir(parents=True, exist_ok=True)
        path = direction_root / (
            f"episode_{self.episode_index:04d}_"
            f"robots_{self.n_agents}_"
            f"profile_{self.ablation_profile.key}_"
            f"mode_{self.communication_mode}_"
            f"seed_{self.seed}.json"
        )
        payload = {
            "episode": self.episode_index,
            "team_size": self.n_agents,
            "communication_mode": self.communication_mode,
            "scenario_id": self.scenario_id,
            "trial": self.trial,
            "ablation_profile": self.ablation_profile.as_dict(),
            "seed": self.seed,
            "bias_steps": int(INITIAL_DIRECTION_BIAS_STEPS),
            "max_bias_weight": float(INITIAL_DIRECTION_MAX_BIAS_WEIGHT),
            "decay": bool(INITIAL_DIRECTION_DECAY),
            "assignments": self.initial_direction_manager.assignment_payload(),
        }
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return str(path)

    # English purpose: Build the dynamic-region event log path.
    def _region_event_path(self):
        output_root = Path(self.output_root)
        event_root = output_root / "region_events"
        event_root.mkdir(parents=True, exist_ok=True)
        path = event_root / (
            f"episode_{self.episode_index:04d}_"
            f"robots_{self.n_agents}_"
            f"profile_{self.ablation_profile.key}_"
            f"mode_{self.communication_mode}_"
            f"seed_{self.seed}.jsonl"
        )
        return str(path)

    # English purpose: Append coordinator events to the JSONL log.
    def _write_region_events(self):
        if self.region_coordinator is None:
            return
        events = self.region_coordinator.drain_events()
        if not events:
            return
        with Path(self.region_event_file).open("a", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    # English purpose: Update region leases from merged maps, robot positions, and contacts.
    def _update_dynamic_regions(self, step, contact_pairs):
        if self.region_coordinator is None:
            return None
        claim_contacts = (
            list(contact_pairs)
            if self.communication_mode != "none"
            else []
        )
        snapshot = self.region_coordinator.update(
            step=int(step),
            robot_maps=[
                self.map_merger.merged_map(robot_id, copy_map=True)
                for robot_id in range(self.n_agents)
            ],
            robot_positions=self.env.robot_locations,
            contact_pairs=claim_contacts,
            env=self.env,
            robots=self.robot_list,
            wait_steps=self.motion_coordinator.wait_steps,
        )
        self._write_region_events()
        return snapshot

    # ------------------------------------------------------------------
    # Merged-map graph construction and frozen DARE inference
    # ------------------------------------------------------------------

