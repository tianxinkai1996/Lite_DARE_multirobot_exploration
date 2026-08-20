from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from classes.multi_robot.evaluation_recorder import EpisodeMetricsRecorder
from mergingmap.multi_test_driver_mergingmap import run_seed
from paper_experiments.chapter4_config import EPISODE_WORKERS, MODELS, RANDOM_SEED
from paper_experiments.common import iter_map_trials, parse_map_repeats, parse_map_selection
from paper_experiments.select_attention_model import select_model


class Chapter4CommonTests(unittest.TestCase):
    def test_map_selection_supports_ranges(self):
        self.assertEqual(parse_map_selection("0,3,7-9", 20), [0, 3, 7, 8, 9])
        self.assertEqual(parse_map_selection("all", 3), [0, 1, 2])

    def test_per_map_repeat_overrides(self):
        overrides = parse_map_repeats("1:5,3:1", 2)
        pairs = list(iter_map_trials([0, 1, 3], 2, overrides))
        self.assertEqual(sum(map_index == 0 for map_index, _ in pairs), 2)
        self.assertEqual(sum(map_index == 1 for map_index, _ in pairs), 5)
        self.assertEqual(sum(map_index == 3 for map_index, _ in pairs), 1)

    def test_all_depth_models_share_one_training_seed(self):
        self.assertEqual({model.encoder_layers for model in MODELS.values()}, {2, 4, 6})
        self.assertEqual({model.training_seed for model in MODELS.values()}, {RANDOM_SEED})

    def test_repeated_trials_derive_from_one_base_seed(self):
        first = run_seed(7, 0, 4, "compressed", base_seed=RANDOM_SEED)
        repeated = run_seed(7, 0, 4, "raw", base_seed=RANDOM_SEED)
        second_trial = run_seed(7, 1, 4, "compressed", base_seed=RANDOM_SEED)
        self.assertEqual(first, repeated)  # paired across communication treatments
        self.assertNotEqual(first, second_trial)  # repeated trials are not identical clones

    def _write_depth_fixture(self, root, rows_by_model):
        for key, depth, coverage, auc, success in rows_by_model:
            path = root / "e1_policy_depth" / key
            path.mkdir(parents=True)
            with (path / "results.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "model_key", "model_name", "encoder_layers", "team_size",
                        "scenario_pair_key", "final_coverage", "coverage_auc", "success",
                    ],
                )
                writer.writeheader()
                for trial in range(12):
                    writer.writerow(
                        {
                            "model_key": key,
                            "model_name": key,
                            "encoder_layers": depth,
                            "team_size": 1,
                            "scenario_pair_key": f"map_0000_trial_{trial:02d}",
                            "final_coverage": coverage,
                            "coverage_auc": auc,
                            "success": success,
                        }
                    )

    def test_auto_selection_chooses_l4_when_l4_is_closer_to_l6(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_depth_fixture(
                root,
                [
                    ("DARE-L6", 6, 0.990, 0.900, 1.00),
                    ("LiteDARE-L4", 4, 0.988, 0.898, 0.99),
                    ("LiteDARE-L2", 2, 0.960, 0.850, 0.92),
                ],
            )
            selected, summary = select_model(
                root, delta_coverage=0.01, delta_success=0.02, bootstrap_samples=200
            )
            self.assertEqual(selected, "LiteDARE-L4")
            l4 = next(row for row in summary if row["encoder_layers"] == 4)
            l2 = next(row for row in summary if row["encoder_layers"] == 2)
            self.assertLess(l4["similarity_distance_to_L6"], l2["similarity_distance_to_L6"])

    def test_auto_selection_chooses_l2_when_l2_is_closer_to_l6(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_depth_fixture(
                root,
                [
                    ("DARE-L6", 6, 0.990, 0.900, 1.00),
                    ("LiteDARE-L4", 4, 0.975, 0.880, 0.95),
                    ("LiteDARE-L2", 2, 0.989, 0.899, 0.99),
                ],
            )
            selected, _ = select_model(
                root, delta_coverage=0.01, delta_success=0.02, bootstrap_samples=200
            )
            self.assertEqual(selected, "LiteDARE-L2")

    def test_auto_selection_prefers_l2_only_when_similarity_is_tied(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_depth_fixture(
                root,
                [
                    ("DARE-L6", 6, 0.990, 0.900, 1.00),
                    ("LiteDARE-L4", 4, 0.985, 0.895, 0.98),
                    ("LiteDARE-L2", 2, 0.985, 0.895, 0.98),
                ],
            )
            selected, _ = select_model(
                root, delta_coverage=0.01, delta_success=0.02, bootstrap_samples=200
            )
            self.assertEqual(selected, "LiteDARE-L2")



    def test_episode_workers_default_is_positive(self):
        self.assertGreaterEqual(EPISODE_WORKERS, 1)

    def test_ablation_parser_accepts_episode_workers(self):
        from mergingmap.run_paper_ablation import build_parser

        args = build_parser().parse_args(["--episode-workers", "4", "--plan-only"])
        self.assertEqual(args.episode_workers, 4)

    def test_map_structure_metrics_are_outcome_independent(self):
        free, occupied = 255, 1
        truth = np.array(
            [
                [occupied, occupied, occupied, occupied, occupied],
                [occupied, free, free, free, occupied],
                [occupied, occupied, free, occupied, occupied],
                [occupied, free, free, free, occupied],
                [occupied, occupied, occupied, occupied, occupied],
            ],
            dtype=np.uint8,
        )
        metrics = EpisodeMetricsRecorder.map_structure_metrics(
            truth, free_value=free, occupied_value=occupied
        )
        self.assertGreater(metrics["map_obstacle_ratio"], 0.0)
        self.assertGreater(metrics["map_narrow_free_ratio"], 0.0)
        self.assertGreaterEqual(metrics["map_structure_difficulty_score"], 0.0)
        self.assertLessEqual(metrics["map_structure_difficulty_score"], 1.0)


if __name__ == "__main__":
    unittest.main()


