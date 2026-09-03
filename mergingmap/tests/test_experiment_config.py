"""Tests for experiment configuration parsing."""

import unittest

from mergingmap.experiment_config import (
    extract_repeat_argument,
    positive_repeat_count,
)


class ExperimentConfigTests(unittest.TestCase):
    def test_space_separated_override(self):
        result = extract_repeat_argument(
            [
                "--runs-per-map",
                "5",
                "--maps",
                "all",
                "--map-count",
                "100",
            ]
        )
        self.assertEqual(result.runs_per_map, 5)
        self.assertEqual(
            result.forwarded_args,
            ("--maps", "all", "--map-count", "100"),
        )

    def test_equals_override(self):
        result = extract_repeat_argument(
            ["--map-repeats=7", "--modes", "compressed"]
        )
        self.assertEqual(result.runs_per_map, 7)
        self.assertEqual(
            result.forwarded_args,
            ("--modes", "compressed"),
        )

    def test_no_override_preserves_all_driver_arguments(self):
        args = ["--maps", "0,1,2", "--map-count", "3"]
        result = extract_repeat_argument(args)
        self.assertIsNone(result.runs_per_map)
        self.assertEqual(result.forwarded_args, tuple(args))

    def test_non_positive_count_is_rejected(self):
        with self.assertRaises(ValueError):
            positive_repeat_count(0, name="runs")
        with self.assertRaises(ValueError):
            extract_repeat_argument(["--runs-per-map", "-2"])

    def test_duplicate_override_is_rejected(self):
        with self.assertRaises(ValueError):
            extract_repeat_argument(
                ["--runs-per-map", "2", "--map-repeats=3"]
            )


if __name__ == "__main__":
    unittest.main()

