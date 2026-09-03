"""Tests for the multi-robot driver configuration."""

import ast
from pathlib import Path
import unittest

from mergingmap.experiment_config import extract_repeat_argument, option_value


ROOT = Path(__file__).resolve().parents[1]


class DriverConfigurationTests(unittest.TestCase):
    def test_repeat_argument_is_removed_before_forwarding(self):
        parsed = extract_repeat_argument(
            ["--maps", "all", "--runs-per-map", "5", "--map-count", "2"]
        )
        self.assertEqual(parsed.runs_per_map, 5)
        self.assertEqual(
            parsed.forwarded_args,
            ("--maps", "all", "--map-count", "2"),
        )

    def test_option_value_supports_both_forms(self):
        self.assertEqual(option_value(["--map-count=4"], "--map-count"), "4")
        self.assertEqual(option_value(["--map-count", "7"], "--map-count"), "7")

    def test_runner_uses_dedicated_folder_driver(self):
        source = (ROOT / "run_mergingmap.py").read_text(encoding="utf-8")
        self.assertIn('DRIVER = HERE / "multi_test_driver_mergingmap.py"', source)
        self.assertNotIn('DRIVER = PROJECT_ROOT / "multi_test_driver.py"', source)

    def test_driver_does_not_pass_shared_start_to_worker(self):
        source_path = ROOT / "multi_test_driver_mergingmap.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        worker_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (
                isinstance(node.func, ast.Name)
                and node.func.id == "MultiRobotTestWorker"
            )
        ]
        self.assertEqual(len(worker_calls), 1)
        keywords = {item.arg for item in worker_calls[0].keywords}
        self.assertNotIn("start_position", keywords)
        self.assertNotIn("shared_start", keywords)


if __name__ == "__main__":
    unittest.main()
