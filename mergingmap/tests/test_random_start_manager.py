import unittest

import numpy as np

from mergingmap.random_start_manager import (
    RandomStartConfig,
    create_environment_with_valid_random_starts,
    validate_random_starts,
)


FREE = 255
OCCUPIED = 1


class DummyEnv:
    def __init__(self, positions):
        self.robot_locations = np.asarray(positions, dtype=float)
        self.ground_truth = np.full((20, 20), FREE, dtype=np.uint8)

    def world_to_cell(self, point):
        return np.asarray(point, dtype=int)


class RandomStartManagerTests(unittest.TestCase):
    def test_duplicate_positions_are_rejected(self):
        result = validate_random_starts(
            DummyEnv([(2, 2), (2, 2)]),
            n_agents=2,
            config=RandomStartConfig(
                free_value=FREE,
                min_separation=1.0,
            ),
        )
        self.assertFalse(result.valid)

    def test_factory_retries_until_distinct_positions(self):
        def factory(seed):
            if seed == 10:
                return DummyEnv([(2, 2), (2, 2)])
            return DummyEnv([(2, 2), (8, 8)])

        selection = create_environment_with_valid_random_starts(
            factory,
            base_seed=10,
            n_agents=2,
            config=RandomStartConfig(
                free_value=FREE,
                min_separation=2.0,
                max_attempts=3,
            ),
        )
        self.assertEqual(selection.attempt_count, 2)
        self.assertTrue(selection.validation.valid)


if __name__ == "__main__":
    unittest.main()