import collections
import unittest

import numpy as np

from mergingmap.initial_direction_manager import (
    InitialDirectionConfig,
    InitialDirectionManager,
)


class InitialDirectionManagerTests(unittest.TestCase):
    def test_assignments_are_deterministic(self):
        first = InitialDirectionManager(
            8,
            seed=123,
            config=InitialDirectionConfig(),
        )
        second = InitialDirectionManager(
            8,
            seed=123,
            config=InitialDirectionConfig(),
        )
        self.assertEqual(first.roles, second.roles)

    def test_eight_robots_receive_balanced_directions(self):
        manager = InitialDirectionManager(
            8,
            seed=42,
            config=InitialDirectionConfig(),
        )
        counts = collections.Counter(role.name for role in manager.roles)
        self.assertEqual(
            counts,
            {"north": 2, "east": 2, "south": 2, "west": 2},
        )

    def test_first_four_directions_are_unique(self):
        manager = InitialDirectionManager(
            4,
            seed=99,
            config=InitialDirectionConfig(),
        )
        self.assertEqual(len({role.name for role in manager.roles}), 4)

    def test_role_aligned_candidate_is_promoted_at_step_zero(self):
        manager = InitialDirectionManager(
            1,
            seed=7,
            config=InitialDirectionConfig(
                bias_steps=4,
                max_bias_weight=1.0,
                decay=False,
            ),
        )
        role = manager.role_for_robot(0)
        direction = np.asarray(role.vector, dtype=np.float32)
        current = np.asarray((0.0, 0.0), dtype=np.float32)
        opposite = current - direction
        aligned = current + direction

        ordered = manager.order_candidates(
            robot_id=0,
            current_position=current,
            ordered_candidates=[opposite, aligned, current],
            step=0,
        )
        np.testing.assert_allclose(ordered[0], aligned)

    def test_bias_expires_and_original_dare_order_returns(self):
        manager = InitialDirectionManager(
            1,
            seed=7,
            config=InitialDirectionConfig(
                bias_steps=2,
                max_bias_weight=1.0,
            ),
        )
        original = [
            np.asarray((1.0, 0.0), dtype=np.float32),
            np.asarray((0.0, 1.0), dtype=np.float32),
        ]
        ordered = manager.order_candidates(
            robot_id=0,
            current_position=(0.0, 0.0),
            ordered_candidates=original,
            step=2,
        )
        np.testing.assert_allclose(ordered, original)


if __name__ == "__main__":
    unittest.main()

