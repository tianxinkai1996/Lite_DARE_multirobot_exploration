"""Tests for the merged belief manager and conservative fusion."""

import unittest
import numpy as np

from mergingmap.merged_belief_manager import (
    MergedBeliefConfig,
    MergedBeliefManager,
)


UNKNOWN = 127
FREE = 255
OCCUPIED = 1


class DummyMapInfo:
    def __init__(self, grid):
        self.map = grid
        self.map_origin_x = -10.0
        self.map_origin_y = 5.0
        self.cell_size = 0.4


class MergedBeliefManagerTests(unittest.TestCase):
    def config(self, max_cells=0):
        return MergedBeliefConfig(
            unknown_value=UNKNOWN,
            free_value=FREE,
            occupied_value=OCCUPIED,
            max_cells_per_packet=max_cells,
        )

    def test_pair_exchange_builds_union(self):
        first = np.full((4, 4), UNKNOWN, dtype=np.uint8)
        second = first.copy()
        first[0, 0] = FREE
        first[0, 1] = OCCUPIED
        second[3, 3] = FREE

        manager = MergedBeliefManager([first, second], self.config())
        changed_0, changed_1, _, _ = manager.exchange_pair(
            0, 1, step=0, mode="delta"
        )

        self.assertEqual(changed_0, 1)
        self.assertEqual(changed_1, 2)
        np.testing.assert_array_equal(
            manager.merged_map(0),
            manager.merged_map(1),
        )
        self.assertEqual(manager.pairwise_agreement(0, 1), 1.0)

    def test_knowledge_propagates_transitively(self):
        maps = [
            np.full((3, 3), UNKNOWN, dtype=np.uint8)
            for _ in range(3)
        ]
        maps[0][0, 0] = FREE
        maps[1][1, 1] = FREE
        maps[2][2, 2] = OCCUPIED

        manager = MergedBeliefManager(maps, self.config())
        manager.exchange_pair(0, 1, step=0, mode="delta")
        manager.exchange_pair(1, 2, step=1, mode="delta")
        manager.exchange_pair(0, 2, step=2, mode="delta")

        for robot_id in range(1, 3):
            np.testing.assert_array_equal(
                manager.merged_map(0),
                manager.merged_map(robot_id),
            )

    def test_occupied_wins_conflict(self):
        first = np.full((2, 2), UNKNOWN, dtype=np.uint8)
        second = first.copy()
        first[0, 0] = FREE
        second[0, 0] = OCCUPIED

        manager = MergedBeliefManager([first, second], self.config())
        manager.exchange_pair(0, 1, step=0, mode="delta")

        self.assertEqual(manager.merged_map(0)[0, 0], OCCUPIED)
        self.assertEqual(manager.merged_map(1)[0, 0], OCCUPIED)
        self.assertGreaterEqual(manager.conflicts, 1)

    def test_bounded_packets_eventually_send_everything(self):
        first = np.full((4, 4), UNKNOWN, dtype=np.uint8)
        second = first.copy()
        first.flat[:10] = FREE

        manager = MergedBeliefManager(
            [first, second],
            self.config(max_cells=3),
        )
        for step in range(4):
            manager.exchange_pair(0, 1, step=step, mode="delta")

        np.testing.assert_array_equal(
            manager.merged_map(0),
            manager.merged_map(1),
        )

    def test_map_info_adapter_preserves_frame(self):
        first = np.full((2, 2), UNKNOWN, dtype=np.uint8)
        first[0, 0] = FREE
        manager = MergedBeliefManager([first], self.config())
        template = DummyMapInfo(
            np.full((2, 2), UNKNOWN, dtype=np.uint8)
        )

        result = manager.make_map_info(0, template)

        self.assertIsNot(result, template)
        self.assertEqual(result.map_origin_x, -10.0)
        self.assertEqual(result.map_origin_y, 5.0)
        self.assertEqual(result.cell_size, 0.4)
        self.assertEqual(result.map[0, 0], FREE)


if __name__ == "__main__":
    unittest.main()

