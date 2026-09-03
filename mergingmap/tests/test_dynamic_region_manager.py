"""Tests for the dynamic region lease lifecycle."""

import unittest

import numpy as np

from mergingmap.dynamic_regions import (
    DynamicRegionConfig,
    DynamicRegionCoordinator,
    RegionLease,
    extract_frontier_regions,
)


UNKNOWN = 127
FREE = 255
OCCUPIED = 1


class DummyEnv:
    def cell_to_world(self, cell):
        return np.asarray(cell, dtype=np.float32)

    def world_to_cell(self, position):
        return np.asarray(position, dtype=np.int32)


class DummyNode:
    def __init__(self, coords, neighbours):
        self.coords = np.asarray(coords, dtype=np.float32)
        self.neighbor_list = [
            np.asarray(value, dtype=np.float32)
            for value in neighbours
        ]


class DummyRecord:
    def __init__(self, node):
        self.data = node


class DummyNodesDict:
    def __init__(self, records):
        self.records = records

    def __iter__(self):
        return iter(self.records)


class DummyNodeManager:
    def __init__(self, records):
        self.nodes_dict = DummyNodesDict(records)


class DummyRobot:
    def __init__(self, coordinates):
        records = []
        for index, coordinate in enumerate(coordinates):
            neighbours = []
            if index > 0:
                neighbours.append(coordinates[index - 1])
            if index + 1 < len(coordinates):
                neighbours.append(coordinates[index + 1])
            records.append(
                DummyRecord(DummyNode(coordinate, neighbours))
            )
        self.node_manager = DummyNodeManager(records)


class DynamicRegionManagerTests(unittest.TestCase):
    def config(self, **overrides):
        values = dict(
            unknown_value=UNKNOWN,
            free_value=FREE,
            node_resolution=1.0,
            min_frontier_cells=1,
            max_frontier_cells_per_region=100,
            region_id_quantization_cells=1,
            region_match_iou_threshold=0.1,
            region_match_centroid_cells=2.0,
            region_conflict_distance=2.0,
            lease_steps=20,
            claim_ttl_steps=20,
            min_commitment_steps=2,
            no_progress_release_steps=8,
            force_progress_after_steps=3,
            progress_known_cells=1,
            arrival_distance=0.25,
            distance_slack=0.25,
            distance_weight=1.0,
            utility_weight=1.0,
            age_weight=0.1,
            debug=False,
            debug_interval=10,
        )
        values.update(overrides)
        return DynamicRegionConfig(**values)

    @staticmethod
    def two_patch_map():
        grid = np.full((12, 12), UNKNOWN, dtype=np.uint8)
        grid[1:3, 1:3] = FREE
        grid[8:10, 8:10] = FREE
        return grid

    def test_extracts_two_current_frontier_tasks(self):
        regions = extract_frontier_regions(
            self.two_patch_map(),
            env=DummyEnv(),
            config=self.config(),
        )

        self.assertEqual(len(regions), 2)
        self.assertTrue(all(region.frontier_count == 4 for region in regions))
        self.assertNotEqual(regions[0].region_id, regions[1].region_id)

    def test_contact_component_receives_unique_regions(self):
        coordinator = DynamicRegionCoordinator(2, self.config())
        grid = self.two_patch_map()

        snapshot = coordinator.update(
            step=0,
            robot_maps=[grid, grid],
            robot_positions=[(1.0, 1.0), (9.0, 9.0)],
            contact_pairs=[(0, 1)],
            env=DummyEnv(),
        )

        first = coordinator.states[0].lease
        second = coordinator.states[1].lease
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertNotEqual(first.region_id, second.region_id)
        self.assertEqual(snapshot["active_leases"], 2)
        self.assertTrue(snapshot["all_current_frontiers_accounted"])

    def test_without_contact_duplicate_claim_is_possible_then_resolved(self):
        coordinator = DynamicRegionCoordinator(2, self.config())
        grid = np.full((8, 8), UNKNOWN, dtype=np.uint8)
        grid[3:5, 3:5] = FREE

        coordinator.update(
            step=0,
            robot_maps=[grid, grid],
            robot_positions=[(2.0, 3.0), (6.0, 3.0)],
            contact_pairs=[],
            env=DummyEnv(),
        )
        self.assertIsNotNone(coordinator.states[0].lease)
        self.assertIsNotNone(coordinator.states[1].lease)

        coordinator.update(
            step=1,
            robot_maps=[grid, grid],
            robot_positions=[(2.0, 3.0), (6.0, 3.0)],
            contact_pairs=[(0, 1)],
            env=DummyEnv(),
        )

        active = sum(
            state.lease is not None for state in coordinator.states
        )
        self.assertEqual(active, 1)
        self.assertEqual(coordinator.claim_conflicts_resolved, 1)

    def test_disappearing_frontier_releases_completed_lease(self):
        coordinator = DynamicRegionCoordinator(1, self.config())
        grid = np.full((8, 8), UNKNOWN, dtype=np.uint8)
        grid[3:5, 3:5] = FREE

        coordinator.update(
            step=0,
            robot_maps=[grid],
            robot_positions=[(3.0, 3.0)],
            contact_pairs=[],
            env=DummyEnv(),
        )
        self.assertIsNotNone(coordinator.states[0].lease)

        no_frontier = np.full((8, 8), FREE, dtype=np.uint8)
        coordinator.update(
            step=1,
            robot_maps=[no_frontier],
            robot_positions=[(3.0, 3.0)],
            contact_pairs=[],
            env=DummyEnv(),
        )

        self.assertIsNone(coordinator.states[0].lease)
        self.assertEqual(
            coordinator.leases_released_completed,
            1,
        )

    def test_no_progress_lease_is_recycled(self):
        coordinator = DynamicRegionCoordinator(
            1,
            self.config(
                lease_steps=100,
                min_commitment_steps=1,
                no_progress_release_steps=2,
            ),
        )
        grid = np.full((8, 8), UNKNOWN, dtype=np.uint8)
        grid[3:5, 3:5] = FREE

        coordinator.update(
            step=0,
            robot_maps=[grid],
            robot_positions=[(1.0, 1.0)],
            contact_pairs=[],
            env=DummyEnv(),
        )
        coordinator.update(
            step=3,
            robot_maps=[grid],
            robot_positions=[(1.0, 1.0)],
            contact_pairs=[],
            env=DummyEnv(),
        )

        self.assertEqual(
            coordinator.leases_released_no_progress,
            1,
        )
        self.assertGreaterEqual(coordinator.assignments_created, 2)
        self.assertIsNotNone(coordinator.states[0].lease)

    def test_candidate_supervisor_prefers_graph_progress(self):
        coordinator = DynamicRegionCoordinator(1, self.config())
        grid = np.full((6, 6), UNKNOWN, dtype=np.uint8)
        grid[0, 3] = FREE

        coordinator.update(
            step=0,
            robot_maps=[grid],
            robot_positions=[(0.0, 0.0)],
            contact_pairs=[],
            env=DummyEnv(),
        )
        self.assertIsNotNone(coordinator.states[0].lease)

        robot = DummyRobot(
            [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
        )
        # Frozen DARE ranks waiting first in this synthetic case. The region
        # supervisor must move the progress candidate ahead without deleting
        # either candidate.
        ordered = coordinator.order_candidates(
            robot_id=0,
            robot=robot,
            current_position=np.asarray((0.0, 0.0)),
            ordered_candidates=[
                np.asarray((0.0, 0.0)),
                np.asarray((1.0, 0.0)),
            ],
            step=0,
        )

        np.testing.assert_allclose(ordered[0], (1.0, 0.0))
        self.assertEqual(len(ordered), 2)
        self.assertEqual(coordinator.region_candidate_overrides, 1)

    def test_sensor_progress_renews_lease(self):
        coordinator = DynamicRegionCoordinator(
            1,
            self.config(lease_steps=5),
        )
        grid = np.full((8, 8), UNKNOWN, dtype=np.uint8)
        grid[3:5, 3:5] = FREE

        coordinator.update(
            step=0,
            robot_maps=[grid],
            robot_positions=[(1.0, 1.0)],
            contact_pairs=[],
            env=DummyEnv(),
        )
        lease = coordinator.states[0].lease
        self.assertIsNotNone(lease)
        original_expiry = lease.expiry_step

        coordinator.report_progress(
            robot_id=0,
            step=4,
            newly_known_cells=3,
        )

        self.assertEqual(lease.last_progress_step, 4)
        self.assertGreater(lease.expiry_step, original_expiry)


if __name__ == "__main__":
    unittest.main()

