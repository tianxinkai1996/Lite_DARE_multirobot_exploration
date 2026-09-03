"""Focused tests for methodology-specific map deltas, motion, and recovery."""
from __future__ import annotations

import unittest

import numpy as np

from collision.motion_exchange import MotionExchangeConfig, MotionIntentExchange
from deadlock.oscillation_tracker import OscillationTracker
from merged_belief_manager import MergedBeliefConfig, MergedBeliefManager
from motion_coordinator import MergingMapMotionCoordinator

UNKNOWN, FREE, OCCUPIED = 127, 255, 1


def point(x, y):
    return np.asarray([x, y], dtype=np.float32)


class MethodologyAlignmentTests(unittest.TestCase):
    def test_deterministic_delta_history_avoids_resending_unchanged_cells(self):
        first = np.full((2, 2), UNKNOWN, dtype=np.uint8)
        second = first.copy()
        first[0, 0] = FREE
        manager = MergedBeliefManager(
            [first, second],
            MergedBeliefConfig(
                unknown_value=UNKNOWN,
                free_value=FREE,
                occupied_value=OCCUPIED,
            ),
        )
        manager.exchange_pair(0, 1, step=0, mode="delta")
        _, _, packet, _ = manager.exchange_pair(0, 1, step=1, mode="delta")
        self.assertEqual(packet["cell_count"], 0)
        self.assertEqual(manager.metrics()["map_packets_sent"], 4)
        self.assertEqual(manager.metrics()["map_packets_delivered"], 4)

    def test_motion_message_is_short_and_priority_breaks_vertex_tie(self):
        exchange = MotionIntentExchange(
            2,
            MotionExchangeConfig(node_resolution=1.0, safe_distance=0.5),
        )
        target = point(1, 1)
        current = [point(0, 0), point(0, 2)]
        plans = [[target, point(2, 1)], [target, point(2, 2)]]
        exchange.exchange(
            step=0,
            contact_pairs=[(0, 1)],
            current_positions=current,
            plans=plans,
            priorities=[5.0, 1.0],
        )
        filtered = exchange.filter_candidates(
            step=0,
            current_positions=current,
            candidate_lists=[[target, point(-1, 0)], [target, point(0, 3)]],
            contact_pairs=[(0, 1)],
            priorities=[5.0, 1.0],
        )
        self.assertTrue(np.allclose(filtered[0][0], target))
        self.assertFalse(np.allclose(filtered[1][0], target))
        self.assertEqual(exchange.metrics()["motion_message_packets"], 2)
        for manager in exchange.managers:
            for packet in manager._packets.values():
                self.assertLessEqual(len(packet["plan"]), 2)
                self.assertEqual(len(packet["trail"]), 0)

    def test_oscillation_is_soft_and_detected_from_actual_positions(self):
        tracker = OscillationTracker(1, base_penalty=3.0)
        tracker.initialise([point(0, 0)])
        tracker.update_after_execution([point(1, 0)])
        detected = tracker.update_after_execution([point(0, 0)])
        self.assertEqual(detected, (0,))
        ordered = tracker.order_candidates(
            robot_id=0,
            current_position=point(0, 0),
            ordered_candidates=[point(1, 0), point(0, 1)],
        )
        self.assertTrue(np.allclose(ordered[0], point(0, 1)))
        self.assertTrue(any(np.allclose(value, point(1, 0)) for value in ordered))

    def test_staged_recovery_requests_soft_relaxation_and_lease_release(self):
        coordinator = MergingMapMotionCoordinator(
            1,
            safe_distance=0.5,
            mode="collision_deadlock",
            deadlock_wait_threshold=2,
            deadlock_soft_relax_threshold=3,
            deadlock_lease_release_threshold=4,
            deadlock_backtrack_threshold=6,
        )
        decision = coordinator.resolve_step(
            [point(0, 0)],
            [[point(1, 0), point(0, 0)]],
            recovery_candidate_lists=[[point(0, 1), point(0, 0)]],
            stall_steps=[4],
            time_step=4,
        )
        self.assertEqual(decision.lease_release_robot_ids, (0,))
        self.assertEqual(decision.resolution_info.recovery_stages, (4,))
        self.assertEqual(
            decision.resolution_info.soft_penalty_relaxed_robot_ids, (0,)
        )


if __name__ == "__main__":
    unittest.main()
