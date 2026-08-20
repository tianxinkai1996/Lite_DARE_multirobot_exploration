"""Integration tests for collision/deadlock modules through MergingMap.

通过 MergingMap 统一接口测试碰撞与死锁模块。
"""
from __future__ import annotations

import unittest

import numpy as np

from mergingmap.motion_coordinator import MergingMapMotionCoordinator


def point(x: float, y: float) -> np.ndarray:
    """Create a float32 test coordinate.

    中文目的：统一测试坐标类型，避免数值类型影响比较。
    English implementation: returns a two-dimensional float32 NumPy array.
    """

    return np.asarray([x, y], dtype=np.float32)


class MergingMapMotionCoordinatorTests(unittest.TestCase):
    """Verify all coordination modes through the MergingMap-facing facade.

    中文：验证测试函数无需直接依赖底层实现即可调用碰撞与死锁逻辑。
    English: verifies collision and deadlock behaviour through the public facade.
    """

    def test_collision_mode_resolves_same_destination(self):
        """Collision-only mode must prevent two robots selecting one node.

        中文目的：验证仅碰撞模式能够消解同点冲突。
        English implementation: both robots prefer one target and exactly one is
        allowed to occupy it.
        """

        coordinator = MergingMapMotionCoordinator(
            2,
            safe_distance=1.0,
            mode="collision",
        )
        target = point(4, 0)
        decision = coordinator.resolve_step(
            [point(0, 0), point(0, 4)],
            [[target, point(-4, 0)], [target, point(0, 8)]],
            time_step=1,
        )
        selected = decision.next_positions
        self.assertGreaterEqual(np.linalg.norm(selected[0] - selected[1]), 1.0)
        self.assertEqual(sum(np.allclose(value, target) for value in selected), 1)
        self.assertEqual(decision.resolution_info.escape_robot_ids, ())

    def test_deadlock_mode_prioritises_longest_waiting_robot(self):
        """Full mode must convert wait history into search priority.

        中文目的：验证最长等待机器人在竞争同一目标时优先。
        English implementation: seeds wait counters and checks the resolver order
        plus the selected target owner.
        """

        coordinator = MergingMapMotionCoordinator(
            2,
            safe_distance=1.0,
            mode="collision_deadlock",
            deadlock_wait_threshold=3,
        )
        coordinator.wait_steps[:] = [1, 5]
        target = point(4, 0)
        decision = coordinator.resolve_step(
            [point(0, 0), point(0, 4)],
            [[target, point(-4, 0)], [target, point(0, 8)]],
            time_step=2,
        )
        self.assertEqual(decision.resolution_info.priority_order[0], 1)
        self.assertEqual(decision.resolution_info.escape_robot_ids, (1,))
        self.assertTrue(np.allclose(decision.next_positions[1], target))

    def test_actual_execution_updates_deadlock_recovery(self):
        """A selected escape robot moving must count as one recovery event.

        中文目的：验证死锁统计基于环境实际执行位置更新。
        English implementation: selects an escape robot, reports actual movement,
        and checks wait reset, token rotation, and recovery count.
        """

        coordinator = MergingMapMotionCoordinator(
            2,
            safe_distance=1.0,
            mode="collision_deadlock",
            deadlock_wait_threshold=3,
        )
        coordinator.wait_steps[:] = [3, 0]
        self.assertEqual(coordinator.escape_robot_ids(), {0})
        coordinator.update_after_execution(
            [point(0, 0), point(4, 0)],
            [point(0, 4), point(4, 0)],
        )
        self.assertEqual(coordinator.wait_steps.tolist(), [0, 1])
        self.assertEqual(coordinator.deadlock_break_events, 1)
        self.assertEqual(coordinator.priority_token, 1)

    def test_ghost_mode_preserves_uncoordinated_preference(self):
        """Ghost ablation must return the original first candidates unchanged.

        中文目的：验证无协调消融模式不会隐式调用碰撞或死锁算法。
        English implementation: conflicting preferred targets are intentionally
        preserved and resolver diagnostics remain absent.
        """

        coordinator = MergingMapMotionCoordinator(
            2,
            safe_distance=1.0,
            mode="ghost",
        )
        target = point(4, 0)
        decision = coordinator.resolve_step(
            [point(0, 0), point(0, 4)],
            [[target], [target]],
            time_step=0,
        )
        self.assertTrue(all(np.allclose(value, target) for value in decision.next_positions))
        self.assertIsNone(decision.resolution_info)
        self.assertEqual(decision.blocked_robot_ids, ())

    def test_shared_depot_serial_fallback_is_available(self):
        """A bounded search may serialise one departure from a shared depot.

        中文目的：验证共享起点在回溯预算不足时不会退化为全体永久等待。
        English implementation: caps search at one node and checks one robot moves
        through the collision resolver fallback.
        """

        coordinator = MergingMapMotionCoordinator(
            4,
            safe_distance=0.75,
            mode="collision_deadlock",
            max_backtracking_nodes=1,
        )
        current = [point(0, 0) for _ in range(4)]
        candidates = [[point(1, 0), current[i]] for i in range(4)]
        decision = coordinator.resolve_step(
            current,
            candidates,
            time_step=5,
            allow_shared_start_step0=True,
        )
        moved = [
            robot_id
            for robot_id in range(4)
            if not np.allclose(decision.next_positions[robot_id], current[robot_id])
        ]
        self.assertEqual(moved, [0])
        self.assertTrue(decision.resolution_info.used_serial_fallback)


if __name__ == "__main__":
    unittest.main()

