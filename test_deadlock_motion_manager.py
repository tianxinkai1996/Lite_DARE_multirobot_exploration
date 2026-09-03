from __future__ import annotations

import numpy as np

from classes.multi_robot.motion_manager import DeadlockAwareMotionManager


def _p(x, y):
    return np.asarray([x, y], dtype=np.float32)


def test_same_destination_conflict_is_resolved():
    manager = DeadlockAwareMotionManager(2, safe_distance=1.0)
    current = [_p(0, 0), _p(0, 4)]
    target = _p(4, 0)
    candidates = [
        [target, _p(-4, 0)],
        [target, _p(0, 8)],
    ]

    selected, _, _ = manager.resolve(current, candidates, time_step=1)

    assert not np.allclose(selected[0], selected[1])
    assert np.linalg.norm(selected[0] - selected[1]) >= 1.0
    assert sum(np.allclose(position, target) for position in selected) == 1


def test_edge_swap_is_prevented():
    manager = DeadlockAwareMotionManager(2, safe_distance=1.0)
    current = [_p(0, 0), _p(4, 0)]
    candidates = [
        [_p(4, 0), _p(0, 4)],
        [_p(0, 0), _p(4, 4)],
    ]

    selected, _, _ = manager.resolve(current, candidates, time_step=1)

    swapped = np.allclose(selected[0], current[1]) and np.allclose(selected[1], current[0])
    assert not swapped
    assert np.linalg.norm(selected[0] - selected[1]) >= 1.0


def test_longest_waiting_robot_gets_priority():
    manager = DeadlockAwareMotionManager(2, safe_distance=1.0)
    manager.wait_steps[:] = [1, 5]

    current = [_p(0, 0), _p(0, 4)]
    target = _p(4, 0)
    candidates = [
        [target, _p(-4, 0)],
        [target, _p(0, 8)],
    ]

    selected, _, info = manager.resolve(current, candidates, time_step=1)

    assert info.priority_order[0] == 1
    assert np.allclose(selected[1], target)
    assert not np.allclose(selected[0], target)


def test_backtracking_revises_an_earlier_choice():
    manager = DeadlockAwareMotionManager(2, safe_distance=1.0, max_backtracking_nodes=100)
    current = [_p(0, 0), _p(4, 0)]

    # Robot 0's first choice enters Robot 1's current node. Robot 1 can then
    # neither swap into Robot 0's node nor wait, so the resolver must backtrack
    # and use Robot 0's second choice.
    candidates = [
        [_p(4, 0), _p(0, 4)],
        [_p(0, 0)],
    ]

    selected, _, info = manager.resolve(current, candidates, time_step=1)

    assert info.backtracking_nodes > 2
    assert np.allclose(selected[0], _p(0, 4))
    assert np.allclose(selected[1], _p(0, 0))


def test_search_limit_uses_conservative_fallback():
    manager = DeadlockAwareMotionManager(2, safe_distance=1.0, max_backtracking_nodes=1)
    current = [_p(0, 0), _p(4, 0)]
    candidates = [[_p(0, 4)], [_p(4, 4)]]

    selected, _, info = manager.resolve(current, candidates, time_step=1)

    assert info.backtracking_nodes == 1
    assert all(np.allclose(selected[i], current[i]) for i in range(2))


def test_escape_selects_only_the_highest_priority_starving_robot():
    manager = DeadlockAwareMotionManager(3, safe_distance=1.0, deadlock_wait_threshold=3)
    manager.wait_steps[:] = [3, 5, 4]

    escape = manager.escape_robot_ids()

    assert escape == {1}


def test_successful_escape_is_recorded_and_wait_counts_update():
    manager = DeadlockAwareMotionManager(2, safe_distance=1.0, deadlock_wait_threshold=3)
    manager.wait_steps[:] = [3, 0]
    assert manager.escape_robot_ids() == {0}

    previous = [_p(0, 0), _p(4, 0)]
    actual = [_p(0, 4), _p(4, 0)]
    manager.update_after_execution(previous, actual)

    assert manager.wait_steps.tolist() == [0, 1]
    assert manager.deadlock_break_events == 1
    assert manager.priority_token == 1


def test_shared_depot_waiting_is_allowed_only_at_step_zero():
    manager = DeadlockAwareMotionManager(2, safe_distance=1.0)
    current = [_p(0, 0), _p(0, 0)]
    candidates = [[_p(0, 0)], [_p(0, 0)]]

    selected, _, _ = manager.resolve(
        current,
        candidates,
        time_step=0,
        allow_shared_start_step0=True,
    )

    assert all(np.allclose(position, _p(0, 0)) for position in selected)


def test_shared_depot_waiting_remains_legal_after_step_zero():
    manager = DeadlockAwareMotionManager(3, safe_distance=0.75)
    current = [np.array([0.0, 0.0], dtype=np.float32) for _ in range(3)]
    candidates = [
        [np.array([1.0, 0.0], dtype=np.float32), current[0]],
        [current[1]],
        [current[2]],
    ]

    selected, _, _ = manager.resolve(
        current,
        candidates,
        time_step=8,
        allow_shared_start_step0=True,
    )

    assert np.allclose(selected[0], [1.0, 0.0])
    assert np.allclose(selected[1], [0.0, 0.0])
    assert np.allclose(selected[2], [0.0, 0.0])


def test_serial_departure_fallback_moves_one_robot_when_joint_search_is_capped():
    manager = DeadlockAwareMotionManager(
        4,
        safe_distance=0.75,
        max_backtracking_nodes=1,
    )
    current = [np.array([0.0, 0.0], dtype=np.float32) for _ in range(4)]
    candidates = [
        [np.array([1.0, 0.0], dtype=np.float32), current[i]]
        for i in range(4)
    ]

    selected, _, info = manager.resolve(
        current,
        candidates,
        time_step=5,
        allow_shared_start_step0=True,
    )

    moved = [i for i in range(4) if not np.allclose(selected[i], current[i])]
    assert moved == [0]
    assert info.used_serial_fallback


def test_new_overlap_is_still_rejected_away_from_shared_depot():
    manager = DeadlockAwareMotionManager(2, safe_distance=0.75)
    current = [
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([2.0, 0.0], dtype=np.float32),
    ]
    candidates = [
        [np.array([1.0, 0.0], dtype=np.float32), current[0]],
        [np.array([1.0, 0.0], dtype=np.float32), current[1]],
    ]

    selected, _, _ = manager.resolve(
        current,
        candidates,
        time_step=9,
        allow_shared_start_step0=True,
    )

    assert not (
        np.allclose(selected[0], [1.0, 0.0])
        and np.allclose(selected[1], [1.0, 0.0])
    )
