from __future__ import annotations

import numpy as np

from parameter import FREE, OCCUPIED
from classes.env.multi_robot_env import MultiRobotEnv


def _fake_env(grid, cell_size=1.0):
    env = MultiRobotEnv.__new__(MultiRobotEnv)
    env.ground_truth = np.asarray(grid, dtype=np.int32)
    env.cell_size = float(cell_size)
    env.belief_origin_x = 0.0
    env.belief_origin_y = 0.0
    env.start_clearance = 0.6
    return env


def test_clear_centre_is_accepted():
    grid = np.full((7, 7), FREE, dtype=np.int32)
    env = _fake_env(grid)
    assert env.is_start_clear_world((3.0, 3.0), 0.6)


def test_wall_intersection_is_rejected():
    grid = np.full((7, 7), FREE, dtype=np.int32)
    grid[3, 4] = OCCUPIED
    env = _fake_env(grid)
    assert not env.is_start_clear_world((3.0, 3.0), 0.6)


def test_map_boundary_is_rejected():
    grid = np.full((7, 7), FREE, dtype=np.int32)
    env = _fake_env(grid)
    assert not env.is_start_clear_world((0.0, 0.0), 0.6)


def test_occupied_centre_is_rejected():
    grid = np.full((7, 7), FREE, dtype=np.int32)
    grid[3, 3] = OCCUPIED
    env = _fake_env(grid)
    assert not env.is_start_clear_world((3.0, 3.0), 0.1)
