"""Persistent progress, waiting, and staged deadlock-recovery state.

"""
from __future__ import annotations

from typing import Sequence, Set, Tuple

import numpy as np

from deadlock.priority_policy import (
    priority_order,
    priority_score,
    select_escape_robot,
)

class DeadlockStateTracker:
    """Track consecutive waits, no-progress state, priority, and recovery stages.

    English implementation: combines executed wait counters with externally
    supplied no-progress counters, generates weighted rotating priority, exposes
    staged recovery thresholds, and records successful recovery movement.
    """

    def __init__(self, n_agents, wait_threshold=3, stall_threshold=None, soft_relax_threshold=None, lease_release_threshold=None, backtrack_threshold=None, wait_weight=1.0, stall_weight=1.0):
        if n_agents <= 0:
            raise ValueError("n_agents must be positive")
        if wait_threshold < 1:
            raise ValueError("wait_threshold must be at least 1")
        self.n_agents = int(n_agents)
        self.wait_threshold = int(wait_threshold)
        self.stall_threshold = int(stall_threshold or wait_threshold)
        self.soft_relax_threshold = int(
            soft_relax_threshold or max(wait_threshold + 1, 2 * wait_threshold)
        )
        self.lease_release_threshold = int(
            lease_release_threshold
            or max(self.soft_relax_threshold + 1, 3 * wait_threshold)
        )
        self.backtrack_threshold = int(
            backtrack_threshold
            or max(self.lease_release_threshold + 1, 4 * wait_threshold)
        )
        thresholds = (
            self.wait_threshold,
            self.stall_threshold,
            self.soft_relax_threshold,
            self.lease_release_threshold,
            self.backtrack_threshold,
        )
        if any(value < 1 for value in thresholds):
            raise ValueError("recovery thresholds must be positive")
        if wait_weight < 0 or stall_weight < 0:
            raise ValueError("priority weights cannot be negative")
        self.wait_weight = float(wait_weight)
        self.stall_weight = float(stall_weight)
        self.wait_steps = np.zeros(self.n_agents, dtype=np.int64)
        self.stall_steps = np.zeros(self.n_agents, dtype=np.int64)
        self.priority_token = 0
        self.deadlock_break_events = 0
        self.max_wait_steps = 0
        self.max_stall_steps = 0
        self._last_escape_robot_ids: Tuple[int, ...] = ()
        self.stage_activation_counts = {stage: 0 for stage in range(1, 7)}

    def set_stall_steps(self, values):
        """Replace no-progress counters supplied by the region/progress layer.

        English implementation: validates and copies one counter per robot.
        """

        if values is None:
            self.stall_steps[:] = 0
            return
        if len(values) != self.n_agents:
            raise ValueError("stall_steps must match n_agents")
        self.stall_steps[:] = np.maximum(0, np.asarray(values, dtype=np.int64))
        self.max_stall_steps = max(
            self.max_stall_steps, int(np.max(self.stall_steps, initial=0))
        )

    def priorities(self):
        """Return numeric priorities transmitted in short motion messages."""

        return [
            priority_score(
                robot_id,
                self.wait_steps,
                self.stall_steps,
                wait_weight=self.wait_weight,
                stall_weight=self.stall_weight,
            )
            for robot_id in range(self.n_agents)
        ]

    def ordered_robot_ids(self):
        """Return collision-search order from wait, stall, token, and ID."""

        return priority_order(
            self.wait_steps,
            self.priority_token,
            self.stall_steps,
            wait_weight=self.wait_weight,
            stall_weight=self.stall_weight,
        )

    def escape_robot_ids(self):
        """Return at most one robot eligible for staged recovery."""

        selected = select_escape_robot(
            self.wait_steps,
            self.priority_token,
            self.wait_threshold,
            stall_steps=self.stall_steps,
            stall_threshold=self.stall_threshold,
            wait_weight=self.wait_weight,
            stall_weight=self.stall_weight,
        )
        self._last_escape_robot_ids = () if selected is None else (selected,)
        return set(self._last_escape_robot_ids)

    def recovery_stage(self, robot_id):
        """Return the current staged-recovery level from zero to six.

        0 normal; 1 short wait; 2 dynamic-priority re-resolution; 3 soft-cost removal;
        4 region-lease release; 5 graph backtracking; 6 hold still when all hard safety
        actions fail.
        """

        robot_id = int(robot_id)
        pressure = max(
            int(self.wait_steps[robot_id]), int(self.stall_steps[robot_id])
        )
        if pressure <= 0:
            return 0
        if pressure < self.wait_threshold:
            return 1
        if pressure < self.soft_relax_threshold:
            return 2
        if pressure < self.lease_release_threshold:
            return 3
        if pressure < self.backtrack_threshold:
            return 4
        return 5

    def recovery_stages(self):
        """Return all robot stages and count active stages for diagnostics."""

        stages = tuple(self.recovery_stage(robot_id) for robot_id in range(self.n_agents))
        for stage in stages:
            if stage > 0:
                self.stage_activation_counts[stage] += 1
        return stages

    def record_forced_stationary(self, robot_ids):
        """Count final-stage waits when every hard-safe moving action fails.

        English implementation: increments stage-six diagnostics after collision
        resolution confirms that a late-stage recovery robot still cannot move.
        """

        self.stage_activation_counts[6] += len({int(value) for value in robot_ids})

    def update_after_execution(self, previous_positions, actual_positions):
        """Update waiting history from actions accepted by the environment."""

        if len(previous_positions) != self.n_agents or len(actual_positions) != self.n_agents:
            raise ValueError("position arrays must match n_agents")
        escaped_and_moved = False
        for robot_id, (previous, actual) in enumerate(
            zip(previous_positions, actual_positions)
        ):
            if np.allclose(previous, actual):
                self.wait_steps[robot_id] += 1
            else:
                if robot_id in self._last_escape_robot_ids:
                    escaped_and_moved = True
                self.wait_steps[robot_id] = 0
        if escaped_and_moved:
            self.deadlock_break_events += 1
        self.max_wait_steps = max(
            self.max_wait_steps, int(np.max(self.wait_steps, initial=0))
        )
        self.priority_token = (self.priority_token + 1) % self.n_agents

    def metrics(self):
        """Return episode-level deadlock and staged-recovery diagnostics."""

        output: dict[str, int | float] = {
            "deadlock_break_events": int(self.deadlock_break_events),
            "deadlock_max_wait_steps": int(self.max_wait_steps),
            "deadlock_max_stall_steps": int(self.max_stall_steps),
            "deadlock_wait_priority_weight": float(self.wait_weight),
            "deadlock_stall_priority_weight": float(self.stall_weight),
        }
        for stage, count in self.stage_activation_counts.items():
            output[f"deadlock_recovery_stage_{stage}_activations"] = int(count)
        return output
