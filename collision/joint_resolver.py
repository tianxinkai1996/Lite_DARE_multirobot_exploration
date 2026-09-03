"""Bounded synchronous collision resolver with deterministic fallback.

"""
from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

from collision.candidates import blocked_robot_ids, normalise_candidates
from collision.geometry import is_wait, pair_conflict, shared_overlap_exists
from collision.models import CollisionResolution

class JointCollisionResolver:
    """Resolve a joint one-step action without vertex or edge-swap conflicts.

    English implementation: performs priority-ordered bounded backtracking. When
    a permitted shared-depot overlap exists and the search budget is exhausted,
    it attempts a one-robot serial-departure fallback.
    """

    def __init__(self, n_agents, safe_distance, max_backtracking_nodes=20000):
        """Initialise one bounded collision resolver.

        English purpose: configure collision resolution for a fixed robot team.
        English implementation: validates limits, stores search parameters, and
        initialises the serial-departure fallback counter.
        """

        if n_agents <= 0:
            raise ValueError("n_agents must be positive")
        if safe_distance <= 0:
            raise ValueError("safe_distance must be positive")
        if max_backtracking_nodes < 1:
            raise ValueError("max_backtracking_nodes must be at least 1")
        self.n_agents = int(n_agents)
        self.safe_distance = float(safe_distance)
        self.max_backtracking_nodes = int(max_backtracking_nodes)
        self.serial_fallback_events = 0

    def _feasible_against_assignment(self, robot_id, candidate, current, assignment, allow_shared_start_wait):
        """Check one candidate against already assigned higher-priority robots.

        English implementation: applies pairwise conflict predicates only to
        entries that already have a selected next position.
        """

        for other_id, other_next in enumerate(assignment):
            if other_next is None:
                continue
            if pair_conflict(
                current[robot_id],
                candidate,
                current[other_id],
                other_next,
                safe_distance=self.safe_distance,
                allow_shared_start_wait=allow_shared_start_wait,
            ):
                return False
        return True

    def _serial_departure(self, current, candidates, priority_order, allow_shared_start_wait):
        """Move at most one priority robot while all peers remain waiting.

        English implementation: scans priority order and returns the first moving
        candidate compatible with every other robot's waiting action.
        """

        selected = [position.copy() for position in current]
        for robot_id in priority_order:
            for candidate in candidates[robot_id]:
                if is_wait(current[robot_id], candidate):
                    continue
                if all(
                    other_id == robot_id
                    or not pair_conflict(
                        current[robot_id],
                        candidate,
                        current[other_id],
                        current[other_id],
                        safe_distance=self.safe_distance,
                        allow_shared_start_wait=allow_shared_start_wait,
                    )
                    for other_id in range(self.n_agents)
                ):
                    selected[robot_id] = candidate.copy()
                    self.serial_fallback_events += 1
                    return selected
        return selected

    def resolve(self, current_positions, candidate_lists, priority_order=None, time_step=0, allow_shared_start_step0=False):
        """Choose one collision-free next position for every robot.

        `time_step` 

        English implementation: normalises candidates, runs bounded recursive
        assignment, and returns conservative waits when no legal assignment is
        found outside a permitted shared-depot cluster.
        """

        del time_step
        current, candidates = normalise_candidates(current_positions, candidate_lists)
        if len(current) != self.n_agents:
            raise ValueError("position arrays must match n_agents")

        order = list(range(self.n_agents)) if priority_order is None else list(priority_order)
        if sorted(order) != list(range(self.n_agents)):
            raise ValueError("priority_order must contain every robot ID exactly once")

        assignment: List[np.ndarray | None] = [None] * self.n_agents
        search_nodes = 0

        def search(depth):
            nonlocal search_nodes
            if depth >= len(order):
                return True
            if search_nodes >= self.max_backtracking_nodes:
                return False

            robot_id = order[depth]
            for candidate in candidates[robot_id]:
                if search_nodes >= self.max_backtracking_nodes:
                    return False
                search_nodes += 1
                if not self._feasible_against_assignment(
                    robot_id,
                    candidate,
                    current,
                    assignment,
                    allow_shared_start_wait=allow_shared_start_step0,
                ):
                    continue
                assignment[robot_id] = candidate
                if search(depth + 1):
                    return True
                assignment[robot_id] = None
            return False

        solved = search(0)
        used_serial_fallback = False
        if solved:
            selected = [
                current[robot_id].copy()
                if assignment[robot_id] is None
                else assignment[robot_id].copy()
                for robot_id in range(self.n_agents)
            ]
        elif allow_shared_start_step0 and shared_overlap_exists(
            current, safe_distance=self.safe_distance
        ):
            selected = self._serial_departure(
                current,
                candidates,
                order,
                allow_shared_start_wait=True,
            )
            used_serial_fallback = any(
                not is_wait(current[robot_id], selected[robot_id])
                for robot_id in range(self.n_agents)
            )
        else:
            selected = [position.copy() for position in current]

        blocked = blocked_robot_ids(current, selected, candidates)
        return CollisionResolution(
            positions=tuple(position.copy() for position in selected),
            blocked_robot_ids=tuple(blocked),
            backtracking_nodes=int(search_nodes),
            used_serial_fallback=bool(used_serial_fallback),
        )

def resolve_synchronous_moves(current_positions, candidate_lists, safe_distance, time_step, allow_shared_start_step0=False):
    """Run stateless collision-only resolution for ablation experiments.

    MergingMap 
    English implementation: creates one temporary resolver using robot-ID order
    and returns selected positions plus dynamically blocked robot IDs.
    """

    resolver = JointCollisionResolver(
        len(current_positions),
        safe_distance=safe_distance,
    )
    result = resolver.resolve(
        current_positions,
        candidate_lists,
        priority_order=range(len(current_positions)),
        time_step=time_step,
        allow_shared_start_step0=allow_shared_start_step0,
    )
    return [position.copy() for position in result.positions], list(result.blocked_robot_ids)
