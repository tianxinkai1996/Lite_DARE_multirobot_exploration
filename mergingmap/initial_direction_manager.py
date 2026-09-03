"""Balanced random cardinal-direction roles for early multi-robot departure."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import random
from typing import List, Sequence

import numpy as np


CARDINAL_DIRECTIONS: tuple[tuple[str, tuple[float, float]], ...] = (
    ("north", (0.0, 1.0)),
    ("east", (1.0, 0.0)),
    ("south", (0.0, -1.0)),
    ("west", (-1.0, 0.0)),
)


@dataclass(frozen=True)
class InitialDirectionConfig:
    enabled: bool = True
    bias_steps: int = 12
    max_bias_weight: float = 0.80
    decay: bool = True

    def __post_init__(self):
        if self.bias_steps < 0:
            raise ValueError("bias_steps cannot be negative")
        if not 0.0 <= self.max_bias_weight <= 1.0:
            raise ValueError("max_bias_weight must be between 0 and 1")


@dataclass(frozen=True)
class DirectionRole:
    robot_id: int
    name: str
    vector: tuple[float, float]


class InitialDirectionManager:
    """Assign reproducible cardinal roles and bias early DARE candidates.

    Assignment is random but balanced. Cardinal directions are shuffled in
    blocks of four. Consequently, up to four robots receive unique directions;
    eight robots receive every direction exactly twice. The selected role is
    fixed for the episode rather than sampled again at every step.
    """

    def __init__(self, n_robots, seed, config):
        if n_robots <= 0:
            raise ValueError("n_robots must be positive")
        self.n_robots = int(n_robots)
        self.seed = int(seed)
        self.config = config
        self.roles = self._assign_roles()
        self.candidate_overrides = 0
        self.steps_applied = 0

    def _assign_roles(self):
        rng = random.Random(self.seed ^ 0x4D524449)
        assignments: list[tuple[str, tuple[float, float]]] = []
        while len(assignments) < self.n_robots:
            block = list(CARDINAL_DIRECTIONS)
            rng.shuffle(block)
            assignments.extend(block)
        return tuple(
            DirectionRole(robot_id, name, vector)
            for robot_id, (name, vector) in enumerate(
                assignments[: self.n_robots]
            )
        )

    def role_for_robot(self, robot_id):
        robot_id = int(robot_id)
        if not 0 <= robot_id < self.n_robots:
            raise IndexError(f"robot_id {robot_id} is out of range")
        return self.roles[robot_id]

    def bias_weight(self, step):
        if not self.config.enabled or self.config.bias_steps <= 0:
            return 0.0
        step = int(step)
        if step < 0 or step >= self.config.bias_steps:
            return 0.0
        if not self.config.decay or self.config.bias_steps == 1:
            return float(self.config.max_bias_weight)
        fraction = 1.0 - step / float(self.config.bias_steps - 1)
        return float(self.config.max_bias_weight * max(0.0, fraction))

    def order_candidates(self, robot_id, current_position, ordered_candidates, step):
        """Blend frozen DARE rank with the robot's early cardinal role."""
        candidates = [
            np.asarray(candidate, dtype=np.float32).copy()
            for candidate in ordered_candidates
        ]
        weight = self.bias_weight(step)
        if weight <= 0.0 or len(candidates) <= 1:
            return candidates

        current = np.asarray(current_position, dtype=float)
        role = self.role_for_robot(robot_id)
        direction = np.asarray(role.vector, dtype=float)
        denominator = max(1, len(candidates) - 1)
        scored = []

        for dare_rank, candidate in enumerate(candidates):
            movement = np.asarray(candidate, dtype=float) - current
            norm = float(np.linalg.norm(movement))
            is_wait = norm <= 1e-8
            if is_wait:
                alignment_cost = 1.25
            else:
                cosine = float(
                    np.clip(np.dot(movement, direction) / norm, -1.0, 1.0)
                )
                alignment_cost = float(math.acos(cosine) / math.pi)

            dare_cost = dare_rank / float(denominator)
            combined = (1.0 - weight) * dare_cost + weight * alignment_cost
            scored.append((combined, 1 if is_wait else 0, dare_rank, candidate))

        scored.sort(key=lambda item: (item[0], item[1], item[2]))
        result = [item[3] for item in scored]
        self.steps_applied += 1
        if not np.allclose(result[0], candidates[0], atol=1e-4):
            self.candidate_overrides += 1
        return result

    def assignment_payload(self):
        return [
            {
                "robot_id": role.robot_id,
                "direction": role.name,
                "vector": [float(role.vector[0]), float(role.vector[1])],
            }
            for role in self.roles
        ]

    def assignments_json(self):
        return json.dumps(self.assignment_payload(), ensure_ascii=False)

    def metrics(self):
        return {
            "initial_direction_enabled": bool(self.config.enabled),
            "initial_direction_bias_steps": int(self.config.bias_steps),
            "initial_direction_max_bias_weight": float(
                self.config.max_bias_weight
            ),
            "initial_direction_assignments": self.assignments_json(),
            "initial_direction_candidate_overrides": int(
                self.candidate_overrides
            ),
            "initial_direction_steps_applied": int(self.steps_applied),
        }
