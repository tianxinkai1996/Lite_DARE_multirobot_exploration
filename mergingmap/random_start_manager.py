from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class RandomStartConfig:
    """Validation and retry settings for multi-robot initial positions."""

    free_value: int
    min_separation: float
    max_attempts: int = 100
    clearance_radius_cells: int = 0

    def __post_init__(self) -> None:
        if self.min_separation < 0:
            raise ValueError("min_separation cannot be negative")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if self.clearance_radius_cells < 0:
            raise ValueError("clearance_radius_cells cannot be negative")


@dataclass(frozen=True)
class StartValidationResult:
    valid: bool
    reasons: tuple[str, ...]
    cell_positions: tuple[tuple[int, int], ...]
    minimum_pairwise_distance: float


@dataclass(frozen=True)
class RandomStartSelection:
    env: object
    selected_seed: int
    attempt_count: int
    validation: StartValidationResult


def _as_grid(value: object) -> Optional[np.ndarray]:
    """Convert a ground-truth container or MapInfo-like object to a 2-D grid."""
    if value is None:
        return None

    if hasattr(value, "map"):
        value = getattr(value, "map")

    try:
        grid = np.asarray(value)
    except Exception:
        return None

    return grid if grid.ndim == 2 else None


def extract_ground_truth_grid(env: object) -> np.ndarray:
    """Return the environment's 2-D ground-truth occupancy grid.

    The implementation accepts the attribute names used by common DARE forks.
    It fails loudly instead of silently validating starts against a local belief.
    """
    for name in (
        "ground_truth",
        "ground_truth_map",
        "ground_truth_belief",
        "ground_truth_info",
        "map_info",
    ):
        if hasattr(env, name):
            grid = _as_grid(getattr(env, name))
            if grid is not None:
                return grid

    raise AttributeError(
        "Cannot validate random starts because MultiRobotEnv exposes no "
        "2-D ground-truth grid. Expected one of: ground_truth, "
        "ground_truth_map, ground_truth_belief, ground_truth_info, map_info."
    )


def _world_to_cell(env: object, position: Sequence[float]) -> tuple[int, int]:
    if not hasattr(env, "world_to_cell"):
        raise AttributeError(
            "MultiRobotEnv must provide world_to_cell() for start validation"
        )
    cell = np.asarray(env.world_to_cell(position), dtype=float).reshape(-1)
    if cell.size < 2 or not np.all(np.isfinite(cell[:2])):
        raise ValueError(f"Invalid cell returned for position {position}: {cell}")
    return int(np.rint(cell[0])), int(np.rint(cell[1]))


def _minimum_pairwise_distance(positions: np.ndarray) -> float:
    if len(positions) < 2:
        return float("inf")
    minimum = float("inf")
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            minimum = min(
                minimum,
                float(np.linalg.norm(positions[i] - positions[j])),
            )
    return minimum


def validate_random_starts(
    env: object,
    *,
    n_agents: int,
    config: RandomStartConfig,
) -> StartValidationResult:
    """Check count, uniqueness, free-space membership, clearance and separation."""
    reasons: list[str] = []

    if not hasattr(env, "robot_locations"):
        return StartValidationResult(
            False,
            ("environment has no robot_locations",),
            (),
            0.0,
        )

    positions = np.asarray(env.robot_locations, dtype=float)
    if positions.shape != (int(n_agents), 2):
        reasons.append(
            f"robot_locations shape is {positions.shape}, "
            f"expected ({int(n_agents)}, 2)"
        )

    if positions.ndim != 2 or positions.shape[1:] != (2,):
        return StartValidationResult(False, tuple(reasons), (), 0.0)

    if not np.all(np.isfinite(positions)):
        reasons.append("one or more start positions are not finite")

    ground_truth = extract_ground_truth_grid(env)
    height, width = ground_truth.shape
    cells: list[tuple[int, int]] = []

    for robot_id, position in enumerate(positions):
        try:
            x, y = _world_to_cell(env, position)
        except Exception as exc:
            reasons.append(f"robot {robot_id}: world_to_cell failed: {exc}")
            continue

        cells.append((x, y))
        if x < 0 or x >= width or y < 0 or y >= height:
            reasons.append(
                f"robot {robot_id}: cell {(x, y)} is outside map bounds "
                f"(width={width}, height={height})"
            )
            continue

        if int(ground_truth[y, x]) != int(config.free_value):
            reasons.append(
                f"robot {robot_id}: start cell {(x, y)} is not FREE "
                f"(value={int(ground_truth[y, x])})"
            )
            continue

        radius = int(config.clearance_radius_cells)
        if radius > 0:
            x0, x1 = max(0, x - radius), min(width - 1, x + radius)
            y0, y1 = max(0, y - radius), min(height - 1, y + radius)
            neighbourhood = ground_truth[y0 : y1 + 1, x0 : x1 + 1]
            if np.any(neighbourhood != int(config.free_value)):
                reasons.append(
                    f"robot {robot_id}: start cell {(x, y)} does not have "
                    f"{radius}-cell free-space clearance"
                )

    if len(cells) == int(n_agents) and len(set(cells)) != len(cells):
        reasons.append("two or more robots occupy the same grid cell")

    # Also reject equal world coordinates even if a custom cell conversion
    # unexpectedly maps them differently.
    rounded_world = {
        tuple(np.round(position, decimals=6))
        for position in positions
    }
    if len(rounded_world) != len(positions):
        reasons.append("two or more robots have the same world position")

    minimum_distance = _minimum_pairwise_distance(positions)
    if (
        len(positions) >= 2
        and minimum_distance + 1e-8 < float(config.min_separation)
    ):
        reasons.append(
            f"minimum pairwise distance {minimum_distance:.3f} is below "
            f"required {float(config.min_separation):.3f}"
        )

    return StartValidationResult(
        valid=not reasons,
        reasons=tuple(reasons),
        cell_positions=tuple(cells),
        minimum_pairwise_distance=float(minimum_distance),
    )


def derived_attempt_seed(base_seed: int, attempt_index: int) -> int:
    """Return a deterministic but well-separated seed for a retry."""
    if attempt_index < 0:
        raise ValueError("attempt_index cannot be negative")
    # The multiplier is prime and keeps different worker seeds from producing
    # the same short retry sequence.
    return int((int(base_seed) + 1_000_003 * attempt_index) % (2**31 - 1))


def create_environment_with_valid_random_starts(
    env_factory: Callable[[int], object],
    *,
    base_seed: int,
    n_agents: int,
    config: RandomStartConfig,
) -> RandomStartSelection:
    """Create environments until all initial positions are distinct and free.

    Reconstructing the environment, rather than editing robot_locations after
    construction, ensures that local belief maps, sensor observations and
    internal robot state are initialized consistently with the selected starts.
    """
    failures: list[str] = []

    for attempt_index in range(config.max_attempts):
        candidate_seed = derived_attempt_seed(base_seed, attempt_index)
        env = env_factory(candidate_seed)
        validation = validate_random_starts(
            env,
            n_agents=n_agents,
            config=config,
        )
        if validation.valid:
            return RandomStartSelection(
                env=env,
                selected_seed=candidate_seed,
                attempt_count=attempt_index + 1,
                validation=validation,
            )

        failures.append(
            f"attempt={attempt_index + 1}, seed={candidate_seed}: "
            + "; ".join(validation.reasons)
        )

    preview = "\n".join(failures[-5:])
    raise RuntimeError(
        "Failed to generate valid distinct random starts after "
        f"{config.max_attempts} attempts. Last failures:\n{preview}"
    )