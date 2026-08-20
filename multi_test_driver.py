"""Evaluate 1/2/4/6/8 robots from four distinct shared depots on each map.

For every selected map, four distinct free graph nodes are sampled once. Each
node defines one trial: all robots in that trial start from exactly the same
node. The same four depots are reused for every team size and communication
mode, enabling paired comparisons.

Examples:
    python multi_test_driver.py --maps all --map-count 100 --profile original_dare
    python multi_test_driver.py --maps 3 --team-sizes 2,4,6,8
    python multi_test_driver.py --maps 1,4,9 --profile coordinated
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import time
from pathlib import Path
from typing import Sequence

import dill
import hydra
import numpy as np
import torch

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from test_parameter import USE_GPU, USE_TEST_DATASET, checkpoint_name, run_path
from classes.env.multi_robot_env import MultiRobotEnv
from multi_test_parameter import (
    BASE_SEED,
    COMMUNICATION_MODES,
    MAP_TEST_MODE,
    MIN_START_SEPARATION,
    SPECIFIED_MAP_INDICES,
    START_SAMPLES_PER_MAP,
    START_CLEARANCE,
    TEAM_DONE_TOLERANCE_CELLS,
    TEAM_SIZES,
    TOTAL_MAP_COUNT,
)
from multi_test_worker import MultiRobotTestWorker


def load_frozen_policy(device: torch.device):
    checkpoint = os.path.join(run_path, "checkpoints", checkpoint_name)
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    # 中文目的：让工作器记录真实加载的 checkpoint 路径和文件大小。
    # English purpose: expose the actual checkpoint identity to runtime metrics.
    os.environ["DARE_CHECKPOINT_PATH"] = str(Path(checkpoint).resolve())
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    workspace_class = hydra.utils.get_class(cfg._target_)
    output_dir = os.path.join(run_path, "multi_robot_inference")
    workspace = workspace_class(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(device).eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return policy


def _numeric_map_indices(directory: Path) -> list[int]:
    if not directory.is_dir():
        return []
    indices: set[int] = set()
    for path in directory.iterdir():
        if path.is_file() and (match := re.search(r"(\d+)", path.stem)):
            indices.add(int(match.group(1)))
    if not indices:
        return []
    ordered = sorted(indices)
    if ordered == list(range(1, len(ordered) + 1)):
        return list(range(len(ordered)))
    if ordered == list(range(len(ordered))):
        return ordered
    return list(range(len(ordered)))


def discover_all_map_indices(explicit_count: int | None) -> list[int]:
    if explicit_count is not None:
        if explicit_count <= 0:
            raise ValueError("map count must be positive")
        return list(range(explicit_count))
    for directory in (
        Path("data/test"), Path("data/test_data"), Path("test_data"),
        Path("maps/test"), Path("dataset/test"), Path("datasets/test"),
    ):
        indices = _numeric_map_indices(directory)
        if indices:
            print(f"Discovered {len(indices)} maps from {directory}")
            return indices
    raise RuntimeError(
        "Could not discover test-map count. Set TOTAL_MAP_COUNT or use --map-count N."
    )


def parse_map_selection(value: str | None, map_count: int | None) -> list[int]:
    if value is None:
        if MAP_TEST_MODE == "all":
            return discover_all_map_indices(map_count if map_count is not None else TOTAL_MAP_COUNT)
        return sorted({int(i) for i in SPECIFIED_MAP_INDICES})
    if value.strip().lower() == "all":
        return discover_all_map_indices(map_count if map_count is not None else TOTAL_MAP_COUNT)
    try:
        indices = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    except ValueError as exc:
        raise ValueError("--maps must be 'all' or comma-separated integer indices") from exc
    if not indices or any(index < 0 for index in indices):
        raise ValueError("Specified map indices must be non-negative")
    return indices


def depot_seed(map_index: int) -> int:
    # Independent of team size and communication mode for paired experiments.
    return int(BASE_SEED + 100_000_000 + 10_000 * map_index)


def run_seed(map_index: int, start_sample: int, team_size: int, mode: str) -> int:
    """Use the same stochastic seed for every mode in a paired scenario."""

    if mode not in {"none", "raw", "compressed"}:
        raise ValueError(f"Unsupported communication mode: {mode}")
    return int(BASE_SEED + 10_000 * map_index + 100 * team_size + start_sample)


def sample_four_distinct_depots(map_index: int) -> np.ndarray:
    """Sample four separated free nodes once for a map.

    A temporary four-agent environment is used only as a simulator-side depot
    sampler. These four nodes are then reused as shared starts in all 2/4/6/8
    robot runs on the same map.
    """
    sampler = MultiRobotEnv(
        episode_index=map_index,
        n_agents=START_SAMPLES_PER_MAP,
        test=USE_TEST_DATASET,
        seed=depot_seed(map_index),
        min_start_separation=MIN_START_SEPARATION,
        done_tolerance_cells=TEAM_DONE_TOLERANCE_CELLS,
        start_clearance=START_CLEARANCE,
    )
    depots = np.asarray(sampler.robot_locations, dtype=np.float32)
    if depots.shape != (4, 2):
        raise RuntimeError(f"Expected four depot positions, got {depots.shape}")
    if len({tuple(np.round(point, 5)) for point in depots}) != 4:
        raise RuntimeError("Depot sampler returned duplicate starts")
    return depots


def write_results(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)



def parse_team_sizes(value: str) -> tuple[int, ...]:
    """Parse the robot-count list used by the original DARE test driver.

    中文目的：允许论文扩展性实验直接指定机器人数量，而不修改配置文件。
    中文实现：按输入顺序解析逗号分隔正整数，并拒绝空列表和重复值。
    English purpose: expose robot counts for original-DARE scalability tests.
    English implementation: parses ordered unique positive integers.
    """

    try:
        values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError("--team-sizes must contain comma-separated integers") from exc
    if not values or any(value <= 0 for value in values):
        raise ValueError("--team-sizes must contain positive integers")
    if len(values) != len(set(values)):
        raise ValueError("--team-sizes must not contain duplicate values")
    return values

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", default=None, help="all, one index, or comma-separated indices")
    parser.add_argument("--map-count", type=int, default=None)
    parser.add_argument("--modes", default=",".join(COMMUNICATION_MODES))
    parser.add_argument(
        "--team-sizes",
        default=",".join(str(value) for value in TEAM_SIZES),
        help="comma-separated robot counts; e.g. 1,2,4,6,8",
    )
    parser.add_argument(
        "--profile",
        choices=("original_dare", "coordinated"),
        default="coordinated",
        help=(
            "original_dare disables all added communication/collision/deadlock "
            "wrappers; coordinated preserves the previous enhanced test"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    team_sizes = parse_team_sizes(args.team_sizes)
    if START_SAMPLES_PER_MAP != 4:
        raise ValueError("START_SAMPLES_PER_MAP must be exactly 4")

    map_indices = parse_map_selection(args.maps, args.map_count)
    modes = tuple(part.strip() for part in args.modes.split(",") if part.strip())
    if args.profile == "original_dare":
        modes = ("none",)
    invalid_modes = set(modes) - {"none", "raw", "compressed"}
    if invalid_modes:
        raise ValueError(f"Unsupported communication modes: {sorted(invalid_modes)}")

    device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
    policy = load_frozen_policy(device)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    map_label = "all" if args.maps == "all" else "maps_" + "-".join(str(index) for index in map_indices)
    mode_label = "-".join(modes)
    session_name = f"run_{stamp}_{map_label}_profile_{args.profile}_modes_{mode_label}"

    # Keep the CSV and all visual outputs for this invocation under one
    # timestamped session directory so repeated experiments never overwrite or
    # mix with one another.
    session_dir = Path(run_path) / "multi_robot_outputs" / session_name
    session_dir.mkdir(parents=True, exist_ok=False)
    csv_path = session_dir / "results.csv"

    depot_by_map = {map_index: sample_four_distinct_depots(map_index) for map_index in map_indices}
    results: list[dict] = []
    total = len(map_indices) * START_SAMPLES_PER_MAP * len(team_sizes) * len(modes)
    run_number = 0

    for mode in modes:
        for map_index in map_indices:
            depots = depot_by_map[map_index]
            for start_sample, depot in enumerate(depots):
                for team_size in team_sizes:
                    run_number += 1
                    seed = run_seed(map_index, start_sample, team_size, mode)
                    print(
                        f"[{run_number}/{total}] map={map_index} trial={start_sample} "
                        f"shared_start=({depot[0]:.3f},{depot[1]:.3f}) "
                        f"robots={team_size} mode={mode} seed={seed}"
                    )
                    start_x = float(depot[0])
                    start_y = float(depot[1])
                    experiment_name = (
                        f"map_{map_index:04d}"
                        f"__start_{start_sample:02d}_x{start_x:.2f}_y{start_y:.2f}"
                        f"__robots_{team_size:02d}"
                        f"__mode_{mode}"
                        f"__seed_{seed}"
                    )
                    experiment_dir = session_dir / experiment_name
                    experiment_dir.mkdir(parents=True, exist_ok=False)

                    worker = MultiRobotTestWorker(
                        policy=policy,
                        episode_index=map_index,
                        n_agents=team_size,
                        device=device,
                        seed=seed,
                        communication_mode=mode,
                        start_position=depot,
                        start_sample=start_sample,
                        visual_output_root=str(experiment_dir),
                        test_profile=args.profile,
                        scenario_id=(
                            f"map_{map_index:04d}_trial_{start_sample:02d}_"
                            f"robots_{team_size:02d}_seed_{seed}"
                        ),
                        trial=start_sample,
                    )
                    metrics = worker.run_episode()
                    results.append(metrics)
                    write_results(csv_path, results)
                    print(
                        "  coverage={:.3f} steps={} success={} max_wait={}".format(
                            metrics["team_coverage"], metrics["steps"], metrics["success"],
                            metrics["max_consecutive_wait_steps"],
                        )
                    )

    print(f"Session folder: {session_dir}")
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()