"""Dedicated driver for random-start merging-map experiments.

Unlike the project's original ``multi_test_driver.py``, this driver never
samples or injects a shared depot. Every worker constructs and validates its
own separated random starts. It also relies on the worker to assign balanced
random north/east/south/west primary directions at episode start.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Sequence

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

import torch

from test_parameter import USE_GPU
from mergingmap.multi_test_parameter import (
    BASE_SEED,
    COMMUNICATION_MODES,
    DARE_CHECKPOINT_PATH,
    MAP_TEST_REPEATS,
    TEAM_SIZES,
)

def load_frozen_policy(device, checkpoint_path=None):
    import dill
    import hydra
    from diffusion_policy.workspace.base_workspace import BaseWorkspace

    checkpoint = Path(checkpoint_path or os.environ.get(
        "DARE_CHECKPOINT_PATH", DARE_CHECKPOINT_PATH
    )).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    with checkpoint.open("rb") as handle:
        payload = torch.load(handle, pickle_module=dill)
    cfg = payload["cfg"]
    workspace_class = hydra.utils.get_class(cfg._target_)
    output_dir = Path(
        os.environ.get(
            "MERGINGMAP_RUN_DIR",
            str(checkpoint.parents[1] / "mergingmap_inference"),
        )
    ) / "policy_workspace"
    output_dir.mkdir(parents=True, exist_ok=True)

    workspace = workspace_class(cfg, output_dir=str(output_dir))
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(device).eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return policy

def _numeric_map_indices(directory):
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

def discover_all_map_indices(explicit_count):
    if explicit_count is not None:
        if explicit_count <= 0:
            raise ValueError("--map-count must be positive")
        return list(range(explicit_count))
    for directory in (
        Path("data/test"),
        Path("data/test_data"),
        Path("test_data"),
        Path("maps/test"),
        Path("dataset/test"),
        Path("datasets/test"),
    ):
        indices = _numeric_map_indices(directory)
        if indices:
            print(f"[MERGINGMAP-DRIVER] discovered {len(indices)} maps from {directory}")
            return indices
    raise RuntimeError(
        "Could not discover test-map count. Supply --map-count N or explicit --maps indices."
    )

def parse_map_selection(value, map_count):
    """Parse ``all``, comma-separated indices, and inclusive ranges.

    English implementation: retains automatic discovery for ``all`` and accepts
    explicit inclusive ranges without changing dataset indexing.
    """

    if value.strip().lower() == "all":
        return discover_all_map_indices(map_count)
    result: set[int] = set()
    try:
        for part in value.split(","):
            token = part.strip()
            if not token:
                continue
            if "-" in token:
                left, right = token.split("-", 1)
                first, last = int(left), int(right)
                if first < 0 or last < first:
                    raise ValueError(f"invalid map range {token!r}")
                result.update(range(first, last + 1))
            else:
                index = int(token)
                if index < 0:
                    raise ValueError("map index must be non-negative")
                result.add(index)
    except ValueError as exc:
        raise ValueError(
            "--maps must be 'all', comma-separated non-negative integers, "
            "or inclusive ranges such as 0,3,7-10"
        ) from exc
    if not result:
        raise ValueError("--maps must contain at least one map index")
    return sorted(result)

def run_seed(map_index, trial, team_size, mode, base_seed=BASE_SEED):
    """Return a mode-independent seed for paired ablation experiments.

    English implementation: one base seed defines the experiment, while deterministic
    offsets produce distinct but paired map/trial/team scenarios.
    """

    if mode not in {"none", "raw", "compressed"}:
        raise ValueError(f"Unsupported communication mode: {mode}")
    return int(
        int(base_seed)
        + 100_000 * int(map_index)
        + 1_000 * int(trial)
        + 10 * int(team_size)
    )

def write_results(path, rows):
    if not rows:
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

def build_parser():
    parser = argparse.ArgumentParser(
        description="Random-start merging-map multi-robot DARE driver"
    )
    parser.add_argument(
        "--maps",
        default="all",
        help="all, one map index, or comma-separated map indices",
    )
    parser.add_argument("--map-count", type=int, default=None)
    parser.add_argument(
        "--modes",
        default=",".join(COMMUNICATION_MODES),
        help="comma-separated subset of none,raw,compressed",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="record an error and continue with later runs",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="print the run matrix without loading the checkpoint",
    )
    return parser

def main(argv=None):
    args = build_parser().parse_args(argv)
    map_indices = parse_map_selection(args.maps, args.map_count)
    modes = tuple(part.strip() for part in args.modes.split(",") if part.strip())
    invalid_modes = set(modes) - {"none", "raw", "compressed"}
    if invalid_modes:
        raise ValueError(f"Unsupported communication modes: {sorted(invalid_modes)}")
    if not modes:
        raise ValueError("At least one communication mode is required")

    total = (
        len(map_indices)
        * int(MAP_TEST_REPEATS)
        * len(TEAM_SIZES)
        * len(modes)
    )
    if args.plan_only:
        print(
            f"[MERGINGMAP-DRIVER] plan_only=True shared_start_loop=False "
            f"total_runs={total}",
            flush=True,
        )
        run_number = 0
        for mode in modes:
            for map_index in map_indices:
                for trial in range(int(MAP_TEST_REPEATS)):
                    for team_size in TEAM_SIZES:
                        run_number += 1
                        seed = run_seed(map_index, trial, team_size, mode)
                        print(
                            f"[{run_number}/{total}] map={map_index} trial={trial} "
                            f"random_starts=worker robots={team_size} "
                            f"mode={mode} seed={seed}",
                            flush=True,
                        )
        return

    device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
    policy = load_frozen_policy(device)
    from mergingmap.multi_robot_test_worker import MultiRobotTestWorker

    result_root = Path(
        os.environ.get(
            "MERGINGMAP_RUN_DIR",
            str(Path(__file__).resolve().parent / "test_outputs" / time.strftime("run_%Y%m%d_%H%M%S")),
        )
    ).resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    csv_path = result_root / "results.csv"
    failure_path = result_root / "failures.jsonl"

    print(
        f"[MERGINGMAP-DRIVER] dedicated_driver={Path(__file__).name} "
        f"random_starts=True shared_start_loop=False total_runs={total}",
        flush=True,
    )

    results: list[dict] = []
    failures: list[dict] = []
    run_number = 0
    for mode in modes:
        for map_index in map_indices:
            for trial in range(int(MAP_TEST_REPEATS)):
                for team_size in TEAM_SIZES:
                    run_number += 1
                    seed = run_seed(map_index, trial, team_size, mode)
                    print(
                        f"[{run_number}/{total}] map={map_index} trial={trial} "
                        f"random_starts=worker robots={team_size} mode={mode} seed={seed}",
                        flush=True,
                    )
                    try:
                        worker = MultiRobotTestWorker(
                            policy=policy,
                            episode_index=map_index,
                            n_agents=int(team_size),
                            device=device,
                            seed=seed,
                            communication_mode=mode,
                        )
                        metrics = worker.run_episode()
                        metrics = {
                            "map_index": int(map_index),
                            "trial": int(trial),
                            "driver_run_number": int(run_number),
                            **metrics,
                        }
                        results.append(metrics)
                        write_results(csv_path, results)
                        print(
                            "  coverage={:.3f} steps={} success={} starts={} directions={}".format(
                                float(metrics.get("team_coverage", 0.0)),
                                metrics.get("steps"),
                                metrics.get("success"),
                                metrics.get("start_positions"),
                                metrics.get("initial_direction_assignments"),
                            ),
                            flush=True,
                        )
                    except Exception as exc:
                        failure = {
                            "map_index": int(map_index),
                            "trial": int(trial),
                            "team_size": int(team_size),
                            "communication_mode": mode,
                            "seed": int(seed),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                        failures.append(failure)
                        with failure_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                        print(f"  ERROR {type(exc).__name__}: {exc}", flush=True)
                        if not args.continue_on_error:
                            raise

    summary = {
        "driver": str(Path(__file__).resolve()),
        "shared_start_loop": False,
        "map_indices": map_indices,
        "tests_per_map": int(MAP_TEST_REPEATS),
        "team_sizes": [int(value) for value in TEAM_SIZES],
        "communication_modes": list(modes),
        "planned_runs": int(total),
        "completed_runs": len(results),
        "failed_runs": len(failures),
        "results_csv": str(csv_path),
    }
    (result_root / "driver_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[MERGINGMAP-DRIVER] results={csv_path}", flush=True)

if __name__ == "__main__":
    main()
