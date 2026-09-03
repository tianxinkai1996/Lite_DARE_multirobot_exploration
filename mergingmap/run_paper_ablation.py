#!/usr/bin/env python3
"""Run controlled MergingMap/Collision/Deadlock paper ablations.

Original-DARE robot baselines are executed separately by
``run_original_dare_reference.py`` using the same scenario list.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

import torch

from mergingmap.ablation_profiles import (
    CORE_ABLATION_PROFILES,
    parse_profile_keys,
    resolve_ablation_profile,
    method_display_name,
)
from mergingmap.multi_test_driver_mergingmap import (
    discover_all_map_indices,
    load_frozen_policy,
    parse_map_selection,
    run_seed,
)
from mergingmap.multi_test_parameter import (
    COMMUNICATION_MODES,
    DARE_CHECKPOINT_PATH,
    MAP_TEST_REPEATS,
    TEAM_SIZES,
)
from test_parameter import USE_GPU
from paper_experiments.common import parse_map_repeats, repeat_count

def _write_csv(path, rows):
    """Write heterogeneous rows while preserving first-seen column order.

    English implementation: unions keys in first-seen order and rewrites one
    complete UTF-8 CSV after every completed episode.
    """

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(str(key))
                seen.add(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def _read_csv(path):
    """Read an existing aggregate CSV for safe resume."""

    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

def _append_jsonl(path, payload):
    """Append one failure or status event to a JSONL file.

    English implementation: serialises one mapping per line without replacing
    earlier failures.
    """

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")

def _parse_positive_ints(value, option):
    """Parse a unique comma-separated positive integer list.

    English implementation: preserves order and validates positivity and
    uniqueness.
    """

    try:
        values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(f"{option} must contain comma-separated integers") from exc
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"{option} must contain positive integers")
    if len(set(values)) != len(values):
        raise ValueError(f"{option} must not contain duplicate values")
    return values

def scenario_id(map_index, trial, team_size, seed, communication_mode):
    """Build the pairing key shared by every method.

    Encodes map, repeat, team size, deterministic seed, and map-communication
    treatment in one stable human-readable identifier.
    """

    if communication_mode not in {"none", "raw", "compressed"}:
        raise ValueError(f"unsupported communication mode: {communication_mode}")
    return (
        f"map_{int(map_index):04d}_trial_{int(trial):02d}_"
        f"robots_{int(team_size):02d}_seed_{int(seed)}_"
        f"mapcomm_{communication_mode}"
    )

def checkpoint_identity(path):
    """Return a resolved path, size, and SHA-256 checkpoint identity.

    English implementation: streams the checkpoint through SHA-256 to avoid
    loading the whole file into additional memory.
    """

    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return {
        "path": str(checkpoint),
        "size_bytes": checkpoint.stat().st_size,
        "sha256": digest.hexdigest(),
    }

def _normalise_start_positions(value):
    """Canonicalise JSON start positions for cross-method integrity checks.

    English implementation: parses JSON when possible and emits compact sorted
    JSON so text comparisons are stable.
    """

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)

def build_parser():
    """Create the paper-ablation command-line interface.

    English implementation: keeps planning mode independent from checkpoints so
    the full run matrix can be validated before expensive inference.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", default="all")
    parser.add_argument("--map-count", type=int, default=None)
    parser.add_argument(
        "--methods",
        default=",".join(CORE_ABLATION_PROFILES),
        help="comma-separated subset of map_only,map_region,map_reservation,full",
    )
    parser.add_argument(
        "--team-sizes",
        default=",".join(str(value) for value in TEAM_SIZES),
        help="team sizes used for this ablation; original DARE can be run later from the scenario manifest",
    )
    parser.add_argument(
        "--modes",
        default="compressed",
        help=(
            "map communication modes: none, raw, compressed; the core paper "
            "ablation defaults to one fixed compressed mode to avoid mixing "
            "communication treatments"
        ),
    )
    parser.add_argument("--runs-per-map", type=int, default=MAP_TEST_REPEATS)
    parser.add_argument(
        "--map-repeats",
        default=None,
        help="per-map repeat overrides, e.g. 0:10,3:5,8:20",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="override the frozen DARE/LiteDARE checkpoint for this run",
    )
    parser.add_argument("--model-key", default="selected-policy")
    parser.add_argument("--model-name", default="Selected DARE/LiteDARE")
    parser.add_argument("--encoder-layers", type=int, default=None)
    parser.add_argument("--training-seed", type=int, default=None)
    parser.add_argument("--base-seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--episode-workers",
        type=int,
        default=1,
        help=(
            "number of independent episode processes; each process loads one "
            "frozen policy copy and reuses it across assigned episodes"
        ),
    )
    parser.add_argument(
        "--persist-every",
        type=int,
        default=25,
        help="rewrite aggregate CSV snapshots every N completed episodes",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--include-extra-supervisors",
        action="store_true",
        help="also enable initial-direction and dynamic-region heuristics; not recommended for the core paper ablation",
    )
    parser.add_argument(
        "--save-visualisation",
        action="store_true",
        help="save frames/GIFs; disabled by default for batch paper runs",
    )
    return parser

# Process-local state used only by the spawn-based parallel episode pool.
# Each child owns one policy/CUDA context and reuses it across assigned episodes.
_EPISODE_POLICY = None
_EPISODE_DEVICE = None
_EPISODE_METHODS_ROOT: Path | None = None
_EPISODE_INCLUDE_EXTRA_SUPERVISORS = False
_EPISODE_SAVE_VISUALISATION = False

def _episode_process_init(checkpoint_path, methods_root, include_extra_supervisors, save_visualisation):
    """Initialise one reusable policy inside a spawned episode process.

    One policy/CUDA context is retained per spawned process and reused for every
    episode assigned to that process.
    """

    global _EPISODE_POLICY
    global _EPISODE_DEVICE
    global _EPISODE_METHODS_ROOT
    global _EPISODE_INCLUDE_EXTRA_SUPERVISORS
    global _EPISODE_SAVE_VISUALISATION

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    worker_runtime_root = Path(methods_root).parent / "_parallel_policy_runtime" / f"pid_{os.getpid()}"
    worker_runtime_root.mkdir(parents=True, exist_ok=True)
    os.environ["DARE_CHECKPOINT_PATH"] = str(checkpoint)
    os.environ["MERGINGMAP_RUN_DIR"] = str(worker_runtime_root)

    _EPISODE_DEVICE = torch.device(
        "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
    )
    _EPISODE_POLICY = load_frozen_policy(
        _EPISODE_DEVICE,
        checkpoint_path=checkpoint,
    )
    _EPISODE_METHODS_ROOT = Path(methods_root).expanduser().resolve()
    _EPISODE_INCLUDE_EXTRA_SUPERVISORS = bool(include_extra_supervisors)
    _EPISODE_SAVE_VISUALISATION = bool(save_visualisation)

def _run_parallel_episode(row):
    """Execute one independent multi-robot episode in a child process.

    Workers compute episodes and unique per-episode artifacts only; the parent
    process remains the sole writer of aggregate tables and manifests.
    """

    if _EPISODE_POLICY is None or _EPISODE_DEVICE is None or _EPISODE_METHODS_ROOT is None:
        raise RuntimeError("parallel episode worker was not initialised")

    from mergingmap.multi_robot_test_worker import MultiRobotTestWorker

    method_key = str(row["method_key"])
    method_root = _EPISODE_METHODS_ROOT / method_key
    method_root.mkdir(parents=True, exist_ok=True)

    worker = MultiRobotTestWorker(
        policy=_EPISODE_POLICY,
        episode_index=int(row["map_index"]),
        n_agents=int(row["team_size"]),
        device=_EPISODE_DEVICE,
        seed=int(row["seed"]),
        communication_mode=str(row["communication_mode"]),
        ablation_profile=method_key,
        include_extra_supervisors=_EPISODE_INCLUDE_EXTRA_SUPERVISORS,
        output_root=method_root,
        scenario_id=str(row["scenario_id"]),
        trial=int(row["trial"]),
        save_visualisation=_EPISODE_SAVE_VISUALISATION,
    )
    return {"row": dict(row), "metrics": worker.run_episode()}

def _policy_metadata(checkpoint_path, run_root):
    """Load one temporary CPU policy to record class and parameter count.

    This keeps manifest metadata identical in serial and parallel modes without
    retaining an extra GPU policy in the parent while child processes run.
    """

    previous_run_dir = os.environ.get("MERGINGMAP_RUN_DIR")
    os.environ["MERGINGMAP_RUN_DIR"] = str(run_root / "_policy_metadata")
    try:
        policy = load_frozen_policy(torch.device("cpu"), checkpoint_path=checkpoint_path)
        return {
            "policy_class": f"{policy.__class__.__module__}.{policy.__class__.__name__}",
            "policy_parameters": int(sum(parameter.numel() for parameter in policy.parameters())),
        }
    finally:
        if previous_run_dir is None:
            os.environ.pop("MERGINGMAP_RUN_DIR", None)
        else:
            os.environ["MERGINGMAP_RUN_DIR"] = previous_run_dir

def _result_row(row, metrics, args, checkpoint_sha256):
    """Build one aggregate results row in the parent process."""

    result: dict[str, object] = {
        "driver_run_number": int(row["run_number"]),
        "map_index": int(row["map_index"]),
        "trial": int(row["trial"]),
        "model_key": str(args.model_key),
        "model_name": str(args.model_name),
        "encoder_layers": args.encoder_layers,
        "training_seed": args.training_seed,
        "base_random_seed": args.base_seed,
        "method_role": str(row["method_key"]),
        "checkpoint_sha256": checkpoint_sha256,
        **dict(metrics),
    }
    result["method"] = method_display_name(str(args.model_name), str(row["method_key"]))
    return result

def _consume_success(row, metrics, args, checkpoint_sha256, results, scenario_rows, start_signatures):
    """Validate pairing and merge one completed episode into parent-owned state."""

    signature = _normalise_start_positions(metrics.get("start_positions"))
    scenario_key = str(row["scenario_id"])
    existing = start_signatures.setdefault(scenario_key, signature)
    if existing != signature:
        raise RuntimeError(
            "paired-start integrity failure for "
            f"{scenario_key}: {existing} != {signature}"
        )

    results.append(
        _result_row(
            row=row,
            metrics=metrics,
            args=args,
            checkpoint_sha256=checkpoint_sha256,
        )
    )
    scenario_rows.setdefault(
        scenario_key,
        {
            "scenario_id": scenario_key,
            "map_index": int(row["map_index"]),
            "trial": int(row["trial"]),
            "team_size": int(row["team_size"]),
            "seed": int(row["seed"]),
            "communication_mode": str(row["communication_mode"]),
            "start_positions": signature,
            "selected_start_seed": metrics.get("selected_start_seed"),
            "start_cells": metrics.get("start_cells"),
            "source_method": str(row["method_key"]),
        },
    )

def _persist_aggregates(results_path, scenario_path, results, scenario_rows):
    """Persist deterministic sorted snapshots from parent-owned aggregate state."""

    ordered_results = sorted(results, key=lambda item: int(item["driver_run_number"]))
    ordered_scenarios = sorted(
        scenario_rows.values(),
        key=lambda item: (
            int(item["map_index"]),
            int(item["trial"]),
            int(item["team_size"]),
            str(item["communication_mode"]),
        ),
    )
    _write_csv(results_path, ordered_results)
    _write_csv(scenario_path, ordered_scenarios)

def _planned_rows(methods, maps, default_trials, trial_overrides, team_sizes, modes, base_seed=None):
    """Yield the deterministic run matrix in execution order.

    English implementation: yields one dictionary per method/scenario without
    loading a model or environment.
    """

    from mergingmap.multi_test_parameter import BASE_SEED

    seed_base = BASE_SEED if base_seed is None else int(base_seed)
    run_number = 0
    for method_key in methods:
        for mode in modes:
            for map_index in maps:
                trials = repeat_count(map_index, default_trials, trial_overrides)
                for trial in range(trials):
                    for team_size in team_sizes:
                        run_number += 1
                        seed = run_seed(
                            map_index, trial, team_size, mode, base_seed=seed_base
                        )
                        yield {
                            "run_number": run_number,
                            "method_key": method_key,
                            "map_index": map_index,
                            "trial": trial,
                            "team_size": team_size,
                            "communication_mode": mode,
                            "seed": seed,
                            "scenario_id": scenario_id(
                                map_index, trial, team_size, seed, mode
                            ),
                        }

def main(argv=None):
    """Execute or print the controlled MergingMap ablation matrix.

    English implementation: shares one policy object, writes per-method outputs,
    verifies paired starts, and persists manifests after every episode.
    """

    args = build_parser().parse_args(argv)
    methods = parse_profile_keys(args.methods)
    map_indices = parse_map_selection(args.maps, args.map_count)
    team_sizes = _parse_positive_ints(args.team_sizes, option="--team-sizes")
    modes = tuple(part.strip() for part in args.modes.split(",") if part.strip())
    invalid_modes = set(modes) - {"none", "raw", "compressed"}
    if invalid_modes or not modes:
        raise ValueError(f"unsupported communication modes: {sorted(invalid_modes)}")
    if args.runs_per_map <= 0:
        raise ValueError("--runs-per-map must be positive")
    map_repeat_overrides = parse_map_repeats(args.map_repeats, args.runs_per_map)
    unknown_repeat_maps = sorted(set(map_repeat_overrides) - set(map_indices))
    if unknown_repeat_maps:
        raise ValueError(
            f"--map-repeats contains maps not selected by --maps: {unknown_repeat_maps}"
        )

    plan = list(
        _planned_rows(
            methods=methods,
            maps=map_indices,
            default_trials=args.runs_per_map,
            trial_overrides=map_repeat_overrides,
            team_sizes=team_sizes,
            modes=modes,
            base_seed=args.base_seed,
        )
    )
    if args.plan_only:
        print(
            f"[PAPER-ABLATION] total_runs={len(plan)} methods={methods} "
            f"extra_supervisors={bool(args.include_extra_supervisors)}"
        )
        for row in plan:
            print(
                "[{run_number}/{total}] method={method_key} map={map_index} "
                "trial={trial} robots={team_size} mode={communication_mode} "
                "seed={seed} scenario={scenario_id}".format(total=len(plan), **row)
            )
        return 0

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = (
        args.output
        or HERE / "test_outputs" / f"paper_ablation_{stamp}"
    ).expanduser().resolve()
    manifest_path = run_root / "run_manifest.json"
    resuming = run_root.exists()
    if resuming and not manifest_path.is_file():
        raise FileExistsError(
            f"output directory exists but has no run_manifest.json: {run_root}; "
            "refusing to mix a new experiment with unknown files"
        )
    run_root.mkdir(parents=True, exist_ok=True)
    methods_root = run_root / "methods"
    methods_root.mkdir(parents=True, exist_ok=True)

    if args.episode_workers <= 0:
        raise ValueError("--episode-workers must be positive")
    if args.persist_every <= 0:
        raise ValueError("--persist-every must be positive")
    if args.save_visualisation and args.episode_workers > 1:
        print(
            "[WARNING] visualisation is enabled with parallel episodes; PNG/GIF I/O "
            "may dominate runtime.",
            flush=True,
        )

    checkpoint_path = Path(args.checkpoint or DARE_CHECKPOINT_PATH).expanduser().resolve()
    os.environ["DARE_CHECKPOINT_PATH"] = str(checkpoint_path)
    checkpoint_info = checkpoint_identity(checkpoint_path)
    device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")

    # Serial mode keeps the original single-policy behaviour. Parallel mode records
    # metadata with a temporary CPU copy, then each child process owns one GPU copy.
    if args.episode_workers == 1:
        policy = load_frozen_policy(device, checkpoint_path=checkpoint_path)
        policy_metadata = {
            "policy_class": f"{policy.__class__.__module__}.{policy.__class__.__name__}",
            "policy_parameters": int(sum(parameter.numel() for parameter in policy.parameters())),
        }
    else:
        policy = None
        policy_metadata = _policy_metadata(checkpoint_path, run_root)

    requested_method_keys = list(methods)
    if resuming:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_method_keys = [
            str(item.get("key", item.get("name", "")))
            for item in manifest.get("methods", [])
        ]
        # Profiles serialise their key in different fields across older revisions;
        # fall back to the originally requested ordering when exact keys are absent.
        if not any(existing_method_keys):
            existing_method_keys = requested_method_keys
        checks = {
            "checkpoint_sha256": (manifest.get("checkpoint", {}).get("sha256"), checkpoint_info["sha256"]),
            "model_key": (str(manifest.get("model_key")), str(args.model_key)),
            "encoder_layers": (manifest.get("encoder_layers"), args.encoder_layers),
            "base_random_seed": (manifest.get("base_random_seed"), args.base_seed),
            "maps": (list(manifest.get("maps", [])), list(map_indices)),
            "runs_per_map": (int(manifest.get("runs_per_map", -1)), int(args.runs_per_map)),
            "map_repeat_overrides": (
                {str(k): int(v) for k, v in manifest.get("map_repeat_overrides", {}).items()},
                {str(k): int(v) for k, v in map_repeat_overrides.items()},
            ),
            "team_sizes": (list(manifest.get("team_sizes", [])), list(team_sizes)),
            "communication_modes": (list(manifest.get("communication_modes", [])), list(modes)),
            "methods": (existing_method_keys, requested_method_keys),
        }
        mismatches = [name for name, (old, new) in checks.items() if old != new]
        if mismatches:
            details = "; ".join(f"{name}: {checks[name][0]!r} != {checks[name][1]!r}" for name in mismatches)
            raise ValueError(f"resume configuration mismatch for {run_root}: {details}")
        manifest.update(
            {
                "status": "running",
                "resumed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "episode_workers": int(args.episode_workers),
                "multiprocessing_start_method": "serial" if args.episode_workers == 1 else "spawn",
            }
        )
    else:
        manifest = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "status": "running",
            "checkpoint": checkpoint_info,
            "policy_class": policy_metadata["policy_class"],
            "policy_parameters": policy_metadata["policy_parameters"],
            "device": str(device),
            "episode_workers": int(args.episode_workers),
            "multiprocessing_start_method": "serial" if args.episode_workers == 1 else "spawn",
            "model_key": str(args.model_key),
            "model_name": str(args.model_name),
            "encoder_layers": args.encoder_layers,
            "training_seed": args.training_seed,
            "base_random_seed": args.base_seed,
            "maps": list(map_indices),
            "runs_per_map": int(args.runs_per_map),
            "map_repeat_overrides": {str(k): int(v) for k, v in map_repeat_overrides.items()},
            "team_sizes": list(team_sizes),
            "communication_modes": list(modes),
            "methods": [
                resolve_ablation_profile(
                    key,
                    include_extra_supervisors=args.include_extra_supervisors,
                ).as_dict()
                for key in methods
            ],
            "core_ablation_warning": (
                "extra supervisors enabled; results do not isolate only MM/Collision/Deadlock"
                if args.include_extra_supervisors
                else "canonical profile-specific supervisors only; no extra supervisor override"
            ),
            "original_dare_reference": {
                "status": "not_run",
                "command": (
                    "python mergingmap/run_original_dare_reference.py "
                    f"{run_root / 'scenario_manifest.csv'}"
                ),
            },
        }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    from mergingmap.multi_robot_test_worker import MultiRobotTestWorker

    failures_path = run_root / "failures.jsonl"
    completed_journal_path = run_root / "completed_results.jsonl"
    results_path = run_root / "results.csv"
    scenario_path = run_root / "scenario_manifest.csv"

    results: list[dict[str, object]] = [dict(row) for row in _read_csv(results_path)] if resuming else []
    existing_scenarios = _read_csv(scenario_path) if resuming else []
    scenario_rows: dict[str, dict[str, object]] = {
        str(row["scenario_id"]): dict(row)
        for row in existing_scenarios
        if str(row.get("scenario_id", "")).strip()
    }
    start_signatures: dict[str, str] = {
        key: _normalise_start_positions(row.get("start_positions"))
        for key, row in scenario_rows.items()
    }
    completed_run_numbers = {
        int(row["driver_run_number"])
        for row in results
        if str(row.get("driver_run_number", "")).strip()
    }
    full_plan = plan
    plan = [row for row in full_plan if int(row["run_number"]) not in completed_run_numbers]
    completed = len(results)
    if resuming:
        print(
            f"[RESUME] root={run_root} completed={completed}/{len(full_plan)} "
            f"pending={len(plan)}",
            flush=True,
        )
    if not plan:
        manifest["status"] = "complete"
        manifest["completed_runs"] = len(results)
        manifest["planned_runs"] = len(full_plan)
        manifest["scenario_count"] = len(scenario_rows)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"run_root={run_root}")
        print(f"results={results_path}")
        print(f"scenarios={scenario_path}")
        print("[RESUME] nothing pending")
        return 0

    def handle_success(row, metrics):
        nonlocal completed
        _consume_success(
            row=row,
            metrics=metrics,
            args=args,
            checkpoint_sha256=str(checkpoint_info["sha256"]),
            results=results,
            scenario_rows=scenario_rows,
            start_signatures=start_signatures,
        )
        completed += 1
        _append_jsonl(
            completed_journal_path,
            {
                "driver_run_number": int(row["run_number"]),
                "scenario_id": str(row["scenario_id"]),
                "method_key": str(row["method_key"]),
                "map_index": int(row["map_index"]),
                "trial": int(row["trial"]),
                "team_size": int(row["team_size"]),
                "communication_mode": str(row["communication_mode"]),
                "coverage": metrics.get("team_coverage"),
                "steps": metrics.get("steps"),
                "success": metrics.get("success"),
            },
        )
        if completed % int(args.persist_every) == 0:
            _persist_aggregates(
                results_path=results_path,
                scenario_path=scenario_path,
                results=results,
                scenario_rows=scenario_rows,
            )
        print(
            f"[{completed}/{len(full_plan)} completed] "
            f"method={row['method_key']} map={row['map_index']} trial={row['trial']} "
            f"robots={row['team_size']} mode={row['communication_mode']} "
            f"coverage={float(metrics.get('team_coverage', 0.0)):.4f} "
            f"steps={metrics.get('steps')} success={metrics.get('success')}",
            flush=True,
        )

    def handle_failure(row, exc):
        failure = {
            **dict(row),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _append_jsonl(failures_path, failure)
        print(
            f"  ERROR method={row['method_key']} map={row['map_index']} "
            f"trial={row['trial']} robots={row['team_size']} "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

    if args.episode_workers == 1:
        assert policy is not None
        for row in plan:
            method_key = str(row["method_key"])
            method_root = methods_root / method_key
            method_root.mkdir(parents=True, exist_ok=True)
            try:
                worker = MultiRobotTestWorker(
                    policy=policy,
                    episode_index=int(row["map_index"]),
                    n_agents=int(row["team_size"]),
                    device=device,
                    seed=int(row["seed"]),
                    communication_mode=str(row["communication_mode"]),
                    ablation_profile=method_key,
                    include_extra_supervisors=args.include_extra_supervisors,
                    output_root=method_root,
                    scenario_id=str(row["scenario_id"]),
                    trial=int(row["trial"]),
                    save_visualisation=args.save_visualisation,
                )
                handle_success(row, worker.run_episode())
            except Exception as exc:
                handle_failure(row, exc)
                if not args.continue_on_error:
                    manifest["status"] = "failed"
                    manifest["completed_runs"] = len(results)
                    manifest_path.write_text(
                        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                    _persist_aggregates(
                        results_path=results_path,
                        scenario_path=scenario_path,
                        results=results,
                        scenario_rows=scenario_rows,
                    )
                    raise
    else:
        # CUDA multiprocessing uses spawn rather than fork. Each process loads one
        # frozen policy exactly once in the initializer and then executes many rows.
        ctx = mp.get_context("spawn")
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=int(args.episode_workers),
            mp_context=ctx,
            initializer=_episode_process_init,
            initargs=(
                str(checkpoint_path),
                str(methods_root),
                bool(args.include_extra_supervisors),
                bool(args.save_visualisation),
            ),
        )
        futures = {executor.submit(_run_parallel_episode, row): row for row in plan}
        abort_error: BaseException | None = None
        try:
            for future in concurrent.futures.as_completed(futures):
                row = futures[future]
                try:
                    payload = future.result()
                    metrics = payload["metrics"]
                    if not isinstance(metrics, Mapping):
                        raise TypeError("episode worker returned non-mapping metrics")
                    handle_success(row, metrics)
                except Exception as exc:
                    handle_failure(row, exc)
                    if not args.continue_on_error:
                        abort_error = exc
                        for pending in futures:
                            pending.cancel()
                        break
        finally:
            executor.shutdown(wait=abort_error is None, cancel_futures=abort_error is not None)

        if abort_error is not None:
            manifest["status"] = "failed"
            manifest["completed_runs"] = len(results)
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            _persist_aggregates(
                results_path=results_path,
                scenario_path=scenario_path,
                results=results,
                scenario_rows=scenario_rows,
            )
            raise RuntimeError("parallel ablation aborted after episode failure") from abort_error

    # Final snapshot is always written, even when the last batch is smaller than
    # --persist-every. Completion order does not affect the deterministic CSV order.
    _persist_aggregates(
        results_path=results_path,
        scenario_path=scenario_path,
        results=results,
        scenario_rows=scenario_rows,
    )

    manifest["status"] = "complete" if len(results) == len(full_plan) else "complete_with_failures"
    manifest["completed_runs"] = len(results)
    manifest["planned_runs"] = len(full_plan)
    manifest["scenario_count"] = len(scenario_rows)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"run_root={run_root}")
    print(f"results={results_path}")
    print(f"scenarios={scenario_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

