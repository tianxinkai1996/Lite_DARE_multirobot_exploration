#!/usr/bin/env python3
"""Evaluate selected DARE/LiteDARE checkpoints on chosen maps and repeats.

No JSON registry is required. Models are defined in ``chapter4_config.py``.
The same base random seed is used for the whole controlled study; deterministic
map/trial/team offsets produce reproducible repeated evaluation scenarios.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_experiments.chapter4_config import (
    MAP_COUNT,
    MAP_REPEATS,
    MAPS,
    MODELS,
    RANDOM_SEED,
    RUNS_PER_MAP,
)
from paper_experiments.common import (
    ModelSpec,
    format_map_repeats,
    iter_map_trials,
    parse_map_repeats,
    parse_map_selection,
    select_models,
    write_csv,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="all")
    parser.add_argument("--maps", default=MAPS)
    parser.add_argument("--map-count", type=int, default=MAP_COUNT)
    parser.add_argument("--runs-per-map", type=int, default=RUNS_PER_MAP)
    parser.add_argument(
        "--map-repeats",
        default=format_map_repeats(MAP_REPEATS),
        help="per-map overrides such as 0:10,7:20",
    )
    parser.add_argument("--team-sizes", default="1")
    parser.add_argument("--base-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-visualisation", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser


def _team_sizes(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError("--team-sizes must contain integers") from exc
    if not result or any(item not in {1, 2, 4, 6, 8} for item in result):
        raise ValueError("team sizes must be selected from 1,2,4,6,8")
    if len(set(result)) != len(result):
        raise ValueError("team sizes must not repeat")
    return result


def _scenario_seed(
    map_index: int,
    trial: int,
    team_size: int,
    *,
    base_seed: int,
) -> int:
    """Deterministic paired scenario seed without importing MergingMap drivers."""
    return int(
        int(base_seed)
        + 100_000 * int(map_index)
        + 1_000 * int(trial)
        + 10 * int(team_size)
    )


def _plan_rows(
    models: Sequence[ModelSpec],
    maps: Sequence[int],
    repeats: int,
    overrides: Mapping[int, int],
    team_sizes: Sequence[int],
    base_seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    number = 0
    for model in models:
        for map_index, trial in iter_map_trials(maps, repeats, overrides):
            for team_size in team_sizes:
                number += 1
                seed = _scenario_seed(
                    map_index,
                    trial,
                    team_size,
                    base_seed=base_seed,
                )
                pair_key = (
                    f"map_{map_index:04d}_trial_{trial:02d}_"
                    f"robots_{team_size:02d}_seed_{seed}"
                )
                rows.append(
                    {
                        "run_number": number,
                        **model.as_dict(),
                        "base_random_seed": int(base_seed),
                        "map_index": map_index,
                        "trial": trial,
                        "team_size": team_size,
                        "seed": seed,
                        "scenario_pair_key": pair_key,
                    }
                )
    return rows




def _load_frozen_policy(device, checkpoint_path: Path):
    """Load a frozen DARE/LiteDARE policy without importing MergingMap drivers.

    Importing ``mergingmap.multi_test_driver_mergingmap`` mutates ``sys.path`` so
    the later top-level import ``multi_test_parameter`` can resolve to
    ``mergingmap/multi_test_parameter.py`` instead of the project-root module.
    Stage-1 depth evaluation uses the original root ``multi_test_worker.py``, so
    its loader is intentionally kept local here to avoid that namespace clash.
    """
    import dill
    import hydra
    import torch
    from diffusion_policy.workspace.base_workspace import BaseWorkspace

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    with checkpoint.open("rb") as handle:
        payload = torch.load(handle, pickle_module=dill)
    cfg = payload["cfg"]
    workspace_class = hydra.utils.get_class(cfg._target_)
    output_dir = checkpoint.parents[1] / "chapter4_policy_inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    workspace = workspace_class(cfg, output_dir=str(output_dir))
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(device).eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return policy


def _device(value: str):
    import torch

    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device=cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.runs_per_map <= 0:
        raise ValueError("--runs-per-map must be positive")
    models = select_models(MODELS, args.models)
    if any(model.training_seed != RANDOM_SEED for model in models):
        raise ValueError(
            "chapter4_config.py must use the same RANDOM_SEED for all L6/L4/L2 models"
        )
    maps = parse_map_selection(args.maps, args.map_count)
    overrides = parse_map_repeats(args.map_repeats, args.runs_per_map)
    unknown = sorted(set(overrides) - set(maps))
    if unknown:
        raise ValueError(f"repeat overrides reference unselected maps: {unknown}")
    team_sizes = _team_sizes(args.team_sizes)
    plan = _plan_rows(
        models,
        maps,
        args.runs_per_map,
        overrides,
        team_sizes,
        args.base_seed,
    )

    if args.plan_only:
        print(f"[POLICY-EVAL] total_runs={len(plan)} base_seed={args.base_seed}")
        for row in plan:
            print(
                "[{run_number}/{total}] model={model_key} map={map_index} "
                "trial={trial} robots={team_size} seed={seed}".format(
                    total=len(plan), **row
                )
            )
        return 0

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    results_path = output / "results.csv"
    manifest_path = output / "run_manifest.json"
    failures_path = output / "failures.jsonl"
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "running",
        "models": [model.as_dict() for model in models],
        "base_random_seed": int(args.base_seed),
        "maps": list(maps),
        "runs_per_map": int(args.runs_per_map),
        "map_repeat_overrides": {str(k): int(v) for k, v in overrides.items()},
        "team_sizes": list(team_sizes),
        "planned_runs": len(plan),
        "purpose": "Chapter 4 single-seed self-attention depth comparison",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    import torch

    device = _device(args.device)
    results: list[dict[str, object]] = []
    for model in models:
        if not model.checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint not found for {model.key}: {model.checkpoint}")
        os.environ["DARE_CHECKPOINT_PATH"] = str(model.checkpoint)
        from multi_test_worker import MultiRobotTestWorker

        policy = _load_frozen_policy(device, checkpoint_path=model.checkpoint)
        checkpoint_sha = _sha256(model.checkpoint)
        model_rows = [row for row in plan if row["model_key"] == model.key]
        model_root = output / "models" / model.key
        model_root.mkdir(parents=True, exist_ok=True)
        for row in model_rows:
            print(
                "[{run_number}/{total}] model={model_key} map={map_index} "
                "trial={trial} robots={team_size}".format(total=len(plan), **row),
                flush=True,
            )
            try:
                test_profile = (
                    "original_dare" if model.family.strip().lower() == "dare" else "policy_only"
                )
                worker = MultiRobotTestWorker(
                    policy=policy,
                    episode_index=int(row["map_index"]),
                    n_agents=int(row["team_size"]),
                    device=device,
                    seed=int(row["seed"]),
                    communication_mode="none",
                    start_sample=int(row["trial"]),
                    visual_output_root=str(model_root),
                    test_profile=test_profile,
                    scenario_id=str(row["scenario_pair_key"]),
                    trial=int(row["trial"]),
                    save_visualisation=args.save_visualisation,
                )
                metrics = worker.run_episode()
                result = {
                    **row,
                    "method": model.display_name,
                    "method_role": "depth_comparison",
                    "checkpoint_sha256": checkpoint_sha,
                    "checkpoint_size_bytes": model.checkpoint.stat().st_size,
                    **metrics,
                }
                result.update(model.as_dict())
                result["scenario_pair_key"] = row["scenario_pair_key"]
                result["base_random_seed"] = int(args.base_seed)
                results.append(result)
                write_csv(results_path, results)
                print(
                    f"  success={metrics.get('success')} "
                    f"coverage={float(metrics.get('final_coverage', metrics.get('team_coverage', 0.0))):.4f} "
                    f"auc={float(metrics.get('coverage_auc', 0.0)):.4f}",
                    flush=True,
                )
            except Exception as exc:
                failure = {**row, "error_type": type(exc).__name__, "error": str(exc)}
                with failures_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                if not args.continue_on_error:
                    manifest["status"] = "failed"
                    manifest["completed_runs"] = len(results)
                    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
                    raise
        del policy
        if device.type == "cuda":
            torch.cuda.empty_cache()

    manifest["status"] = "complete" if len(results) == len(plan) else "complete_with_failures"
    manifest["completed_runs"] = len(results)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"results={results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

