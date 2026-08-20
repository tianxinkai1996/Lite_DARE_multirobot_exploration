#!/usr/bin/env python3
"""Run Chapter 4 in two ordered stages using only Python configuration.

Stage 1 compares DARE-L6, LiteDARE-L4 and LiteDARE-L2 with one shared base
random seed. The selection script then chooses whichever LiteDARE variant is closest to DARE-L6 in paired exploration outcomes.
Only after that selection is written does Stage 2 run multi-robot ablation and
communication experiments with the selected checkpoint.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_experiments.chapter4_config import (
    ATTENTION_COMPARISON_MODELS,
    BOOTSTRAP_SAMPLES,
    COMMUNICATION_TEAM_SIZES,
    CONTINUE_ON_ERROR,
    EXPERIMENT_PROFILE,
    EPISODE_WORKERS,
    DEFAULT_OUTPUT,
    DELTA_COVERAGE_AUC,
    DELTA_FINAL_COVERAGE,
    DELTA_SUCCESS_RATE,
    SELECTION_TIE_TOLERANCE,
    MAP_COUNT,
    MAP_REPEATS,
    MAPS,
    MODELS,
    PARALLEL_WORKERS,
    RANDOM_SEED,
    RUN_ORIGINAL_DARE_REFERENCE,
    RUNS_PER_MAP,
    SAVE_VISUALISATION,
    TEAM_SIZES,
)
from paper_experiments.common import format_map_repeats, select_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("compare", "downstream", "all"), default="all")
    parser.add_argument("--models", default=",".join(ATTENTION_COMPARISON_MODELS))
    parser.add_argument("--maps", default=MAPS)
    parser.add_argument("--map-count", type=int, default=MAP_COUNT)
    parser.add_argument("--runs-per-map", type=int, default=RUNS_PER_MAP)
    parser.add_argument("--map-repeats", default=format_map_repeats(MAP_REPEATS))
    parser.add_argument("--team-sizes", default=",".join(str(v) for v in TEAM_SIZES))
    parser.add_argument(
        "--communication-team-sizes",
        default=",".join(str(v) for v in COMMUNICATION_TEAM_SIZES),
    )
    parser.add_argument("--base-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--parallel-workers", type=int, default=PARALLEL_WORKERS)
    parser.add_argument(
        "--episode-workers",
        type=int,
        default=EPISODE_WORKERS,
        help=(
            "independent episode processes used inside downstream ablation runs; "
            "use 1 for efficiency benchmarks"
        ),
    )
    parser.add_argument(
        "--experiment-profile",
        choices=("fast_exploration", "efficiency_benchmark"),
        default=EXPERIMENT_PROFILE,
        help=(
            "fast_exploration disables expensive per-step hardware profiling; "
            "efficiency_benchmark restores synchronized timing/memory/energy metrics"
        ),
    )
    parser.add_argument("--continue-on-error", action="store_true", default=CONTINUE_ON_ERROR)
    parser.add_argument("--save-visualisation", action="store_true", default=SAVE_VISUALISATION)
    parser.add_argument("--plan-only", action="store_true")
    return parser


def _map_args(args: argparse.Namespace) -> list[str]:
    command = [
        "--maps", args.maps,
        "--map-count", str(args.map_count),
        "--runs-per-map", str(args.runs_per_map),
        "--base-seed", str(args.base_seed),
    ]
    if args.map_repeats:
        command.extend(["--map-repeats", args.map_repeats])
    if args.continue_on_error:
        command.append("--continue-on-error")
    if args.save_visualisation:
        command.append("--save-visualisation")
    return command


def _run(
    command: Sequence[str],
    *,
    plan_only: bool = False,
    experiment_profile: str = EXPERIMENT_PROFILE,
) -> int:
    print("$ " + " ".join(str(part) for part in command), flush=True)
    if plan_only:
        return 0
    env = os.environ.copy()
    env["LITEDARE_EXPERIMENT_PROFILE"] = str(experiment_profile)
    return subprocess.run(
        list(command),
        cwd=str(PROJECT_ROOT),
        check=False,
        env=env,
    ).returncode


def _compare_task(args: argparse.Namespace, model_key: str) -> tuple[str, list[str]]:
    output = args.output / "e1_policy_depth" / model_key
    return model_key, [
        sys.executable,
        "paper_experiments/evaluate_policy.py",
        "--models", model_key,
        "--team-sizes", "1",
        "--output", str(output),
        *_map_args(args),
    ]


def _execute_parallel(tasks: list[tuple[str, list[str]]], args: argparse.Namespace) -> list[tuple[str, int]]:
    failures: list[tuple[str, int]] = []
    if args.parallel_workers <= 1 or len(tasks) <= 1:
        for name, command in tasks:
            code = _run(command, plan_only=args.plan_only, experiment_profile=args.experiment_profile)
            if code:
                failures.append((name, code))
                if not args.continue_on_error:
                    break
        return failures
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel_workers) as pool:
        futures = {
            pool.submit(_run, command, plan_only=args.plan_only): name
            for name, command in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            code = future.result()
            if code:
                failures.append((name, code))
    return failures


def _select(args: argparse.Namespace) -> int:
    command = [
        sys.executable,
        "paper_experiments/select_attention_model.py",
        str(args.output),
        "--delta-coverage", str(DELTA_FINAL_COVERAGE),
        "--delta-success", str(DELTA_SUCCESS_RATE),
        "--delta-auc", str(DELTA_COVERAGE_AUC),
        "--tie-tolerance", str(SELECTION_TIE_TOLERANCE),
        "--bootstrap-samples", str(BOOTSTRAP_SAMPLES),
    ]
    return _run(command, plan_only=False, experiment_profile=args.experiment_profile)


def _load_selected(output: Path):
    selection_file = output / "selected_model.py"
    if not selection_file.is_file():
        raise FileNotFoundError(
            f"{selection_file} not found. Run --stage compare first so L6/L4/L2 can be compared."
        )
    payload = runpy.run_path(str(selection_file))
    key = str(payload["SELECTED_MODEL_KEY"])
    if key not in MODELS:
        raise ValueError(f"selected model {key!r} is not defined in chapter4_config.py")
    return MODELS[key]


def _multi_commands(args: argparse.Namespace, model) -> tuple[str, list[list[str]]]:
    output = args.output / "multi_ablation"
    ablation = [
        sys.executable,
        "mergingmap/run_paper_ablation.py",
        "--checkpoint", str(model.checkpoint),
        "--model-key", model.key,
        "--model-name", model.display_name,
        "--encoder-layers", str(model.encoder_layers),
        "--training-seed", str(RANDOM_SEED),
        "--methods", "map_only,map_region,map_reservation,full",
        "--episode-workers", str(args.episode_workers),
        "--team-sizes", args.team_sizes,
        "--modes", "compressed",
        "--output", str(output),
        *_map_args(args),
    ]
    policy_only = [
        sys.executable,
        "mergingmap/run_original_dare_reference.py",
        str(output / "scenario_manifest.csv"),
        "--profile", "policy_only",
        "--method-label", f"{model.display_name}-only",
        "--checkpoint", str(model.checkpoint),
        "--model-key", model.key,
        "--model-name", model.display_name,
        "--encoder-layers", str(model.encoder_layers),
        "--training-seed", str(RANDOM_SEED),
    ]
    if args.continue_on_error:
        policy_only.append("--continue-on-error")
    if args.save_visualisation:
        policy_only.append("--save-visualisation")
    commands = [ablation, policy_only]

    original = MODELS["DARE-L6"]
    if RUN_ORIGINAL_DARE_REFERENCE and model.key != original.key:
        commands.append([
            sys.executable,
            "mergingmap/run_original_dare_reference.py",
            str(output / "scenario_manifest.csv"),
            "--profile", "original_dare",
            "--method-label", original.display_name,
            "--checkpoint", str(original.checkpoint),
            "--model-key", original.key,
            "--model-name", original.display_name,
            "--encoder-layers", str(original.encoder_layers),
            "--training-seed", str(RANDOM_SEED),
        ])
    return "multi", commands


def _communication_commands(args: argparse.Namespace, model) -> tuple[str, list[list[str]]]:
    output = args.output / "communication"
    command = [
        sys.executable,
        "mergingmap/run_paper_ablation.py",
        "--checkpoint", str(model.checkpoint),
        "--model-key", model.key,
        "--model-name", model.display_name,
        "--encoder-layers", str(model.encoder_layers),
        "--training-seed", str(RANDOM_SEED),
        "--methods", "full",
        "--episode-workers", str(args.episode_workers),
        "--team-sizes", args.communication_team_sizes,
        "--modes", "none,raw,compressed",
        "--output", str(output),
        *_map_args(args),
    ]
    return "communication", [command]


def _execute_command_group(task: tuple[str, list[list[str]]], args: argparse.Namespace) -> tuple[str, int]:
    name, commands = task
    for command in commands:
        code = _run(command, plan_only=args.plan_only, experiment_profile=args.experiment_profile)
        if code:
            return name, code
    return name, 0


def _run_downstream(args: argparse.Namespace, model) -> list[tuple[str, int]]:
    """Run large downstream groups sequentially; parallelism is inside each group.

    中文目的：避免 multi-ablation 与 communication 同时各启动一组 CUDA 进程，
    从而把真实 GPU 进程上限稳定控制在 ``episode_workers``。
    English: each ``run_paper_ablation.py`` command owns its own episode pool, so
    the two large experiment groups run sequentially rather than multiplying CUDA
    contexts by running both pools at once.
    """

    tasks = [_multi_commands(args, model), _communication_commands(args, model)]
    failures: list[tuple[str, int]] = []
    for task in tasks:
        name, code = _execute_command_group(task, args)
        if code:
            failures.append((name, code))
            if not args.continue_on_error:
                break
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.parallel_workers <= 0:
        raise ValueError("--parallel-workers must be positive")
    if args.episode_workers <= 0:
        raise ValueError("--episode-workers must be positive")
    if args.experiment_profile == "efficiency_benchmark" and (
        args.parallel_workers != 1 or args.episode_workers != 1
    ):
        raise ValueError(
            "efficiency_benchmark must use --parallel-workers 1 and "
            "--episode-workers 1 so timing/resource measurements are not "
            "contaminated by concurrent model processes"
        )
    if args.runs_per_map <= 0:
        raise ValueError("--runs-per-map must be positive")
    if args.base_seed != RANDOM_SEED:
        raise ValueError(
            f"this controlled pipeline uses one RANDOM_SEED={RANDOM_SEED}; edit chapter4_config.py to change it"
        )
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    requested_models = select_models(MODELS, args.models)
    selected_depths = {model.encoder_layers for model in requested_models}
    if args.stage == "all" and selected_depths != {2, 4, 6}:
        raise ValueError("--stage all requires L6, L4 and L2 so the downstream model can be selected first")

    plan = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "stage": args.stage,
        "models": [model.key for model in requested_models],
        "base_random_seed": RANDOM_SEED,
        "maps": args.maps,
        "map_count": args.map_count,
        "runs_per_map": args.runs_per_map,
        "map_repeats": args.map_repeats,
        "team_sizes": args.team_sizes,
        "parallel_workers": args.parallel_workers,
        "episode_workers": args.episode_workers,
        "experiment_profile": args.experiment_profile,
        "selection_rule": "closest LiteDARE model to L6 by paired success/final-coverage/AUC distance",
        "selection_scales": {"success": DELTA_SUCCESS_RATE, "final_coverage": DELTA_FINAL_COVERAGE, "coverage_auc": DELTA_COVERAGE_AUC},
        "selection_tie_tolerance": SELECTION_TIE_TOLERANCE,
    }
    (args.output / "chapter4_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    if args.stage in {"compare", "all"}:
        compare_tasks = [_compare_task(args, model.key) for model in requested_models]
        failures = _execute_parallel(compare_tasks, args)
        if failures:
            print(f"comparison_failures={failures}")
            return 1
        if args.plan_only:
            print("selection deferred until comparison results exist")
            return 0
        if {2, 4, 6}.issubset(selected_depths):
            if _select(args) != 0:
                return 1
        elif args.stage == "compare":
            print("comparison complete; automatic selection skipped because L6/L4/L2 were not all requested")
            return 0

    if args.stage in {"downstream", "all"}:
        selected_model = _load_selected(args.output)
        print(
            f"[SELECTED] {selected_model.key}: L{selected_model.encoder_layers} "
            f"checkpoint={selected_model.checkpoint}",
            flush=True,
        )
        failures = _run_downstream(args, selected_model)
        if failures:
            print(f"downstream_failures={failures}")
            return 1

    print(f"chapter4_output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
