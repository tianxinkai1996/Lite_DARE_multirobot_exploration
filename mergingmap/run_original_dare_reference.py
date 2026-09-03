#!/usr/bin/env python3
"""Run a saved policy/reference profile on the exact paper scenario manifest.

This version is package-import safe and resumable. It is designed for long
Chapter 4 runs: already completed scenarios are kept and skipped on restart,
so fixing an import error does not require repeating finished episodes.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

# Root project modules such as multi_test_worker.py and multi_test_parameter.py
# must resolve from /DARE, not from /DARE/mergingmap. Internal MergingMap code is
# imported explicitly through the mergingmap package below.
_PROJECT_ROOT_STR = str(PROJECT_ROOT)
while _PROJECT_ROOT_STR in sys.path:
    sys.path.remove(_PROJECT_ROOT_STR)
sys.path.insert(0, _PROJECT_ROOT_STR)


def _read_csv(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    """Persist all completed rows after each episode using first-seen fields."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            key = str(key)
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _parse_filter(value):
    if value is None:
        return None
    try:
        result = {int(part.strip()) for part in value.split(",") if part.strip()}
    except ValueError as exc:
        raise ValueError("filter values must be comma-separated integers") from exc
    if not result:
        raise ValueError("filter cannot be empty")
    return result


def _scenario_key(row):
    """Return the stable scenario identity used for resume de-duplication."""
    for name in ("scenario_id", "scenario_pair_key"):
        value = str(row.get(name, "") or "").strip()
        if value and value.lower() not in {"nan", "none"}:
            return value
    return "|".join(
        str(row.get(name, "NA"))
        for name in ("map_index", "trial", "team_size", "seed")
    )


def _existing_results(path, restart):
    if restart or not path.is_file() or path.stat().st_size == 0:
        return []
    return _read_csv(path)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario_manifest", type=Path)
    parser.add_argument(
        "--profile",
        choices=("original_dare", "policy_only"),
        default="original_dare",
    )
    parser.add_argument("--method-label", default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--model-key", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--encoder-layers", type=int, default=None)
    parser.add_argument("--training-seed", type=int, default=None)
    parser.add_argument("--maps", default=None)
    parser.add_argument("--team-sizes", default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--save-visualisation", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="discard existing results for this profile and rerun every selected scenario",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    scenario_path = args.scenario_manifest.expanduser().resolve()
    if not scenario_path.is_file():
        raise FileNotFoundError(f"scenario manifest not found: {scenario_path}")
    run_root = scenario_path.parent
    manifest_path = run_root / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(args.checkpoint or manifest["checkpoint"]["path"]).expanduser().resolve()
    if not checkpoint_path.is_file() and not args.plan_only:
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    os.environ["DARE_CHECKPOINT_PATH"] = str(checkpoint_path)

    map_filter = _parse_filter(args.maps)
    team_filter = _parse_filter(args.team_sizes)
    selected_rows = [
        row
        for row in _read_csv(scenario_path)
        if (map_filter is None or int(row["map_index"]) in map_filter)
        and (team_filter is None or int(row["team_size"]) in team_filter)
    ]
    if not selected_rows:
        raise ValueError("no scenarios remain after filtering")

    output_root = run_root / "methods" / args.profile
    results_path = output_root / "results.csv"
    failures_path = output_root / "failures.jsonl"
    results = _existing_results(results_path, restart=args.restart)
    completed = {_scenario_key(row) for row in results}
    pending = [row for row in selected_rows if _scenario_key(row) not in completed]

    print(f"run_root={run_root}")
    print(f"results={results_path}")
    print(f"profile={args.profile}")
    print(
        f"selected={len(selected_rows)} completed={len(selected_rows) - len(pending)} "
        f"pending={len(pending)} restart={args.restart}",
        flush=True,
    )

    if args.plan_only:
        for index, row in enumerate(pending, start=1):
            print(
                f"[{index}/{len(pending)}] scenario={_scenario_key(row)} "
                f"map={row['map_index']} trial={row['trial']} robots={row['team_size']}"
            )
        return 0

    if not pending:
        print("[RESUME] all selected scenarios are already complete; nothing to run")
        return 0

    import torch

    # Package-qualify MergingMap imports; keep original project worker imports at
    # project root. This prevents mergingmap/multi_test_parameter.py from
    # shadowing /DARE/multi_test_parameter.py.
    from mergingmap.multi_test_driver_mergingmap import load_frozen_policy
    from test_parameter import USE_GPU
    from multi_test_worker import MultiRobotTestWorker

    device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
    policy = load_frozen_policy(device, checkpoint_path=checkpoint_path)
    output_root.mkdir(parents=True, exist_ok=True)

    failures = 0
    for index, row in enumerate(pending, start=1):
        scenario = _scenario_key(row)
        print(
            f"[{index}/{len(pending)}] {args.profile} scenario={scenario} "
            f"robots={row['team_size']}",
            flush=True,
        )
        try:
            starts = json.loads(row["start_positions"])
            worker = MultiRobotTestWorker(
                policy=policy,
                episode_index=int(row["map_index"]),
                n_agents=int(row["team_size"]),
                device=device,
                seed=int(row["seed"]),
                communication_mode="none",
                start_positions=starts,
                start_sample=int(row["trial"]),
                visual_output_root=str(output_root),
                test_profile=args.profile,
                scenario_id=row.get("scenario_id") or scenario,
                trial=int(row["trial"]),
                save_visualisation=args.save_visualisation,
            )
            metrics = worker.run_episode()

            expected = json.dumps(starts, separators=(",", ":"), ensure_ascii=False)
            actual = json.dumps(
                json.loads(str(metrics["start_positions"])),
                separators=(",", ":"),
                ensure_ascii=False,
            )
            if expected != actual:
                raise RuntimeError(f"start mismatch for {scenario}: {expected} != {actual}")

            metrics["comparison_communication_mode"] = row.get("communication_mode", "compressed")
            metrics["scenario_source_method"] = row.get("source_method", "")
            metrics["model_key"] = args.model_key or manifest.get("model_key", "")
            metrics["model_name"] = args.model_name or manifest.get("model_name", "")
            metrics["encoder_layers"] = (
                args.encoder_layers if args.encoder_layers is not None else manifest.get("encoder_layers")
            )
            metrics["training_seed"] = (
                args.training_seed if args.training_seed is not None else manifest.get("training_seed")
            )
            metrics["base_random_seed"] = manifest.get("base_random_seed")
            metrics["method_role"] = "policy_only" if args.profile == "policy_only" else "original_dare"
            if args.method_label:
                metrics["method"] = str(args.method_label)
            results.append(metrics)
            _write_csv(results_path, results)
            completed.add(scenario)
            print(
                f"  coverage={float(metrics.get('team_coverage', 0.0)):.4f} "
                f"steps={metrics.get('steps')} success={metrics.get('success')}",
                flush=True,
            )
        except Exception as exc:
            failures += 1
            failure = {**row, "error_type": type(exc).__name__, "error": str(exc)}
            with failures_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
            print(f"  ERROR {type(exc).__name__}: {exc}", flush=True)
            if not args.continue_on_error:
                raise

    reference_key = f"{args.profile}_reference"
    reference = manifest.setdefault(reference_key, {})
    reference.update(
        {
            "status": "complete" if failures == 0 else "complete_with_failures",
            "completed_runs": sum(_scenario_key(row) in completed for row in selected_rows),
            "planned_runs": len(selected_rows),
            "output": str(output_root),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "profile": args.profile,
            "method_label": args.method_label,
            "checkpoint": str(checkpoint_path),
            "model_key": args.model_key,
            "model_name": args.model_name,
            "encoder_layers": args.encoder_layers,
            "training_seed": args.training_seed,
            "effective_communication_mode": "none",
            "resume_safe": True,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"policy_reference_output={output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



