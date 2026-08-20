#!/usr/bin/env python3
"""Resume the existing Chapter 4 communication experiment from its aggregate results."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _methods(manifest: dict) -> list[str]:
    result = []
    for item in manifest.get("methods", []):
        if not isinstance(item, dict):
            continue
        value = item.get("key")
        if value:
            result.append(str(value))
    if result:
        return result
    # This helper is intentionally communication-only.
    return ["full"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "communication_root",
        nargs="?",
        type=Path,
        default=Path("/root/lite dare/DARE/paper_outputs/chapter4_single_seed/communication"),
    )
    parser.add_argument("--episode-workers", type=int, default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    root = args.communication_root.expanduser().resolve()
    manifest_path = root / "run_manifest.json"
    results_path = root / "results.csv"

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing run manifest: {manifest_path}")
    if not results_path.is_file():
        raise FileNotFoundError(
            f"Missing results.csv: {results_path}\n"
            "Restore the clean pre-break aggregate before resuming."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = _csv_rows(results_path)
    completed = sorted(
        {
            int(row["driver_run_number"])
            for row in rows
            if str(row.get("driver_run_number", "")).strip()
        }
    )
    if completed:
        print(
            f"[PRECHECK] aggregate rows={len(rows)} "
            f"completed_run_numbers={len(completed)} "
            f"min={completed[0]} max={completed[-1]}"
        )
    else:
        print("[PRECHECK] no completed driver_run_number found")

    project_root = Path(__file__).resolve().parent
    runner = project_root / "mergingmap" / "run_paper_ablation.py"
    if not runner.is_file():
        raise FileNotFoundError(
            f"Expected to run this helper from the DARE project root; missing {runner}"
        )

    checkpoint = manifest.get("checkpoint", {}).get("path")
    if not checkpoint:
        raise ValueError("run_manifest.json has no checkpoint.path")

    maps = [int(v) for v in manifest.get("maps", [])]
    if not maps:
        raise ValueError("run_manifest.json has no maps")
    map_arg = ",".join(str(v) for v in maps)

    team_sizes = [int(v) for v in manifest.get("team_sizes", [])]
    modes = [str(v) for v in manifest.get("communication_modes", [])]
    methods = _methods(manifest)

    cmd = [
        sys.executable,
        str(runner),
        "--checkpoint", str(checkpoint),
        "--model-key", str(manifest.get("model_key")),
        "--model-name", str(manifest.get("model_name")),
        "--methods", ",".join(methods),
        "--maps", map_arg,
        "--runs-per-map", str(int(manifest.get("runs_per_map", 4))),
        "--team-sizes", ",".join(str(v) for v in team_sizes),
        "--modes", ",".join(modes),
        "--output", str(root),
        "--persist-every", "25",
    ]

    if manifest.get("encoder_layers") is not None:
        cmd += ["--encoder-layers", str(int(manifest["encoder_layers"]))]
    if manifest.get("training_seed") is not None:
        cmd += ["--training-seed", str(int(manifest["training_seed"]))]
    if manifest.get("base_random_seed") is not None:
        cmd += ["--base-seed", str(int(manifest["base_random_seed"]))]

    overrides = manifest.get("map_repeat_overrides", {}) or {}
    if overrides:
        map_repeats = ",".join(
            f"{int(k)}:{int(v)}"
            for k, v in sorted(overrides.items(), key=lambda kv: int(kv[0]))
        )
        cmd += ["--map-repeats", map_repeats]

    workers = (
        args.episode_workers
        if args.episode_workers is not None
        else int(manifest.get("episode_workers", 1))
    )
    cmd += ["--episode-workers", str(workers)]

    # Deliberately DO NOT pass --save-visualisation.
    if args.continue_on_error:
        cmd.append("--continue-on-error")

    env = os.environ.copy()
    env.setdefault("LITEDARE_EXPERIMENT_PROFILE", "fast_exploration")

    print("[RESUME COMMAND]")
    print(" ".join(f'"{part}"' if " " in part else part for part in cmd))
    print("[INFO] visualisation disabled for the resumed batch")
    return subprocess.run(cmd, cwd=str(project_root), env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
