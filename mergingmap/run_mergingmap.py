#!/usr/bin/env python3
"""Run the dedicated mergingmap driver with isolated project modules.

Examples:

    # Use MANUAL_TESTS_PER_MAP from mergingmap/multi_test_parameter.py
    python -u mergingmap/run_mergingmap.py \
        --maps all --map-count 100 --modes compressed

    # Override the repetition count for this launch only
    python -u mergingmap/run_mergingmap.py \
        --runs-per-map 5 \
        --maps all --map-count 100 --modes compressed

No existing project Python file is overwritten. The wrapper removes its custom
``--runs-per-map`` argument, sets an environment override, then forwards every
other argument unchanged to the dedicated driver in this folder.
"""
from __future__ import annotations

import importlib
import json
import os
import runpy
import sys
from datetime import datetime
from pathlib import Path

from mergingmap.experiment_config import extract_repeat_argument, option_value


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DRIVER = HERE / "multi_test_driver_mergingmap.py"


def _prepare_run_directory() -> Path:
    run_root = HERE / "test_outputs"
    run_root.mkdir(parents=True, exist_ok=True)
    if "MERGINGMAP_RUN_DIR" not in os.environ:
        stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        os.environ["MERGINGMAP_RUN_DIR"] = str(run_root / stamp)
    result = Path(os.environ["MERGINGMAP_RUN_DIR"]).resolve()
    result.mkdir(parents=True, exist_ok=True)
    return result


def main() -> int:
    if not DRIVER.exists():
        raise FileNotFoundError(
            f"Cannot find {DRIVER}. Copy the complete mergingmap folder "
            "into the DARE project root."
        )

    custom = extract_repeat_argument(sys.argv[1:])
    if custom.runs_per_map is not None:
        os.environ["MERGINGMAP_MAP_TEST_REPEATS"] = str(
            custom.runs_per_map
        )

    # Prevent the unchanged project driver from seeing the custom argument.
    sys.argv = [sys.argv[0], *custom.forwarded_args]

    run_directory = _prepare_run_directory()

    # Isolated modules take precedence over the files in the original project.
    sys.path.insert(0, str(HERE))
    sys.path.insert(1, str(PROJECT_ROOT))
    os.chdir(PROJECT_ROOT)

    # Import after applying the command-line environment override.
    config = importlib.import_module("multi_test_parameter")
    repeats = int(config.MAP_TEST_REPEATS)

    map_count_text = option_value(custom.forwarded_args, "--map-count")
    map_count = None
    if map_count_text is not None:
        try:
            map_count = int(map_count_text)
        except ValueError:
            map_count = None

    modes_text = option_value(custom.forwarded_args, "--modes")
    modes = (
        [item.strip() for item in modes_text.split(",") if item.strip()]
        if modes_text
        else list(config.COMMUNICATION_MODES)
    )

    estimated_runs = None
    if map_count is not None and map_count > 0:
        estimated_runs = (
            map_count
            * repeats
            * len(config.TEAM_SIZES)
            * max(1, len(modes))
        )

    run_configuration = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "result_root": str(run_directory),
        "tests_per_map": repeats,
        "tests_per_map_source": (
            "command_line"
            if custom.runs_per_map is not None
            else (
                "environment"
                if "MERGINGMAP_MAP_TEST_REPEATS" in os.environ
                else "multi_test_parameter.py"
            )
        ),
        "team_sizes": [int(value) for value in config.TEAM_SIZES],
        "communication_modes": modes,
        "map_count": map_count,
        "estimated_total_runs": estimated_runs,
        "forwarded_driver_arguments": list(custom.forwarded_args),
    }
    (run_directory / "run_configuration.json").write_text(
        json.dumps(run_configuration, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"[MERGINGMAP] project_root={PROJECT_ROOT}", flush=True)
    print(f"[MERGINGMAP] result_root={run_directory}", flush=True)
    print(
        f"[MERGINGMAP] tests_per_map={repeats} "
        f"team_sizes={tuple(config.TEAM_SIZES)} "
        f"modes={tuple(modes)}",
        flush=True,
    )
    if estimated_runs is not None:
        print(
            f"[MERGINGMAP] estimated_total_runs={estimated_runs} "
            f"({map_count} maps x {repeats} repetitions x "
            f"{len(config.TEAM_SIZES)} team sizes x {max(1, len(modes))} modes)",
            flush=True,
        )
    print("[MERGINGMAP] dedicated_driver=mergingmap/multi_test_driver_mergingmap.py; existing project files are not overwritten", flush=True)

    runpy.run_path(str(DRIVER), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


