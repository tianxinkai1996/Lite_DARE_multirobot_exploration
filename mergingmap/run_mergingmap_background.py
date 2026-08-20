#!/usr/bin/env python3
"""Start the isolated mergingmap experiment in the background.

Usage from the DARE project root:

    python mergingmap/run_mergingmap_background.py \
        --maps all --map-count 100 --modes compressed

The child process uses the current Python interpreter, so an activated conda/venv
is preserved. Every launch creates a new directory below
``mergingmap/test_outputs`` and writes the log and PID there.
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
RUNNER = HERE / "run_mergingmap.py"
OUTPUT_ROOT = HERE / "test_outputs"


def make_unique_run_directory() -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = OUTPUT_ROOT / f"run_{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = OUTPUT_ROOT / f"run_{stamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def main() -> int:
    if not RUNNER.exists():
        raise FileNotFoundError(f"Cannot find isolated runner: {RUNNER}")
    dedicated_driver = HERE / "multi_test_driver_mergingmap.py"
    if not dedicated_driver.exists():
        raise FileNotFoundError(
            f"Missing dedicated mergingmap driver: {dedicated_driver}"
        )

    run_dir = make_unique_run_directory()
    log_path = run_dir / "mergingmap_test.log"
    pid_path = run_dir / "mergingmap_test.pid"

    env = os.environ.copy()
    env["MERGINGMAP_RUN_DIR"] = str(run_dir)
    env["PYTHONUNBUFFERED"] = "1"

    command = [sys.executable, "-u", str(RUNNER), *sys.argv[1:]]
    with log_path.open("ab", buffering=0) as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )

    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    (OUTPUT_ROOT / "latest_test.pid").write_text(
        f"{process.pid}\n", encoding="utf-8"
    )
    (OUTPUT_ROOT / "latest_run_path.txt").write_text(
        f"{run_dir}\n", encoding="utf-8"
    )

    print("Started mergingmap test")
    print(f"PID: {process.pid}")
    print(f"Output: {run_dir}")
    print(f"Log: {log_path}")
    print(
        "Monitor: "
        f"{sys.executable} {HERE / 'monitor_mergingmap.py'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

