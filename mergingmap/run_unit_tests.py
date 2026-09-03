"""Run all standalone mergingmap tests and syntax checks."""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUTPUT_ROOT = HERE / "test_outputs"

# English purpose: Create a unit-test output directory without overwriting prior runs.
def make_unique_result_directory():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = OUTPUT_ROOT / f"unit_tests_{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = OUTPUT_ROOT / f"unit_tests_{stamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate

# English purpose: Execute a command and persist its combined output.
def run_and_record(command, output_path):
    completed = subprocess.run(
        command,
        cwd=HERE,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(HERE), str(PROJECT_ROOT))),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output_path.write_text(completed.stdout, encoding="utf-8")
    print(completed.stdout, end="")
    return int(completed.returncode)

# English purpose: Run MergingMap tests, syntax checks, and write a summary.
def main():
    result_dir = make_unique_result_directory()
    print(f"[MERGINGMAP TEST] result_dir={result_dir}")

    unit_result = run_and_record(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        result_dir / "UNIT_TEST_RESULTS.txt",
    )

    # Compile MergingMap plus the modular collision/deadlock packages.
    syntax_paths = [
        *sorted(HERE.rglob("*.py")),
        *sorted((PROJECT_ROOT / "collision").rglob("*.py")),
        *sorted((PROJECT_ROOT / "deadlock").rglob("*.py")),
    ]
    syntax_targets = [str(path) for path in syntax_paths if "test_outputs" not in path.parts]
    syntax_result = run_and_record(
        [sys.executable, "-m", "py_compile", *syntax_targets],
        result_dir / "SYNTAX_CHECK_RESULTS.txt",
    )

    overall = unit_result == 0 and syntax_result == 0
    summary = (
        f"unit_tests_return_code={unit_result}\n"
        f"syntax_check_return_code={syntax_result}\n"
        f"overall={'PASS' if overall else 'FAIL'}\n"
    )
    (result_dir / "SUMMARY.txt").write_text(summary, encoding="utf-8")
    (OUTPUT_ROOT / "latest_unit_test_path.txt").write_text(
        f"{result_dir}\n", encoding="utf-8"
    )
    print(summary, end="")
    print(f"Saved results to: {result_dir}")
    return 0 if overall else 1

if __name__ == "__main__":
    raise SystemExit(main())
