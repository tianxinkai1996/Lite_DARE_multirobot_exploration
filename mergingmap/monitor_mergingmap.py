"""Display the latest mergingmap process status and optionally follow its log."""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE / "test_outputs"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def show_process(pid: int) -> None:
    if not process_exists(pid):
        print(f"Process {pid} is not running.")
        return
    completed = subprocess.run(
        [
            "ps",
            "-p",
            str(pid),
            "-o",
            "pid,etime,%cpu,%mem,stat,cmd",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    print(completed.stdout.strip())


def print_last_lines(log_path: Path, line_count: int) -> None:
    if not log_path.exists():
        print(f"Log not found: {log_path}")
        return
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines[-line_count:]:
        print(line)


def follow_file(log_path: Path) -> None:
    while not log_path.exists():
        time.sleep(0.5)
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(0, os.SEEK_END)
        try:
            while True:
                line = handle.readline()
                if line:
                    print(line, end="", flush=True)
                else:
                    time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopped following the log.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lines", type=int, default=50)
    parser.add_argument(
        "--follow",
        action="store_true",
        help="continue following the latest log until Ctrl+C",
    )
    args = parser.parse_args()

    latest_run_file = OUTPUT_ROOT / "latest_run_path.txt"
    latest_pid_file = OUTPUT_ROOT / "latest_test.pid"
    if not latest_run_file.exists():
        print("No mergingmap run has been recorded yet.")
        return 1

    run_dir = Path(read_text(latest_run_file))
    log_path = run_dir / "mergingmap_test.log"
    print(f"Latest run: {run_dir}")

    if latest_pid_file.exists():
        try:
            show_process(int(read_text(latest_pid_file)))
        except ValueError:
            print(f"Invalid PID file: {latest_pid_file}")

    print(f"\nLast {args.lines} log lines:")
    print_last_lines(log_path, max(0, args.lines))

    if args.follow:
        print(f"\nFollowing {log_path}; press Ctrl+C to stop.\n")
        follow_file(log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
