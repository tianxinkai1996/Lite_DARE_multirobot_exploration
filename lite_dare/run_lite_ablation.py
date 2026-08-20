from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially retrain discrete-policy Lite-DARE L4 and L2 "
            "from scratch using train_exploration_transformer_node_discrete.yaml."
        )
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        choices=(4, 2),
        default=[4, 2],
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help=(
            "Defaults to the root-level "
            "train_exploration_transformer_node_discrete.yaml."
        ),
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-num-workers", type=int, default=0)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-metric-plots", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failed: list[int] = []

    for layers in args.layers:
        command = [
            sys.executable,
            "-m",
            "lite_dare.train_lite_dare",
            "--encoder-layers",
            str(layers),
            "--epochs",
            str(args.epochs),
            "--warmup-steps",
            str(args.warmup_steps),
            "--train-batch-size",
            str(args.train_batch_size),
            "--val-batch-size",
            str(args.val_batch_size),
            "--num-workers",
            str(args.num_workers),
            "--val-num-workers",
            str(args.val_num_workers),
        ]

        if args.config_file is not None:
            command.extend(
                ["--config-file", str(args.config_file)]
            )
        if args.dry_run:
            command.append("--dry-run")
        if args.no_metric_plots:
            command.append("--no-metric-plots")
        for override in args.override:
            command.extend(["--override", override])

        print("=" * 78, flush=True)
        print("Running:", " ".join(command), flush=True)
        print("=" * 78, flush=True)

        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            failed.append(layers)
            if not args.continue_on_error:
                break

    if failed:
        print(f"Failed Lite-DARE depths: {failed}")
        return 1

    print("All requested Lite-DARE variants completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())