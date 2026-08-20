from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Heavy-DARE and Lite-DARE run folders using saved "
            "metadata, runtime measurements, and extracted training metrics."
        )
    )
    parser.add_argument(
        "--runs",
        type=Path,
        nargs="+",
        required=True,
        help="Run directories to compare.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels corresponding to --runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lite_dare/comparisons/latest"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def nested_metric(
    metrics_summary: dict[str, Any],
    candidate_names: tuple[str, ...],
    statistic: str,
) -> float | None:
    metrics = metrics_summary.get("metrics", {})
    if not isinstance(metrics, dict):
        return None

    for name in candidate_names:
        payload = metrics.get(name)
        if isinstance(payload, dict):
            value = payload.get(statistic)
            if isinstance(value, (int, float)):
                return float(value)

    # Fall back to suffix matching because some loggers prefix metrics.
    for metric_name, payload in metrics.items():
        if not isinstance(payload, dict):
            continue
        lowered = metric_name.lower()
        if any(lowered.endswith(name.lower()) for name in candidate_names):
            value = payload.get(statistic)
            if isinstance(value, (int, float)):
                return float(value)

    return None


def row_for_run(run_dir: Path, label: str | None) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    architecture = read_json(run_dir / "architecture.json")
    runtime = read_json(run_dir / "run_summary.json")
    metrics = read_json(
        run_dir / "comparison_artifacts" / "metrics_summary.json"
    )

    return {
        "label": (
            label
            or architecture.get("architecture_name")
            or run_dir.name
        ),
        "run_dir": str(run_dir),
        "graph_encoder_layers": architecture.get("graph_encoder_layers"),
        "graph_encoder_heads": architecture.get("graph_encoder_heads"),
        "embedding_dimension": architecture.get("embedding_dimension"),
        "total_parameters": runtime.get("total_parameters"),
        "trainable_parameters": runtime.get("trainable_parameters"),
        "training_duration_seconds": runtime.get(
            "training_duration_seconds"
        ),
        "peak_gpu_memory_allocated_mb": runtime.get(
            "peak_gpu_memory_allocated_mb"
        ),
        "peak_gpu_memory_reserved_mb": runtime.get(
            "peak_gpu_memory_reserved_mb"
        ),
        "best_train_loss": nested_metric(
            metrics,
            ("train_loss", "training_loss"),
            "minimum",
        ),
        "final_train_loss": nested_metric(
            metrics,
            ("train_loss", "training_loss"),
            "last",
        ),
        "best_val_loss": nested_metric(
            metrics,
            ("val_loss", "validation_loss"),
            "minimum",
        ),
        "final_val_loss": nested_metric(
            metrics,
            ("val_loss", "validation_loss"),
            "last",
        ),
        "status": runtime.get("status"),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    keys = list(rows[0].keys()) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def generate_bar_charts(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = (
        ("total_parameters", "Total parameters"),
        ("training_duration_seconds", "Training duration (seconds)"),
        (
            "peak_gpu_memory_allocated_mb",
            "Peak GPU memory allocated (MB)",
        ),
        ("best_train_loss", "Best training loss"),
        ("best_val_loss", "Best validation loss"),
    )

    generated: list[str] = []
    labels = [str(row["label"]) for row in rows]

    for key, title in metrics:
        values = [row.get(key) for row in rows]
        if not any(isinstance(value, (int, float)) for value in values):
            continue

        numeric_values = [
            float(value) if isinstance(value, (int, float)) else math.nan
            for value in values
        ]

        figure = plt.figure()
        axis = figure.add_subplot(1, 1, 1)
        axis.bar(labels, numeric_values)
        axis.set_ylabel(title)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=20)
        figure.tight_layout()

        output_path = output_dir / f"{key}.png"
        figure.savefig(output_path, dpi=160)
        plt.close(figure)
        generated.append(str(output_path))

    return generated


def main() -> int:
    args = parse_args()

    if args.labels is not None and len(args.labels) not in (0, len(args.runs)):
        raise ValueError(
            "--labels must be omitted or contain one label per run."
        )

    labels = (
        args.labels
        if args.labels
        else [None] * len(args.runs)
    )
    rows = [
        row_for_run(run_dir, label)
        for run_dir, label in zip(args.runs, labels)
    ]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(rows, output_dir / "comparison_summary.csv")
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )
    charts = generate_bar_charts(rows, output_dir / "plots")

    print(json.dumps(
        {
            "comparison_summary_csv": str(
                output_dir / "comparison_summary.csv"
            ),
            "comparison_summary_json": str(
                output_dir / "comparison_summary.json"
            ),
            "plots": charts,
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
