from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


PREFERRED_LOG_NAMES = (
    "logs.json.txt",
    "metrics.jsonl",
    "metrics.json",
    "training_metrics.jsonl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract scalar training metrics from a DARE run and generate "
            "comparison-ready CSV files and individual plots."
        )
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Extract CSV data without creating PNG charts.",
    )
    return parser.parse_args()


def is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, bool)) and not isinstance(value, complex)


def flatten_scalars(
    value: Any,
    prefix: str = "",
) -> dict[str, float | int | bool]:
    result: dict[str, float | int | bool] = {}

    if is_scalar(value):
        result[prefix or "value"] = value
        return result

    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_scalars(child, child_prefix))

    return result


def load_json_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                row = flatten_scalars(value)
                row["_source_file"] = str(path)
                row["_source_line"] = line_number
                rows.append(row)
    return rows


def load_json_file(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    if isinstance(value, list):
        rows = []
        for index, item in enumerate(value):
            if isinstance(item, dict):
                row = flatten_scalars(item)
                row["_source_file"] = str(path)
                row["_source_line"] = index + 1
                rows.append(row)
        return rows

    if isinstance(value, dict):
        row = flatten_scalars(value)
        row["_source_file"] = str(path)
        row["_source_line"] = 1
        return [row]

    return []


def discover_metric_files(run_dir: Path) -> list[Path]:
    files: list[Path] = []

    for name in PREFERRED_LOG_NAMES:
        files.extend(run_dir.rglob(name))

    for pattern in ("*.jsonl", "*.json.txt"):
        files.extend(run_dir.rglob(pattern))

    unique = []
    seen = set()
    for path in files:
        resolved = path.resolve()
        if resolved not in seen and path.is_file():
            seen.add(resolved)
            unique.append(path)

    return sorted(unique)


def load_rows(run_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    metric_files = discover_metric_files(run_dir)
    rows: list[dict[str, Any]] = []

    for path in metric_files:
        if path.suffix == ".json" and not path.name.endswith(".json.txt"):
            rows.extend(load_json_file(path))
        else:
            rows.extend(load_json_lines(path))

    # Preserve stable record ordering for plotting.
    for index, row in enumerate(rows):
        row["_record_index"] = index

    return rows, [str(path) for path in metric_files]


def write_metrics_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    keys = sorted({key for row in rows for key in row.keys()})
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def numeric_metric_keys(rows: list[dict[str, Any]]) -> list[str]:
    excluded = {
        "_record_index",
        "_source_line",
    }
    keys = []
    for key in sorted({key for row in rows for key in row}):
        if key in excluded or key.startswith("_source"):
            continue
        values = [row.get(key) for row in rows if is_scalar(row.get(key))]
        if len(values) >= 2:
            keys.append(key)
    return keys


def choose_x_key(rows: list[dict[str, Any]]) -> str:
    candidates = (
        "epoch",
        "global_step",
        "step",
        "train_step",
        "batch",
    )
    available = {key for row in rows for key in row}
    for candidate in candidates:
        if candidate in available:
            return candidate
    return "_record_index"


def safe_filename(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace("\\", "_")
        .replace(".", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def generate_plots(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    x_key = choose_x_key(rows)
    generated: list[str] = []

    # Prioritise loss, learning rate, time, and memory metrics.
    keys = numeric_metric_keys(rows)
    priority_tokens = ("loss", "lr", "learning_rate", "time", "memory")
    priority = [
        key for key in keys
        if any(token in key.lower() for token in priority_tokens)
    ]
    remaining = [key for key in keys if key not in priority]
    plot_keys = (priority + remaining)[:20]

    for metric_key in plot_keys:
        pairs = []
        for row in rows:
            x_value = row.get(x_key, row.get("_record_index"))
            y_value = row.get(metric_key)
            if is_scalar(x_value) and is_scalar(y_value):
                y_float = float(y_value)
                if math.isfinite(y_float):
                    pairs.append((float(x_value), y_float))

        if len(pairs) < 2:
            continue

        pairs.sort(key=lambda pair: pair[0])
        x_values = [pair[0] for pair in pairs]
        y_values = [pair[1] for pair in pairs]

        figure = plt.figure()
        axis = figure.add_subplot(1, 1, 1)
        axis.plot(x_values, y_values)
        axis.set_xlabel(x_key)
        axis.set_ylabel(metric_key)
        axis.set_title(metric_key)
        axis.grid(True)
        figure.tight_layout()

        output_path = output_dir / f"{safe_filename(metric_key)}.png"
        figure.savefig(output_path, dpi=160)
        plt.close(figure)
        generated.append(str(output_path))

    return generated


def build_metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "record_count": len(rows),
        "metrics": {},
    }

    for key in numeric_metric_keys(rows):
        values = [
            float(row[key])
            for row in rows
            if is_scalar(row.get(key)) and math.isfinite(float(row[key]))
        ]
        if not values:
            continue

        summary["metrics"][key] = {
            "count": len(values),
            "first": values[0],
            "last": values[-1],
            "minimum": min(values),
            "maximum": max(values),
            "mean": sum(values) / len(values),
        }

    return summary


def extract_run_metrics(
    run_dir: Path,
    *,
    create_plots: bool = True,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    rows, source_files = load_rows(run_dir)

    artifacts_dir = run_dir / "comparison_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = artifacts_dir / "metrics.csv"
    if rows:
        write_metrics_csv(rows, metrics_csv)

    plots = (
        generate_plots(rows, artifacts_dir / "plots")
        if rows and create_plots
        else []
    )

    summary = build_metric_summary(rows)
    summary.update(
        {
            "source_metric_files": source_files,
            "metrics_csv": str(metrics_csv) if rows else None,
            "plots": plots,
        }
    )

    summary_path = artifacts_dir / "metrics_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return summary


def main() -> int:
    args = parse_args()
    summary = extract_run_metrics(
        args.run_dir,
        create_plots=not args.no_plots,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())