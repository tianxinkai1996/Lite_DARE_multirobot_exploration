#!/usr/bin/env python3
"""Select the LiteDARE self-attention depth used downstream in Chapter 4.

DARE-L6 is the reference. LiteDARE-L4 and LiteDARE-L2 are evaluated on the
same paired single-robot map/trial scenarios using one shared training/base
random seed. The downstream model is whichever LiteDARE variant is closest to
L6 in exploration behaviour (success rate, final coverage and Coverage AUC).
Computational cost is reported separately and never makes a model look closer.

Non-inferiority margins are still evaluated and written to the summary because
they support the paper's performance-preservation claim, but they are not the
automatic model-selection gate. If L4 and L2 have effectively equal similarity
scores, the shallower L2 model is preferred.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_experiments.chapter4_config import (
    BOOTSTRAP_SAMPLES,
    DELTA_COVERAGE_AUC,
    DELTA_FINAL_COVERAGE,
    DELTA_SUCCESS_RATE,
    MODELS,
    RANDOM_SEED,
    SELECTION_TIE_TOLERANCE,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Chapter 4 output root")
    parser.add_argument("--delta-coverage", type=float, default=DELTA_FINAL_COVERAGE)
    parser.add_argument("--delta-success", type=float, default=DELTA_SUCCESS_RATE)
    parser.add_argument("--delta-auc", type=float, default=DELTA_COVERAGE_AUC)
    parser.add_argument("--tie-tolerance", type=float, default=SELECTION_TIE_TOLERANCE)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    return parser


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _success(value: object) -> float:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return 1.0
    if text in {"false", "0", "no"}:
        return 0.0
    return _float(value)


def _read_depth_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted((root / "e1_policy_depth").rglob("results.csv")):
        if path.stat().st_size == 0:
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    if not rows:
        raise FileNotFoundError(f"no E1 results.csv found below {root / 'e1_policy_depth'}")
    return rows


def _bootstrap_ci(values: np.ndarray, samples: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan, math.nan
    mean = float(values.mean())
    if values.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(RANDOM_SEED)
    draws = rng.choice(values, size=(int(samples), values.size), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return mean, float(low), float(high)


def _group(rows: Sequence[dict[str, str]]) -> dict[int, dict[str, dict[str, float]]]:
    grouped: dict[int, dict[str, dict[str, float]]] = {}
    for row in rows:
        if int(float(row.get("team_size", 1))) != 1:
            continue
        depth = int(float(row["encoder_layers"]))
        key = row.get("scenario_pair_key") or row.get("scenario_id")
        if not key:
            continue
        grouped.setdefault(depth, {})[str(key)] = {
            "final_coverage": _float(row.get("final_coverage", row.get("team_coverage"))),
            "coverage_auc": _float(row.get("coverage_auc")),
            "success": _success(row.get("success")),
            "policy_parameters": _float(row.get("policy_parameters", row.get("total_parameters"))),
            "latency_ms": _float(
                row.get(
                    "policy_inference_per_robot_mean_ms",
                    row.get("policy_inference_team_mean_ms", row.get("policy_inference_team_ms_mean")),
                )
            ),
        }
    missing = [depth for depth in (6, 4, 2) if depth not in grouped]
    if missing:
        raise ValueError(f"automatic selection requires L6/L4/L2 results; missing depths={missing}")
    return grouped


def _mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else math.nan


def _normalised_similarity_distance(
    success_diff: float,
    coverage_diff: float,
    auc_diff: float,
    *,
    delta_success: float,
    delta_coverage: float,
    delta_auc: float,
) -> float:
    """Return exploration distance to L6; lower means more similar.

    The scales are pre-declared practical comparison scales. They only make the
    three metrics dimensionless; they are not weights and do not use model cost.
    """
    scales = (float(delta_success), float(delta_coverage), float(delta_auc))
    if any(scale <= 0 for scale in scales):
        raise ValueError("all similarity scales must be positive")
    diffs = np.asarray([success_diff, coverage_diff, auc_diff], dtype=float)
    if not np.all(np.isfinite(diffs)):
        return math.inf
    scaled = diffs / np.asarray(scales, dtype=float)
    return float(np.linalg.norm(scaled, ord=2))


def select_model(
    root: Path,
    *,
    delta_coverage: float,
    delta_success: float,
    bootstrap_samples: int,
    delta_auc: float = DELTA_COVERAGE_AUC,
    tie_tolerance: float = SELECTION_TIE_TOLERANCE,
) -> tuple[str, list[dict[str, object]]]:
    rows = _read_depth_rows(root)
    grouped = _group(rows)
    baseline = grouped[6]
    summary: list[dict[str, object]] = []

    for depth in (6, 4, 2):
        part = grouped[depth]
        common = sorted(set(baseline).intersection(part))
        if not common:
            raise ValueError(f"L{depth} has no paired scenarios with L6")
        coverage = [part[key]["final_coverage"] for key in common]
        success = [part[key]["success"] for key in common]
        auc = [part[key]["coverage_auc"] for key in common]
        coverage_diff = np.asarray(
            [part[key]["final_coverage"] - baseline[key]["final_coverage"] for key in common], dtype=float
        )
        success_diff = np.asarray(
            [part[key]["success"] - baseline[key]["success"] for key in common], dtype=float
        )
        auc_diff = np.asarray(
            [part[key]["coverage_auc"] - baseline[key]["coverage_auc"] for key in common], dtype=float
        )
        cov_mean, cov_low, cov_high = _bootstrap_ci(coverage_diff, bootstrap_samples)
        suc_mean, suc_low, suc_high = _bootstrap_ci(success_diff, bootstrap_samples)
        auc_mean, auc_low, auc_high = _bootstrap_ci(auc_diff, bootstrap_samples)
        distance = 0.0 if depth == 6 else _normalised_similarity_distance(
            suc_mean,
            cov_mean,
            auc_mean,
            delta_success=delta_success,
            delta_coverage=delta_coverage,
            delta_auc=delta_auc,
        )
        preservation = depth == 6 or (
            np.isfinite(cov_low)
            and np.isfinite(suc_low)
            and cov_low >= -float(delta_coverage)
            and suc_low >= -float(delta_success)
        )
        summary.append(
            {
                "encoder_layers": depth,
                "model_key": next(model.key for model in MODELS.values() if model.encoder_layers == depth),
                "paired_episodes": len(common),
                "mean_final_coverage": _mean(coverage),
                "mean_success_rate": _mean(success),
                "mean_coverage_auc": _mean(auc),
                "coverage_diff_vs_L6": cov_mean,
                "coverage_diff_ci_low": cov_low,
                "coverage_diff_ci_high": cov_high,
                "success_diff_vs_L6": suc_mean,
                "success_diff_ci_low": suc_low,
                "success_diff_ci_high": suc_high,
                "auc_diff_vs_L6": auc_mean,
                "auc_diff_ci_low": auc_low,
                "auc_diff_ci_high": auc_high,
                "abs_coverage_diff_vs_L6": abs(cov_mean),
                "abs_success_diff_vs_L6": abs(suc_mean),
                "abs_auc_diff_vs_L6": abs(auc_mean),
                "similarity_distance_to_L6": distance,
                "mean_policy_parameters": _mean([part[key]["policy_parameters"] for key in common]),
                "mean_inference_latency_ms": _mean([part[key]["latency_ms"] for key in common]),
                "noninferior_coverage": bool(depth == 6 or cov_low >= -float(delta_coverage)),
                "noninferior_success": bool(depth == 6 or suc_low >= -float(delta_success)),
                "performance_preserved": bool(preservation),
            }
        )

    lite_rows = [row for row in summary if int(row["encoder_layers"]) in {2, 4}]
    lite_rows.sort(key=lambda row: float(row["similarity_distance_to_L6"]))
    best = lite_rows[0]
    second = lite_rows[1]
    if abs(float(best["similarity_distance_to_L6"]) - float(second["similarity_distance_to_L6"])) <= float(tie_tolerance):
        selected_row = min(lite_rows, key=lambda row: int(row["encoder_layers"]))
    else:
        selected_row = best
    for row in summary:
        row["selected_for_downstream"] = bool(row["model_key"] == selected_row["model_key"])
    return str(selected_row["model_key"]), summary


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_selected_python(
    path: Path,
    model_key: str,
    summary: Sequence[dict[str, object]],
    *,
    delta_coverage: float,
    delta_success: float,
    delta_auc: float,
    tie_tolerance: float,
) -> None:
    model = MODELS[model_key]
    row = next(item for item in summary if item["model_key"] == model_key)
    reason = (
        "LiteDARE variant with the smallest normalised exploration distance to DARE-L6 "
        "using paired success-rate, final-coverage and Coverage-AUC differences; "
        "model cost is excluded from the distance, and L2 is preferred only for an effective tie."
    )
    text = f'''"""Auto-generated by select_attention_model.py; do not edit manually."""\n\nSELECTED_MODEL_KEY = {model.key!r}\nSELECTED_MODEL_NAME = {model.display_name!r}\nSELECTED_ENCODER_LAYERS = {model.encoder_layers!r}\nSELECTED_CHECKPOINT = {str(model.checkpoint)!r}\nRANDOM_SEED = {RANDOM_SEED!r}\nDELTA_FINAL_COVERAGE = {float(delta_coverage)!r}\nDELTA_SUCCESS_RATE = {float(delta_success)!r}\nDELTA_COVERAGE_AUC = {float(delta_auc)!r}\nSELECTION_TIE_TOLERANCE = {float(tie_tolerance)!r}\nSIMILARITY_DISTANCE_TO_L6 = {float(row['similarity_distance_to_L6'])!r}\nPERFORMANCE_PRESERVED = {bool(row['performance_preserved'])!r}\nSELECTION_REASON = {reason!r}\n'''
    path.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.expanduser().resolve()
    selected, summary = select_model(
        root,
        delta_coverage=args.delta_coverage,
        delta_success=args.delta_success,
        delta_auc=args.delta_auc,
        tie_tolerance=args.tie_tolerance,
        bootstrap_samples=args.bootstrap_samples,
    )
    _write_csv(root / "attention_selection_summary.csv", summary)
    _write_selected_python(
        root / "selected_model.py",
        selected,
        summary,
        delta_coverage=args.delta_coverage,
        delta_success=args.delta_success,
        delta_auc=args.delta_auc,
        tie_tolerance=args.tie_tolerance,
    )
    print(f"selected_model={selected}")
    print(f"selection_file={root / 'selected_model.py'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


