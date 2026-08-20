"""Comprehensive Chapter 4 reporting without changing experiment execution.

The report has two simultaneous views:
1) an Overall view across N=2/4/6/8, and
2) explicit N=2,4,6,8 views under matched conditions.

No inference-time, GPU-memory or VRAM analysis is performed. The single-robot
lightweight evidence uses behaviour plus model size (parameter count) only.

Main-text figures are deliberately diverse (bar, box/scatter, forest, line,
heatmap and trade-off scatter). Additional diagnostic tables/figures are written
for optional appendix use when the underlying columns are available.
"""
from __future__ import annotations

import ast
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from paper_experiments.report_data import (
    complete_communication_pairs,
    complete_multi_pairs,
    pair_key,
)
from paper_experiments.report_scope import (
    PRIMARY_COMM_MODES,
    PRIMARY_MULTI_LABELS,
    PRIMARY_TEAM_SIZES,
    SINGLE_ROBOT_MODEL_ORDER,
)

OVERALL_LABEL = "Overall"
TEAM_SCOPE_ORDER = (OVERALL_LABEL,) + tuple(f"N={n}" for n in PRIMARY_TEAM_SIZES)


# Model-level metadata used only for structural parameter counting.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_CHECKPOINTS = {
    "DARE-L6": PROJECT_ROOT / (
        "runs/2026.06.25/"
        "12.50.20_train_diffusion_transformer_node_exploration_node/"
        "checkpoints/epoch=0180-val_loss=0.071.ckpt"
    ),
    "LiteDARE-L4": PROJECT_ROOT / (
        "lite_dare/runs/"
        "DiscreteLiteDARE_NodeEncSA_L4_H4_D256_DecL1_H4/"
        "seed_42_20260726_004608/"
        "checkpoints/epoch=0150-val_loss=0.058.ckpt"
    ),
    "LiteDARE-L2": PROJECT_ROOT / (
        "lite_dare/runs/"
        "DiscreteLiteDARE_NodeEncSA_L2_H4_D256_DecL1_H4/"
        "seed_42_20260730_095849/"
        "checkpoints/epoch=0150-val_loss=0.058.ckpt"
    ),
}


@lru_cache(maxsize=1)
def _checkpoint_parameter_counts() -> dict[str, float]:
    """Load each frozen checkpoint once on CPU and return total parameter counts."""
    import torch
    from mergingmap.multi_test_driver_mergingmap import load_frozen_policy

    counts: dict[str, float] = {}
    device = torch.device("cpu")

    for model_name, checkpoint in MODEL_CHECKPOINTS.items():
        if not checkpoint.is_file():
            print(
                f"[REPORT] parameter count unavailable for {model_name}: "
                f"checkpoint not found: {checkpoint}"
            )
            counts[model_name] = math.nan
            continue

        try:
            model = load_frozen_policy(device, checkpoint_path=checkpoint)
            counts[model_name] = float(
                sum(parameter.numel() for parameter in model.parameters())
            )
        except Exception as exc:
            print(
                f"[REPORT] parameter count unavailable for {model_name}: "
                f"{type(exc).__name__}: {exc}"
            )
            counts[model_name] = math.nan

    return counts


def _model_parameter_count(part: pd.DataFrame, model_name: str) -> float:
    """Prefer numeric CSV metadata; otherwise recover the checkpoint count."""
    param_column = _first(part, ["policy_parameters", "total_parameters"])
    if param_column:
        values = pd.to_numeric(part[param_column], errors="coerce").dropna()
        if not values.empty:
            return float(values.mean())
    return float(_checkpoint_parameter_counts().get(str(model_name), math.nan))


def _plt():
    import matplotlib.pyplot as plt
    return plt


# Main-text visual policy:
# bootstrap confidence intervals are still computed and exported, but the
# dissertation figures show means/trends without CI error bars for readability.
# Key 95% CI values remain available in the chapter text and supplementary CSVs.

def _save_figure(fig, out: Path, stem: str, *, preview_png: bool = True) -> None:
    """Save a dissertation figure as vector PDF, with optional PNG preview.

    PDF is the authoritative dissertation artefact.  The PNG copy is only for
    quick inspection and is not intended for inclusion in the final thesis.
    """
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / f"{stem}.pdf", bbox_inches="tight")
    if preview_png:
        fig.savefig(figures / f"{stem}.png", dpi=220, bbox_inches="tight")


def _save(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _first(data: pd.DataFrame, names: Sequence[str]) -> str | None:
    return next((name for name in names if name in data.columns), None)


def _success_float(series: pd.Series) -> pd.Series:
    mapped = series.astype(str).str.lower().map(
        {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0}
    )
    return mapped.fillna(pd.to_numeric(series, errors="coerce"))


def _metric_numeric(series: pd.Series, metric: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if metric.startswith("steps_to_"):
        values = values.mask(values.lt(0))
    return values


def _as_percent(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size and np.nanmax(np.abs(finite)) <= 1.5:
        return arr * 100.0
    return arr


def _bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan, math.nan
    mean = float(values.mean())
    if values.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, values.size), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return mean, float(low), float(high)


def _bootstrap_macro_ci(groups: Sequence[np.ndarray], samples: int, seed: int) -> tuple[float, float, float]:
    """Equal-weight team-size macro mean and bootstrap CI."""
    clean = []
    for values in groups:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            clean.append(arr)
    if not clean:
        return math.nan, math.nan, math.nan
    mean = float(np.mean([arr.mean() for arr in clean]))
    if all(arr.size == 1 for arr in clean):
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    boot_means = np.zeros(samples, dtype=float)
    for arr in clean:
        idx = rng.integers(0, arr.size, size=(samples, arr.size))
        boot_means += arr[idx].mean(axis=1)
    boot_means /= float(len(clean))
    low, high = np.quantile(boot_means, [0.025, 0.975])
    return mean, float(low), float(high)


def _add_derived_metrics(data: pd.DataFrame) -> pd.DataFrame:
    work = data.copy()
    if "success" in work.columns:
        work["success_rate"] = _success_float(work["success"])
    n = pd.to_numeric(work.get("team_size", pd.Series(index=work.index, dtype=float)), errors="coerce")

    path = _first(work, ["team_travel_distance_recorded", "team_travel_distance"])
    if path:
        work["team_path_per_robot"] = pd.to_numeric(work[path], errors="coerce") / n

    collisions = _first(work, ["actual_collision_pairs", "actual_collision_pairs_recorded", "actual_collision_steps"])
    if collisions:
        c = pd.to_numeric(work[collisions], errors="coerce")
        work["collisions_per_robot"] = c / n
        pair_den = n * (n - 1.0) / 2.0
        work["collisions_per_robot_pair"] = c / pair_den.where(pair_den.gt(0))

    bytes_col = _first(
        work,
        [
            "communication_bytes_recorded", "map_bytes_recorded", "map_bytes_sent",
            "communication_payload_bytes_recorded", "communication_payload_bytes",
        ],
    )
    if bytes_col:
        b = pd.to_numeric(work[bytes_col], errors="coerce")
        work["map_payload_kib"] = b / 1024.0
        work["map_payload_kib_per_robot"] = b / 1024.0 / n

    packets = _first(work, ["communication_packets_recorded", "map_packets_recorded", "map_packets_sent", "packets_sent"])
    if packets:
        p = pd.to_numeric(work[packets], errors="coerce")
        work["packets_per_robot"] = p / n

    retrans = _first(work, ["retransmission_packets", "map_retransmission_packets", "retransmission_packets_recorded"])
    if retrans and packets:
        r = pd.to_numeric(work[retrans], errors="coerce")
        p = pd.to_numeric(work[packets], errors="coerce")
        work["retransmission_rate"] = r / p.where(p.gt(0))

    for threshold in (90, 95, 99):
        col = _first(work, [f"steps_to_{threshold}_coverage", f"steps_to_{threshold}", f"steps{threshold}"])
        if col:
            raw = pd.to_numeric(work[col], errors="coerce")
            work[f"reached_{threshold}_rate"] = raw.ge(0).astype(float).where(raw.notna())
            work[f"steps_to_{threshold}_conditional"] = raw.mask(raw.lt(0))
    return work


def _summary_overall_and_team(
    data: pd.DataFrame,
    *,
    treatment_column: str,
    treatments: Sequence[str],
    metrics: Sequence[str],
    samples: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    work = _add_derived_metrics(data)
    for t_idx, treatment in enumerate(treatments):
        tdata = work.loc[work[treatment_column].astype(str).eq(str(treatment))]
        for m_idx, metric in enumerate(metrics):
            if metric not in tdata.columns:
                continue
            team_groups: list[np.ndarray] = []
            for team_size in PRIMARY_TEAM_SIZES:
                part = tdata.loc[pd.to_numeric(tdata["team_size"], errors="coerce").eq(team_size)]
                values = _metric_numeric(part[metric], metric).to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if values.size:
                    team_groups.append(values)
                mean, low, high = _bootstrap_ci(
                    values,
                    samples=samples,
                    seed=seed + 100*m_idx + team_size,
                )
                rows.append({
                    "scope": f"N={team_size}", "team_size": team_size,
                    treatment_column: treatment, "metric": metric,
                    "n": int(values.size), "mean": mean, "ci_low": low, "ci_high": high,
                })
            mean, low, high = _bootstrap_macro_ci(
                team_groups,
                samples=samples,
                seed=seed + 50000 + 100*m_idx,
            )
            rows.append({
                "scope": OVERALL_LABEL, "team_size": np.nan,
                treatment_column: treatment, "metric": metric,
                "n": int(sum(len(v) for v in team_groups)), "mean": mean, "ci_low": low, "ci_high": high,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["scope"] = pd.Categorical(out["scope"], categories=TEAM_SCOPE_ORDER, ordered=True)
        out = out.sort_values(["scope", treatment_column, "metric"]).reset_index(drop=True)
        out["scope"] = out["scope"].astype(str)
    return out


def _pivot_summary(long_summary: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    if long_summary.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for group_values, part in long_summary.groupby(list(group_columns), dropna=False, observed=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = dict(zip(group_columns, group_values))
        row["episodes"] = int(part["n"].max()) if "n" in part else 0
        for _, metric_row in part.iterrows():
            metric = str(metric_row["metric"])
            row[metric] = metric_row["mean"]
            row[f"{metric}_n"] = int(metric_row["n"])
            row[f"{metric}_ci_low"] = metric_row["ci_low"]
            row[f"{metric}_ci_high"] = metric_row["ci_high"]
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_effects_overall_and_team(
    data: pd.DataFrame,
    *,
    treatment_column: str,
    reference: str,
    candidates: Sequence[str],
    metrics: Sequence[str],
    samples: int,
    seed: int,
) -> pd.DataFrame:
    if data.empty or "_report_pair" not in data.columns:
        return pd.DataFrame()
    work = _add_derived_metrics(data)
    rows: list[dict] = []
    for c_idx, candidate in enumerate(candidates):
        for m_idx, metric in enumerate(metrics):
            if metric not in work.columns:
                continue
            by_team: list[np.ndarray] = []
            for team_size in PRIMARY_TEAM_SIZES:
                team = work.loc[pd.to_numeric(work["team_size"], errors="coerce").eq(team_size)]
                ref = team.loc[team[treatment_column].astype(str).eq(reference)]
                cand = team.loc[team[treatment_column].astype(str).eq(candidate)]
                left = cand[["_report_pair", metric]].dropna().rename(columns={metric: "candidate"})
                right = ref[["_report_pair", metric]].dropna().rename(columns={metric: "reference"})
                paired = left.merge(right, on="_report_pair", how="inner")
                cand_v = _metric_numeric(paired["candidate"], metric)
                ref_v = _metric_numeric(paired["reference"], metric)
                diff = (cand_v - ref_v).to_numpy(dtype=float)
                valid = np.isfinite(diff)
                diff = diff[valid]
                if diff.size:
                    by_team.append(diff)
                mean, low, high = _bootstrap_ci(diff, samples=samples, seed=seed + 1000*c_idx + 100*m_idx + team_size)
                # relative change uses paired reference mean; left blank when undefined or near zero.
                ref_valid = ref_v.to_numpy(dtype=float)[valid]
                ref_mean = float(np.nanmean(ref_valid)) if ref_valid.size else math.nan
                rel = mean / ref_mean * 100.0 if np.isfinite(ref_mean) and abs(ref_mean) > 1e-12 else math.nan
                wins = int(np.sum(diff > 0)); ties = int(np.sum(np.isclose(diff, 0.0))); losses = int(np.sum(diff < 0))
                rows.append({
                    "scope": f"N={team_size}", "team_size": team_size,
                    "comparison": f"{candidate} - {reference}", "metric": metric,
                    "pairs": int(diff.size), "mean_diff": mean, "ci_low": low, "ci_high": high,
                    "relative_change_percent": rel,
                    "candidate_better_raw_positive": wins, "ties": ties, "candidate_lower_raw": losses,
                })
            mean, low, high = _bootstrap_macro_ci(by_team, samples=samples, seed=seed + 50000 + 1000*c_idx + 100*m_idx)
            rows.append({
                "scope": OVERALL_LABEL, "team_size": np.nan,
                "comparison": f"{candidate} - {reference}", "metric": metric,
                "pairs": int(sum(len(v) for v in by_team)), "mean_diff": mean, "ci_low": low, "ci_high": high,
                "relative_change_percent": math.nan,
                "candidate_better_raw_positive": int(sum(np.sum(v > 0) for v in by_team)),
                "ties": int(sum(np.sum(np.isclose(v, 0.0)) for v in by_team)),
                "candidate_lower_raw": int(sum(np.sum(v < 0) for v in by_team)),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["scope"] = pd.Categorical(out["scope"], categories=TEAM_SCOPE_ORDER, ordered=True)
        out = out.sort_values(["metric", "scope", "comparison"]).reset_index(drop=True)
        out["scope"] = out["scope"].astype(str)
    return out


def _quantile_table(data: pd.DataFrame, *, group_columns: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    work = _add_derived_metrics(data)
    rows: list[dict] = []
    for group_values, part in work.groupby(list(group_columns), dropna=False):
        if not isinstance(group_values, tuple): group_values = (group_values,)
        base = dict(zip(group_columns, group_values))
        for metric in metrics:
            if metric not in part.columns: continue
            v = _metric_numeric(part[metric], metric).dropna().to_numpy(dtype=float)
            if not v.size: continue
            rows.append({**base, "metric": metric, "n": len(v), "p10": np.quantile(v, .10), "median": np.median(v), "p90": np.quantile(v, .90), "iqr": np.quantile(v,.75)-np.quantile(v,.25)})
    return pd.DataFrame(rows)


def _scaling_table(summary_long: pd.DataFrame, treatment_column: str) -> pd.DataFrame:
    rows: list[dict] = []
    if summary_long.empty: return pd.DataFrame()
    for (treatment, metric), part in summary_long.groupby([treatment_column, "metric"], dropna=False):
        n2 = part.loc[part["scope"].eq("N=2"), "mean"]
        n8 = part.loc[part["scope"].eq("N=8"), "mean"]
        if n2.empty or n8.empty: continue
        a, b = float(n2.iloc[0]), float(n8.iloc[0])
        rows.append({treatment_column: treatment, "metric": metric, "N2_mean": a, "N8_mean": b, "N8_minus_N2": b-a, "N8_vs_N2_percent": (b-a)/a*100 if abs(a)>1e-12 else math.nan})
    return pd.DataFrame(rows)


def _mean_percent(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if not values.size:
        return math.nan
    mean = float(values.mean())
    return mean * 100.0 if np.nanmax(np.abs(values)) <= 1.5 else mean


def _single_robot_absolute_table(single: pd.DataFrame) -> pd.DataFrame:
    """Return the exact numerical columns required by dissertation Table 4.1."""
    if single.empty:
        return pd.DataFrame()
    work = _add_derived_metrics(single)
    final_cov = _first(work, ["final_coverage", "team_coverage"])
    rows: list[dict] = []
    for model in SINGLE_ROBOT_MODEL_ORDER:
        part = work.loc[
            work.get(
                "model_name",
                pd.Series(index=work.index, dtype=object),
            ).astype(str).eq(model)
        ]
        if part.empty:
            continue

        parameter_count = _model_parameter_count(part, model)

        row = {
            "Model": model,
            "Runs": int(len(part)),
            "Success (%)": _mean_percent(part["success_rate"]) if "success_rate" in part else math.nan,
            "Final cov. (%)": _mean_percent(part[final_cov]) if final_cov else math.nan,
            "Cov. AUC": float(pd.to_numeric(part["coverage_auc"], errors="coerce").mean()) if "coverage_auc" in part else math.nan,
            "Reach@95 (%)": _mean_percent(part["reached_95_rate"]) if "reached_95_rate" in part else math.nan,
            "Steps@90": float(pd.to_numeric(part["steps_to_90_conditional"], errors="coerce").mean()) if "steps_to_90_conditional" in part else math.nan,
            "Steps@95": float(pd.to_numeric(part["steps_to_95_conditional"], errors="coerce").mean()) if "steps_to_95_conditional" in part else math.nan,
            "Steps@99": float(pd.to_numeric(part["steps_to_99_conditional"], errors="coerce").mean()) if "steps_to_99_conditional" in part else math.nan,
            "Params (M)": parameter_count / 1e6 if np.isfinite(parameter_count) else math.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def table_4_2_selection(selection_file: Path, out: Path, notes: list[str]) -> None:
    """Create Table 4.2 with paired means, CI bounds and selection metadata."""
    if not selection_file.is_file() or selection_file.stat().st_size == 0:
        notes.append("Table 4.2 skipped: attention_selection_summary.csv unavailable")
        return

    data = pd.read_csv(selection_file)
    required = {
        "model_key",
        "success_diff_vs_L6", "success_diff_ci_low", "success_diff_ci_high",
        "coverage_diff_vs_L6", "coverage_diff_ci_low", "coverage_diff_ci_high",
        "auc_diff_vs_L6", "auc_diff_ci_low", "auc_diff_ci_high",
        "similarity_distance_to_L6",
        "performance_preserved",
        "selected_for_downstream",
    }
    if not required.issubset(data.columns):
        notes.append("Table 4.2 skipped: selection summary lacks CI/selection columns")
        return

    try:
        from paper_experiments.chapter4_config import (
            DELTA_SUCCESS_RATE,
            DELTA_FINAL_COVERAGE,
            DELTA_COVERAGE_AUC,
        )
    except Exception:
        DELTA_SUCCESS_RATE, DELTA_FINAL_COVERAGE, DELTA_COVERAGE_AUC = 0.02, 0.01, 0.02

    model_order = {"LiteDARE-L4": 0, "LiteDARE-L2": 1}
    metric_specs = (
        ("Success", "success_diff_vs_L6", "success_diff_ci_low", "success_diff_ci_high", DELTA_SUCCESS_RATE, "preservation"),
        ("Final coverage", "coverage_diff_vs_L6", "coverage_diff_ci_low", "coverage_diff_ci_high", DELTA_FINAL_COVERAGE, "preservation"),
        ("Coverage AUC", "auc_diff_vs_L6", "auc_diff_ci_low", "auc_diff_ci_high", DELTA_COVERAGE_AUC, "D_L scale"),
    )

    rows: list[dict[str, object]] = []
    lite = data.loc[data["model_key"].astype(str).isin(model_order)].copy()
    lite["_order"] = lite["model_key"].map(model_order)
    lite = lite.sort_values("_order")

    for _, model_row in lite.iterrows():
        for metric, mean_col, low_col, high_col, scale, role in metric_specs:
            rows.append({
                "Candidate": str(model_row["model_key"]),
                "Metric": metric,
                "Mean difference": pd.to_numeric(pd.Series([model_row[mean_col]]), errors="coerce").iloc[0],
                "95% CI lower": pd.to_numeric(pd.Series([model_row[low_col]]), errors="coerce").iloc[0],
                "95% CI upper": pd.to_numeric(pd.Series([model_row[high_col]]), errors="coerce").iloc[0],
                "Practical scale": float(scale),
                "Scale role": role,
                "D_L": float(model_row["similarity_distance_to_L6"]),
                "Preserved": bool(model_row["performance_preserved"]),
                "Downstream": bool(model_row["selected_for_downstream"]),
            })

    _save(
        pd.DataFrame(rows),
        out / "tables" / "table_4_2_paired_ci_and_selection.csv",
    )


def figure_4_1(single: pd.DataFrame, out: Path, notes: list[str]) -> None:
    """Figure 4.1 plus the exact absolute-results Table 4.1."""
    needed = {"encoder_layers", "model_name", "coverage_auc", "success"}
    if single.empty or not needed.issubset(single.columns):
        notes.append("Figure/Table 4.1 skipped: incomplete single-robot depth results")
        return

    work = _add_derived_metrics(single)
    table = _single_robot_absolute_table(single)
    _save(table, out / "tables" / "table_4_1_single_robot_absolute.csv")

    final_cov = _first(work, ["final_coverage", "team_coverage"])
    param = _first(work, ["policy_parameters", "total_parameters"])
    metrics = [
        m for m in ("success_rate", final_cov, "coverage_auc")
        if m and m in work.columns
    ]

    agg = {m: "mean" for m in metrics}
    if param:
        agg[param] = "mean"

    summary = (
        work.groupby(["encoder_layers", "model_name"], as_index=False)
        .agg(agg)
    )

    order = [
        m for m in SINGLE_ROBOT_MODEL_ORDER
        if m in set(summary["model_name"])
    ]
    summary["_order"] = summary["model_name"].map(
        {name: i for i, name in enumerate(order)}
    )
    summary = summary.sort_values("_order").drop(columns="_order")

    checkpoint_counts = _checkpoint_parameter_counts()
    fallback = pd.to_numeric(
        summary["model_name"].map(checkpoint_counts),
        errors="coerce",
    )

    if param:
        csv_counts = pd.to_numeric(summary[param], errors="coerce")
        summary["_parameter_count"] = csv_counts.where(
            csv_counts.notna(),
            fallback,
        )
    else:
        summary["_parameter_count"] = fallback

    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    x = np.arange(len(summary))
    width = 0.24
    labels = {
        "success_rate": "Success",
        "final_coverage": "Final coverage",
        "team_coverage": "Final coverage",
        "coverage_auc": "Coverage AUC",
    }

    for j, metric in enumerate(metrics):
        values = _as_percent(summary[metric])
        axes[0].bar(
            x + (j - (len(metrics) - 1) / 2) * width,
            values,
            width=width,
            label=labels.get(metric, metric),
        )

    axes[0].set_xticks(x, summary["model_name"], rotation=0)
    axes[0].set_ylabel("Performance (%)")
    axes[0].set_title("(a) Exploration outcomes")
    axes[0].grid(axis="y", alpha=.25)
    axes[0].legend(fontsize=8)

    parameter_values = (
        pd.to_numeric(summary["_parameter_count"], errors="coerce")
        .to_numpy(dtype=float)
        / 1e6
    )

    if np.isfinite(parameter_values).any():
        bars = axes[1].bar(summary["model_name"], parameter_values)
        axes[1].set_ylabel("Parameters (million)")
        axes[1].set_title("(b) Parameter count")
        axes[1].grid(axis="y", alpha=.25)

        for bar, value in zip(bars, parameter_values):
            if np.isfinite(value):
                axes[1].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height(),
                    f"{value:.3f} M",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        missing_models = [
            str(name)
            for name, value in zip(summary["model_name"], parameter_values)
            if not np.isfinite(value)
        ]
        if missing_models:
            notes.append(
                "Figure 4.1 parameter count unavailable for: "
                + ", ".join(missing_models)
            )
    else:
        axes[1].text(
            .5,
            .5,
            "Parameter count unavailable",
            ha="center",
            va="center",
        )
        axes[1].axis("off")
        notes.append(
            "Figure 4.1 parameter count unavailable: "
            "no numeric CSV metadata and checkpoint counting failed"
        )

    fig.suptitle(
        "Figure 4.1. DARE/LiteDARE behaviour and structural model size"
    )
    fig.tight_layout()
    _save_figure(fig, out, "figure_4_1_depth_performance_model_size")
    plt.close(fig)


def figure_4_2(single: pd.DataFrame, out: Path, notes: list[str], *, samples: int, seed: int) -> None:
    """Paired behavioural-deviation forest plot required as Figure 4.2."""
    needed = {"model_name", "coverage_auc", "success"}
    final_cov = _first(single, ["final_coverage", "team_coverage"])
    if single.empty or not needed.issubset(single.columns) or not final_cov:
        notes.append("Figure 4.2 skipped: paired single-robot metrics unavailable")
        return
    work = _add_derived_metrics(single)
    work["_pair"] = pair_key(work)
    ref = work.loc[work["model_name"].astype(str).eq("DARE-L6")]
    if ref.empty:
        notes.append("Figure 4.2 skipped: DARE-L6 reference unavailable")
        return
    rows = []
    for c_idx, candidate in enumerate(("LiteDARE-L4", "LiteDARE-L2")):
        cand = work.loc[work["model_name"].astype(str).eq(candidate)]
        for m_idx, metric in enumerate(("success_rate", final_cov, "coverage_auc")):
            if cand.empty or metric not in cand.columns:
                continue
            paired = cand[["_pair", metric]].dropna().rename(columns={metric: "candidate"}).merge(
                ref[["_pair", metric]].dropna().rename(columns={metric: "reference"}), on="_pair"
            )
            diff = (pd.to_numeric(paired["candidate"], errors="coerce") - pd.to_numeric(paired["reference"], errors="coerce")).to_numpy(dtype=float)
            diff = diff[np.isfinite(diff)]
            mean, low, high = _bootstrap_ci(diff, samples, seed + 100 * c_idx + m_idx)
            rows.append({"candidate": candidate, "metric": metric, "pairs": len(diff), "mean_diff": mean, "ci_low": low, "ci_high": high})
    summary = pd.DataFrame(rows)
    _save(summary, out / "tables" / "figure_4_2_paired_differences.csv")
    if summary.empty:
        notes.append("Figure 4.2 skipped: no paired rows")
        return

    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    candidates = [c for c in ("LiteDARE-L4", "LiteDARE-L2") if c in set(summary["candidate"])]
    y = np.arange(len(candidates))
    titles = (("success_rate", "Success rate"), (final_cov, "Final coverage"), ("coverage_auc", "Coverage AUC"))
    try:
        from paper_experiments.chapter4_config import (
            DELTA_SUCCESS_RATE,
            DELTA_FINAL_COVERAGE,
        )
    except Exception:
        DELTA_SUCCESS_RATE, DELTA_FINAL_COVERAGE = 0.02, 0.01
    # The performance-preservation gate uses the lower CI bounds of success
    # and final coverage only. Coverage AUC contributes to D_L but is not part
    # of the preservation gate.
    margins = {
        "success_rate": float(DELTA_SUCCESS_RATE),
        final_cov: float(DELTA_FINAL_COVERAGE),
    }
    for ax, (metric, title) in zip(axes, titles):
        part = summary.loc[summary["metric"].eq(metric)].set_index("candidate").reindex(candidates)
        means = part["mean_diff"].to_numpy(float)
        low = part["ci_low"].to_numpy(float)
        high = part["ci_high"].to_numpy(float)
        # Main-text figure shows paired mean differences only.
        # The 95% bootstrap CIs remain available in the exported CSV and are
        # reported numerically in the chapter text.
        ax.scatter(means, y, marker="o", s=42)
        ax.axvline(0, ls="--", lw=1)
        if metric in margins:
            margin = float(margins[metric])
            ax.axvline(-margin, ls=":", lw=1)
            finite_low = low[np.isfinite(low)]
            finite_high = high[np.isfinite(high)]
            left = min(float(finite_low.min()) if finite_low.size else 0.0, -margin)
            right = max(float(finite_high.max()) if finite_high.size else 0.0, 0.0)
            span = max(right - left, 1e-9)
            ax.set_xlim(left - 0.06 * span, right + 0.06 * span)
        ax.set_title(title)
        ax.set_xlabel("Candidate - DARE-L6")
        ax.grid(axis="x", alpha=.25)
    axes[0].set_yticks(y, candidates)
    fig.suptitle("Figure 4.2. Paired mean deviations from DARE-L6")
    fig.tight_layout()
    _save_figure(fig, out, "figure_4_2_paired_difference_forest")
    plt.close(fig)


def _multi_metric_columns(work: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    final_cov = _first(work, ["final_coverage", "team_coverage"])
    auc = "coverage_auc" if "coverage_auc" in work.columns else None
    overlap = _first(work, ["trajectory_overlap_ratio", "overlap_node_ratio", "revisit_ratio"])
    collisions = _first(work, ["actual_collision_pairs", "actual_collision_pairs_recorded", "actual_collision_steps"])
    deadlock = _first(work, ["deadlock_duration_robot_steps", "deadlock_event_duration_mean_steps", "deadlock_count", "deadlock_rate"])
    path = _first(work, ["team_travel_distance_recorded", "team_travel_distance"])
    metrics = [m for m in (final_cov, auc, "reached_95_rate", "steps_to_95_conditional", overlap, path, "team_path_per_robot", collisions, "collisions_per_robot_pair", deadlock) if m and m in work.columns]
    labels = {
        final_cov: "Final coverage", auc: "Coverage AUC", "reached_95_rate": "Reach@95 rate",
        "steps_to_95_conditional": "Steps@95 (reached only)", overlap: "Trajectory overlap", path: "Team path",
        "team_path_per_robot": "Path / robot", collisions: "Collisions", "collisions_per_robot_pair": "Collisions / robot pair",
        deadlock: "Deadlock / waiting",
    }
    return metrics, labels


def _summary_metric(summary: pd.DataFrame, metric: str | None, target: str) -> pd.Series:
    if not metric or metric not in summary.columns:
        return pd.Series(np.nan, index=summary.index, name=target)
    return pd.to_numeric(summary[metric], errors="coerce").rename(target)


def _wide_effect_table(effects: pd.DataFrame, mapping: Sequence[tuple[str, str | None]]) -> pd.DataFrame:
    rows = []
    for scope in TEAM_SCOPE_ORDER:
        row: dict[str, object] = {"Scope": scope}
        for label, metric in mapping:
            if not metric or effects.empty:
                row[label] = math.nan
                continue
            value = effects.loc[effects["scope"].eq(scope) & effects["metric"].eq(metric), "mean_diff"]
            row[label] = float(value.iloc[0]) if not value.empty else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_treatment_scaling(ax, long_summary: pd.DataFrame, metric: str, treatment_column: str, treatments: Sequence[str], labels: dict[str, str], ylabel: str, *, percent: bool = False) -> None:
    part = long_summary.loc[long_summary["metric"].eq(metric) & ~long_summary["scope"].eq(OVERALL_LABEL)]

    # Different marker/line combinations make exactly overlapping series visible.
    style_map = {
        "none": {"marker": "o", "linestyle": "-"},
        "raw": {"marker": "s", "linestyle": "--", "markerfacecolor": "none", "zorder": 4},
        "compressed": {"marker": "^", "linestyle": ":", "zorder": 3},
        "map_only": {"marker": "o", "linestyle": "-"},
        "full": {"marker": "s", "linestyle": "--"},
    }

    for treatment in treatments:
        r = part.loc[part[treatment_column].astype(str).eq(str(treatment))].copy()
        r["team_size"] = pd.to_numeric(r["team_size"], errors="coerce")
        r = r.sort_values("team_size")
        if r.empty:
            continue
        mean = r["mean"].to_numpy(float)
        low = r["ci_low"].to_numpy(float)
        high = r["ci_high"].to_numpy(float)
        if percent:
            mean, low, high = _as_percent(mean), _as_percent(low), _as_percent(high)
        style = style_map.get(str(treatment), {"marker": "o", "linestyle": "-"})
        ax.plot(
            r["team_size"],
            mean,
            label=labels.get(str(treatment), str(treatment)),
            **style,
        )
    ax.set_xticks(PRIMARY_TEAM_SIZES)
    ax.set_xlabel("Robots, N")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.25)


def figure_4_3(multi: pd.DataFrame, out: Path, notes: list[str], *, samples: int, seed: int) -> None:
    """Generate Figure 4.2 primary scaling and Figure 4.3 standalone blocking trend."""
    if multi.empty or not {"team_size", "method_role_normalised", "coverage_auc"}.issubset(multi.columns):
        notes.append("Figure 4.2 skipped: Map-only/Full results incomplete")
        return
    work = complete_multi_pairs(multi)
    if work.empty:
        notes.append("Figure 4.2 skipped: no complete matched Map-only/Full pairs")
        return
    work = _add_derived_metrics(work)
    metrics, labels = _multi_metric_columns(work)
    final_cov = _first(work, ["final_coverage", "team_coverage"])
    overlap = _first(work, ["trajectory_overlap_ratio", "overlap_node_ratio", "revisit_ratio"])
    deadlock = _first(work, ["deadlock_duration_robot_steps", "deadlock_event_duration_mean_steps", "deadlock_count", "deadlock_rate"])
    collision_pair = "collisions_per_robot_pair" if "collisions_per_robot_pair" in work.columns else None

    long_summary = _summary_overall_and_team(
        work, treatment_column="method_role_normalised", treatments=("map_only", "full"),
        metrics=metrics, samples=samples, seed=seed,
    )
    summary = _pivot_summary(long_summary, ("scope", "method_role_normalised"))
    summary["Method"] = summary["method_role_normalised"].map(PRIMARY_MULTI_LABELS)
    summary["_scope_order"] = summary["scope"].map({s: i for i, s in enumerate(TEAM_SCOPE_ORDER)})
    summary["_method_order"] = summary["method_role_normalised"].map({"map_only": 0, "full": 1})
    summary = summary.sort_values(["_scope_order", "_method_order"]).drop(columns=["_scope_order", "_method_order"])
    _save(long_summary, out / "tables" / "figure_4_2_absolute_mean_ci_long.csv")

    final_series = _summary_metric(summary, final_cov, "Final cov. (%)")
    reach_series = _summary_metric(summary, "reached_95_rate", "Reach@95 (%)")
    appendix = pd.DataFrame({
        "Scope": summary["scope"],
        "Method": summary["Method"],
        "Final cov. (%)": _as_percent(final_series),
        "AUC": _summary_metric(summary, "coverage_auc", "AUC"),
        "Reach@95 (%)": _as_percent(reach_series),
        "Steps@95": _summary_metric(summary, "steps_to_95_conditional", "Steps@95"),
        "Overlap": _summary_metric(summary, overlap, "Overlap"),
        "Path/robot": _summary_metric(summary, "team_path_per_robot", "Path/robot"),
        "Coll./pair": _summary_metric(summary, collision_pair, "Coll./pair"),
        "Deadlock/wait": _summary_metric(summary, deadlock, "Deadlock/wait"),
    })
    _save(appendix, out / "tables" / "supp_multi_absolute.csv")

    effects = _paired_effects_overall_and_team(
        work, treatment_column="method_role_normalised", reference="map_only", candidates=("full",),
        metrics=metrics, samples=samples, seed=seed,
    )
    _save(effects, out / "tables" / "supp_multi_full_minus_maponly_effects_long.csv")
    # Main-text table keeps only the metrics needed to support the core claims.
    # Final coverage and path length remain available in diagnostic exports but
    # are omitted from the thesis table to avoid redundant detail.
    main_effects = _wide_effect_table(effects, (
        ("Delta AUC", "coverage_auc"),
        ("Delta Steps@95", "steps_to_95_conditional"),
        ("Delta Overlap", overlap),
        ("Delta Coll.", collision_pair),
        ("Delta Deadlock", deadlock),
    ))
    _save(main_effects, out / "tables" / "supp_multi_full_minus_maponly_key_effects.csv")

    _save(_quantile_table(work, group_columns=("team_size", "method_role_normalised"), metrics=metrics), out / "tables" / "supp_multi_distribution_quantiles.csv")
    _save(_scaling_table(long_summary, "method_role_normalised"), out / "tables" / "supp_multi_N2_to_N8_scaling.csv")
    _save(_win_tie_loss_table(work, treatment_column="method_role_normalised", reference="map_only", candidate="full", metrics=metrics), out / "tables" / "supp_multi_win_tie_loss.csv")
    _save(_map_level_effects(work, treatment_column="method_role_normalised", reference="map_only", candidate="full", metrics=metrics), out / "tables" / "supp_multi_map_level_paired_effects.csv")
    for descriptor, frame in _difficulty_tables(work, treatment_column="method_role_normalised", metrics=metrics).items():
        _save(frame, out / "tables" / f"supp_multi_difficulty_{descriptor}.csv")

    required_panels = ["coverage_auc", overlap, collision_pair, "steps_to_95_conditional"]
    if any(metric is None or metric not in metrics for metric in required_panels):
        notes.append("Figure 4.2: one or more preferred panel metrics are unavailable; unavailable panels are annotated")

    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.5))
    role_labels = {"map_only": "Map-only", "full": "Full"}
    if "coverage_auc" in metrics:
        _plot_treatment_scaling(axes[0, 0], long_summary, "coverage_auc", "method_role_normalised", ("map_only", "full"), role_labels, "Coverage AUC")
        axes[0, 0].set_title("(a) Coverage AUC")
    else:
        axes[0, 0].text(.5, .5, "Coverage AUC unavailable", ha="center", va="center"); axes[0, 0].axis("off")
    if overlap and overlap in metrics:
        _plot_treatment_scaling(axes[0, 1], long_summary, overlap, "method_role_normalised", ("map_only", "full"), role_labels, "Trajectory overlap")
        axes[0, 1].set_title("(b) Trajectory overlap")
    else:
        axes[0, 1].text(.5, .5, "Overlap unavailable", ha="center", va="center"); axes[0, 1].axis("off")
    if collision_pair and collision_pair in metrics:
        _plot_treatment_scaling(axes[1, 0], long_summary, collision_pair, "method_role_normalised", ("map_only", "full"), role_labels, "Collisions / robot pair")
        axes[1, 0].set_title("(c) Normalised collisions")
    else:
        axes[1, 0].text(.5, .5, "Collision/pair unavailable", ha="center", va="center"); axes[1, 0].axis("off")

    ax = axes[1, 1]
    if "steps_to_95_conditional" in metrics:
        _plot_treatment_scaling(
            ax,
            long_summary,
            "steps_to_95_conditional",
            "method_role_normalised",
            ("map_only", "full"),
            role_labels,
            "Steps to 95% coverage",
        )
        ax.set_title("(d) Steps to 95% coverage")
        ax.legend(fontsize=8)
    else:
        ax.text(.5, .5, "Steps@95 unavailable", ha="center", va="center")
        ax.axis("off")

    for pane in (axes[0, 0], axes[0, 1], axes[1, 0]):
        if pane.axison:
            pane.legend(fontsize=8)

    fig.suptitle("Figure 4.2. Map-only vs Full under compressed communication")
    fig.tight_layout()
    _save_figure(fig, out, "figure_4_2_maponly_full_primary_scaling")
    plt.close(fig)

    # Figure 4.3 isolates blocking/deadlock so it does not share a second y-axis
    # with progress. Exact means are labelled directly to avoid a redundant table.
    if deadlock and deadlock in metrics:
        fig, ax = plt.subplots(1, 1, figsize=(7.4, 4.5))
        part = long_summary.loc[
            long_summary["metric"].eq(deadlock)
            & ~long_summary["scope"].eq(OVERALL_LABEL)
        ].copy()

        style_map = {
            "map_only": {"marker": "o", "linestyle": "-"},
            "full": {"marker": "s", "linestyle": "--"},
        }

        plotted = False
        for role in ("map_only", "full"):
            r = part.loc[part["method_role_normalised"].eq(role)].copy()
            r["team_size"] = pd.to_numeric(r["team_size"], errors="coerce")
            r = r.sort_values("team_size")
            if r.empty:
                continue

            x = r["team_size"].to_numpy(float)
            y = r["mean"].to_numpy(float)
            line = ax.plot(
                x,
                y,
                label=role_labels[role],
                **style_map[role],
            )[0]
            colour = line.get_color()

            # Direct numeric labels make the standalone figure self-contained.
            for xi, yi in zip(x, y):
                offset = 8 if role == "map_only" else -13
                ax.annotate(
                    f"{yi:.3f}" if abs(yi) < 1 else f"{yi:.2f}",
                    (xi, yi),
                    xytext=(0, offset),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if offset > 0 else "top",
                    fontsize=8,
                    color=colour,
                )
            plotted = True

        if plotted:
            ax.set_xticks(list(PRIMARY_TEAM_SIZES))
            ax.set_xlabel("Robots, N")
            ax.set_ylabel(labels.get(deadlock, "Deadlock / waiting"))
            ax.set_title("Figure 4.3. Deadlock / waiting scaling")
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            _save_figure(fig, out, "figure_4_3_maponly_full_deadlock_scaling")
        else:
            notes.append("Figure 4.5 skipped: deadlock/waiting plotting data unavailable")
        plt.close(fig)
    else:
        notes.append("Figure 4.5 skipped: deadlock/waiting metric unavailable")




def _communication_metric_columns(work: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    final_cov = _first(work, ["final_coverage", "team_coverage"])
    iou = _first(work, ["mean_robot_map_free_iou", "team_map_free_iou", "map_free_iou", "mean_pairwise_map_iou"])
    packets = _first(work, ["communication_packets_recorded", "map_packets_recorded", "map_packets_sent", "packets_sent"])
    retrans = _first(work, ["retransmission_packets", "map_retransmission_packets", "retransmission_packets_recorded"])
    metrics = [m for m in (final_cov, "coverage_auc", iou, "reached_95_rate", "steps_to_95_conditional", "map_payload_kib", "map_payload_kib_per_robot", packets, "packets_per_robot", retrans, "retransmission_rate") if m and m in work.columns]
    labels = {final_cov: "Final coverage", "coverage_auc": "Coverage AUC", iou: "Map IoU", "reached_95_rate": "Reach@95 rate", "steps_to_95_conditional": "Steps@95 (reached only)", "map_payload_kib": "Map payload (KiB)", "map_payload_kib_per_robot": "Payload / robot (KiB)", packets: "Packets", "packets_per_robot": "Packets / robot", retrans: "Retransmissions", "retransmission_rate": "Retransmission rate"}
    return metrics, labels


def _communication_savings(work: pd.DataFrame) -> pd.DataFrame:
    """Raw-vs-compressed payload savings, Overall and per team size."""
    if "map_payload_kib" not in work.columns:
        return pd.DataFrame()
    rows = []
    for scope, n in [(OVERALL_LABEL, None)] + [(f"N={n}", n) for n in PRIMARY_TEAM_SIZES]:
        part = work if n is None else work.loc[pd.to_numeric(work["team_size"], errors="coerce").eq(n)]
        means = {}
        for mode in ("raw", "compressed"):
            m = part.loc[part["communication_mode"].astype(str).eq(mode)]
            if n is None:
                per_n = []
                for tn in PRIMARY_TEAM_SIZES:
                    v = pd.to_numeric(m.loc[pd.to_numeric(m["team_size"], errors="coerce").eq(tn), "map_payload_kib"], errors="coerce").dropna()
                    if len(v):
                        per_n.append(v.mean())
                means[mode] = float(np.mean(per_n)) if per_n else math.nan
            else:
                v = pd.to_numeric(m["map_payload_kib"], errors="coerce").dropna()
                means[mode] = float(v.mean()) if len(v) else math.nan
        raw, comp = means.get("raw", math.nan), means.get("compressed", math.nan)
        rows.append({
            "Scope": scope, "Raw KiB": raw, "Compressed KiB": comp,
            "Saved KiB": raw - comp if np.isfinite(raw) and np.isfinite(comp) else math.nan,
            "Saving (%)": (raw - comp) / raw * 100 if np.isfinite(raw) and abs(raw) > 1e-12 and np.isfinite(comp) else math.nan,
        })
    return pd.DataFrame(rows)


def _communication_common(comm: pd.DataFrame, *, samples: int, seed: int):
    work = complete_communication_pairs(comm)
    if work.empty:
        return work, [], {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    work = _add_derived_metrics(work)
    metrics, labels = _communication_metric_columns(work)
    long_summary = _summary_overall_and_team(
        work, treatment_column="communication_mode", treatments=PRIMARY_COMM_MODES,
        metrics=metrics, samples=samples, seed=seed,
    )
    summary = _pivot_summary(long_summary, ("scope", "communication_mode"))
    order = {m: i for i, m in enumerate(PRIMARY_COMM_MODES)}
    summary["_scope_order"] = summary["scope"].map({s: i for i, s in enumerate(TEAM_SCOPE_ORDER)})
    summary["_mode_order"] = summary["communication_mode"].map(order)
    summary = summary.sort_values(["_scope_order", "_mode_order"]).drop(columns=["_scope_order", "_mode_order"])
    effects = _paired_effects_overall_and_team(
        work, treatment_column="communication_mode", reference="compressed", candidates=("none", "raw"),
        metrics=metrics, samples=samples, seed=seed,
    )
    savings = _communication_savings(work)
    return work, metrics, labels, long_summary, summary, effects, savings


def figure_4_5(comm: pd.DataFrame, out: Path, notes: list[str], *, samples: int, seed: int) -> None:
    """Compact Figure 4.4: communication performance scaling (AUC and Map IoU)."""
    if comm.empty or not {"team_size", "communication_mode", "coverage_auc"}.issubset(comm.columns):
        notes.append("Figure 4.5 skipped: incomplete Full communication results")
        return

    common = _communication_common(comm, samples=samples, seed=seed)
    if len(common) != 7:
        notes.append("Figure 4.5 skipped: internal communication reporting error")
        return

    work, metrics, labels, long_summary, summary, effects, savings = common
    if work.empty:
        notes.append("Figure 4.5 skipped: no complete none/raw/compressed matched sets")
        return

    iou = _first(work, [
        "mean_robot_map_free_iou",
        "team_map_free_iou",
        "map_free_iou",
        "mean_pairwise_map_iou",
    ])
    if not iou or iou not in metrics:
        notes.append("Figure 4.5 skipped: Map IoU unavailable")
        return

    # Compact main-text table: raw and compressed are identical on AUC and IoU,
    # so report them once as a shared-map result rather than duplicating columns.
    rows = []
    for scope in TEAM_SCOPE_ORDER:
        part = summary.loc[summary["scope"].astype(str).eq(scope)]
        none_row = part.loc[part["communication_mode"].astype(str).eq("none")]
        raw_row = part.loc[part["communication_mode"].astype(str).eq("raw")]
        comp_row = part.loc[part["communication_mode"].astype(str).eq("compressed")]
        if none_row.empty or raw_row.empty or comp_row.empty:
            continue

        none_auc = _summary_metric(none_row, "coverage_auc", "AUC")
        raw_auc = _summary_metric(raw_row, "coverage_auc", "AUC")
        comp_auc = _summary_metric(comp_row, "coverage_auc", "AUC")
        none_iou = _summary_metric(none_row, iou, "Map IoU")
        raw_iou = _summary_metric(raw_row, iou, "Map IoU")
        comp_iou = _summary_metric(comp_row, iou, "Map IoU")

        rows.append({
            "Scope": scope,
            "AUC none": float(none_auc.iloc[0]),
            "AUC raw=compressed": float((raw_auc.iloc[0] + comp_auc.iloc[0]) / 2.0),
            "Map IoU none": float(none_iou.iloc[0]),
            "Map IoU raw=compressed": float((raw_iou.iloc[0] + comp_iou.iloc[0]) / 2.0),
        })

    key_table = pd.DataFrame(rows)
    _save(key_table, out / "tables" / "supp_communication_key_summary.csv")

    # Retain detailed diagnostics outside the thesis-facing tables for auditability.
    _save(long_summary, out / "tables" / "supp_communication_absolute_mean_ci_long.csv")
    _save(effects, out / "tables" / "supp_communication_mode_minus_compressed_effects.csv")
    _save(savings, out / "tables" / "supp_communication_raw_vs_compressed_payload_saving.csv")
    _save(
        _win_tie_loss_table(
            work,
            treatment_column="communication_mode",
            reference="raw",
            candidate="compressed",
            metrics=metrics,
        ),
        out / "tables" / "supp_communication_win_tie_loss_compressed_vs_raw.csv",
    )

    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    mode_labels = {m: m for m in PRIMARY_COMM_MODES}

    _plot_treatment_scaling(
        axes[0],
        long_summary,
        "coverage_auc",
        "communication_mode",
        PRIMARY_COMM_MODES,
        mode_labels,
        "Coverage AUC",
    )
    axes[0].set_title("(a) Coverage AUC")
    axes[0].legend(fontsize=8)

    _plot_treatment_scaling(
        axes[1],
        long_summary,
        iou,
        "communication_mode",
        PRIMARY_COMM_MODES,
        mode_labels,
        "Map IoU",
    )
    axes[1].set_title("(b) Map agreement")
    axes[1].legend(fontsize=8)

    fig.suptitle("Figure 4.4. Communication-mode scaling with Full coordination fixed")
    fig.tight_layout()
    _save_figure(fig, out, "figure_4_4_communication_primary_scaling")
    plt.close(fig)



def figure_4_6(comm: pd.DataFrame, out: Path, notes: list[str], *, samples: int, seed: int) -> None:
    """Figure 4.5: total-payload trade-off plus payload-per-robot scaling."""
    if comm.empty:
        notes.append("Figure 4.5 skipped: communication results unavailable")
        return

    work, metrics, labels, long_summary, summary, effects, savings = _communication_common(
        comm, samples=samples, seed=seed
    )
    required = {"map_payload_kib", "map_payload_kib_per_robot", "coverage_auc"}
    if work.empty or not required.issubset(metrics):
        notes.append("Figure 4.5 skipped: complete payload/AUC results unavailable")
        return

    # Retain exact data for auditability, but do not use an additional thesis table.
    _save(savings, out / "tables" / "supp_communication_raw_vs_compressed_payload_saving.csv")

    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2))

    scope_specs = [(OVERALL_LABEL, None)] + [(f"N={n}", n) for n in PRIMARY_TEAM_SIZES]
    scope_colours = {
        OVERALL_LABEL: "black",
        "N=2": "C1",
        "N=4": "C0",
        "N=6": "C2",
        "N=8": "C4",
    }

    def _point(scope_label: str, team_size, mode: str, metric: str):
        part = long_summary.loc[
            long_summary["metric"].eq(metric)
            & long_summary["communication_mode"].astype(str).eq(mode)
            & long_summary["scope"].astype(str).eq(scope_label)
        ]
        if team_size is not None and "team_size" in part.columns:
            part = part.loc[pd.to_numeric(part["team_size"], errors="coerce").eq(team_size)]
        if part.empty:
            return None
        return float(part.iloc[0]["mean"])

    # --------------------------------------------------------
    # (a) Total payload vs AUC.
    # --------------------------------------------------------
    ax = axes[0]
    x_min, x_max = math.inf, -math.inf
    y_min, y_max = math.inf, -math.inf
    left_labels = []

    for scope_label, team_size in scope_specs:
        raw_x = _point(scope_label, team_size, "raw", "map_payload_kib")
        comp_x = _point(scope_label, team_size, "compressed", "map_payload_kib")
        y = _point(scope_label, team_size, "raw", "coverage_auc")
        if y is None:
            y = _point(scope_label, team_size, "compressed", "coverage_auc")
        if raw_x is None or comp_x is None or y is None:
            continue

        row = savings.loc[savings["Scope"].astype(str).eq(scope_label)]
        saving_pct = float(row.iloc[0]["Saving (%)"]) if not row.empty else math.nan

        colour = scope_colours.get(scope_label, "C0")
        linestyle = "--" if scope_label == OVERALL_LABEL else "-"
        linewidth = 1.8 if scope_label == OVERALL_LABEL else 1.5

        ax.plot([comp_x, raw_x], [y, y], linestyle=linestyle, linewidth=linewidth,
                color=colour, alpha=0.95)
        ax.scatter([raw_x], [y], marker="s", s=64,
                   facecolors=("black" if scope_label == OVERALL_LABEL else "none"),
                   edgecolors=colour, linewidths=1.5, zorder=3)
        ax.scatter([comp_x], [y], marker="^", s=72, color=colour, zorder=3)

        x_mid = (raw_x * comp_x) ** 0.5
        label = f"Overall saved {saving_pct:.1f}%" if scope_label == OVERALL_LABEL else f"{saving_pct:.1f}% saved"
        ax.annotate(label, (x_mid, y), xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, color=colour)

        ax.annotate(f"{comp_x:.0f}", (comp_x, y), xytext=(0, -12),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=7.5, color=colour)
        ax.annotate(f"{raw_x:.0f}", (raw_x, y), xytext=(0, -12),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=7.5, color=colour)

        left_labels.append((scope_label, y, colour, team_size))
        x_min, x_max = min(x_min, comp_x, raw_x), max(x_max, comp_x, raw_x)
        y_min, y_max = min(y_min, y), max(y_max, y)

    ax.set_xscale("log")
    ax.set_xlabel("Map payload (KiB / episode)")
    ax.set_ylabel("Coverage AUC")
    ax.set_title("(a) Total-payload trade-off")

    # Keep the logarithmic scale because payload spans roughly 0.1--3.5 MiB,
    # but simplify the axis to a small set of human-readable major ticks.
    # Exact endpoint payloads are already annotated next to the markers.
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    payload_ticks = [100, 200, 500, 1000, 2000, 4000]
    ax.xaxis.set_major_locator(FixedLocator(payload_ticks))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, pos: f"{int(value)}" if value in payload_ticks else "")
    )
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(alpha=.25, which="major")

    if np.isfinite(x_min) and np.isfinite(y_min):
        ax.set_xlim(90, 4200)
        ax.set_ylim(max(0.58, y_min - 0.03), min(0.92, y_max + 0.05))
        x_text = 105
        for scope_label, y, colour, team_size in sorted(left_labels, key=lambda item: item[1]):
            ax.annotate(scope_label, (x_text, y), xytext=(0, 4),
                        textcoords="offset points", ha="left", va="bottom",
                        fontsize=8.5 if scope_label != OVERALL_LABEL else 9.5,
                        fontweight="bold", color=colour)

    # --------------------------------------------------------
    # (b) Payload per robot.
    # --------------------------------------------------------
    ax = axes[1]
    team_sizes = list(PRIMARY_TEAM_SIZES)
    raw_values, comp_values = [], []
    for n in team_sizes:
        raw_values.append(_point(f"N={n}", n, "raw", "map_payload_kib_per_robot"))
        comp_values.append(_point(f"N={n}", n, "compressed", "map_payload_kib_per_robot"))

    ax.plot(team_sizes, raw_values, marker="s", linestyle="--", label="raw")
    ax.plot(team_sizes, comp_values, marker="^", linestyle=":", label="compressed")

    # Label each point directly so the panel can stand without a separate table.
    for n, raw_v, comp_v in zip(team_sizes, raw_values, comp_values):
        if raw_v is not None:
            ax.annotate(f"{raw_v:.1f}", (n, raw_v), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=8)
        if comp_v is not None:
            ax.annotate(f"{comp_v:.1f}", (n, comp_v), xytext=(0, -12),
                        textcoords="offset points", ha="center", fontsize=8)

    # Overall mean reference lines make the average communication burden explicit.
    raw_overall = _point(OVERALL_LABEL, None, "raw", "map_payload_kib_per_robot")
    comp_overall = _point(OVERALL_LABEL, None, "compressed", "map_payload_kib_per_robot")
    if raw_overall is not None:
        ax.axhline(raw_overall, linestyle="--", linewidth=1, alpha=.55)
        ax.text(team_sizes[-1] + 0.08, raw_overall, f"raw avg {raw_overall:.1f}",
                va="center", fontsize=7.5)
    if comp_overall is not None:
        ax.axhline(comp_overall, linestyle=":", linewidth=1, alpha=.55)
        ax.text(team_sizes[-1] + 0.08, comp_overall, f"comp. avg {comp_overall:.1f}",
                va="center", fontsize=7.5)

    ax.set_xticks(team_sizes)
    ax.set_xlabel("Robots, N")
    ax.set_ylabel("Map payload (KiB / robot)")
    ax.set_title("(b) Payload per robot")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)

    fig.suptitle("Figure 4.5. Raw-to-compressed communication efficiency")
    fig.tight_layout()
    _save_figure(fig, out, "figure_4_5_communication_tradeoff_and_per_robot")
    plt.close(fig)

def _parse_serialised(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (list, tuple, dict)):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(text)
        except Exception:
            pass
    return None


def _xy_array(value) -> np.ndarray | None:
    obj = _parse_serialised(value)
    try:
        arr = np.asarray(obj, dtype=float)
    except Exception:
        return None
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, :2]
    return None


def _extract_trajectories(row: pd.Series) -> list[np.ndarray]:
    candidates = ("robot_trajectories", "team_trajectories", "trajectories", "trajectory_visits", "robot_paths", "paths", "trajectory_positions")
    for column in candidates:
        if column not in row.index:
            continue
        obj = _parse_serialised(row[column])
        if isinstance(obj, dict):
            result = []
            for value in obj.values():
                arr = _xy_array(value)
                if arr is not None and len(arr):
                    result.append(arr)
            if result:
                return result
        if isinstance(obj, (list, tuple)):
            one = _xy_array(obj)
            if one is not None:
                return [one]
            result = []
            for value in obj:
                arr = _xy_array(value)
                if arr is not None and len(arr):
                    result.append(arr)
            if result:
                return result
    return []


def _extract_series(row: pd.Series, names: Sequence[str]) -> np.ndarray | None:
    for name in names:
        if name not in row.index:
            continue
        obj = _parse_serialised(row[name])
        try:
            arr = np.asarray(obj, dtype=float).reshape(-1)
        except Exception:
            continue
        arr = arr[np.isfinite(arr)]
        if arr.size:
            return arr
    return None


def _case_interaction_scores(work: pd.DataFrame) -> pd.DataFrame:
    work = _add_derived_metrics(work)
    baseline = work.loc[work["method_role_normalised"].astype(str).eq("map_only")].copy()
    candidates = [
        _first(baseline, ["trajectory_overlap_ratio", "overlap_node_ratio", "revisit_ratio"]),
        "collisions_per_robot_pair" if "collisions_per_robot_pair" in baseline.columns else None,
        _first(baseline, ["deadlock_duration_robot_steps", "deadlock_event_duration_mean_steps", "deadlock_count", "deadlock_rate"]),
    ]
    candidates = [c for c in candidates if c and c in baseline.columns]
    if not candidates:
        baseline["interaction_score"] = 0.0
        return baseline
    rank_parts = []
    for metric in candidates:
        values = pd.to_numeric(baseline[metric], errors="coerce")
        rank_parts.append(values.rank(pct=True, method="average"))
    baseline["interaction_score"] = pd.concat(rank_parts, axis=1).mean(axis=1, skipna=True).fillna(0.0)
    return baseline


def _select_qualitative_cases(multi: pd.DataFrame, comm: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    mw = complete_multi_pairs(multi)
    if not mw.empty:
        scored = _case_interaction_scores(mw)
        if not scored.empty:
            median = float(scored["interaction_score"].median())
            typical = scored.loc[(scored["interaction_score"] - median).abs().idxmin()]
            high = scored.loc[scored["interaction_score"].idxmax()]
            for case, item, rule in (
                ("coordination_typical", typical, "Map-only interaction score closest to the median; outcome difference is not used"),
                ("coordination_high_interaction", high, "Highest pre-declared Map-only interaction score; outcome difference is not used"),
            ):
                rows.append({
                    "case": case, "pair_key": item.get("_report_pair", ""), "team_size": item.get("team_size", np.nan),
                    "map_index": item.get("map_index", np.nan), "trial": item.get("trial", np.nan),
                    "interaction_score": item.get("interaction_score", np.nan), "selection_rule": rule,
                })
    cw = complete_communication_pairs(comm)
    if not cw.empty:
        cw = _add_derived_metrics(cw)
        raw = cw.loc[cw["communication_mode"].astype(str).eq("raw")].copy()
        if not raw.empty:
            available_n = sorted(pd.to_numeric(raw["team_size"], errors="coerce").dropna().astype(int).unique())
            target_n = 4 if 4 in available_n else available_n[len(available_n) // 2]
            part = raw.loc[pd.to_numeric(raw["team_size"], errors="coerce").eq(target_n)].copy()
            if "map_payload_kib" in part.columns and pd.to_numeric(part["map_payload_kib"], errors="coerce").notna().any():
                payload = pd.to_numeric(part["map_payload_kib"], errors="coerce")
                median = float(payload.median())
                item = part.loc[(payload - median).abs().idxmin()]
                rule = f"N={target_n} raw-payload episode closest to the median; AUC/IoU differences are not used"
            else:
                part = part.sort_values([c for c in ("map_index", "trial", "seed") if c in part.columns])
                item = part.iloc[0]
                rule = f"First deterministic N={target_n} matched raw/compressed episode because payload history is unavailable"
            rows.append({
                "case": "communication_typical", "pair_key": item.get("_report_pair", ""), "team_size": item.get("team_size", np.nan),
                "map_index": item.get("map_index", np.nan), "trial": item.get("trial", np.nan),
                "interaction_score": np.nan, "selection_rule": rule,
            })
    return pd.DataFrame(rows)


def _pair_rows(data: pd.DataFrame, pair_key_value: str, column: str, values: Sequence[str]) -> dict[str, pd.Series]:
    work = data.loc[data.get("_report_pair", pd.Series(index=data.index, dtype=object)).astype(str).eq(str(pair_key_value))]
    result = {}
    for value in values:
        part = work.loc[work[column].astype(str).eq(str(value))]
        if not part.empty:
            result[str(value)] = part.iloc[0]
    return result


def _explicit_image_path(row: pd.Series, root: Path) -> Path | None:
    for name in ("visualisation_path", "visualization_path", "trajectory_plot", "episode_figure", "render_path", "figure_path", "image_path"):
        if name not in row.index:
            continue
        text = str(row[name]).strip()
        if not text or text.lower() in {"nan", "none"}:
            continue
        path = Path(text).expanduser()
        if not path.is_absolute():
            source = Path(str(row.get("_source_file", "")))
            base = source.parent if source.exists() else root
            path = base / path
        if path.is_file():
            return path
    return None


def _build_image_index(root: Path) -> list[Path]:
    images = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        images.extend(root.rglob(ext))
    return images


def _discover_image(row: pd.Series, root: Path, image_index: Sequence[Path]) -> Path | None:
    direct = _explicit_image_path(row, root)
    if direct:
        return direct
    pair_value = str(row.get("_report_pair", row.get("scenario_pair_key", row.get("scenario_id", ""))))
    scenario = str(row.get("scenario_pair_key", row.get("scenario_id", "")))
    role = str(row.get("method_role_normalised", row.get("communication_mode", ""))).lower()
    map_index = row.get("map_index", "")
    trial = row.get("trial", "")
    team = row.get("team_size", "")
    best, best_score = None, 0
    for path in image_index:
        text = str(path).lower()
        score = 0
        if pair_value and pair_value not in {"nan", "None"} and pair_value.lower() in text:
            score += 100
        if scenario and scenario not in {"nan", "None"} and scenario.lower() in text:
            score += 80
        if role and role not in {"nan", "none"} and role in text:
            score += 20
        for token, weight in ((f"map_{int(float(map_index)):04d}" if str(map_index) not in {"", "nan"} else "", 8),
                              (f"trial_{int(float(trial)):02d}" if str(trial) not in {"", "nan"} else "", 6),
                              (f"robots_{int(float(team)):02d}" if str(team) not in {"", "nan"} else "", 4)):
            if token and token in text:
                score += weight
        if score > best_score:
            best, best_score = path, score
    return best if best_score >= 20 else None


def _plot_trajectory_comparison(ax, rows: dict[str, pd.Series], labels: Sequence[str], title: str) -> bool:
    plotted = False
    linestyles = ("--", "-")
    for method, ls in zip(labels, linestyles):
        row = rows.get(method)
        if row is None:
            continue
        trajectories = _extract_trajectories(row)
        for idx, traj in enumerate(trajectories):
            ax.plot(traj[:, 0], traj[:, 1], ls=ls, alpha=.75, label=method if idx == 0 else None)
            ax.scatter([traj[0, 0]], [traj[0, 1]], marker="o", s=16)
            ax.scatter([traj[-1, 0]], [traj[-1, 1]], marker="x", s=20)
            plotted = True
    if plotted:
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    return plotted


def _show_image_pair(ax, rows: dict[str, pd.Series], labels: Sequence[str], root: Path, image_index: Sequence[Path], title: str) -> bool:
    found = []
    for label in labels:
        row = rows.get(label)
        if row is None:
            found.append((label, None)); continue
        found.append((label, _discover_image(row, root, image_index)))
    if not all(path for _, path in found):
        return False
    ax.axis("off")
    for idx, (label, path) in enumerate(found):
        inset = ax.inset_axes([0.01 + 0.5 * idx, 0.08, 0.48, 0.82])
        inset.imshow(_plt().imread(path))
        inset.set_title(label, fontsize=8)
        inset.axis("off")
    ax.set_title(title)
    return True


def _plot_communication_history(ax, rows: dict[str, pd.Series]) -> bool:
    plotted = False
    history_names = ("known_map_growth", "known_cells_history", "known_cell_history", "known_map_cells_history", "coverage_history")
    contact_names = ("contact_steps", "communication_contact_steps", "map_contact_steps")
    for mode in ("raw", "compressed"):
        row = rows.get(mode)
        if row is None:
            continue
        hist = _extract_series(row, history_names)
        if hist is not None:
            ax.plot(np.arange(len(hist)), hist, label=mode)
            plotted = True
        contacts = _extract_series(row, contact_names)
        if contacts is not None and mode == "raw":
            for step in contacts[:30]:
                ax.axvline(step, lw=.5, alpha=.2)
    if plotted:
        ax.set_xlabel("Decision step")
        ax.set_ylabel("Known-map growth / coverage")
        ax.set_title("(b) Raw vs compressed communication case")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    return plotted


def figure_4_7(multi: pd.DataFrame, comm: pd.DataFrame, root: Path, out: Path, notes: list[str]) -> None:
    """Optional supplementary qualitative helper; not used by the six-figure main Chapter 4 pipeline.

    The selection manifest is always written. The figure itself is written only when either
    serialised trajectory/history telemetry or discoverable saved visualisations are available;
    missing qualitative telemetry is never fabricated.
    """
    mw = complete_multi_pairs(multi)
    cw = complete_communication_pairs(comm)
    manifest = _select_qualitative_cases(multi, comm)
    _save(manifest, out / "tables" / "figure_4_7_qualitative_case_selection.csv")
    if manifest.empty:
        notes.append("Figure 4.7 skipped: no complete matched qualitative cases")
        return
    if not mw.empty:
        mw = _add_derived_metrics(mw)
    if not cw.empty:
        cw = _add_derived_metrics(cw)
    image_index = _build_image_index(root)
    plt = _plt()
    fig = plt.figure(figsize=(12, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0])
    left = gs[0].subgridspec(1, 2)
    ax_typical = fig.add_subplot(left[0])
    ax_high = fig.add_subplot(left[1])
    ax_comm = fig.add_subplot(gs[1])
    success_count = 0

    for ax, case_name, title in (
        (ax_typical, "coordination_typical", "Typical matched case"),
        (ax_high, "coordination_high_interaction", "High-interaction matched case"),
    ):
        selected = manifest.loc[manifest["case"].eq(case_name)]
        if selected.empty or mw.empty:
            ax.text(.5, .5, "Case unavailable", ha="center", va="center"); ax.axis("off"); continue
        key = str(selected.iloc[0]["pair_key"])
        rows = _pair_rows(mw, key, "method_role_normalised", ("map_only", "full"))
        if _plot_trajectory_comparison(ax, rows, ("map_only", "full"), title):
            success_count += 1
        elif _show_image_pair(ax, rows, ("map_only", "full"), root, image_index, title):
            success_count += 1
        else:
            ax.text(.5, .5, f"{title}\nselected, but trajectory/image telemetry not found", ha="center", va="center", wrap=True); ax.axis("off")

    selected = manifest.loc[manifest["case"].eq("communication_typical")]
    if not selected.empty and not cw.empty:
        key = str(selected.iloc[0]["pair_key"])
        rows = _pair_rows(cw, key, "communication_mode", ("raw", "compressed"))
        if _plot_communication_history(ax_comm, rows):
            success_count += 1
        elif _show_image_pair(ax_comm, rows, ("raw", "compressed"), root, image_index, "(b) Raw vs compressed communication case"):
            success_count += 1
        else:
            ax_comm.text(.5, .5, "Communication case selected, but known-map/contact history or images were not found", ha="center", va="center", wrap=True); ax_comm.axis("off")
    else:
        ax_comm.text(.5, .5, "Communication case unavailable", ha="center", va="center"); ax_comm.axis("off")

    fig.suptitle("Figure 4.7. Representative matched qualitative cases selected by a pre-declared rule")
    fig.tight_layout()
    if success_count:
        _save_figure(fig, out, "figure_4_7_qualitative_matched_cases")
    else:
        notes.append("Figure 4.7 case manifest generated, but no trajectory/history telemetry or discoverable saved images were available")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Optional appendix diagnostics. These functions never change the primary claim.
# ---------------------------------------------------------------------------

def _higher_is_better(metric: str) -> bool:
    name = metric.lower()
    if any(token in name for token in ("step", "overlap", "revisit", "collision", "deadlock", "wait", "path", "payload", "packet", "retrans")):
        return False
    return True


def _win_tie_loss_table(
    data: pd.DataFrame,
    *,
    treatment_column: str,
    reference: str,
    candidate: str,
    metrics: Sequence[str],
) -> pd.DataFrame:
    if data.empty or "_report_pair" not in data.columns:
        return pd.DataFrame()
    work = _add_derived_metrics(data)
    rows=[]
    for metric in metrics:
        if metric not in work.columns:
            continue
        higher=_higher_is_better(metric)
        team_rows=[]
        for n in PRIMARY_TEAM_SIZES:
            team=work.loc[pd.to_numeric(work.team_size,errors="coerce").eq(n)]
            a=team.loc[team[treatment_column].astype(str).eq(candidate),["_report_pair",metric]].rename(columns={metric:"candidate"})
            b=team.loc[team[treatment_column].astype(str).eq(reference),["_report_pair",metric]].rename(columns={metric:"reference"})
            p=a.merge(b,on="_report_pair").dropna()
            d=_metric_numeric(p.candidate,metric)-_metric_numeric(p.reference,metric)
            d=d[np.isfinite(d)]
            better=(d>0) if higher else (d<0); worse=(d<0) if higher else (d>0); tie=np.isclose(d,0)
            row={"scope":f"N={n}","metric":metric,"comparison":f"{candidate} vs {reference}","pairs":len(d),"candidate_better":int(np.sum(better)),"tie":int(np.sum(tie)),"candidate_worse":int(np.sum(worse))}
            rows.append(row); team_rows.append(row)
        if team_rows:
            rows.append({"scope":OVERALL_LABEL,"metric":metric,"comparison":f"{candidate} vs {reference}","pairs":sum(r["pairs"] for r in team_rows),"candidate_better":sum(r["candidate_better"] for r in team_rows),"tie":sum(r["tie"] for r in team_rows),"candidate_worse":sum(r["candidate_worse"] for r in team_rows)})
    return pd.DataFrame(rows)


def _map_level_effects(
    data: pd.DataFrame,
    *,
    treatment_column: str,
    reference: str,
    candidate: str,
    metrics: Sequence[str],
) -> pd.DataFrame:
    if "map_index" not in data.columns or "_report_pair" not in data.columns:
        return pd.DataFrame()
    work=_add_derived_metrics(data); rows=[]
    for n in PRIMARY_TEAM_SIZES:
        team=work.loc[pd.to_numeric(work.team_size,errors="coerce").eq(n)]
        for metric in metrics:
            if metric not in team.columns: continue
            a=team.loc[team[treatment_column].astype(str).eq(candidate),["_report_pair","map_index",metric]].rename(columns={metric:"candidate"})
            b=team.loc[team[treatment_column].astype(str).eq(reference),["_report_pair",metric]].rename(columns={metric:"reference"})
            p=a.merge(b,on="_report_pair").dropna(); p["difference"]=_metric_numeric(p.candidate,metric)-_metric_numeric(p.reference,metric)
            for map_index,part in p.groupby("map_index"):
                v=part.difference[np.isfinite(part.difference)]
                if len(v): rows.append({"team_size":n,"map_index":map_index,"metric":metric,"trials":len(v),"mean_paired_difference":float(v.mean()),"median_paired_difference":float(v.median())})
    return pd.DataFrame(rows)


def _difficulty_tables(
    data: pd.DataFrame,
    *,
    treatment_column: str,
    metrics: Sequence[str],
) -> dict[str,pd.DataFrame]:
    """Outcome-independent tertiles for existing environment descriptors only."""
    candidates={
        "obstacle":["obstacle_ratio","map_obstacle_ratio","obstacle_density"],
        "narrow":["narrow_ratio","narrow_area_ratio","narrow_passage_ratio","narrow_passage_fraction"],
        "branch":["branch_ratio","graph_branch_ratio","branch_node_ratio"],
    }
    work=_add_derived_metrics(data); outputs={}
    for label,names in candidates.items():
        col=_first(work,names)
        if not col: continue
        values=pd.to_numeric(work[col],errors="coerce")
        # Define bins from unique map-level descriptor values so treatment rows do not duplicate weight.
        if "map_index" in work.columns:
            base=work.assign(_descriptor=values).dropna(subset=["_descriptor"]).groupby("map_index")["_descriptor"].first()
        else:
            base=values.dropna()
        if base.nunique()<3: continue
        q=base.quantile([1/3,2/3]).to_numpy(float)
        if not np.isfinite(q).all() or q[0]>=q[1]: continue
        desc=pd.to_numeric(work[col],errors="coerce")
        work2=work.copy(); work2["difficulty_stratum"]=pd.cut(desc,[-np.inf,q[0],q[1],np.inf],labels=["low","medium","high"],include_lowest=True)
        rows=[]
        for (n,treatment,stratum),part in work2.groupby(["team_size",treatment_column,"difficulty_stratum"],dropna=True,observed=True):
            for metric in metrics:
                if metric not in part.columns: continue
                v=_metric_numeric(part[metric],metric).dropna().to_numpy(float)
                if len(v): rows.append({"descriptor":col,"team_size":n,treatment_column:treatment,"stratum":str(stratum),"metric":metric,"n":len(v),"mean":float(v.mean()),"median":float(np.median(v))})
        outputs[label]=pd.DataFrame(rows)
    return outputs


def _payload_correlations(data: pd.DataFrame, iou_metric: str | None) -> pd.DataFrame:
    work=_add_derived_metrics(data)
    if "map_payload_kib" not in work.columns: return pd.DataFrame()
    rows=[]
    for n in PRIMARY_TEAM_SIZES:
        for mode in ("raw","compressed"):
            part=work.loc[pd.to_numeric(work.team_size,errors="coerce").eq(n)&work.communication_mode.astype(str).eq(mode)]
            for target in ["coverage_auc"]+([iou_metric] if iou_metric else []):
                if not target or target not in part.columns: continue
                x=pd.to_numeric(part.map_payload_kib,errors="coerce"); y=pd.to_numeric(part[target],errors="coerce"); valid=x.notna()&y.notna()
                if valid.sum()<3: continue
                xv=x[valid].astype(float); yv=y[valid].astype(float)
                pearson=float(xv.corr(yv,method="pearson")) if xv.nunique()>1 and yv.nunique()>1 else math.nan
                spearman=float(xv.corr(yv,method="spearman")) if xv.nunique()>1 and yv.nunique()>1 else math.nan
                rows.append({"team_size":n,"communication_mode":mode,"target":target,"n":int(valid.sum()),"pearson_r":pearson,"spearman_rho":spearman})
    return pd.DataFrame(rows)