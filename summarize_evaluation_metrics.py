#!/usr/bin/env python3
"""Aggregate evaluation metrics into paper-ready tables and paired statistics.

汇总原始 DARE、MergingMap、碰撞和死锁实验，生成均值/标准差/Bootstrap 置信
区间、配对 Wilcoxon 检验、Holm 校正、效应量、覆盖曲线与场景完整性检查。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

try:
    from scipy.stats import rankdata, wilcoxon
except Exception:  # pragma: no cover - target environment normally has SciPy.
    rankdata = None
    wilcoxon = None


DEFAULT_METRICS = (
    "success",
    "final_coverage",
    "coverage_auc",
    "steps_to_90_coverage",
    "steps_to_95_coverage",
    "steps_to_99_coverage",
    "team_travel_distance_recorded",
    "path_balance_cv",
    "revisit_ratio",
    "overlap_node_ratio",
    "trajectory_overlap_ratio",
    "region_assignment_overlap_rate",
    "new_free_cells_per_travel_distance",
    "preferred_vertex_conflicts",
    "preferred_swap_conflicts",
    "actual_collision_pairs",
    "dynamic_safety_blocks_recorded",
    "waiting_robot_steps",
    "deadlock_count",
    "deadlock_duration_robot_steps",
    "deadlock_recovery_rate",
    "deadlock_max_wait_steps",
    "deadlock_max_stall_steps",
    "oscillation_events",
    "region_reassignments_created",
    "mean_pairwise_map_agreement",
    "min_pairwise_map_agreement",
    "mean_robot_map_free_iou",
    "team_map_free_iou",
    "map_structure_difficulty_score",
    "communication_bytes_recorded",
    "communication_payload_bytes_recorded",
    "communication_packets_recorded",
    "mean_communication_payload_bytes_per_packet",
    "policy_inference_team_mean_ms",
    "policy_inference_per_robot_mean_ms",
    "environment_step_mean_ms",
    "map_local_sync_and_exchange_mean_ms",
    "collision_resolution_total_ms",
    "deadlock_state_update_total_ms",
    "metric_runtime_ms",
    "wall_clock_episode_ms",
)

# 中文目的：把所有可能用于论文取舍的开销字段单独输出，避免主性能表过宽。
# English purpose: keep comprehensive overhead metrics in dedicated tables while
# preserving a readable core performance summary.
COMPUTATION_METRICS = (
    "detailed_runtime_episode_wall_ms",
    "metric_runtime_ms",
    "wall_clock_episode_ms",
    "policy_inference_team_total_ms",
    "policy_inference_team_mean_ms",
    "policy_inference_team_p95_ms",
    "policy_inference_team_max_ms",
    "observation_graph_refresh_total_ms",
    "observation_graph_refresh_mean_ms",
    "candidate_generation_total_ms",
    "candidate_generation_and_reservation_filter_total_ms",
    "map_local_sync_and_exchange_total_ms",
    "post_step_local_map_sync_total_ms",
    "motion_coordination_facade_total_ms",
    "collision_resolution_total_ms",
    "motion_exchange_total_ms",
    "motion_filter_total_ms",
    "graph_backtrack_total_ms",
    "dynamic_region_update_total_ms",
    "deadlock_priority_total_ms",
    "deadlock_escape_selection_total_ms",
    "deadlock_state_update_total_ms",
    "environment_step_total_ms",
    "environment_step_mean_ms",
    "visualization_frame_total_ms",
    "visualization_gif_build_total_ms",
    "recorded_peak_process_rss_bytes",
    "recorded_peak_process_peak_rss_bytes",
    "recorded_peak_python_tracemalloc_peak_bytes",
    "recorded_peak_gpu_memory_allocated_bytes",
    "recorded_peak_gpu_memory_reserved_bytes",
    "recorded_peak_gpu_peak_memory_allocated_bytes",
    "recorded_peak_gpu_peak_memory_reserved_bytes",
    "cpu_package_energy_joules",
    "gpu_energy_joules",
    "total_measured_energy_joules",
)

COMMUNICATION_METRICS = (
    "communication_packets_recorded",
    "communication_payload_bytes_recorded",
    "communication_payload_bytes_per_step",
    "communication_payload_bytes_per_robot",
    "communication_payload_bytes_per_new_free_cell",
    "mean_communication_payload_bytes_per_packet",
    "communication_packets_per_step",
    "communication_payload_cells_recorded",
    "communication_cells_received_recorded",
    "communication_encode_ms_recorded",
    "communication_decode_apply_ms_recorded",
    "communication_exchange_wall_ms_recorded",
    "communication_delivery_events_recorded",
    "communication_conflicts_recorded",
    "map_mean_cells_per_packet",
    "map_mean_inprocess_delivery_ms",
    "map_first_contact_packets",
    "map_repeated_contact_packets",
    "map_mean_first_contact_packet_bytes",
    "map_mean_repeated_contact_packet_bytes",
    "map_sent_mask_memory_bytes",
    "region_message_packets",
    "region_message_bytes",
    "motion_message_packets",
    "motion_message_bytes",
    "mean_pairwise_map_agreement",
    "min_pairwise_map_agreement",
)

MODEL_COMPLEXITY_METRICS = (
    "policy_parameters",
    "policy_trainable_parameters",
    "policy_buffers",
    "policy_parameter_bytes",
    "policy_buffer_bytes",
    "checkpoint_size_bytes",
    "policy_flops_profiled",
    "policy_macs_estimated_from_flops",
    "policy_profile_wall_ms",
)

LOWER_IS_BETTER = {
    "steps_to_90_coverage",
    "steps_to_95_coverage",
    "steps_to_99_coverage",
    "team_travel_distance_recorded",
    "path_balance_cv",
    "revisit_ratio",
    "overlap_node_ratio",
    "trajectory_overlap_ratio",
    "region_assignment_overlap_rate",
    "preferred_vertex_conflicts",
    "preferred_swap_conflicts",
    "actual_collision_pairs",
    "dynamic_safety_blocks_recorded",
    "waiting_robot_steps",
    "deadlock_count",
    "deadlock_duration_robot_steps",
    "deadlock_max_wait_steps",
    "deadlock_max_stall_steps",
    "oscillation_events",
    "region_reassignments_created",
    "communication_bytes_recorded",
    "communication_payload_bytes_recorded",
    "communication_packets_recorded",
    "mean_communication_payload_bytes_per_packet",
    "policy_inference_team_mean_ms",
    "policy_inference_per_robot_mean_ms",
    "environment_step_mean_ms",
    "map_local_sync_and_exchange_mean_ms",
    "collision_resolution_total_ms",
    "deadlock_state_update_total_ms",
    "metric_runtime_ms",
    "wall_clock_episode_ms",
}

CORE_METHOD_ORDER = (
    "Original-DARE",
    "LiteDARE-only",
    "LiteDARE-MapOnly",
    "LiteDARE-Map-Region",
    "LiteDARE-Map-Reservation",
    "LiteDARE-Full-ContactAware",
)


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    """Write heterogeneous dictionaries with stable first-seen columns.

    中文目的：让新增指标不会破坏旧结果，并输出可直接用于论文制表的 CSV。
    English implementation: unions keys in first-seen order and writes UTF-8.
    """

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(str(key))
                seen.add(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _number(value: object) -> float | None:
    """Convert one finite scalar to float and reject booleans/missing values.

    中文目的：统一处理 JSON、CSV 中的空值、字符串和非有限数。
    English implementation: returns ``None`` for booleans, parse failures, NaN,
    and infinities.
    """

    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _metric_number(row: Mapping, metric: str) -> float | None:
    """Read one metric while treating boolean success as 0/1.

    中文目的：在保持普通布尔值不被误当作数值的同时，允许统计任务成功率。
    English implementation: maps only the named ``success`` metric to 0/1 and
    delegates every other field to the finite-number parser.
    """

    value = row.get(metric)
    if metric == "success" and isinstance(value, bool):
        return float(value)
    if metric == "success" and isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes"}:
            return 1.0
        if lowered in {"false", "no"}:
            return 0.0
    return _number(value)


def _stable_seed(*parts: object) -> int:
    """Derive a deterministic NumPy seed from text fields.

    中文目的：使 Bootstrap 置信区间在不同机器上重复运行时保持一致。
    English implementation: hashes joined fields with SHA-256 and uses 64 bits.
    """

    text = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little")


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float | str, float | str]:
    """Return a percentile Bootstrap interval for the sample mean.

    中文目的：在不强依赖正态分布的情况下报告均值 95% 置信区间。
    English implementation: resamples observations with replacement and returns
    lower/upper percentiles; empty input yields blank fields.
    """

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return "", ""
    if array.size == 1 or samples <= 0:
        value = float(array[0])
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(int(samples), array.size))
    means = np.mean(array[indices], axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    lower, upper = np.quantile(means, [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def load_rows(root: Path) -> list[dict]:
    """Load every episode summary below one experiment root.

    中文目的：自动合并 methods/ 下的原 DARE 与三种模块消融结果。
    English implementation: recursively reads ``episode_metrics.json`` and adds
    the source path for traceability.
    """

    rows: list[dict] = []
    for path in sorted(root.rglob("episode_metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["source_file"] = str(path)
        rows.append(payload)
    return rows


def _metric_values(rows: Sequence[dict], metric: str) -> tuple[list[float], int, int]:
    """Extract valid values and threshold reach counts for one metric.

    中文目的：Steps@X 中的 -1 表示未达到，不应混入条件步数均值。
    English implementation: excludes negative threshold times while returning
    total and reached counts for a separate reach-rate field.
    """

    raw = [value for row in rows if (value := _metric_number(row, metric)) is not None]
    if metric.startswith("steps_to_"):
        reached = [value for value in raw if value >= 0]
        return reached, len(raw), len(reached)
    return raw, len(raw), len(raw)


def grouped_summary(
    rows: Sequence[dict],
    metrics: Iterable[str],
    *,
    group_fields: Sequence[str],
    bootstrap_samples: int,
) -> list[dict]:
    """Summarise metrics by arbitrary method/scenario fields.

    中文目的：同时生成总体方法表和按机器人数量分层的扩展性表。
    English implementation: computes n, mean, sample standard deviation, median,
    and deterministic Bootstrap confidence intervals for every group/metric.
    """

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field, "") for field in group_fields)].append(row)

    output: list[dict] = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        result = {field: value for field, value in zip(group_fields, key)}
        result["episodes"] = len(group_rows)
        for metric in metrics:
            values, total_n, reached_n = _metric_values(group_rows, metric)
            lower, upper = bootstrap_mean_ci(
                values,
                samples=bootstrap_samples,
                seed=_stable_seed(*key, metric, "summary"),
            )
            result[f"{metric}_n"] = len(values)
            if metric.startswith("steps_to_"):
                result[f"{metric}_reached_rate"] = (
                    0.0 if total_n == 0 else reached_n / total_n
                )
            result[f"{metric}_mean"] = statistics.fmean(values) if values else ""
            result[f"{metric}_std"] = (
                statistics.stdev(values)
                if len(values) >= 2
                else (0.0 if values else "")
            )
            result[f"{metric}_median"] = statistics.median(values) if values else ""
            result[f"{metric}_ci95_low"] = lower
            result[f"{metric}_ci95_high"] = upper
        output.append(result)
    return output


def _pair_key(row: Mapping) -> tuple:
    """Return the strongest available paired-scenario identifier.

    中文目的：优先使用统一 ``scenario_id``，同时兼容旧结果文件。
    English implementation: falls back to the legacy episode/seed/team/mode key.
    """

    scenario = row.get("scenario_id")
    if scenario not in (None, ""):
        return ("scenario_id", str(scenario))
    return (
        "legacy",
        row.get("episode"),
        row.get("seed"),
        row.get("team_size"),
        row.get("communication_mode"),
    )


def _rank_biserial(differences: Sequence[float]) -> float | str:
    """Compute paired rank-biserial correlation with positive=better.

    中文目的：除 p 值外报告提升幅度方向和强度。
    English implementation: ranks absolute nonzero differences and contrasts
    positive versus negative rank sums.
    """

    array = np.asarray([value for value in differences if not math.isclose(value, 0.0)], dtype=float)
    if array.size == 0:
        return 0.0
    ranks = rankdata(np.abs(array)) if rankdata is not None else np.arange(1, array.size + 1)
    positive = float(np.sum(ranks[array > 0]))
    negative = float(np.sum(ranks[array < 0]))
    denominator = positive + negative
    return 0.0 if denominator <= 0 else (positive - negative) / denominator


def _wilcoxon_p(differences: Sequence[float]) -> float | str:
    """Return a two-sided paired Wilcoxon p-value.

    中文目的：检验方法间配对差值是否系统偏离零。
    English implementation: removes exact zeros, returns 1 for all ties, and
    leaves a blank only when SciPy is unavailable.
    """

    nonzero = [value for value in differences if not math.isclose(value, 0.0, abs_tol=1e-12)]
    if not nonzero:
        return 1.0
    if wilcoxon is None:
        return ""
    return float(wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox").pvalue)


def _holm_adjust(rows: list[dict], p_field: str = "wilcoxon_p") -> None:
    """Add Holm-adjusted p-values across all valid pairwise tests.

    中文目的：控制多指标、多方法比较造成的第一类错误累积。
    English implementation: applies the monotone step-down Holm correction in
    place and writes ``holm_p`` to every row.
    """

    indexed = [
        (index, float(row[p_field]))
        for index, row in enumerate(rows)
        if _number(row.get(p_field)) is not None
    ]
    indexed.sort(key=lambda item: item[1])
    count = len(indexed)
    adjusted_sorted: list[tuple[int, float]] = []
    running = 0.0
    for rank, (index, p_value) in enumerate(indexed):
        adjusted = min(1.0, (count - rank) * p_value)
        running = max(running, adjusted)
        adjusted_sorted.append((index, running))
    for row in rows:
        row["holm_p"] = ""
    for index, adjusted in adjusted_sorted:
        rows[index]["holm_p"] = adjusted


def default_comparisons(methods: set[str]) -> list[tuple[str, str]]:
    """Choose comparisons that directly support the paper contribution chain.

    中文目的：默认检验原 DARE→MM、MM→Collision、Collision→Deadlock 和完整方法
    相对原 DARE 的提升。
    English implementation: includes only pairs whose methods are present.
    """

    candidates = [
        ("Original-DARE", "LiteDARE-only"),
        ("LiteDARE-only", "LiteDARE-MapOnly"),
        ("Original-DARE", "LiteDARE-MapOnly"),
        ("LiteDARE-MapOnly", "LiteDARE-Map-Region"),
        ("LiteDARE-MapOnly", "LiteDARE-Map-Reservation"),
        ("LiteDARE-Map-Reservation", "LiteDARE-Full-ContactAware"),
        ("LiteDARE-Map-Region", "LiteDARE-Full-ContactAware"),
        ("Original-DARE", "LiteDARE-Full-ContactAware"),
    ]
    return [pair for pair in candidates if pair[0] in methods and pair[1] in methods]


def parse_comparisons(value: str | None, methods: set[str]) -> list[tuple[str, str]]:
    """Parse ``baseline:candidate`` pairs or choose contribution-driven defaults.

    中文目的：允许用户补充特定比较，同时保证方法名存在。
    English implementation: comma-splits pairs, validates both sides, and
    rejects self-comparisons.
    """

    if value is None:
        return default_comparisons(methods)
    pairs: list[tuple[str, str]] = []
    for item in value.split(","):
        if not item.strip():
            continue
        if ":" not in item:
            raise ValueError("--comparisons entries must be baseline:candidate")
        baseline, candidate = (part.strip() for part in item.split(":", 1))
        if baseline not in methods or candidate not in methods:
            raise ValueError(f"comparison method not found: {baseline}:{candidate}")
        if baseline == candidate:
            raise ValueError("comparison methods must differ")
        pairs.append((baseline, candidate))
    if not pairs:
        raise ValueError("--comparisons produced no pairs")
    return pairs


def paired_comparisons(
    rows: Sequence[dict],
    comparisons: Sequence[tuple[str, str]],
    metrics: Iterable[str],
    *,
    bootstrap_samples: int,
) -> list[dict]:
    """Compute paired improvements, confidence intervals, tests, and effects.

    中文目的：正值始终表示候选方法更优，便于跨“越大越好/越小越好”指标解释。
    English implementation: aligns rows by scenario, orients differences by
    metric direction, and reports win/tie/loss, relative gain, Wilcoxon, and
    rank-biserial effect size.
    """

    by_method: dict[str, dict[tuple, dict]] = defaultdict(dict)
    for row in rows:
        by_method[str(row.get("method", "unknown"))][_pair_key(row)] = row

    output: list[dict] = []
    for baseline, candidate in comparisons:
        baseline_rows = by_method[baseline]
        candidate_rows = by_method[candidate]
        shared = sorted(set(baseline_rows) & set(candidate_rows), key=str)
        for metric in metrics:
            improvements: list[float] = []
            relative: list[float] = []
            wins = ties = losses = 0
            lower_is_better = metric in LOWER_IS_BETTER
            start_mismatches = 0
            baseline_reached = candidate_reached = both_reached = 0
            baseline_only_reached = candidate_only_reached = neither_reached = 0
            for key in shared:
                base_row = baseline_rows[key]
                candidate_row = candidate_rows[key]
                if (
                    base_row.get("start_positions") not in (None, "")
                    and candidate_row.get("start_positions") not in (None, "")
                    and str(base_row.get("start_positions"))
                    != str(candidate_row.get("start_positions"))
                ):
                    start_mismatches += 1
                base_value = _metric_number(base_row, metric)
                candidate_value = _metric_number(candidate_row, metric)
                if base_value is None or candidate_value is None:
                    continue
                if metric.startswith("steps_to_"):
                    base_hit = base_value >= 0
                    candidate_hit = candidate_value >= 0
                    baseline_reached += int(base_hit)
                    candidate_reached += int(candidate_hit)
                    both_reached += int(base_hit and candidate_hit)
                    baseline_only_reached += int(base_hit and not candidate_hit)
                    candidate_only_reached += int(candidate_hit and not base_hit)
                    neither_reached += int(not base_hit and not candidate_hit)
                    if not (base_hit and candidate_hit):
                        continue
                improvement = (
                    base_value - candidate_value
                    if lower_is_better
                    else candidate_value - base_value
                )
                improvements.append(improvement)
                if abs(base_value) > 1e-12:
                    relative.append(improvement / abs(base_value))
                if math.isclose(improvement, 0.0, abs_tol=1e-12):
                    ties += 1
                elif improvement > 0:
                    wins += 1
                else:
                    losses += 1

            ci_low, ci_high = bootstrap_mean_ci(
                improvements,
                samples=bootstrap_samples,
                seed=_stable_seed(baseline, candidate, metric, "paired"),
            )
            output.append(
                {
                    "baseline": baseline,
                    "candidate": candidate,
                    "metric": metric,
                    "paired_n": len(improvements),
                    "shared_scenarios": len(shared),
                    "start_mismatches": start_mismatches,
                    "mean_improvement": statistics.fmean(improvements) if improvements else "",
                    "median_improvement": statistics.median(improvements) if improvements else "",
                    "mean_improvement_ci95_low": ci_low,
                    "mean_improvement_ci95_high": ci_high,
                    "mean_relative_improvement_percent": (
                        100.0 * statistics.fmean(relative) if relative else ""
                    ),
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "win_rate": (
                        0.0 if not improvements else wins / len(improvements)
                    ),
                    "wilcoxon_p": _wilcoxon_p(improvements),
                    "rank_biserial": _rank_biserial(improvements),
                    "baseline_reached": (
                        baseline_reached if metric.startswith("steps_to_") else ""
                    ),
                    "candidate_reached": (
                        candidate_reached if metric.startswith("steps_to_") else ""
                    ),
                    "both_reached": (
                        both_reached if metric.startswith("steps_to_") else ""
                    ),
                    "baseline_only_reached": (
                        baseline_only_reached if metric.startswith("steps_to_") else ""
                    ),
                    "candidate_only_reached": (
                        candidate_only_reached if metric.startswith("steps_to_") else ""
                    ),
                    "neither_reached": (
                        neither_reached if metric.startswith("steps_to_") else ""
                    ),
                }
            )
    _holm_adjust(output)
    return output


def scenario_integrity(rows: Sequence[dict]) -> list[dict]:
    """Check seeds, team sizes, and starts across methods for every scenario.

    中文目的：在统计前发现起点不一致、重复方法或场景字段冲突。
    English implementation: groups by pairing key and counts unique metadata
    values, emitting one explicit pass/fail row per scenario.
    """

    expected_methods = {str(row.get("method", "unknown")) for row in rows}
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[_pair_key(row)].append(row)
    output: list[dict] = []
    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        methods = [str(row.get("method", "unknown")) for row in group]
        starts = {
            str(row.get("start_positions"))
            for row in group
            if row.get("start_positions") not in (None, "")
        }
        seeds = {row.get("seed") for row in group}
        teams = {row.get("team_size") for row in group}
        episodes = {row.get("episode") for row in group}
        method_set = set(methods)
        missing_methods = sorted(expected_methods - method_set)
        duplicate_methods = sorted(
            method for method in method_set if methods.count(method) > 1
        )
        valid = (
            not missing_methods
            and not duplicate_methods
            and len(starts) <= 1
            and len(seeds) <= 1
            and len(teams) <= 1
            and len(episodes) <= 1
        )
        output.append(
            {
                "scenario_key": json.dumps(key, ensure_ascii=False),
                "scenario_id": group[0].get("scenario_id", ""),
                "method_count": len(methods),
                "methods": "|".join(sorted(methods)),
                "expected_methods": "|".join(sorted(expected_methods)),
                "missing_methods": "|".join(missing_methods),
                "duplicate_methods": "|".join(duplicate_methods),
                "unique_start_count": len(starts),
                "unique_seed_count": len(seeds),
                "unique_team_size_count": len(teams),
                "unique_episode_count": len(episodes),
                "integrity_pass": valid,
            }
        )
    return output


def coverage_curves(
    rows: Sequence[dict],
    *,
    bootstrap_samples: int,
) -> list[dict]:
    """Aggregate per-step coverage with final-value carry-forward.

    中文目的：生成 Coverage-Step 均值与 95% 区间，展示最终覆盖率相近时的过程差异。
    中文实现：从每个 episode 的 step_metrics.csv 读取曲线，加入 step 0 初始覆盖率；
    提前完成的回合保持最终覆盖率直到该方法/机器人组的最大步数。

    English implementation: reconstructs each episode curve, adds its recorded
    initial coverage, forward-fills completed episodes, and summarises by method,
    team size, and step.
    """

    episode_curves: list[tuple[str, str, int, str, dict[int, float]]] = []
    for row in rows:
        summary_path = Path(str(row["source_file"]))
        step_path = summary_path.with_name("step_metrics.csv")
        if not step_path.is_file():
            continue
        curve = {0: float(row.get("initial_coverage", 0.0))}
        with step_path.open("r", newline="", encoding="utf-8") as handle:
            for step_row in csv.DictReader(handle):
                step = int(step_row["step"])
                coverage = _number(step_row.get("coverage"))
                if coverage is not None:
                    curve[step] = coverage
        if curve:
            episode_curves.append(
                (
                    str(row.get("method", "unknown")),
                    str(row.get("communication_mode", "unknown")),
                    int(row.get("team_size", 0)),
                    str(row.get("scenario_id", _pair_key(row))),
                    curve,
                )
            )

    grouped_curves: dict[
        tuple[str, str, int], list[tuple[str, dict[int, float]]]
    ] = defaultdict(list)
    for method, communication_mode, team_size, scenario, curve in episode_curves:
        grouped_curves[(method, communication_mode, team_size)].append((scenario, curve))

    output: list[dict] = []
    for (method, communication_mode, team_size), curves in sorted(grouped_curves.items()):
        max_step = max(max(curve) for _, curve in curves)
        for step in range(max_step + 1):
            values: list[float] = []
            for _, curve in curves:
                available = [index for index in curve if index <= step]
                if available:
                    values.append(curve[max(available)])
            low, high = bootstrap_mean_ci(
                values,
                samples=bootstrap_samples,
                seed=_stable_seed(
                    method, communication_mode, team_size, step, "coverage_curve"
                ),
            )
            output.append(
                {
                    "method": method,
                    "communication_mode": communication_mode,
                    "team_size": team_size,
                    "step": step,
                    "episodes": len(values),
                    "coverage_mean": statistics.fmean(values) if values else "",
                    "coverage_std": (
                        statistics.stdev(values)
                        if len(values) >= 2
                        else (0.0 if values else "")
                    ),
                    "coverage_ci95_low": low,
                    "coverage_ci95_high": high,
                }
            )
    return output


def build_parser() -> argparse.ArgumentParser:
    """Create the paper-statistics command-line interface.

    中文目的：兼容旧的单根目录用法，并允许控制比较对与 Bootstrap 次数。
    English implementation: keeps the original positional run root and adds
    optional comparison and resampling controls.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--comparisons",
        default=None,
        help="comma-separated baseline:candidate pairs; defaults follow the contribution chain",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="legacy shortcut: compare every other method against this baseline",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate all paper-ready CSV outputs for one run root.

    中文目的：用一个命令完成总体、分层、配对、覆盖曲线和完整性分析。
    English implementation: loads summaries once, derives comparison pairs,
    writes core, computation, communication, complexity, paired, and integrity
    tables, and reports integrity failures.
    """

    args = build_parser().parse_args(argv)
    root = args.run_root.expanduser().resolve()
    output = (args.output or root / "evaluation_summary").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples cannot be negative")

    rows = load_rows(root)
    if not rows:
        raise FileNotFoundError(f"No episode_metrics.json files found below {root}")
    methods = {str(row.get("method", "unknown")) for row in rows}
    if args.baseline:
        if args.baseline not in methods:
            raise ValueError(f"baseline method not found: {args.baseline!r}")
        comparisons = [
            (args.baseline, method)
            for method in sorted(methods)
            if method != args.baseline
        ]
    else:
        comparisons = parse_comparisons(args.comparisons, methods)

    integrity = scenario_integrity(rows)
    _write_csv(output / "all_episode_metrics.csv", rows)
    _write_csv(
        output / "method_summary.csv",
        grouped_summary(
            rows,
            DEFAULT_METRICS,
            group_fields=("method",),
            bootstrap_samples=args.bootstrap_samples,
        ),
    )
    _write_csv(
        output / "method_mode_summary.csv",
        grouped_summary(
            rows,
            DEFAULT_METRICS,
            group_fields=("method", "communication_mode"),
            bootstrap_samples=args.bootstrap_samples,
        ),
    )
    _write_csv(
        output / "method_team_summary.csv",
        grouped_summary(
            rows,
            DEFAULT_METRICS,
            group_fields=("method", "team_size"),
            bootstrap_samples=args.bootstrap_samples,
        ),
    )
    _write_csv(
        output / "method_mode_team_summary.csv",
        grouped_summary(
            rows,
            DEFAULT_METRICS,
            group_fields=("method", "communication_mode", "team_size"),
            bootstrap_samples=args.bootstrap_samples,
        ),
    )
    _write_csv(
        output / "computation_overhead_summary.csv",
        grouped_summary(
            rows,
            COMPUTATION_METRICS,
            group_fields=("method", "team_size"),
            bootstrap_samples=args.bootstrap_samples,
        ),
    )
    _write_csv(
        output / "communication_overhead_summary.csv",
        grouped_summary(
            rows,
            COMMUNICATION_METRICS,
            group_fields=("method", "communication_mode", "team_size"),
            bootstrap_samples=args.bootstrap_samples,
        ),
    )
    _write_csv(
        output / "model_complexity_summary.csv",
        grouped_summary(
            rows,
            MODEL_COMPLEXITY_METRICS,
            group_fields=("method",),
            bootstrap_samples=args.bootstrap_samples,
        ),
    )
    _write_csv(
        output / "paired_comparison.csv",
        paired_comparisons(
            rows,
            comparisons,
            DEFAULT_METRICS,
            bootstrap_samples=args.bootstrap_samples,
        ),
    )
    _write_csv(output / "scenario_integrity.csv", integrity)
    _write_csv(
        output / "coverage_curves.csv",
        coverage_curves(rows, bootstrap_samples=args.bootstrap_samples),
    )

    failed_integrity = sum(not bool(row["integrity_pass"]) for row in integrity)
    print(f"episodes={len(rows)}")
    print(f"methods={sorted(methods)}")
    print(f"comparisons={comparisons}")
    print(f"scenario_integrity_failures={failed_integrity}")
    print(f"output={output}")
    return 0 if failed_integrity == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
