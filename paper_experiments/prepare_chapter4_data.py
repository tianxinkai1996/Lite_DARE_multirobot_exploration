#!/usr/bin/env python3
"""Prepare all Chapter 4 / Appendix result artefacts used by the dissertation.

This script does not change experiment execution or raw results. It reads the
existing Chapter 4 result tree, keeps only the declared primary comparisons,
and produces the exact main-text tables/figures plus Appendix Tables 7.1/7.2.

The main dissertation now uses Figures 4.1--4.6 only.  Every main figure is
required as a vector PDF; optional PNG copies are generated only for preview.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from paper_experiments.chapter4_config import BOOTSTRAP_SAMPLES, RANDOM_SEED
except ModuleNotFoundError:
    BOOTSTRAP_SAMPLES = 5000
    RANDOM_SEED = 42

from paper_experiments.report_data import (
    coverage_report,
    numeric_columns,
    partition_reporting_scope,
    primary_communication,
    primary_multi,
    primary_single,
    read_all_results,
)
from paper_experiments.report_figures import (
    figure_4_1,
    figure_4_3,
    figure_4_5,
    figure_4_6,
    table_4_2_selection,
)


EXPECTED_MAIN_FIGURES = (
    "figure_4_1_depth_performance_model_size.pdf",
    "figure_4_2_maponly_full_primary_scaling.pdf",
    "figure_4_3_maponly_full_deadlock_scaling.pdf",
    "figure_4_4_communication_primary_scaling.pdf",
    "figure_4_5_communication_tradeoff_and_per_robot.pdf",
)

EXPECTED_MAIN_TABLES = (
    "table_4_1_single_robot_absolute.csv",
    "table_4_2_paired_ci_and_selection.csv",
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="existing chapter4_single_seed result root")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument(
        "--include-supplementary",
        action="store_true",
        help="also export non-primary episode rows for appendix/supervisor analysis",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail if one of the three primary evidence blocks has no data",
    )
    parser.add_argument(
        "--strict-artifacts",
        action="store_true",
        help=(
            "also fail if an expected dissertation vector figure/table cannot be generated"
        ),
    )
    return parser


def main():
    args = build_parser().parse_args()
    root = args.root.expanduser().resolve()
    out = (args.output or root / "chapter4_prepared_primary").expanduser().resolve()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    data = read_all_results(root)
    numeric_columns(
        data,
        [
            "encoder_layers", "training_seed", "map_index", "trial", "team_size", "seed",
            "coverage_auc", "final_coverage", "team_coverage",
            "steps_to_90_coverage", "steps_to_95_coverage", "steps_to_99_coverage",
            "steps_to_90", "steps_to_95", "steps_to_99",
            "policy_parameters", "total_parameters",
            "team_travel_distance_recorded", "team_travel_distance",
            "actual_collision_pairs", "actual_collision_pairs_recorded", "actual_collision_steps",
            "deadlock_duration_robot_steps", "deadlock_event_duration_mean_steps", "deadlock_count", "deadlock_rate",
            "trajectory_overlap_ratio", "overlap_node_ratio", "revisit_ratio",
            "mean_robot_map_free_iou", "team_map_free_iou", "map_free_iou", "mean_pairwise_map_iou",
            "communication_payload_bytes_recorded", "communication_payload_bytes",
            "communication_bytes_recorded", "map_bytes_recorded", "map_bytes_sent",
            "communication_packets_recorded", "map_packets_recorded", "map_packets_sent", "packets_sent",
            "retransmission_packets", "map_retransmission_packets", "retransmission_packets_recorded",
        ],
    )

    primary, supplementary = partition_reporting_scope(data)
    data.to_csv(out / "tables" / "all_recorded_episode_results.csv", index=False)
    primary.to_csv(out / "tables" / "thesis_primary_episode_results.csv", index=False)
    if args.include_supplementary:
        supplementary.to_csv(out / "tables" / "supplementary_episode_results.csv", index=False)

    for name in ("attention_selection_summary.csv", "selected_model.py"):
        source = root / name
        if source.is_file():
            shutil.copy2(source, out / "tables" / name)

    single = primary_single(data)
    multi = primary_multi(data)
    comm = primary_communication(data)
    notes: list[str] = []

    # Exact dissertation outputs.
    figure_4_1(single, out, notes)
    table_4_2_selection(root / "attention_selection_summary.csv", out, notes)
    # figure_4_3 renders Figure 4.2 (primary trends) and Figure 4.3 (deadlock/waiting).
    figure_4_3(multi, out, notes, samples=args.bootstrap_samples, seed=RANDOM_SEED)
    # figure_4_5 renders compact communication Figure 4.4.
    figure_4_5(comm, out, notes, samples=args.bootstrap_samples, seed=RANDOM_SEED)
    figure_4_6(comm, out, notes, samples=args.bootstrap_samples, seed=RANDOM_SEED)

    coverage = coverage_report(data)
    missing_blocks = []
    if single.empty:
        missing_blocks.append("single_robot")
    if multi.empty:
        missing_blocks.append("map_only_vs_full")
    if comm.empty:
        missing_blocks.append("communication")
    if args.strict and missing_blocks:
        raise RuntimeError(f"missing primary evidence blocks: {', '.join(missing_blocks)}")

    generated_figures = sorted(path.name for path in (out / "figures").glob("*.pdf"))
    preview_figures = sorted(path.name for path in (out / "figures").glob("*.png"))
    generated_tables = sorted(path.name for path in (out / "tables").glob("*.csv"))
    missing_figures = [name for name in EXPECTED_MAIN_FIGURES if name not in generated_figures]
    missing_tables = [name for name in EXPECTED_MAIN_TABLES if name not in generated_tables]

    if args.strict_artifacts and (missing_figures or missing_tables):
        raise RuntimeError(
            "missing dissertation artefacts: "
            f"figures={missing_figures or 'none'}, tables={missing_tables or 'none'}"
        )

    report = {
        "source_root": str(root),
        "output": str(out),
        "reporting_policy": (
            "exact dissertation Chapter 4 outputs; paired LiteDARE selection reported by Table 4.2 only; Overall plus N=2/4/6/8; "
            "matched comparisons only; no latency/VRAM analysis; raw results retained"
        ),
        "primary_blocks": [
            "single_robot_L6_L4_L2",
            "multi_map_only_vs_full_compressed_Overall_and_N_2_4_6_8",
            "communication_full_none_raw_compressed_Overall_and_N_2_4_6_8",
        ],
        "coverage": coverage,
        "supplementary_exported": bool(args.include_supplementary),
        "supplementary_rows": int(len(supplementary)),
        "notes": notes,
        "generated_figures": generated_figures,
        "preview_png_figures": preview_figures,
        "generated_tables": generated_tables,
        "expected_main_figures": list(EXPECTED_MAIN_FIGURES),
        "expected_main_tables": list(EXPECTED_MAIN_TABLES),
        "missing_main_figures": missing_figures,
        "missing_main_tables": missing_tables,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())