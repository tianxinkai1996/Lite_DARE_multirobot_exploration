"""Synthetic smoke tests for reduced reporting with N=2/4/6/8 scaling."""
from __future__ import annotations

import tempfile
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from paper_experiments.report_data import (
    complete_communication_pairs,
    complete_multi_pairs,
    partition_reporting_scope,
    primary_communication,
    primary_multi,
    primary_single,
    read_all_results,
)
from paper_experiments.report_scope import PRIMARY_TEAM_SIZES


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _seed(map_index: int, trial: int, team_size: int) -> int:
    return 42 + 100_000 * map_index + 1_000 * trial + 10 * team_size


def test_scope_keeps_extra_runs_out_of_primary_and_preserves_team_sizes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(
            root / "e1_policy_depth" / "LiteDARE-L2" / "results.csv",
            [{"team_size": 1, "encoder_layers": 2, "model_name": "LiteDARE-L2", "map_index": 0, "trial": 0}],
        )

        multi_rows: list[dict] = []
        for team_size in PRIMARY_TEAM_SIZES:
            for role in ("map_only", "map_region", "map_reservation", "full"):
                multi_rows.append(
                    {
                        "team_size": team_size,
                        "method_role": role,
                        "communication_mode": "compressed",
                        "map_index": 0,
                        "trial": 0,
                        "seed": _seed(0, 0, team_size),
                        "start_positions": f"starts_N{team_size}",
                        "coverage_auc": 0.8,
                    }
                )
        _write(root / "multi_ablation" / "results.csv", multi_rows)

        comm_rows: list[dict] = []
        for team_size in PRIMARY_TEAM_SIZES:
            for mode in ("none", "raw", "compressed"):
                comm_rows.append(
                    {
                        "team_size": team_size,
                        "method_role": "full",
                        "communication_mode": mode,
                        "map_index": 0,
                        "trial": 0,
                        "seed": _seed(0, 0, team_size),
                        "start_positions": f"starts_N{team_size}",
                        "coverage_auc": 0.8,
                    }
                )
        _write(root / "communication" / "results.csv", comm_rows)

        data = read_all_results(root)
        primary, supplementary = partition_reporting_scope(data)
        assert len(primary_single(data)) == 1
        assert set(primary_multi(data)["team_size"].astype(int)) == set(PRIMARY_TEAM_SIZES)
        assert set(primary_multi(data)["method_role_normalised"]) == {"map_only", "full"}
        assert set(primary_communication(data)["team_size"].astype(int)) == set(PRIMARY_TEAM_SIZES)
        assert set(primary_communication(data)["communication_mode"]) == {"none", "raw", "compressed"}
        assert len(complete_multi_pairs(primary_multi(data))) == 2 * len(PRIMARY_TEAM_SIZES)
        assert len(complete_communication_pairs(primary_communication(data))) == 3 * len(PRIMARY_TEAM_SIZES)
        assert set(supplementary["method_role_normalised"]) == {"map_region", "map_reservation"}
        assert len(primary) == 1 + 2 * len(PRIMARY_TEAM_SIZES) + 3 * len(PRIMARY_TEAM_SIZES)


def test_incomplete_or_start_mismatched_treatments_are_excluded() -> None:
    rows = pd.DataFrame(
        [
            {"team_size": 4, "method_role_normalised": "map_only", "map_index": 0, "trial": 0, "seed": 82, "start_positions": "A"},
            {"team_size": 4, "method_role_normalised": "full", "map_index": 0, "trial": 0, "seed": 82, "start_positions": "B"},
            {"team_size": 6, "method_role_normalised": "map_only", "map_index": 0, "trial": 0, "seed": 102, "start_positions": "C"},
            # Full missing for N=6.
        ]
    )
    assert complete_multi_pairs(rows).empty



def test_overall_macro_scope_is_generated() -> None:
    from paper_experiments.report_figures import _summary_overall_and_team
    rows=[]
    for n in PRIMARY_TEAM_SIZES:
        for role,bonus in (("map_only",0.0),("full",0.1)):
            for trial in range(2):
                rows.append({"team_size":n,"method_role_normalised":role,"coverage_auc":0.5+0.01*n+bonus})
    frame=pd.DataFrame(rows)
    summary=_summary_overall_and_team(
        frame,
        treatment_column="method_role_normalised",
        treatments=("map_only","full"),
        metrics=("coverage_auc",),
        samples=100,
        seed=42,
    )
    assert set(summary["scope"]) == {"Overall","N=2","N=4","N=6","N=8"}
    assert len(summary.loc[summary["scope"].eq("Overall")]) == 2

if __name__ == "__main__":
    test_scope_keeps_extra_runs_out_of_primary_and_preserves_team_sizes()
    test_incomplete_or_start_mismatched_treatments_are_excluded()
    test_overall_macro_scope_is_generated()
    print("reporting scope + Overall + team-size smoke tests: PASS")
