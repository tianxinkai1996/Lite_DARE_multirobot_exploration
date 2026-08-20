from __future__ import annotations

import csv
import json
from pathlib import Path

from summarize_evaluation_metrics import main


METHODS = (
    "Original-DARE",
    "LiteDARE-MapOnly",
    "LiteDARE-Map-Region",
    "LiteDARE-Map-Reservation",
    "LiteDARE-Full-ContactAware",
)


def _write_episode(root: Path, method: str, scenario: str, seed: int, coverage: float) -> None:
    method_dir = root / method.replace("/", "_") / scenario
    method_dir.mkdir(parents=True)
    summary = {
        "episode": 0,
        "method": method,
        "seed": seed,
        "team_size": 2,
        "communication_mode": "none" if method == "Original-DARE" else "compressed",
        "scenario_id": scenario,
        "start_positions": "[[0.0, 0.0], [8.0, 0.0]]",
        "initial_coverage": 0.1,
        "final_coverage": coverage,
        "coverage_auc": coverage - 0.1,
        "steps_to_90_coverage": 10,
        "steps_to_95_coverage": 12,
        "steps_to_99_coverage": -1,
        "team_travel_distance_recorded": 20.0,
        "path_balance_cv": 0.1,
        "revisit_ratio": 0.2,
        "overlap_node_ratio": 0.1,
        "new_free_cells_per_travel_distance": 4.0,
        "preferred_vertex_conflicts": 1,
        "preferred_swap_conflicts": 0,
        "actual_collision_pairs": 0,
        "dynamic_safety_blocks_recorded": 0,
        "waiting_robot_steps": 1,
        "deadlock_count": 0,
        "deadlock_duration_robot_steps": 0,
        "deadlock_recovery_rate": 0.0,
        "communication_bytes_recorded": 0,
        "wall_clock_episode_ms": 5.0,
    }
    (method_dir / "episode_metrics.json").write_text(json.dumps(summary), encoding="utf-8")
    with (method_dir / "step_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("step", "coverage"))
        writer.writeheader()
        writer.writerow({"step": 1, "coverage": 0.5})
        writer.writerow({"step": 2, "coverage": coverage})


def test_paper_summary_generates_all_tables(tmp_path: Path) -> None:
    for scenario_index in range(2):
        scenario = f"scenario_{scenario_index}"
        for method_index, method in enumerate(METHODS):
            _write_episode(
                tmp_path,
                method,
                scenario,
                seed=100 + scenario_index,
                coverage=0.80 + 0.03 * method_index,
            )

    return_code = main([str(tmp_path), "--bootstrap-samples", "50"])
    assert return_code == 0
    output = tmp_path / "evaluation_summary"
    expected = {
        "all_episode_metrics.csv",
        "method_summary.csv",
        "method_mode_summary.csv",
        "method_team_summary.csv",
        "method_mode_team_summary.csv",
        "paired_comparison.csv",
        "scenario_integrity.csv",
        "coverage_curves.csv",
        "computation_overhead_summary.csv",
        "communication_overhead_summary.csv",
        "model_complexity_summary.csv",
    }
    assert expected == {path.name for path in output.iterdir()}

    paired = list(csv.DictReader((output / "paired_comparison.csv").open(encoding="utf-8")))
    assert paired
    assert all(row["start_mismatches"] == "0" for row in paired)
    assert any(
        row["baseline"] == "Original-DARE"
        and row["candidate"] == "LiteDARE-Full-ContactAware"
        for row in paired
    )
