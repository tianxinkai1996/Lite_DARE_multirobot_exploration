from __future__ import annotations

import csv
import json

import numpy as np

from classes.multi_robot.evaluation_recorder import EpisodeMetricsRecorder


def test_episode_metrics_recorder_writes_process_metrics(tmp_path):
    recorder = EpisodeMetricsRecorder(
        output_root=tmp_path,
        episode=7,
        method="LiteDARE-MM+CD",
        seed=42,
        team_size=2,
        communication_mode="compressed",
        initial_positions=[[0.0, 0.0], [4.0, 0.0]],
        initial_coverage=0.10,
        initial_known_free_cells=10,
        node_resolution=4.0,
        safe_distance=3.0,
        deadlock_wait_threshold=2,
        coverage_thresholds=(0.50, 0.90),
    )

    # Both policies prefer the same destination, but coordination resolves it.
    recorder.record_step(
        step=1,
        preferred_positions=[[4.0, 4.0], [4.0, 4.0]],
        proposed_positions=[[4.0, 4.0], [4.0, 0.0]],
        actual_positions=[[4.0, 4.0], [4.0, 0.0]],
        coverage=0.50,
        known_free_cells=50,
        dynamic_blocked_robot_ids=[1],
        contact_pairs=[(0, 1)],
        map_packets_cumulative=2,
        map_bytes_cumulative=128,
        communication_cumulative={
            "communication_packets": 2,
            "communication_payload_bytes": 128,
            "communication_encode_ms": 0.4,
            "communication_decode_apply_ms": 0.2,
        },
        step_runtime_ms=3.5,
        extra_step_metrics={
            "policy_inference_team_ms": 1.25,
            "process_rss_bytes": 1024,
        },
    )

    # Two consecutive waits start a deadlock for both robots.
    for step in (2, 3):
        current = recorder.previous_positions.copy()
        recorder.record_step(
            step=step,
            preferred_positions=current,
            proposed_positions=current,
            actual_positions=current,
            coverage=0.60,
            known_free_cells=60,
            map_packets_cumulative=2,
            map_bytes_cumulative=128,
        )

    # Robot 0 moves and is counted as a recovery.
    current = recorder.previous_positions.copy()
    actual = current.copy()
    actual[0] = np.asarray([8.0, 4.0], dtype=np.float32)
    recorder.record_step(
        step=4,
        preferred_positions=actual,
        proposed_positions=actual,
        actual_positions=actual,
        coverage=0.92,
        known_free_cells=92,
        map_packets_cumulative=4,
        map_bytes_cumulative=256,
    )

    summary = recorder.finalise(success=True)

    assert summary["preferred_vertex_conflicts"] == 1
    assert summary["deadlock_count"] == 2
    assert summary["deadlock_recovery_count"] == 1
    assert summary["steps_to_50_coverage"] == 1
    assert summary["steps_to_90_coverage"] == 4
    assert summary["communication_bytes_recorded"] == 256
    assert summary["communication_payload_bytes_recorded"] == 256
    assert summary["mean_communication_payload_bytes_per_packet"] == 64
    assert summary["communication_payload_bytes_per_step"] == 64
    assert summary["team_travel_distance_recorded"] > 0
    assert 0.0 <= summary["coverage_auc"] <= 1.0

    with recorder.step_metrics_path.open(newline="", encoding="utf-8") as handle:
        step_rows = list(csv.DictReader(handle))
    assert len(step_rows) == 4
    assert step_rows[0]["policy_inference_team_ms"] == "1.25"
    assert step_rows[0]["communication_encode_ms_delta"] == "0.4"

    with recorder.trajectory_path.open(newline="", encoding="utf-8") as handle:
        trajectory_rows = list(csv.DictReader(handle))
    assert len(trajectory_rows) == 2 + 4 * 2

    payload = json.loads(recorder.episode_metrics_path.read_text(encoding="utf-8"))
    assert payload["method"] == "LiteDARE-MM+CD"
