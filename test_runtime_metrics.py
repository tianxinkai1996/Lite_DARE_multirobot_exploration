from __future__ import annotations

import time

import torch

from classes.multi_robot.runtime_metrics import DetailedRuntimeMetrics


def test_detailed_runtime_metrics_records_time_memory_and_model_complexity() -> None:
    policy = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 2),
    )
    collector = DetailedRuntimeMetrics(
        device="cpu",
        policy=policy,
        team_size=4,
        synchronize_cuda=True,
        track_python_memory=True,
        sample_energy=False,
    )
    profile = collector.profile_policy_once(
        lambda: policy(torch.ones(1, 4)),
        enabled=True,
    )
    assert profile["policy_profile_status"] in {
        "ok",
        "no_operator_flop_estimates",
    }

    collector.start_episode()
    collector.start_step()
    with collector.measure("policy_inference_team"):
        policy(torch.ones(1, 4))
        time.sleep(0.001)
    collector.add_step_value("collision_resolution_ms", 0.25)
    step = collector.finish_step()

    assert step["policy_inference_team_ms"] > 0
    assert step["collision_resolution_ms"] == 0.25
    assert step["profiled_step_wall_ms"] > 0
    assert "process_rss_bytes" in step

    summary = collector.finalise()
    assert summary["policy_parameters"] == 58
    assert summary["runtime_team_size"] == 4
    assert summary["checkpoint_sha256"] == ""
    assert summary["policy_inference_team_calls"] == 1
    assert summary["policy_inference_team_total_ms"] > 0
    assert summary["python_memory_tracking_enabled"] is True
    assert summary["cpu_energy_probe_available"] in {True, False}
