from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
import time
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, TextIO

import torch
torch.cuda.set_per_process_memory_fraction(0.3, device=0)


class TeeStream:
    """Write console output to both the terminal and a persistent log file."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


class ConsoleCapture(AbstractContextManager["ConsoleCapture"]):
    """Capture stdout and stderr without hiding them from the terminal."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self._file: TextIO | None = None
        self._stdout: TextIO | None = None
        self._stderr: TextIO | None = None

    def __enter__(self) -> "ConsoleCapture":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.log_path.open("a", encoding="utf-8", buffering=1)
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = TeeStream(self._stdout, self._file)
        sys.stderr = TeeStream(self._stderr, self._file)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._stdout is not None:
            sys.stdout = self._stdout
        if self._stderr is not None:
            sys.stderr = self._stderr
        if self._file is not None:
            self._file.flush()
            self._file.close()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def collect_environment_info(project_root: Path) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    gpu_names = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            gpu_names.append(torch.cuda.get_device_name(index))

    return {
        "captured_at_utc": utc_now_iso(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "working_directory": os.getcwd(),
        "project_root": str(project_root),
        "git_commit": safe_git_commit(project_root),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version() if cuda_available else None
        ),
        "gpu_count": torch.cuda.device_count() if cuda_available else 0,
        "gpu_names": gpu_names,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, default=str) + "\n")


def find_torch_modules(workspace: Any) -> dict[str, torch.nn.Module]:
    """Find likely model modules without assuming one workspace implementation."""
    modules: dict[str, torch.nn.Module] = {}
    preferred_names = (
        "model",
        "policy",
        "net",
        "ema_model",
        "ema_policy",
    )

    for name in preferred_names:
        value = getattr(workspace, name, None)
        if isinstance(value, torch.nn.Module):
            modules[name] = value

    if not modules:
        for name, value in vars(workspace).items():
            if isinstance(value, torch.nn.Module):
                modules[name] = value

    return modules


def parameter_statistics(module: torch.nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
    }


def collect_parameter_statistics(workspace: Any) -> dict[str, Any]:
    modules = find_torch_modules(workspace)
    result: dict[str, Any] = {}

    for name, module in modules.items():
        result[name] = parameter_statistics(module)

    # Prefer the main trained model for top-level comparison fields.
    primary_name = next(
        (name for name in ("model", "policy", "net") if name in result),
        next(iter(result), None),
    )
    if primary_name is not None:
        result["primary_module"] = primary_name
        result.update(result[primary_name])

    return result


class TrainingRuntimeTracker:
    """Measure total training time and CUDA peak memory."""

    def __init__(self) -> None:
        self.started_at_utc: str | None = None
        self.finished_at_utc: str | None = None
        self.start_time: float | None = None

    def start(self) -> None:
        self.started_at_utc = utc_now_iso()
        self.start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

    def finish(self) -> dict[str, Any]:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        duration = (
            time.perf_counter() - self.start_time
            if self.start_time is not None
            else None
        )
        self.finished_at_utc = utc_now_iso()

        result: dict[str, Any] = {
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "training_duration_seconds": duration,
        }

        if torch.cuda.is_available():
            result.update(
                {
                    "peak_gpu_memory_allocated_mb": (
                        torch.cuda.max_memory_allocated() / (1024 ** 2)
                    ),
                    "peak_gpu_memory_reserved_mb": (
                        torch.cuda.max_memory_reserved() / (1024 ** 2)
                    ),
                }
            )
        else:
            result.update(
                {
                    "peak_gpu_memory_allocated_mb": None,
                    "peak_gpu_memory_reserved_mb": None,
                }
            )

        return result


def write_run_manifest(
    run_dir: Path,
    *,
    command: list[str],
    environment: dict[str, Any],
) -> None:
    write_json(
        run_dir / "run_manifest.json",
        {
            "command": command,
            "environment": environment,
        },
    )