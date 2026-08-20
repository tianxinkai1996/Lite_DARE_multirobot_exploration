from __future__ import annotations

"""Detailed runtime, memory, model-complexity, and energy instrumentation.

详细记录评估阶段的模块耗时、内存、模型复杂度和可用硬件能耗。

The module uses only optional, best-effort probes. Missing NVML, RAPL, CUDA,
or profiler FLOP support never stops an experiment; availability and failure
reasons are written to the result instead of being silently guessed.
"""

import contextlib
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import sys
import time
import tracemalloc
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterator, Mapping, MutableMapping

import torch


Numeric = int | float


@lru_cache(maxsize=16)
def _checkpoint_digest(
    path_text: str,
    file_size: int,
    modified_ns: int,
) -> str:
    """Return a cached SHA-256 digest for one checkpoint file.

    中文目的：确认 DARE、LiteDARE 与各消融实验是否真正加载同一检查点。
    English implementation: hashes the file in chunks and caches by path, size,
    and modification time so repeated episodes do not repeatedly read the model.
    """

    del file_size, modified_ns  # Included in the cache key to invalidate stale hashes.
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_model_name() -> str:
    """Return the first available CPU model description.

    中文目的：把论文运行硬件写入每个 episode，便于复现实验。
    English implementation: reads Linux ``/proc/cpuinfo`` and falls back to
    ``platform.processor`` on other systems.
    """

    try:
        for line in Path("/proc/cpuinfo").read_text(
            encoding="utf-8", errors="ignore"
        ).splitlines():
            if line.lower().startswith(("model name", "hardware")):
                return line.split(":", 1)[-1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _finite_number(value: object) -> float | None:
    """Convert one scalar to a finite float.

    中文目的：过滤空值、布尔值、NaN 与无穷值，避免污染统计结果。
    English implementation: returns ``None`` for non-finite or non-numeric input.
    """

    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _percentile(values: list[float], percentile: float) -> float:
    """Return a dependency-free linear percentile.

    中文目的：在不依赖 NumPy 的情况下计算模块耗时分位数。
    English implementation: linearly interpolates between adjacent sorted samples.
    """

    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(percentile)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _current_rss_bytes() -> int | None:
    """Read current process resident memory on Linux.

    中文目的：记录评估进程当前占用的物理内存。
    English implementation: reads ``/proc/self/statm`` and converts pages to bytes.
    """

    try:
        fields = Path("/proc/self/statm").read_text(encoding="utf-8").split()
        resident_pages = int(fields[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError):
        return None


def _peak_rss_bytes() -> int | None:
    """Read maximum resident set size for the current process.

    中文目的：记录回合运行期间进程达到的历史峰值内存。
    English implementation: converts Linux ``ru_maxrss`` KiB to bytes.
    """

    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS reports bytes. The target DARE environment is Linux.
        return value * 1024 if os.name == "posix" else value
    except (OSError, ValueError):
        return None


class _RaplEnergyProbe:
    """Best-effort Intel/AMD powercap energy counter reader.

    中文目的：若系统公开 RAPL/powercap 计数器，则记录 CPU/封装能耗差值。
    English implementation: sums package-level ``energy_uj`` counters and handles wrap.
    """

    def __init__(self) -> None:
        self._domains: list[tuple[Path, int | None]] = []
        root = Path("/sys/class/powercap")
        if not root.exists():
            self.reason = "powercap_sysfs_not_available"
            return

        for energy_path in sorted(root.glob("intel-rapl*/energy_uj")):
            max_path = energy_path.with_name("max_energy_range_uj")
            try:
                maximum = int(max_path.read_text().strip()) if max_path.exists() else None
                int(energy_path.read_text().strip())
            except (OSError, ValueError):
                continue
            self._domains.append((energy_path, maximum))

        # Some systems expose AMD domains or nested package domains only.
        if not self._domains:
            for energy_path in sorted(root.glob("**/energy_uj")):
                max_path = energy_path.with_name("max_energy_range_uj")
                try:
                    maximum = int(max_path.read_text().strip()) if max_path.exists() else None
                    int(energy_path.read_text().strip())
                except (OSError, ValueError):
                    continue
                self._domains.append((energy_path, maximum))

        self.reason = "" if self._domains else "rapl_energy_counter_not_available"

    @property
    def available(self) -> bool:
        return bool(self._domains)

    def read(self) -> list[int] | None:
        if not self.available:
            return None
        values: list[int] = []
        try:
            for path, _ in self._domains:
                values.append(int(path.read_text().strip()))
        except (OSError, ValueError):
            return None
        return values

    def delta_joules(self, start: list[int] | None, end: list[int] | None) -> float | None:
        if start is None or end is None or len(start) != len(end):
            return None
        total_uj = 0
        for start_value, end_value, (_, maximum) in zip(start, end, self._domains):
            delta = end_value - start_value
            if delta < 0 and maximum is not None:
                delta += maximum
            if delta < 0:
                return None
            total_uj += delta
        return float(total_uj) / 1_000_000.0


class _NvmlProbe:
    """Best-effort NVIDIA telemetry reader through optional ``pynvml``.

    中文目的：尽可能记录 GPU 能耗、功率、利用率、显存利用率与温度。
    中文实现：NVML 初始化成功后分别探测各接口；能耗接口不可用时仍保留
    功率和利用率采样，避免把“无总能耗计数”误判为“无任何 GPU 遥测”。

    English implementation: keeps the NVML handle when general telemetry works,
    while tracking total-energy support independently because older GPUs often
    expose power/utilisation but not cumulative energy.
    """

    def __init__(self, device: torch.device) -> None:
        self._pynvml = None
        self._handle = None
        self.reason = "cuda_not_available"
        self.energy_reason = "cuda_not_available"
        self.energy_supported = False
        if device.type != "cuda" or not torch.cuda.is_available():
            return
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            index = torch.cuda.current_device() if device.index is None else int(device.index)
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            self._pynvml = pynvml
            self.reason = ""
            try:
                pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
                self.energy_supported = True
                self.energy_reason = ""
            except Exception as exc:
                self.energy_reason = f"nvml_total_energy_unavailable:{type(exc).__name__}"
        except Exception as exc:  # NVML exposes implementation-specific exceptions.
            self._pynvml = None
            self._handle = None
            self.reason = f"nvml_unavailable:{type(exc).__name__}"
            self.energy_reason = self.reason

    @property
    def available(self) -> bool:
        return self._pynvml is not None and self._handle is not None

    def read_millijoules(self) -> int | None:
        if not self.available or not self.energy_supported:
            return None
        try:
            return int(self._pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle))
        except Exception:
            return None

    def read_power_watts(self) -> float | None:
        if not self.available:
            return None
        try:
            milliwatts = float(self._pynvml.nvmlDeviceGetPowerUsage(self._handle))
            return milliwatts / 1000.0
        except Exception:
            return None

    def read_utilisation(self) -> tuple[float | None, float | None]:
        """Return GPU-core and memory-controller utilisation percentages.

        中文目的：区分“耗时增加”是算力饱和还是等待/通信造成。
        English implementation: returns ``(gpu_percent, memory_percent)`` when
        the device and driver expose NVML utilisation counters.
        """

        if not self.available:
            return None, None
        try:
            value = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            return float(value.gpu), float(value.memory)
        except Exception:
            return None, None

    def read_temperature_c(self) -> float | None:
        """Return current GPU temperature in Celsius when available."""

        if not self.available:
            return None
        try:
            sensor = self._pynvml.NVML_TEMPERATURE_GPU
            return float(self._pynvml.nvmlDeviceGetTemperature(self._handle, sensor))
        except Exception:
            return None


class DetailedRuntimeMetrics:
    """Collect detailed evaluation metrics without changing policy inputs or actions.

    中文目的：统一记录各模块耗时、CPU/GPU 内存、模型规模、FLOPs/MACs 和能耗。
    中文实现：通过上下文计时器记录非侵入式耗时；每步采样资源；对一次额外推理
    使用 Torch profiler 做尽力而为的 FLOPs 估算，并恢复随机数状态以保持实验可复现。

    English implementation: provides named timers, per-step resource snapshots,
    model metadata, optional one-shot profiler FLOPs, and optional RAPL/NVML energy.
    """

    def __init__(
        self,
        *,
        device: torch.device | str,
        policy: torch.nn.Module | object,
        team_size: int = 1,
        checkpoint_path: str | Path | None = None,
        synchronize_cuda: bool = True,
        track_python_memory: bool = True,
        sample_energy: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.policy = policy
        self.team_size = max(1, int(team_size))
        self.synchronize_cuda = bool(synchronize_cuda)
        self.track_python_memory = bool(track_python_memory)
        self.sample_energy = bool(sample_energy)

        self._samples: MutableMapping[str, list[float]] = defaultdict(list)
        self._step_values: dict[str, float] = {}
        self._step_top_level_time_keys: set[str] = set()
        self._step_started_at: float | None = None
        self._step_process_cpu_started: float | None = None
        self._episode_started_at: float | None = None
        self._episode_finished = False
        self._final_summary: dict[str, object] | None = None
        self._python_tracing_started_here = False
        self._step_count = 0

        self._rapl = _RaplEnergyProbe()
        self._nvml = _NvmlProbe(self.device)
        self._rapl_start: list[int] | None = None
        self._nvml_start_mj: int | None = None
        self._process_cpu_start: float | None = None
        self._rusage_start: resource.struct_rusage | None = None
        self._episode_start_resources: dict[str, object] = {}

        self._resource_peaks: dict[str, float] = defaultdict(float)
        self._static_metrics = {
            **self._runtime_environment_metrics(),
            **self._model_metrics(checkpoint_path),
        }
        self._profile_metrics: dict[str, object] = {
            "policy_flops_profiled": None,
            "policy_flops_profiled_per_robot": None,
            "policy_flops_estimated_per_team_step": None,
            "policy_macs_estimated_from_flops": None,
            "policy_macs_estimated_per_robot": None,
            "policy_macs_estimated_per_team_step": None,
            "policy_profile_wall_ms": None,
            "policy_profile_status": "not_requested",
        }

    def _sync_cuda(self) -> None:
        """Synchronise the selected CUDA device when precise GPU timing is requested.

        中文目的：避免异步 CUDA 内核导致推理耗时被低估。
        English implementation: calls ``torch.cuda.synchronize`` only for valid CUDA runs.
        """

        if self.synchronize_cuda and self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def _runtime_environment_metrics(self) -> dict[str, object]:
        """Return software, CPU, CUDA, and thread metadata for reproducibility.

        中文目的：把实验运行环境直接写入结果，避免论文整理时遗漏硬件版本。
        English implementation: collects only local runtime metadata and never
        changes Torch thread counts, device placement, or model behaviour.
        """

        metrics: dict[str, object] = {
            "runtime_platform": platform.platform(),
            "runtime_python_version": sys.version.split()[0],
            "runtime_torch_version": str(torch.__version__),
            "runtime_cuda_version": str(torch.version.cuda or ""),
            "runtime_cudnn_version": (
                None
                if not torch.backends.cudnn.is_available()
                else int(torch.backends.cudnn.version() or 0)
            ),
            "runtime_cpu_model": _cpu_model_name(),
            "runtime_logical_cpu_count": int(os.cpu_count() or 0),
            "runtime_torch_num_threads": int(torch.get_num_threads()),
            "runtime_torch_num_interop_threads": int(torch.get_num_interop_threads()),
            "runtime_team_size": int(self.team_size),
            "runtime_device": str(self.device),
            "runtime_gpu_name": "",
            "runtime_gpu_compute_capability": "",
        }
        if self.device.type == "cuda" and torch.cuda.is_available():
            index = torch.cuda.current_device() if self.device.index is None else int(self.device.index)
            try:
                metrics["runtime_gpu_name"] = torch.cuda.get_device_name(index)
                major, minor = torch.cuda.get_device_capability(index)
                metrics["runtime_gpu_compute_capability"] = f"{major}.{minor}"
            except (AssertionError, RuntimeError):
                pass
        return metrics

    def _model_metrics(self, checkpoint_path: str | Path | None) -> dict[str, object]:
        """Compute static parameter, buffer, and checkpoint-size metrics.

        中文目的：为 DARE 与 LiteDARE 的轻量化比较记录模型规模。
        English implementation: counts tensor elements and storage bytes without a forward pass.
        """

        parameters = list(self.policy.parameters()) if hasattr(self.policy, "parameters") else []
        buffers = list(self.policy.buffers()) if hasattr(self.policy, "buffers") else []
        checkpoint = checkpoint_path or os.environ.get("DARE_CHECKPOINT_PATH")
        checkpoint_resolved = ""
        checkpoint_size = None
        checkpoint_sha256 = ""
        if checkpoint:
            path = Path(checkpoint).expanduser().resolve()
            checkpoint_resolved = str(path)
            if path.is_file():
                stat = path.stat()
                checkpoint_size = int(stat.st_size)
                try:
                    checkpoint_sha256 = _checkpoint_digest(
                        str(path), int(stat.st_size), int(stat.st_mtime_ns)
                    )
                except OSError:
                    checkpoint_sha256 = ""

        module_type_counts: dict[str, int] = defaultdict(int)
        if hasattr(self.policy, "modules"):
            for module in self.policy.modules():
                module_type_counts[type(module).__name__] += 1
        attention_layers = sum(
            count
            for name, count in module_type_counts.items()
            if "attention" in name.lower() or "transformerencoderlayer" in name.lower()
        )
        linear_layers = sum(
            count for name, count in module_type_counts.items() if name.lower() == "linear"
        )

        return {
            "policy_class": type(self.policy).__name__,
            "policy_module_count": int(sum(module_type_counts.values())),
            "policy_attention_like_layer_count": int(attention_layers),
            "policy_linear_layer_count": int(linear_layers),
            "policy_module_type_counts_json": json.dumps(
                dict(sorted(module_type_counts.items())), sort_keys=True
            ),
            "policy_parameters": int(sum(parameter.numel() for parameter in parameters)),
            "policy_trainable_parameters": int(
                sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
            ),
            "policy_buffers": int(sum(buffer.numel() for buffer in buffers)),
            "policy_parameter_bytes": int(
                sum(parameter.numel() * parameter.element_size() for parameter in parameters)
            ),
            "policy_buffer_bytes": int(
                sum(buffer.numel() * buffer.element_size() for buffer in buffers)
            ),
            "checkpoint_path_recorded": checkpoint_resolved,
            "checkpoint_size_bytes": checkpoint_size,
            "checkpoint_sha256": checkpoint_sha256,
        }

    def profile_policy_once(self, callable_: Callable[[], object], *, enabled: bool = True) -> dict[str, object]:
        """Profile one extra inference and estimate FLOPs/MACs when Torch supports it.

        中文目的：记录一次代表性策略推理的运算量，便于论文比较网络复杂度。
        中文实现：保存并恢复 CPU/CUDA RNG 状态，使用 ``torch.profiler`` 的
        ``with_flops`` 统计；失败时记录原因而不是伪造数值。

        English implementation: preserves RNG states, profiles one no-grad call,
        sums available operator FLOPs, estimates MACs as FLOPs/2, and records status.
        """

        if not enabled:
            self._profile_metrics["policy_profile_status"] = "disabled"
            return dict(self._profile_metrics)

        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        started = time.perf_counter()
        try:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if self.device.type == "cuda" and torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self._sync_cuda()
            with torch.no_grad(), torch.profiler.profile(
                activities=activities,
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
                with_flops=True,
            ) as profiler:
                callable_()
                self._sync_cuda()
            events = profiler.key_averages()
            flops = float(sum(float(getattr(event, "flops", 0) or 0) for event in events))
            status = "ok" if flops > 0 else "no_operator_flop_estimates"
            self._profile_metrics.update(
                {
                    "policy_flops_profiled": flops if flops > 0 else None,
                    "policy_flops_profiled_per_robot": flops if flops > 0 else None,
                    "policy_flops_estimated_per_team_step": (
                        flops * self.team_size if flops > 0 else None
                    ),
                    "policy_macs_estimated_from_flops": (
                        flops / 2.0 if flops > 0 else None
                    ),
                    "policy_macs_estimated_per_robot": (
                        flops / 2.0 if flops > 0 else None
                    ),
                    "policy_macs_estimated_per_team_step": (
                        flops * self.team_size / 2.0 if flops > 0 else None
                    ),
                    "policy_profile_status": status,
                }
            )
        except Exception as exc:
            self._profile_metrics.update(
                {
                    "policy_flops_profiled": None,
                    "policy_flops_profiled_per_robot": None,
                    "policy_flops_estimated_per_team_step": None,
                    "policy_macs_estimated_from_flops": None,
                    "policy_macs_estimated_per_robot": None,
                    "policy_macs_estimated_per_team_step": None,
                    "policy_profile_status": f"failed:{type(exc).__name__}:{exc}",
                }
            )
        finally:
            self._profile_metrics["policy_profile_wall_ms"] = (
                time.perf_counter() - started
            ) * 1000.0
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        return dict(self._profile_metrics)

    def start_episode(self) -> None:
        """Reset resource peaks and start the measured episode window.

        中文目的：排除一次性 FLOPs profiling 与初始化开销，单独测量正式回合。
        English implementation: resets CUDA peaks, starts optional tracemalloc and energy baselines.
        """

        self._episode_started_at = time.perf_counter()
        self._episode_finished = False
        self._final_summary = None
        self._process_cpu_start = time.process_time()
        self._rusage_start = resource.getrusage(resource.RUSAGE_SELF)
        self._step_count = 0
        self._samples.clear()
        self._resource_peaks.clear()
        if self.track_python_memory and not tracemalloc.is_tracing():
            tracemalloc.start()
            self._python_tracing_started_here = True
        if self.device.type == "cuda" and torch.cuda.is_available():
            self._sync_cuda()
            torch.cuda.reset_peak_memory_stats(self.device)
        if self.sample_energy:
            self._rapl_start = self._rapl.read()
            self._nvml_start_mj = self._nvml.read_millijoules()
        self._episode_start_resources = self.resource_snapshot()
        self._update_resource_peaks(self._episode_start_resources)

    def start_step(self) -> None:
        """Open one step-local timing bucket.

        中文目的：确保每个控制步输出独立的模块耗时列。
        English implementation: clears the step dictionary and stores a wall-clock start.
        """

        self._step_values = {}
        self._step_top_level_time_keys = set()
        self._step_started_at = time.perf_counter()
        self._step_process_cpu_started = time.process_time()

    @contextlib.contextmanager
    def measure(self, name: str, *, cuda_sync: bool = False) -> Iterator[None]:
        """Measure a named code block and add it to step and episode totals.

        中文目的：以低侵入方式记录策略、地图、碰撞、死锁和环境模块耗时。
        English implementation: optionally synchronises CUDA, times the block, and accumulates samples.
        """

        if cuda_sync:
            self._sync_cuda()
        started = time.perf_counter()
        try:
            yield
        finally:
            if cuda_sync:
                self._sync_cuda()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            key = str(name).strip().replace(" ", "_")
            time_key = f"{key}_ms"
            self._step_values[time_key] = self._step_values.get(time_key, 0.0) + elapsed_ms
            self._step_top_level_time_keys.add(time_key)
            self._samples[key].append(float(elapsed_ms))

    def add_step_value(
        self,
        key: str,
        value: object,
        *,
        aggregate_name: str | None = None,
    ) -> None:
        """Attach one already measured scalar to the current step.

        中文目的：接收 collision/deadlock 内部计时或派生指标。
        English implementation: stores finite numeric values without replacing existing timers.
        """

        number = _finite_number(value)
        if number is not None:
            self._step_values[str(key)] = number
            if aggregate_name:
                self._samples[str(aggregate_name)].append(number)

    def resource_snapshot(self) -> dict[str, object]:
        """Return current CPU, Python, CUDA, and GPU-power readings.

        中文目的：逐步记录内存变化和可用的 GPU 功率。
        English implementation: samples standard-library, tracemalloc, Torch, and optional NVML probes.
        """

        gpu_utilisation, gpu_memory_utilisation = self._nvml.read_utilisation()
        snapshot: dict[str, object] = {
            "process_rss_bytes": _current_rss_bytes(),
            "process_peak_rss_bytes": _peak_rss_bytes(),
            "python_tracemalloc_current_bytes": None,
            "python_tracemalloc_peak_bytes": None,
            "gpu_memory_allocated_bytes": None,
            "gpu_memory_reserved_bytes": None,
            "gpu_peak_memory_allocated_bytes": None,
            "gpu_peak_memory_reserved_bytes": None,
            "gpu_power_watts": self._nvml.read_power_watts() if self.sample_energy else None,
            "gpu_utilisation_percent": gpu_utilisation,
            "gpu_memory_utilisation_percent": gpu_memory_utilisation,
            "gpu_temperature_c": self._nvml.read_temperature_c(),
        }
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            snapshot["python_tracemalloc_current_bytes"] = int(current)
            snapshot["python_tracemalloc_peak_bytes"] = int(peak)
        if self.device.type == "cuda" and torch.cuda.is_available():
            snapshot.update(
                {
                    "gpu_memory_allocated_bytes": int(torch.cuda.memory_allocated(self.device)),
                    "gpu_memory_reserved_bytes": int(torch.cuda.memory_reserved(self.device)),
                    "gpu_peak_memory_allocated_bytes": int(
                        torch.cuda.max_memory_allocated(self.device)
                    ),
                    "gpu_peak_memory_reserved_bytes": int(
                        torch.cuda.max_memory_reserved(self.device)
                    ),
                }
            )
        return snapshot

    def _update_resource_peaks(self, snapshot: Mapping[str, object]) -> None:
        for key, value in snapshot.items():
            number = _finite_number(value)
            if number is not None:
                self._resource_peaks[key] = max(self._resource_peaks[key], number)

    def finish_step(self) -> dict[str, object]:
        """Close the current step and return flattened timing/resource fields.

        中文目的：生成可直接写入 ``step_metrics.csv`` 的完整开销字典。
        English implementation: adds wall time, resource samples, and derived timing coverage.
        """

        if self._step_started_at is None:
            raise RuntimeError("start_step() must be called before finish_step()")
        wall_ms = (time.perf_counter() - self._step_started_at) * 1000.0
        process_cpu_ms = (
            0.0
            if self._step_process_cpu_started is None
            else (time.process_time() - self._step_process_cpu_started) * 1000.0
        )
        self._step_values["profiled_step_wall_ms"] = float(wall_ms)
        self._step_values["step_process_cpu_ms"] = float(process_cpu_ms)
        self._step_values["step_process_cpu_utilisation_percent"] = (
            0.0 if wall_ms <= 1e-12 else float(process_cpu_ms / wall_ms * 100.0)
        )
        timed_components = sum(
            float(self._step_values.get(key, 0.0))
            for key in self._step_top_level_time_keys
        )
        self._step_values["timed_components_sum_ms"] = float(timed_components)
        self._step_values["unattributed_step_ms"] = float(max(0.0, wall_ms - timed_components))

        output_overhead_keys = {
            "visualization_frame_ms",
            "debug_diagnostics_ms",
            "region_event_io_ms",
            "metric_state_snapshot_ms",
        }
        output_overhead_ms = sum(
            float(self._step_values.get(key, 0.0)) for key in output_overhead_keys
        )
        self._step_values["recording_and_output_overhead_ms"] = float(output_overhead_ms)
        self._step_values["control_step_wall_excluding_output_ms"] = float(
            max(0.0, wall_ms - output_overhead_ms)
        )
        for sample_name, value in (
            ("profiled_step_wall", wall_ms),
            ("step_process_cpu", process_cpu_ms),
            (
                "step_process_cpu_utilisation_percent",
                self._step_values["step_process_cpu_utilisation_percent"],
            ),
            ("timed_components_sum", timed_components),
            ("unattributed_step", self._step_values["unattributed_step_ms"]),
            ("recording_and_output_overhead", output_overhead_ms),
            (
                "control_step_wall_excluding_output",
                self._step_values["control_step_wall_excluding_output_ms"],
            ),
        ):
            self._samples[sample_name].append(float(value))
        resources = self.resource_snapshot()
        self._update_resource_peaks(resources)
        result: dict[str, object] = {**self._step_values, **resources}
        self._step_started_at = None
        self._step_process_cpu_started = None
        self._step_count += 1
        return result

    def finalise(self) -> dict[str, object]:
        """Return episode-level timing, memory, FLOPs, and energy metrics.

        中文目的：汇总每个模块的总耗时、均值、P95、最大值及硬件资源峰值。
        English implementation: aggregates named timer samples and closes optional probes.
        """

        if self._episode_finished and self._final_summary is not None:
            return dict(self._final_summary)
        self._episode_finished = True
        self._sync_cuda()
        elapsed_ms = (
            0.0
            if self._episode_started_at is None
            else (time.perf_counter() - self._episode_started_at) * 1000.0
        )
        final_resources = self.resource_snapshot()
        self._update_resource_peaks(final_resources)
        process_cpu_ms = (
            0.0
            if self._process_cpu_start is None
            else (time.process_time() - self._process_cpu_start) * 1000.0
        )
        rusage_end = resource.getrusage(resource.RUSAGE_SELF)
        user_cpu_ms = None
        system_cpu_ms = None
        if self._rusage_start is not None:
            user_cpu_ms = float(
                (rusage_end.ru_utime - self._rusage_start.ru_utime) * 1000.0
            )
            system_cpu_ms = float(
                (rusage_end.ru_stime - self._rusage_start.ru_stime) * 1000.0
            )

        summary: dict[str, object] = {
            **self._static_metrics,
            **self._profile_metrics,
            "detailed_runtime_episode_wall_ms": float(elapsed_ms),
            "episode_process_cpu_ms": float(process_cpu_ms),
            "episode_user_cpu_ms": user_cpu_ms,
            "episode_system_cpu_ms": system_cpu_ms,
            "episode_process_cpu_utilisation_percent": (
                0.0 if elapsed_ms <= 1e-12 else float(process_cpu_ms / elapsed_ms * 100.0)
            ),
            "runtime_steps_profiled": int(self._step_count),
            "cuda_synchronised_timing": bool(self.synchronize_cuda),
            "python_memory_tracking_enabled": bool(self.track_python_memory),
            "cpu_energy_probe_available": bool(self._rapl.available),
            "cpu_energy_probe_reason": self._rapl.reason,
            "gpu_telemetry_probe_available": bool(self._nvml.available),
            "gpu_telemetry_probe_reason": self._nvml.reason,
            "gpu_energy_probe_available": bool(
                self._nvml.available and self._nvml.energy_supported
            ),
            "gpu_energy_probe_reason": self._nvml.energy_reason,
        }

        for name, samples in sorted(self._samples.items()):
            if not samples:
                continue
            summary[f"{name}_calls"] = int(len(samples))
            summary[f"{name}_total_ms"] = float(sum(samples))
            summary[f"{name}_mean_ms"] = float(statistics.fmean(samples))
            summary[f"{name}_p95_ms"] = float(_percentile(samples, 0.95))
            summary[f"{name}_max_ms"] = float(max(samples))

        inference_samples = self._samples.get("policy_inference_team", [])
        if inference_samples:
            per_robot_samples = [value / self.team_size for value in inference_samples]
            summary.update(
                {
                    "policy_inference_per_robot_total_ms": float(sum(per_robot_samples)),
                    "policy_inference_per_robot_mean_ms": float(
                        statistics.fmean(per_robot_samples)
                    ),
                    "policy_inference_per_robot_p95_ms": float(
                        _percentile(per_robot_samples, 0.95)
                    ),
                    "policy_inference_per_robot_max_ms": float(max(per_robot_samples)),
                }
            )

        for key, value in sorted(self._resource_peaks.items()):
            summary[f"recorded_peak_{key}"] = value
        for key, value in final_resources.items():
            summary[f"final_{key}"] = value
        for key, value in self._episode_start_resources.items():
            summary[f"episode_start_{key}"] = value
        start_rss = _finite_number(
            self._episode_start_resources.get("process_rss_bytes")
        )
        peak_rss = _finite_number(self._resource_peaks.get("process_rss_bytes"))
        summary["episode_process_rss_growth_bytes"] = (
            None
            if start_rss is None or peak_rss is None
            else float(max(0.0, peak_rss - start_rss))
        )

        if self.sample_energy:
            rapl_end = self._rapl.read()
            nvml_end = self._nvml.read_millijoules()
            cpu_joules = self._rapl.delta_joules(self._rapl_start, rapl_end)
            gpu_joules = None
            if self._nvml_start_mj is not None and nvml_end is not None:
                delta_mj = nvml_end - self._nvml_start_mj
                if delta_mj >= 0:
                    gpu_joules = float(delta_mj) / 1000.0
            summary["cpu_package_energy_joules"] = cpu_joules
            summary["gpu_energy_joules"] = gpu_joules
            summary["total_measured_energy_joules"] = (
                None
                if cpu_joules is None and gpu_joules is None
                else float((cpu_joules or 0.0) + (gpu_joules or 0.0))
            )
            total_energy = _finite_number(summary["total_measured_energy_joules"])
            summary["measured_energy_per_step_joules"] = (
                None
                if total_energy is None or self._step_count <= 0
                else float(total_energy / self._step_count)
            )

        if self._python_tracing_started_here and tracemalloc.is_tracing():
            tracemalloc.stop()
        self._final_summary = dict(summary)
        return dict(summary)
