"""Shared utilities for the single-seed Chapter 4 experiment pipeline.

第4章实验的模型元数据、地图选择和重复次数工具。模型定义全部放在
``chapter4_config.py`` 中，不再使用 JSON registry。
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ModelSpec:
    """Describe one DARE/LiteDARE checkpoint in the controlled depth study."""

    key: str
    display_name: str
    checkpoint: Path
    encoder_layers: int
    training_seed: int
    family: str = "LiteDARE"

    def as_dict(self) -> dict[str, object]:
        return {
            "model_key": self.key,
            "model_name": self.display_name,
            "checkpoint": str(self.checkpoint),
            "encoder_layers": int(self.encoder_layers),
            "training_seed": int(self.training_seed),
            "model_family": self.family,
        }


def select_models(registry: Mapping[str, ModelSpec], value: str) -> tuple[ModelSpec, ...]:
    """Select all or comma-separated model keys while preserving requested order."""

    if value.strip().lower() == "all":
        return tuple(registry.values())
    keys = tuple(part.strip() for part in value.split(",") if part.strip())
    if not keys:
        raise ValueError("--models must be 'all' or comma-separated model keys")
    if len(set(keys)) != len(keys):
        raise ValueError("--models must not contain duplicate keys")
    missing = [key for key in keys if key not in registry]
    if missing:
        raise ValueError(f"unknown model keys: {missing}; available={list(registry)}")
    return tuple(registry[key] for key in keys)


def parse_map_selection(value: str, map_count: int | None = None) -> list[int]:
    """Parse map selectors such as ``0,3,7-10`` or ``all``."""

    text = value.strip().lower()
    if text == "all":
        if map_count is None or map_count <= 0:
            raise ValueError("--map-count is required and must be positive when --maps=all")
        return list(range(int(map_count)))
    selected: set[int] = set()
    for token in (part.strip() for part in value.split(",")):
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start, end = int(left), int(right)
            if start < 0 or end < start:
                raise ValueError(f"invalid map range: {token}")
            selected.update(range(start, end + 1))
        else:
            index = int(token)
            if index < 0:
                raise ValueError("map indices must be non-negative")
            selected.add(index)
    if not selected:
        raise ValueError("no maps selected")
    return sorted(selected)


def parse_map_repeats(value: str | None, default_repeats: int) -> dict[int, int]:
    """Parse per-map overrides, e.g. ``0:10,3:5,8:20``."""

    if default_repeats <= 0:
        raise ValueError("default repeats must be positive")
    if not value:
        return {}
    result: dict[int, int] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError("--map-repeats entries must use MAP:COUNT")
        map_text, count_text = item.split(":", 1)
        map_index, count = int(map_text), int(count_text)
        if map_index < 0 or count <= 0:
            raise ValueError("map repeat indices must be >=0 and counts >0")
        result[map_index] = count
    return result


def format_map_repeats(overrides: Mapping[int, int]) -> str | None:
    """Convert Python per-map overrides to the CLI form used by worker scripts."""

    if not overrides:
        return None
    return ",".join(f"{int(k)}:{int(v)}" for k, v in sorted(overrides.items()))


def repeat_count(map_index: int, default_repeats: int, overrides: Mapping[int, int]) -> int:
    """Return the effective repeat count for one map."""

    return int(overrides.get(int(map_index), int(default_repeats)))


def iter_map_trials(
    maps: Sequence[int], default_repeats: int, overrides: Mapping[int, int]
) -> Iterable[tuple[int, int]]:
    """Yield ``(map_index, trial)`` pairs using effective per-map repeats."""

    for map_index in maps:
        for trial in range(repeat_count(map_index, default_repeats, overrides)):
            yield int(map_index), int(trial)


def write_csv(path: str | Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write heterogeneous result rows using first-seen column order."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(str(key))
                seen.add(str(key))
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


