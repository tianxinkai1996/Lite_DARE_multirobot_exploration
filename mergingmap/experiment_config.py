"""Small command-line helpers shared by MergingMap launchers.

MergingMap 启动脚本共享的轻量命令行参数工具。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class RepeatArgumentResult:
    """Store a parsed repetition override and untouched forwarded arguments.

    中文目的：同时保存重复次数覆盖值与需要继续传递给主驱动的参数。
    English implementation: separates the custom repetition option from all
    remaining driver arguments.
    """

    runs_per_map: int | None
    forwarded_args: tuple[str, ...]


def positive_repeat_count(value: int, *, name: str = "runs-per-map") -> int:
    """Validate and return one strictly positive repetition count.

    中文目的：统一校验测试重复次数，防止零次或负数实验配置。
    English implementation: converts the value to ``int`` and raises a clear
    ``ValueError`` when the resulting count is not positive.
    """

    count = int(value)
    if count <= 0:
        raise ValueError(f"{name} must be positive")
    return count


def extract_repeat_argument(arguments: Sequence[str]) -> RepeatArgumentResult:
    """Remove one repetition override while preserving all other arguments.

    中文目的：从启动参数中提取 ``--runs-per-map``/``--map-repeats``，并拒绝
    重复覆盖，避免实际实验次数含糊。
    English implementation: supports space-separated and ``--option=value``
    forms, validates positivity, and forwards every unrelated argument unchanged.
    """

    args = list(arguments)
    forwarded: list[str] = []
    runs_per_map: int | None = None
    index = 0

    def store(value: str) -> None:
        nonlocal runs_per_map
        if runs_per_map is not None:
            raise ValueError("repeat count may be supplied only once")
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError("--runs-per-map must be an integer") from exc
        runs_per_map = positive_repeat_count(parsed, name="--runs-per-map")

    while index < len(args):
        item = args[index]
        if item in {"--runs-per-map", "--map-repeats"}:
            if index + 1 >= len(args):
                raise ValueError("--runs-per-map requires an integer value")
            store(args[index + 1])
            index += 2
            continue
        if item.startswith("--runs-per-map=") or item.startswith("--map-repeats="):
            store(item.split("=", 1)[1])
            index += 1
            continue
        forwarded.append(item)
        index += 1

    return RepeatArgumentResult(runs_per_map, tuple(forwarded))


def option_value(arguments: Sequence[str], option: str) -> str | None:
    """Return the last value supplied for a simple command-line option.

    中文目的：兼容 ``--name value`` 与 ``--name=value`` 两种形式读取参数。
    English implementation: scans left to right and returns the final occurrence.
    """

    result: str | None = None
    args = list(arguments)
    index = 0
    while index < len(args):
        item = args[index]
        if item == option and index + 1 < len(args):
            result = args[index + 1]
            index += 2
            continue
        prefix = option + "="
        if item.startswith(prefix):
            result = item[len(prefix):]
        index += 1
    return result
