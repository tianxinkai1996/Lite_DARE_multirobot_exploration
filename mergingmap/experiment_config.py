"""Shared experiment configuration for MergingMap multi-robot tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class RepeatArgumentResult:
    """Store a parsed repetition override and untouched forwarded arguments. separates the custom repetition option from all
    remaining driver arguments.
    """

    runs_per_map: int | None
    forwarded_args: tuple[str, ...]


def positive_repeat_count(value, name="runs-per-map"):
    """Validate and return one strictly positive repetition count.
    """

    count = int(value)
    if count <= 0:
        raise ValueError(f"{name} must be positive")
    return count


def extract_repeat_argument(arguments):
    """Remove one repetition override while preserving all other arguments.
    supports space-separated and ``--option=value``
    forms, validates positivity, and forwards every unrelated argument unchanged.
    """

    args = list(arguments)
    forwarded: list[str] = []
    runs_per_map: int | None = None
    index = 0

    def store(value):
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


def option_value(arguments, option):
    """Return the last value supplied for a simple command-line option.
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
