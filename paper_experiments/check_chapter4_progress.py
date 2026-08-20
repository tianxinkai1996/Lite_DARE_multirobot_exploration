#!/usr/bin/env python3
"""Summarise existing Chapter 4 progress without launching any experiment."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _group_counts(rows: list[dict[str, str]], keys: tuple[str, ...]) -> list[tuple[tuple[str, ...], int]]:
    counts: dict[tuple[str, ...], int] = {}
    for row in rows:
        key = tuple(str(row.get(name, "")) for name in keys)
        counts[key] = counts.get(key, 0) + 1
    return sorted(counts.items())


def _print_group(root: Path, name: str) -> None:
    group = root / name
    results = _rows(group / "results.csv")
    scenarios = _rows(group / "scenario_manifest.csv")
    manifest_path = group / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    print(f"\n[{name}]")
    print(f"  exists={group.exists()} results={len(results)} scenarios={len(scenarios)} status={manifest.get('status', 'n/a')}")
    if manifest:
        print(f"  planned={manifest.get('planned_runs', 'n/a')} completed={manifest.get('completed_runs', 'n/a')}")
    if results:
        keys = tuple(key for key in ("method_role", "communication_mode", "team_size") if key in results[0])
        if keys:
            for values, count in _group_counts(results, keys):
                print("  " + ", ".join(f"{key}={value}" for key, value in zip(keys, values)) + f": {count}")
    for profile in ("policy_only", "original_dare"):
        ref = _rows(group / "methods" / profile / "results.csv")
        if ref:
            print(f"  reference {profile}: {len(ref)} rows")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    print(f"chapter4_root={root}")
    for group in ("e1_policy_depth", "multi_ablation", "communication", "multi_maponly_full", "communication_reduced"):
        if (root / group).exists():
            if group == "e1_policy_depth":
                print("\n[e1_policy_depth]")
                for result in sorted((root / group).glob("*/results.csv")):
                    print(f"  {result.parent.name}: {len(_rows(result))} rows")
            else:
                _print_group(root, group)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
