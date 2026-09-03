"""Load, normalise and partition Chapter 4 results for reporting.

The raw experiment tree is never modified. This module creates a reporting view
that keeps broad experimental collection separate from the narrower main-text scope.
For multi-robot comparisons, only complete matched treatment sets are used in the
primary tables/figures so that method or communication differences are not caused by
missing scenarios.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from paper_experiments.report_scope import (
    COMM_GROUP_NAMES,
    MULTI_GROUP_NAMES,
    PRIMARY_COMM_MODES,
    PRIMARY_COMM_ROLE,
    PRIMARY_MULTI_MODE,
    PRIMARY_MULTI_ROLES,
    PRIMARY_TEAM_SIZES,
    ROLE_LABELS,
    SINGLE_GROUP_NAME,
    SINGLE_ROBOT_DEPTHS,
)


def _top_group(root, path):
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return "unknown"
    return rel.parts[0] if rel.parts else "unknown"


def _infer_role(row):
    role = str(row.get("method_role", "") or "").strip()
    if role and role.lower() != "nan":
        return role
    method = str(row.get("method", "") or "").lower()
    if "map-reservation" in method or "map_reservation" in method:
        return "map_reservation"
    if "map-region" in method or "map_region" in method:
        return "map_region"
    if "maponly" in method or "map-only" in method or "map_only" in method:
        return "map_only"
    if "full" in method:
        return "full"
    if "policy" in method or method.endswith("-only"):
        return "policy_only"
    if "original" in method and "dare" in method:
        return "original_dare"
    return role or "unknown"


def _normalise_mode(data):
    if "communication_mode" not in data.columns:
        data["communication_mode"] = np.nan
    if "comparison_communication_mode" in data.columns:
        missing = data["communication_mode"].isna() | (
            data["communication_mode"].astype(str).str.strip().isin(["", "nan"])
        )
        data.loc[missing, "communication_mode"] = data.loc[
            missing, "comparison_communication_mode"
        ]


def read_all_results(root):
    """Read all recorded results while preserving their source experiment group."""
    frames: list[pd.DataFrame] = []
    seen: set[Path] = set()
    for path in sorted(root.rglob("results.csv")):
        resolved = path.resolve()
        if resolved in seen or not path.is_file() or path.stat().st_size == 0:
            continue
        seen.add(resolved)
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["_source_file"] = str(path)
        frame["_experiment_group"] = _top_group(root, path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no non-empty results.csv found below {root}")

    data = pd.concat(frames, ignore_index=True, sort=False)
    _normalise_mode(data)
    data["method_role_normalised"] = data.apply(_infer_role, axis=1)
    data["method_label_report"] = data["method_role_normalised"].map(ROLE_LABELS).fillna(
        data.get("method", pd.Series(index=data.index, dtype=object)).astype(str)
    )

    identity = [
        key
        for key in (
            "model_key",
            "method_role_normalised",
            "map_index",
            "trial",
            "team_size",
            "communication_mode",
            "scenario_id",
            "scenario_pair_key",
            "_experiment_group",
        )
        if key in data.columns
    ]
    if identity:
        data = data.drop_duplicates(subset=identity, keep="last")
    return data.reset_index(drop=True)


def numeric_columns(data, columns):
    for column in columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")


def _team_size_mask(data):
    if "team_size" not in data.columns:
        return pd.Series(False, index=data.index)
    return pd.to_numeric(data["team_size"], errors="coerce").isin(PRIMARY_TEAM_SIZES)


def primary_single(data):
    mask = data["_experiment_group"].eq(SINGLE_GROUP_NAME)
    if "team_size" in data.columns:
        mask &= pd.to_numeric(data["team_size"], errors="coerce").eq(1)
    if "encoder_layers" in data.columns:
        mask &= pd.to_numeric(data["encoder_layers"], errors="coerce").isin(SINGLE_ROBOT_DEPTHS)
    return data.loc[mask].copy()


def primary_multi(data):
    mask = (
        data["_experiment_group"].isin(MULTI_GROUP_NAMES)
        & data["method_role_normalised"].isin(PRIMARY_MULTI_ROLES)
        & data["communication_mode"].astype(str).eq(PRIMARY_MULTI_MODE)
        & _team_size_mask(data)
    )
    return data.loc[mask].copy()


def primary_communication(data):
    mask = (
        data["_experiment_group"].isin(COMM_GROUP_NAMES)
        & data["method_role_normalised"].eq(PRIMARY_COMM_ROLE)
        & data["communication_mode"].astype(str).isin(PRIMARY_COMM_MODES)
        & _team_size_mask(data)
    )
    return data.loc[mask].copy()


def partition_reporting_scope(data):
    single_idx = set(primary_single(data).index)
    multi_idx = set(primary_multi(data).index)
    comm_idx = set(primary_communication(data).index)
    primary_idx = single_idx | multi_idx | comm_idx
    primary = data.loc[sorted(primary_idx)].copy()
    supplementary = data.drop(index=sorted(primary_idx)).copy()
    return primary, supplementary


def pair_key(data):
    """Return the most stable paired-scenario key available in recorded outputs.

    This is suitable when the treatment itself does not change scenario_id, e.g.
    L6/L4/L2 with scenario_pair_key or Map-only/Full under one fixed communication mode.
    """
    if "scenario_pair_key" in data.columns:
        key = data["scenario_pair_key"].astype(str)
        valid = ~key.isin(["", "nan", "None"])
        if valid.any():
            fallback = comparison_pair_key(data)
            return key.where(valid, fallback)
    if "scenario_id" in data.columns:
        key = data["scenario_id"].astype(str)
        valid = ~key.isin(["", "nan", "None"])
        if valid.any():
            fallback = comparison_pair_key(data)
            return key.where(valid, fallback)
    return comparison_pair_key(data)


def comparison_pair_key(data):
    """Return a treatment-independent key for matched multi-robot scenarios.

    Communication mode is deliberately excluded because none/raw/compressed must be
    compared on the same map/trial/team/seed. Team size remains in the key: N=2,4,6,8
    are controlled scaling conditions, not identical-start paired treatments.
    """
    pieces: list[pd.Series] = []
    for name in ("map_index", "trial", "team_size", "seed"):
        if name in data.columns:
            pieces.append(data[name].astype(str))
        else:
            pieces.append(pd.Series(["NA"] * len(data), index=data.index))
    key = pieces[0]
    for piece in pieces[1:]:
        key = key.str.cat(piece, sep="|")
    return key


def _normalise_start_positions(series):
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"\s+", "", regex=True)
        .replace({"nan": "", "None": ""})
    )


def complete_treatment_pairs(data, treatment_column, treatments):
    """Keep only scenario keys that contain every requested treatment.

    If start_positions is present, a scenario is retained only when non-empty start
    signatures agree across treatments. This makes the report stricter than a simple
    row-count comparison and prevents missing/mismatched starts from entering the
    main paired effect estimates.
    """
    if data.empty or treatment_column not in data.columns:
        return data.iloc[0:0].copy()
    expected = {str(value) for value in treatments}
    work = data.copy()
    work["_report_pair"] = comparison_pair_key(work)
    work["_report_treatment"] = work[treatment_column].astype(str)

    present = work.groupby("_report_pair")["_report_treatment"].agg(lambda s: set(s))
    good = set(present.index[present.map(lambda values: expected.issubset(values))])

    if "start_positions" in work.columns and good:
        starts = work.loc[work["_report_pair"].isin(good), ["_report_pair", "start_positions"]].copy()
        starts["_start"] = _normalise_start_positions(starts["start_positions"])
        start_counts = starts.loc[starts["_start"].ne("")].groupby("_report_pair")["_start"].nunique()
        bad_starts = set(start_counts.index[start_counts.gt(1)])
        good.difference_update(bad_starts)

    return work.loc[work["_report_pair"].isin(good)].copy()


def complete_multi_pairs(data):
    return complete_treatment_pairs(
        data,
        treatment_column="method_role_normalised",
        treatments=PRIMARY_MULTI_ROLES,
    )


def complete_communication_pairs(data):
    return complete_treatment_pairs(
        data,
        treatment_column="communication_mode",
        treatments=PRIMARY_COMM_MODES,
    )


def coverage_report(data):
    """Summarise reporting coverage, including complete matched sets by team size."""
    single = primary_single(data)
    multi = primary_multi(data)
    comm = primary_communication(data)
    multi_complete = complete_multi_pairs(multi)
    comm_complete = complete_communication_pairs(comm)
    report: dict[str, object] = {
        "all_rows": int(len(data)),
        "single_rows": int(len(single)),
        "multi_primary_rows": int(len(multi)),
        "multi_complete_rows": int(len(multi_complete)),
        "communication_primary_rows": int(len(comm)),
        "communication_complete_rows": int(len(comm_complete)),
    }
    if not multi.empty:
        report["multi_counts"] = (
            multi.groupby(["team_size", "method_role_normalised"], dropna=False)
            .size()
            .rename("rows")
            .reset_index()
            .to_dict("records")
        )
    if not multi_complete.empty:
        report["multi_complete_pair_counts"] = (
            multi_complete[["team_size", "_report_pair"]]
            .drop_duplicates()
            .groupby("team_size")
            .size()
            .rename("complete_pairs")
            .reset_index()
            .to_dict("records")
        )
    if not comm.empty:
        report["communication_counts"] = (
            comm.groupby(["team_size", "communication_mode"], dropna=False)
            .size()
            .rename("rows")
            .reset_index()
            .to_dict("records")
        )
    if not comm_complete.empty:
        report["communication_complete_pair_counts"] = (
            comm_complete[["team_size", "_report_pair"]]
            .drop_duplicates()
            .groupby("team_size")
            .size()
            .rename("complete_pairs")
            .reset_index()
            .to_dict("records")
        )
    return report

