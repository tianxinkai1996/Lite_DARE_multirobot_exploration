"""Methodology-aligned ablation profiles for the MergingMap worker.

与方法章节一致的 MergingMap 消融实验配置。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from mergingmap.motion_coordinator import CoordinationMode


@dataclass(frozen=True)
class MergingMapAblationProfile:
    """Bind a paper method name to map, task, and motion layers.

    中文目的：显式区分 Map-only、Map+Region、Map+Reservation 与 Full，确保
    每种方法标签与实际开关一致。English implementation: stores immutable layer
    switches so manifests can prove which knowledge, task, and safety mechanisms ran.
    """

    key: str
    method_name: str
    coordination_mode: CoordinationMode
    enable_initial_direction: bool = False
    enable_dynamic_regions: bool = False
    enable_map_merging: bool = True

    def with_extra_supervisors(self, enabled: bool) -> "MergingMapAblationProfile":
        """Optionally enable both engineering supervisors without disabling defaults.

        中文目的：仅在明确请求额外工程实验时开启初始方向和区域监督；False 保留
        配置本身的论文定义，避免意外关闭 Full 方法的区域层。
        English implementation: returns the original profile unless explicitly enabled.
        """

        if not enabled:
            return self
        return replace(
            self,
            enable_initial_direction=True,
            enable_dynamic_regions=True,
        )

    def as_dict(self) -> dict[str, str | bool]:
        """Return a JSON/CSV-serialisable profile description."""

        return {
            "key": self.key,
            "method_name": self.method_name,
            "coordination_mode": self.coordination_mode,
            "enable_map_merging": bool(self.enable_map_merging),
            "enable_initial_direction": bool(self.enable_initial_direction),
            "enable_dynamic_regions": bool(self.enable_dynamic_regions),
        }


CORE_ABLATION_PROFILES: Mapping[str, MergingMapAblationProfile] = {
    "map_only": MergingMapAblationProfile(
        key="map_only",
        method_name="LiteDARE-MapOnly",
        coordination_mode="ghost",
    ),
    "map_region": MergingMapAblationProfile(
        key="map_region",
        method_name="LiteDARE-Map-Region",
        coordination_mode="ghost",
        enable_dynamic_regions=True,
    ),
    "map_reservation": MergingMapAblationProfile(
        key="map_reservation",
        method_name="LiteDARE-Map-Reservation",
        coordination_mode="collision",
    ),
    "full": MergingMapAblationProfile(
        key="full",
        method_name="LiteDARE-Full-ContactAware",
        coordination_mode="collision_deadlock",
        enable_initial_direction=True,
        enable_dynamic_regions=True,
    ),
}

# Previous command-line keys remain accepted, but no obsolete forwarding files are
# reintroduced. 旧命令行名称仅作为配置别名保留，不恢复已删除旧实现文件。
PROFILE_ALIASES: Mapping[str, str] = {
    "mm": "map_only",
    "mm_region": "map_region",
    "mm_collision": "map_reservation",
    "mm_collision_deadlock": "full",
}


def method_display_name(model_name: str, method_key: str) -> str:
    """Return a model-aware display label for one stable ablation role."""
    suffix = {
        "map_only": "MapOnly", "map_region": "Map-Region",
        "map_reservation": "Map-Reservation", "full": "Full-ContactAware",
    }[method_key]
    return f"{model_name}-{suffix}"

DEFAULT_PROFILE_KEY = "full"


def resolve_ablation_profile(
    value: str | MergingMapAblationProfile,
    *,
    include_extra_supervisors: bool = False,
) -> MergingMapAblationProfile:
    """Resolve canonical/legacy keys or validate an explicit profile object."""

    if isinstance(value, MergingMapAblationProfile):
        profile = value
    else:
        requested = str(value).strip().lower()
        key = PROFILE_ALIASES.get(requested, requested)
        try:
            profile = CORE_ABLATION_PROFILES[key]
        except KeyError as exc:
            supported = ", ".join(
                sorted(set(CORE_ABLATION_PROFILES) | set(PROFILE_ALIASES))
            )
            raise ValueError(
                f"unsupported MergingMap ablation profile {value!r}; "
                f"choose one of: {supported}"
            ) from exc
    return profile.with_extra_supervisors(include_extra_supervisors)


def parse_profile_keys(value: str) -> tuple[str, ...]:
    """Parse, canonicalise, and de-duplicate a comma-separated profile list."""

    requested = tuple(
        part.strip().lower() for part in str(value).split(",") if part.strip()
    )
    if not requested:
        raise ValueError("at least one ablation method is required")
    canonical = tuple(resolve_ablation_profile(key).key for key in requested)
    if len(set(canonical)) != len(canonical):
        raise ValueError("ablation method keys must not repeat the same profile")
    return canonical

