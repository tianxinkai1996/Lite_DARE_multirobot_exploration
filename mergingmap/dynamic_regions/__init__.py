"""Public dynamic-region API for MergingMap.

MergingMap 动态区域模块的公共接口。
"""
from .coordinator import DynamicRegionCoordinator
from .extraction import extract_frontier_regions
from .models import (
    DynamicRegionConfig,
    FrontierRegion,
    PeerClaim,
    RegionLease,
    RobotRegionState,
)

__all__ = [
    "DynamicRegionConfig",
    "DynamicRegionCoordinator",
    "FrontierRegion",
    "PeerClaim",
    "RegionLease",
    "RobotRegionState",
    "extract_frontier_regions",
]