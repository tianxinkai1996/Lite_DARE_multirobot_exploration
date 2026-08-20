"""Runtime profiles for the original DARE multi-robot test worker.

原始 DARE 多机器人测试工作器的运行配置。该模块允许论文基线关闭后来加入的
通信、预留、覆盖协调和死锁恢复，从而保持“每台机器人独立运行原 DARE”语义。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from mergingmap.motion_coordinator import CoordinationMode


@dataclass(frozen=True)
class DareTestProfile:
    """Describe which non-network coordination layers are active.

    中文目的：明确区分纯原始 DARE 基线与已有的协调增强版本。
    中文实现：配置只控制测试时外部模块，不修改 DARE 权重、输入或网络结构。

    English purpose: separate the independent original-DARE reference from the
    existing coordinated extension. English implementation: toggles only
    test-time wrappers and never changes the frozen policy itself.
    """

    key: str
    method_name: str
    coordination_mode: CoordinationMode
    enable_messages: bool
    enable_reservations: bool
    enable_coverage_exchange: bool

    def as_dict(self) -> dict[str, str | bool]:
        """Return primitive fields for experiment manifests.

        中文目的：把真实基线开关写入结果，避免仅依赖方法名称推断配置。
        English implementation: exposes a JSON-serialisable flat dictionary.
        """

        return {
            "key": self.key,
            "method_name": self.method_name,
            "coordination_mode": self.coordination_mode,
            "enable_messages": self.enable_messages,
            "enable_reservations": self.enable_reservations,
            "enable_coverage_exchange": self.enable_coverage_exchange,
        }


DARE_TEST_PROFILES: Mapping[str, DareTestProfile] = {
    "original_dare": DareTestProfile(
        key="original_dare",
        method_name="Original-DARE",
        coordination_mode="ghost",
        enable_messages=False,
        enable_reservations=False,
        enable_coverage_exchange=False,
    ),
    "policy_only": DareTestProfile(
        key="policy_only",
        method_name="LiteDARE-only",
        coordination_mode="ghost",
        enable_messages=False,
        enable_reservations=False,
        enable_coverage_exchange=False,
    ),
    "coordinated": DareTestProfile(
        key="coordinated",
        method_name="LiteDARE-Collision-Deadlock-SparseCoverage",
        coordination_mode="collision_deadlock",
        enable_messages=True,
        enable_reservations=True,
        enable_coverage_exchange=True,
    ),
}


def resolve_dare_test_profile(value: str | DareTestProfile) -> DareTestProfile:
    """Resolve one profile key or return a validated explicit profile.

    中文目的：为原 DARE 驱动和论文基线适配器提供统一配置入口。
    English implementation: accepts immutable profile objects or validates a
    lowercase key against the supported registry.
    """

    if isinstance(value, DareTestProfile):
        return value
    key = str(value).strip().lower()
    try:
        return DARE_TEST_PROFILES[key]
    except KeyError as exc:
        supported = ", ".join(sorted(DARE_TEST_PROFILES))
        raise ValueError(
            f"unsupported DARE test profile {value!r}; choose one of: {supported}"
        ) from exc