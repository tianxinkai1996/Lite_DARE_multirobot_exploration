"""Runtime profiles for the original DARE multi-robot test worker.

Defines test-time toggles that let the paper baseline disable the
communication, reservation, coverage-coordination and deadlock-recovery
layers added later, preserving the "each robot runs original DARE
independently" semantics without touching the frozen policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from mergingmap.motion_coordinator import CoordinationMode


@dataclass(frozen=True)
class DareTestProfile:
    """Describe which non-network coordination layers are active.

    Separates the independent original-DARE reference from the existing
    coordinated extension. Toggles only test-time wrappers and never changes
    the frozen policy weights, inputs, or network structure.
    """

    key: str
    method_name: str
    coordination_mode: CoordinationMode
    enable_messages: bool
    enable_reservations: bool
    enable_coverage_exchange: bool

    def as_dict(self):
        """Return primitive fields for experiment manifests.

        Exposes a JSON-serialisable flat dictionary so the true baseline
        toggles are recorded in results rather than inferred from names.
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


def resolve_dare_test_profile(value):
    """Resolve one profile key or return a validated explicit profile.

    Provides a unified configuration entry point for the original-DARE driver
    and the paper baseline adapters. Accepts immutable profile objects or
    validates a lowercase key against the supported registry.
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