"""Configuration helpers for persistent merged-map exchange.

"""
from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class MergedBeliefConfig:
    """Configure contact-triggered sparse occupancy-map fusion.

    English purpose: configure sparse map encoding and conservative conflict handling.
    English implementation: the dissertation experiments use deterministic in-process
    delivery. Per-peer sent history supports a first-contact known-cell snapshot and
    later sparse deltas under deterministic in-process delivery.
    """

    unknown_value: int = 127
    free_value: int = 255
    occupied_value: int = 1
    max_cells_per_packet: int = 0
    conflict_policy: str = "occupied_wins"
    packet_header_bytes: int = 64

    def __post_init__(self):
        if self.max_cells_per_packet < 0:
            raise ValueError("max_cells_per_packet cannot be negative")
        if self.conflict_policy not in {"occupied_wins", "keep_existing"}:
            raise ValueError(
                "conflict_policy must be 'occupied_wins' or 'keep_existing'"
            )
        if int(self.packet_header_bytes) < 0:
            raise ValueError("packet_header_bytes cannot be negative")
