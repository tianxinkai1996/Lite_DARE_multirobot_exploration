"""Configuration helpers for persistent merged-map exchange.

持久融合地图交换的配置辅助函数。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MergedBeliefConfig:
    """Configure contact-triggered sparse occupancy-map fusion.

    中文目的：集中定义地图状态、数据包预算和冲突策略。
    中文实现：当前论文实验采用确定性的进程内传递；每个有向机器人对维护
    已发送历史，用于首遇完整已知图快照和后续稀疏增量。

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

    def __post_init__(self) -> None:
        if self.max_cells_per_packet < 0:
            raise ValueError("max_cells_per_packet cannot be negative")
        if self.conflict_policy not in {"occupied_wins", "keep_existing"}:
            raise ValueError(
                "conflict_policy must be 'occupied_wins' or 'keep_existing'"
            )
        if int(self.packet_header_bytes) < 0:
            raise ValueError("packet_header_bytes cannot be negative")
