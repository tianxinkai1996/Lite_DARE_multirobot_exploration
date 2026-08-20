"""Persistent per-robot occupancy belief with deterministic per-peer deltas.

每台机器人持久融合占据图及确定性的逐同伴增量同步。
"""
from __future__ import annotations

import copy
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # Standalone tests execute with ``mergingmap`` as the working directory.
    from map_diagnostics import MergedBeliefDiagnosticsMixin
    from map_protocol import MergedBeliefConfig
except ImportError:  # Package imports execute from the project root.
    from mergingmap.map_diagnostics import MergedBeliefDiagnosticsMixin
    from mergingmap.map_protocol import MergedBeliefConfig


class MergedBeliefManager(MergedBeliefDiagnosticsMixin):
    """Maintain one persistent merged map per robot and exchange sparse updates.

    中文目的：把本地感知与同伴知识累积到独立的持久地图，并支持首遇快照、
    后续逐同伴增量、受限数据包和传递式知识传播。
    中文实现：每个有向机器人对维护已发送掩码；确定性传递成功后推进该掩码。
    收到第三方的新知识后，该单元对其他同伴重新变为待发送状态。

    English implementation: maintains a sent-history mask for every ordered
    sender-receiver pair. Deterministically delivered cells are excluded from later
    deltas until their local value changes; peer-learned cells are reopened for all
    other peers, enabling gossip over changing contacts.
    """

    def __init__(
        self,
        local_maps: Sequence[np.ndarray],
        config: MergedBeliefConfig,
    ) -> None:
        if not local_maps:
            raise ValueError("At least one local map is required")
        arrays = [np.asarray(grid) for grid in local_maps]
        shape = arrays[0].shape
        if len(shape) != 2:
            raise ValueError(f"Expected 2-D occupancy maps, got {shape}")
        if any(grid.shape != shape for grid in arrays):
            raise ValueError("All maps must use the same shared grid frame")

        self.config = config
        self.n_robots = len(arrays)
        self.shape = shape
        self._maps: List[np.ndarray] = [grid.copy() for grid in arrays]
        self._clock = [0 for _ in range(self.n_robots)]
        self._cell_version = [np.zeros(shape, dtype=np.int64) for _ in arrays]
        for robot_id, grid in enumerate(self._maps):
            flat_known = np.flatnonzero(grid != self.config.unknown_value)
            if flat_known.size:
                versions = np.arange(1, flat_known.size + 1, dtype=np.int64)
                self._cell_version[robot_id].flat[flat_known] = versions
                self._clock[robot_id] = int(flat_known.size)

        self._sent_to_peer: List[Dict[int, np.ndarray]] = [
            {} for _ in range(self.n_robots)
        ]
        self._packet_sequence: List[Dict[int, int]] = [
            {} for _ in range(self.n_robots)
        ]
        self._contact_count: List[Dict[int, int]] = [
            {} for _ in range(self.n_robots)
        ]
        self._last_received_sequence: List[Dict[int, int]] = [
            {} for _ in range(self.n_robots)
        ]
        self._reset_counters()

    def _reset_counters(self) -> None:
        """Initialise cumulative protocol and map-fusion counters."""

        for name in (
            "exchange_events", "packets_sent", "packets_delivered",
            "first_contact_packets", "repeated_contact_packets",
            "first_contact_bytes", "repeated_contact_bytes", "first_contact_cells",
            "repeated_contact_cells", "cells_sent", "cells_received",
            "cells_changed_received", "bytes_sent", "conflicts", "delivery_events",
            "header_bytes_sent", "index_bytes_sent", "value_bytes_sent",
            "full_known_sparse_reference_bytes", "dense_grid_reference_bytes",
            "deferred_cells",
        ):
            setattr(self, name, 0)
        self.packet_encode_ms = 0.0
        self.packet_apply_ms = 0.0
        self.exchange_wall_ms = 0.0
        self.local_sync_ms = 0.0

    def _validate_robot_id(self, robot_id: int) -> int:
        robot_id = int(robot_id)
        if not 0 <= robot_id < self.n_robots:
            raise IndexError(f"robot_id {robot_id} is out of range")
        return robot_id

    def _sent_mask(self, sender_id: int, receiver_id: int) -> np.ndarray:
        masks = self._sent_to_peer[sender_id]
        if receiver_id not in masks:
            masks[receiver_id] = np.zeros(self.shape, dtype=bool)
        return masks[receiver_id]

    def sent_mask_memory_bytes(self) -> int:
        """Return allocated dense per-peer sent-history memory in bytes."""

        return int(
            sum(mask.nbytes for masks in self._sent_to_peer for mask in masks.values())
        )

    def _mark_new_information(
        self,
        robot_id: int,
        flat_indices: np.ndarray,
        *,
        source_peer: Optional[int] = None,
    ) -> None:
        if flat_indices.size == 0:
            return
        start = self._clock[robot_id] + 1
        stop = start + int(flat_indices.size)
        self._cell_version[robot_id].flat[flat_indices] = np.arange(
            start, stop, dtype=np.int64
        )
        self._clock[robot_id] = stop - 1
        peer_ids = set(self._sent_to_peer[robot_id])
        for peer_id in peer_ids:
            sent_mask = self._sent_mask(robot_id, peer_id)
            sent_mask.flat[flat_indices] = False
            if source_peer is not None and peer_id == int(source_peer):
                sent_mask.flat[flat_indices] = True

    def _resolve_conflicts(
        self,
        existing: np.ndarray,
        incoming: np.ndarray,
    ) -> np.ndarray:
        if self.config.conflict_policy == "keep_existing":
            return existing
        result = existing.copy()
        result[incoming == self.config.occupied_value] = self.config.occupied_value
        return result

    def sync_local_map(self, robot_id: int, local_map: np.ndarray) -> int:
        """Insert newest direct sensing and return the number of changed cells."""

        started_at = time.perf_counter()
        robot_id = self._validate_robot_id(robot_id)
        incoming = np.asarray(local_map)
        if incoming.shape != self.shape:
            raise ValueError(f"Local map shape {incoming.shape} does not match {self.shape}")
        merged = self._maps[robot_id]
        unknown = self.config.unknown_value
        incoming_known = incoming != unknown
        existing_unknown = merged == unknown
        changed = incoming_known & existing_unknown
        merged[changed] = incoming[changed]

        conflict = incoming_known & (~existing_unknown) & (incoming != merged)
        if np.any(conflict):
            self.conflicts += int(np.count_nonzero(conflict))
            resolved = self._resolve_conflicts(merged[conflict], incoming[conflict])
            conflict_indices = np.flatnonzero(conflict)
            resolution_changed = resolved != merged[conflict]
            if np.any(resolution_changed):
                resolved_indices = conflict_indices[resolution_changed]
                merged.flat[resolved_indices] = resolved[resolution_changed]
                changed.flat[resolved_indices] = True

        flat_changed = np.flatnonzero(changed)
        self._mark_new_information(robot_id, flat_changed)
        self.local_sync_ms += (time.perf_counter() - started_at) * 1000.0
        return int(flat_changed.size)

    def _next_sequence(self, sender_id: int, receiver_id: int) -> int:
        current = self._packet_sequence[sender_id].get(receiver_id, 0) + 1
        self._packet_sequence[sender_id][receiver_id] = current
        return current

    def make_packet(
        self,
        sender_id: int,
        receiver_id: int,
        *,
        step: int,
        mode: str = "delta",
    ) -> dict:
        """Build one sparse full/delta packet without mutating the receiver.

        中文目的：首遇时空的逐同伴已发送历史自然形成完整已知图快照，后续仅发送
        尚未向该同伴发送的当前版本；受限数据包优先选择版本较新的单元。
        English implementation: constructs a sequenced first-contact or repeated-contact
        packet. Sent history advances after deterministic in-process delivery.
        """

        started_at = time.perf_counter()
        sender_id = self._validate_robot_id(sender_id)
        receiver_id = self._validate_robot_id(receiver_id)
        if sender_id == receiver_id:
            raise ValueError("sender_id and receiver_id must differ")
        if mode not in {"delta", "full"}:
            raise ValueError("mode must be 'delta' or 'full'")

        contact_index = self._contact_count[sender_id].get(receiver_id, 0)
        self._contact_count[sender_id][receiver_id] = contact_index + 1
        first_contact = contact_index == 0
        effective_mode = mode
        merged = self._maps[sender_id]
        known = merged != self.config.unknown_value
        sent_mask = self._sent_mask(sender_id, receiver_id)
        eligible = known if effective_mode == "full" else (known & (~sent_mask))
        flat_indices = np.flatnonzero(eligible)
        eligible_count = int(flat_indices.size)
        known_count = int(np.count_nonzero(known))

        limit = int(self.config.max_cells_per_packet)
        if limit > 0 and flat_indices.size > limit:
            versions = self._cell_version[sender_id].flat[flat_indices]
            selected = np.argpartition(versions, -limit)[-limit:]
            flat_indices = flat_indices[selected]
            order = np.argsort(
                self._cell_version[sender_id].flat[flat_indices], kind="stable"
            )[::-1]
            flat_indices = flat_indices[order]

        indices = flat_indices.astype(np.int32, copy=True)
        values = merged.flat[flat_indices].astype(np.uint8, copy=True)
        header_bytes = int(self.config.packet_header_bytes)
        index_bytes = int(indices.nbytes)
        value_bytes = int(values.nbytes)
        byte_count = header_bytes + index_bytes + value_bytes
        full_sparse_bytes = header_bytes + known_count * 5
        dense_bytes = header_bytes + int(merged.nbytes)
        deferred = max(0, eligible_count - int(indices.size))
        sequence_id = self._next_sequence(sender_id, receiver_id)


        packet = {
            "type": "merged_map_delta",
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "step": int(step),
            "sequence_id": sequence_id,
            "shape": tuple(int(value) for value in self.shape),
            "indices": indices,
            "values": values,
            "cell_count": int(indices.size),
            "byte_count": int(byte_count),
            "header_bytes": header_bytes,
            "index_bytes": index_bytes,
            "value_bytes": value_bytes,
            "eligible_cell_count": eligible_count,
            "known_cell_count": known_count,
            "deferred_cell_count": deferred,
            "mode": effective_mode,
            "requested_mode": mode,
            "first_contact": bool(first_contact),
            "contact_index": int(contact_index),
            "full_known_sparse_reference_bytes": int(full_sparse_bytes),
            "dense_grid_reference_bytes": int(dense_bytes),
            "payload_ratio_vs_full_known_sparse": (
                1.0 if full_sparse_bytes <= 0 else float(byte_count / full_sparse_bytes)
            ),
            "payload_ratio_vs_dense_grid": (
                1.0 if dense_bytes <= 0 else float(byte_count / dense_bytes)
            ),
        }
        self._record_encoded_packet(packet)
        self.packet_encode_ms += (time.perf_counter() - started_at) * 1000.0
        return packet

    def _record_encoded_packet(self, packet: dict) -> None:
        self.packets_sent += 1
        self.cells_sent += int(packet["cell_count"])
        self.bytes_sent += int(packet["byte_count"])
        self.header_bytes_sent += int(packet["header_bytes"])
        self.index_bytes_sent += int(packet["index_bytes"])
        self.value_bytes_sent += int(packet["value_bytes"])
        self.full_known_sparse_reference_bytes += int(
            packet["full_known_sparse_reference_bytes"]
        )
        self.dense_grid_reference_bytes += int(packet["dense_grid_reference_bytes"])
        self.deferred_cells += int(packet["deferred_cell_count"])
        if packet["first_contact"]:
            self.first_contact_packets += 1
            self.first_contact_bytes += int(packet["byte_count"])
            self.first_contact_cells += int(packet["cell_count"])
        else:
            self.repeated_contact_packets += 1
            self.repeated_contact_bytes += int(packet["byte_count"])
            self.repeated_contact_cells += int(packet["cell_count"])

    def apply_packet(self, receiver_id: int, packet: dict) -> int:
        """Apply one received packet and return its number of changed cells."""

        started_at = time.perf_counter()
        receiver_id = self._validate_robot_id(receiver_id)
        if packet.get("type") != "merged_map_delta":
            raise ValueError("Unsupported packet type")
        if int(packet["receiver_id"]) != receiver_id:
            raise ValueError("Packet receiver does not match receiver_id")
        if tuple(packet["shape"]) != tuple(self.shape):
            raise ValueError("Map frames/shapes do not match; alignment is required")

        sender_id = self._validate_robot_id(int(packet["sender_id"]))
        sequence = int(packet.get("sequence_id", 0))
        self._last_received_sequence[receiver_id][sender_id] = max(
            sequence,
            self._last_received_sequence[receiver_id].get(sender_id, 0),
        )
        indices = np.asarray(packet["indices"], dtype=np.int32)
        values = np.asarray(packet["values"], dtype=np.uint8)
        if indices.ndim != 1 or values.ndim != 1 or indices.size != values.size:
            raise ValueError("Packet indices and values must be equal-length vectors")
        if indices.size and (
            int(indices.min()) < 0 or int(indices.max()) >= int(np.prod(self.shape))
        ):
            raise ValueError("Packet contains an out-of-range flat cell index")

        merged = self._maps[receiver_id]
        current = merged.flat[indices].copy()
        incoming = values.astype(merged.dtype, copy=False)
        unknown = self.config.unknown_value
        current_unknown = current == unknown
        incoming_known = incoming != unknown
        result = current.copy()
        result[current_unknown & incoming_known] = incoming[current_unknown & incoming_known]
        conflict = (~current_unknown) & incoming_known & (current != incoming)
        if np.any(conflict):
            self.conflicts += int(np.count_nonzero(conflict))
            result[conflict] = self._resolve_conflicts(current[conflict], incoming[conflict])

        changed_mask = result != current
        changed_indices = indices[changed_mask]
        if changed_indices.size:
            merged.flat[changed_indices] = result[changed_mask]
            self._mark_new_information(
                receiver_id, changed_indices.astype(np.int64), source_peer=sender_id
            )
        # Every received cell is already known by its sender, so no immediate echo is needed.
        if indices.size:
            self._sent_mask(receiver_id, sender_id).flat[indices] = True
        changed_count = int(changed_indices.size)
        self.cells_received += int(indices.size)
        self.cells_changed_received += changed_count
        self.packet_apply_ms += (time.perf_counter() - started_at) * 1000.0
        self.delivery_events += 1
        return changed_count

    def _deliver(self, receiver_id: int, packet: dict) -> int:
        """Apply one directed packet using deterministic in-process delivery."""

        changed = self.apply_packet(receiver_id, packet)
        self.packets_delivered += 1
        sender_id = self._validate_robot_id(int(packet["sender_id"]))
        indices = np.asarray(packet["indices"], dtype=np.int64)
        if indices.size:
            self._sent_mask(sender_id, receiver_id).flat[indices] = True
        packet["delivered"] = True
        return changed

    def exchange_pair(
        self,
        robot_i: int,
        robot_j: int,
        *,
        step: int,
        mode: str = "delta",
    ) -> Tuple[int, int, dict, dict]:
        """Exchange both directions from pre-merge snapshots with deterministic deltas."""

        started_at = time.perf_counter()
        packet_i = self.make_packet(robot_i, robot_j, step=step, mode=mode)
        packet_j = self.make_packet(robot_j, robot_i, step=step, mode=mode)
        changed_j = self._deliver(robot_j, packet_i)
        changed_i = self._deliver(robot_i, packet_j)
        self.exchange_events += 1
        self.exchange_wall_ms += (time.perf_counter() - started_at) * 1000.0
        return changed_i, changed_j, packet_i, packet_j

    def merged_map(self, robot_id: int, *, copy_map: bool = True) -> np.ndarray:
        robot_id = self._validate_robot_id(robot_id)
        return self._maps[robot_id].copy() if copy_map else self._maps[robot_id]

    def make_map_info(self, robot_id: int, template_map_info):
        """Clone a DARE MapInfo and replace only its occupancy grid."""

        robot_id = self._validate_robot_id(robot_id)
        info = copy.copy(template_map_info)
        if not hasattr(info, "map"):
            raise TypeError("template_map_info must expose a .map attribute")
        info.map = self._maps[robot_id].copy()
        return info


__all__ = ["MergedBeliefConfig", "MergedBeliefManager"]
