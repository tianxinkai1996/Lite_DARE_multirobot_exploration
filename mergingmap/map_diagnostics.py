"""Diagnostics for persistent merged occupancy beliefs.

"""
from __future__ import annotations

import numpy as np

class MergedBeliefDiagnosticsMixin:
    """Expose map agreement, storage, packet-class, and payload metrics."""

    def known_cell_count(self, robot_id):
        """Return the number of currently known cells for one robot."""

        robot_id = self._validate_robot_id(robot_id)
        return int(
            np.count_nonzero(
                self._maps[robot_id] != self.config.unknown_value
            )
        )

    def pairwise_agreement(self, robot_i, robot_j):
        """Return agreement over the union of cells known by either robot."""

        robot_i = self._validate_robot_id(robot_i)
        robot_j = self._validate_robot_id(robot_j)
        map_i = self._maps[robot_i]
        map_j = self._maps[robot_j]
        unknown = self.config.unknown_value
        union_known = (map_i != unknown) | (map_j != unknown)
        total = int(np.count_nonzero(union_known))
        if total == 0:
            return 1.0
        return float(np.count_nonzero((map_i == map_j) & union_known) / total)

    def mean_pairwise_agreement(self):
        values = [
            self.pairwise_agreement(i, j)
            for i in range(self.n_robots)
            for j in range(i + 1, self.n_robots)
        ]
        return 1.0 if not values else float(np.mean(values))

    def min_pairwise_agreement(self):
        values = [
            self.pairwise_agreement(i, j)
            for i in range(self.n_robots)
            for j in range(i + 1, self.n_robots)
        ]
        return 1.0 if not values else float(np.min(values))

    def _mean_or_zero(self, total, count):
        return 0.0 if count <= 0 else float(total / count)

    def metrics(self):
        """Return deterministic map-exchange and payload diagnostics.

        English implementation: exposes cumulative payload and map-quality counters
        for the generic evaluation recorder under the deterministic communication model.
        """

        known_counts = [self.known_cell_count(i) for i in range(self.n_robots)]
        return {
            "map_exchange_events": int(self.exchange_events),
            "map_packets_sent": int(self.packets_sent),
            "map_packets_delivered": int(self.packets_delivered),
            "map_first_contact_packets": int(self.first_contact_packets),
            "map_repeated_contact_packets": int(self.repeated_contact_packets),
            "map_first_contact_bytes": int(self.first_contact_bytes),
            "map_repeated_contact_bytes": int(self.repeated_contact_bytes),
            "map_first_contact_cells": int(self.first_contact_cells),
            "map_repeated_contact_cells": int(self.repeated_contact_cells),
            "map_mean_first_contact_packet_bytes": self._mean_or_zero(
                self.first_contact_bytes, self.first_contact_packets
            ),
            "map_mean_repeated_contact_packet_bytes": self._mean_or_zero(
                self.repeated_contact_bytes, self.repeated_contact_packets
            ),
            "map_cells_sent": int(self.cells_sent),
            "map_cells_received": int(self.cells_received),
            "map_cells_changed_received": int(self.cells_changed_received),
            "map_bytes_sent": int(self.bytes_sent),
            "map_conflicts": int(self.conflicts),
            "map_packet_encode_ms": float(self.packet_encode_ms),
            "map_packet_apply_ms": float(self.packet_apply_ms),
            "map_exchange_wall_ms": float(self.exchange_wall_ms),
            "map_delivery_events": int(self.delivery_events),
            "map_local_sync_ms": float(self.local_sync_ms),
            "map_header_bytes_sent": int(self.header_bytes_sent),
            "map_index_bytes_sent": int(self.index_bytes_sent),
            "map_value_bytes_sent": int(self.value_bytes_sent),
            "map_full_known_sparse_reference_bytes": int(
                self.full_known_sparse_reference_bytes
            ),
            "map_dense_grid_reference_bytes": int(self.dense_grid_reference_bytes),
            "map_deferred_cells": int(self.deferred_cells),
            "map_sent_mask_memory_bytes": int(self.sent_mask_memory_bytes()),
            "map_payload_ratio_vs_full_known_sparse": (
                1.0
                if self.full_known_sparse_reference_bytes <= 0
                else float(self.bytes_sent / self.full_known_sparse_reference_bytes)
            ),
            "map_payload_ratio_vs_dense_grid": (
                1.0
                if self.dense_grid_reference_bytes <= 0
                else float(self.bytes_sent / self.dense_grid_reference_bytes)
            ),
            "map_mean_packet_bytes": self._mean_or_zero(
                self.bytes_sent, self.packets_sent
            ),
            "map_mean_cells_per_packet": self._mean_or_zero(
                self.cells_sent, self.packets_sent
            ),
            "map_mean_inprocess_delivery_ms": self._mean_or_zero(
                self.exchange_wall_ms, self.delivery_events
            ),
            "mean_merged_known_cells": float(np.mean(known_counts)),
            "min_merged_known_cells": int(min(known_counts, default=0)),
            "max_merged_known_cells": int(max(known_counts, default=0)),
            "mean_pairwise_map_agreement": self.mean_pairwise_agreement(),
            "min_pairwise_map_agreement": self.min_pairwise_agreement(),
        }
