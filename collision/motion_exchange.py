"""Short-horizon motion-intent exchange for local reservation filtering.

"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable, Sequence

import numpy as np

from collision.reservation_manager import ReservationManager

@dataclass(frozen=True)
class MotionExchangeConfig:
    """Configure one/two-step motion packets and local cache behaviour."""

    node_resolution: float
    safe_distance: float
    reservation_horizon: int = 2
    cache_ttl_steps: int = 2
    packet_header_bytes: int = 40

    def __post_init__(self):
        if self.node_resolution <= 0 or self.safe_distance <= 0:
            raise ValueError("node_resolution and safe_distance must be positive")
        if not 1 <= int(self.reservation_horizon) <= 2:
            raise ValueError("reservation_horizon must be one or two steps")
        if int(self.cache_ttl_steps) < 0:
            raise ValueError("cache_ttl_steps cannot be negative")

class MotionIntentExchange:
    """Exchange minimal motion messages and filter local DARE candidates.

    Creates a compact direct packet on every contact edge, caches it at the
    receiver, and applies local current-position/vertex/edge checks before the
    global synchronous resolver handles any remaining simultaneous conflict.
    """

    def __init__(self, n_agents, config):
        if n_agents <= 0:
            raise ValueError("n_agents must be positive")
        self.n_agents = int(n_agents)
        self.config = config
        self.managers = [
            ReservationManager(
                robot_id,
                node_resolution=config.node_resolution,
                reservation_horizon=config.reservation_horizon,
                cache_ttl_steps=config.cache_ttl_steps,
                safe_distance=config.safe_distance,
                trail_avoid_radius=config.safe_distance,
                trail_penalty_weight=0.0,
            )
            for robot_id in range(self.n_agents)
        ]
        self.packets_sent = 0
        self.packets_delivered = 0
        self.bytes_sent = 0
        self.local_candidates_rejected = 0
        self.local_candidate_reorders = 0
        self.exchange_ms = 0.0
        self.filter_ms = 0.0

    def make_packet(self, sender_id, step, current_position, plan, priority):
        """Build one minimal one/two-step motion-intent packet."""

        sender_id = int(sender_id)
        if not 0 <= sender_id < self.n_agents:
            raise IndexError("sender_id is out of range")
        current = np.asarray(current_position, dtype=np.float32).reshape(2)
        plan_array = np.asarray(list(plan), dtype=np.float32).reshape(-1, 2)
        plan_array = plan_array[: self.config.reservation_horizon]
        if len(plan_array) == 0:
            plan_array = current.reshape(1, 2)
        byte_count = int(
            self.config.packet_header_bytes + current.nbytes + plan_array.nbytes + 8
        )
        self.packets_sent += 1
        self.bytes_sent += byte_count
        return {
            "type": "short_motion_intent",
            "sender_id": sender_id,
            "step": int(step),
            "current": current.copy(),
            "plan": plan_array.copy(),
            "priority": float(priority),
            "byte_count": byte_count,
        }

    def exchange(self, step, contact_pairs, current_positions, plans, priorities):
        """Send directed short plans across all current contact edges."""

        started_at = time.perf_counter()
        for first, second in contact_pairs:
            for sender, receiver in ((int(first), int(second)), (int(second), int(first))):
                packet = self.make_packet(
                    sender,
                    step=step,
                    current_position=current_positions[sender],
                    plan=plans[sender],
                    priority=float(priorities[sender]),
                )
                self.managers[receiver].receive_packet(packet)
                self.packets_delivered += 1
        self.exchange_ms += (time.perf_counter() - started_at) * 1000.0

    def filter_candidates(self, step, current_positions, candidate_lists, contact_pairs, remove_soft_costs_for=(), priorities=None):
        """Apply local hard reservations while retaining waiting as final fallback."""

        started_at = time.perf_counter()
        visible: list[list[np.ndarray]] = [[] for _ in range(self.n_agents)]
        for first, second in contact_pairs:
            first, second = int(first), int(second)
            visible[first].append(np.asarray(current_positions[second], dtype=np.float32))
            visible[second].append(np.asarray(current_positions[first], dtype=np.float32))

        soft_disabled = {int(robot_id) for robot_id in remove_soft_costs_for}
        priority_values = (
            [0.0] * self.n_agents if priorities is None else [float(v) for v in priorities]
        )
        if len(priority_values) != self.n_agents:
            raise ValueError("priorities must match n_agents")
        output: list[list[np.ndarray]] = []
        for robot_id in range(self.n_agents):
            original = [
                np.asarray(candidate, dtype=np.float32).copy()
                for candidate in candidate_lists[robot_id]
            ]
            filtered = self.managers[robot_id].filter_candidates(
                current_position=current_positions[robot_id],
                ordered_candidates=original,
                current_step=step,
                visible_peer_positions=visible[robot_id],
                ignore_peer_reservations=False,
                ignore_soft_costs=robot_id in soft_disabled,
                own_priority=priority_values[robot_id],
            )
            self.local_candidates_rejected += max(0, len(original) - len(filtered))
            if original and filtered and not np.allclose(original[0], filtered[0]):
                self.local_candidate_reorders += 1
            output.append(filtered)
        self.filter_ms += (time.perf_counter() - started_at) * 1000.0
        return output

    def metrics(self):
        """Return motion-message payload and local-filter diagnostics."""

        return {
            "motion_message_packets": int(self.packets_sent),
            "motion_message_packets_delivered": int(self.packets_delivered),
            "motion_message_bytes": int(self.bytes_sent),
            "motion_message_mean_bytes": (
                0.0 if self.packets_sent == 0 else self.bytes_sent / self.packets_sent
            ),
            "motion_local_candidates_rejected": int(self.local_candidates_rejected),
            "motion_local_candidate_reorders": int(self.local_candidate_reorders),
            "motion_exchange_total_ms": float(self.exchange_ms),
            "motion_filter_total_ms": float(self.filter_ms),
        }

