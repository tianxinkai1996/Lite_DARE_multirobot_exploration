"""Local trajectory-reservation conflict filtering.

"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np

from classes.multi_robot.trajectory_codec import decode_packet

class ReservationManager:
    """Filter local candidates with short-horizon peer reservations.

    CoverageManager 

    English purpose: prevent local motion conflicts using recently communicated
    teammate plans. English implementation: caches each peer's newest packet,
    removes candidates violating hard reservations, and reranks safe candidates
    with trail and optional coverage/goal costs.
    """

    # Purpose: initialise this robot's cache, reservation horizon, and safety thresholds.
    # English purpose: Initialise peer packet cache, reservation horizon, and safety thresholds.
    def __init__(self, robot_id, node_resolution, reservation_horizon, cache_ttl_steps, safe_distance, trail_avoid_radius, trail_penalty_weight):
        """Initialise one robot's local reservation cache.

        English purpose: configure the local peer-reservation filter.
        English implementation: stores horizon and safety parameters and creates
        the newest-packet cache keyed by peer robot ID.
        """

        self.robot_id = int(robot_id)
        self.node_resolution = float(node_resolution)
        self.reservation_horizon = int(reservation_horizon)
        self.cache_ttl_steps = int(cache_ttl_steps)
        self.safe_distance = float(safe_distance)
        self.trail_avoid_radius = float(trail_avoid_radius)
        self.trail_penalty_weight = float(trail_penalty_weight)

        # Latest decoded packet from each directly contacted peer.
        self._packets: dict[int, dict] = {}

    # Purpose: receive and retain the peer's latest short-horizon trajectory packet.
    # English purpose: Receive and retain the newest short-horizon packet from a peer.
    def receive_packet(self, packet):
        """Store the newest valid packet from a directly contacted peer.

        English purpose: retain the newest peer state for later candidate filtering.
        English implementation: decodes the packet, ignores self-messages, and
        replaces the cache entry only when its step is not older.
        """

        if packet.get("type") == "short_motion_intent":
            decoded = {
                "sender_id": int(packet["sender_id"]),
                "step": int(packet["step"]),
                "current": np.asarray(packet["current"], dtype=np.float32).copy(),
                "plan": np.asarray(packet.get("plan", []), dtype=np.float32).reshape(-1, 2),
                "trail": np.empty((0, 2), dtype=np.float32),
                "priority": float(packet.get("priority", 0.0)),
                "byte_count": int(packet.get("byte_count", 0)),
            }
        else:
            decoded = decode_packet(packet, self.node_resolution)
        peer_id = decoded["sender_id"]
        if peer_id == self.robot_id:
            return
        previous = self._packets.get(peer_id)
        if previous is None or decoded["step"] >= previous["step"]:
            self._packets[peer_id] = decoded

    # Purpose: drop peer short-horizon plans older than the validity window.
    # English purpose: Remove peer short-term plans that exceeded their cache TTL.
    def prune(self, current_step):
        """Remove short-term plans whose cache lifetime has expired.

        ``cache_ttl_steps`` 
        CoverageManager 
        English purpose: stop stale peer plans from blocking current candidates.
        English implementation: removes packets older than the configured TTL;
        long-lived coverage knowledge remains outside this module.
        """

        stale = [
            peer_id
            for peer_id, packet in self._packets.items()
            if int(current_step) - int(packet["step"]) > self.cache_ttl_steps
        ]
        for peer_id in stale:
            del self._packets[peer_id]

    # Purpose: read a peer's previous and reserved nodes at a given future time.
    # English purpose: Read a peer previous and reserved node at one future time.
    def _peer_plan_at(self, packet, time_step):
        """Return a peer edge reservation for one future time step.

        Returns ``None`` when no reservation covers the requested step.
        Converts absolute time to a plan index and returns the previous/next
        nodes when the reservation is available.
        """

        # plan[0] is intended for packet_step + 1.
        plan_index = int(time_step - int(packet["step"]) - 1)
        plan = packet["plan"][: self.reservation_horizon]
        if plan_index < 0 or plan_index >= len(plan):
            return None
        previous = packet["current"] if plan_index == 0 else plan[plan_index - 1]
        return np.asarray(previous, dtype=np.float32), np.asarray(plan[plan_index], dtype=np.float32)

    @staticmethod
    # Purpose: detect two agents swapping edge endpoints at the same time step.
    # English purpose: Detect two robots swapping edge endpoints in one time step.
    def _segment_swap(current, candidate, peer_previous, peer_next, tolerance):
        """Detect a synchronous edge swap between two robots.

        English purpose: reject opposite traversal of the same edge in one step.
        English implementation: checks both crossed endpoint distances against
        the configured tolerance.
        """

        return (
            np.linalg.norm(current - peer_next) < tolerance
            and np.linalg.norm(candidate - peer_previous) < tolerance
        )

    # Purpose: detect hard conflicts from visible robots, same-node reservations, and edge swaps.
    # English purpose: Detect hard conflicts from visible peers, vertex reservations, and edge swaps.
    def _hard_conflict(self, current, candidate, current_step, visible_peer_positions, ignore_peer_reservations=False, own_priority=0.0):
        """Return whether one candidate violates a local hard safety rule.

        English purpose: detect immediate and reserved hard conflicts for a move.
        English implementation: checks visible peers first, then cached vertex and
        edge reservations unless peer-plan filtering is explicitly disabled.
        """

        # A locally visible teammate is treated as a dynamic obstacle.
        for peer_pos in visible_peer_positions:
            if np.linalg.norm(candidate - np.asarray(peer_pos, dtype=np.float32)) < self.safe_distance:
                return True

        if ignore_peer_reservations:
            return False

        next_time = int(current_step) + 1
        for peer_id, packet in self._packets.items():
            peer_priority = float(packet.get("priority", 0.0))
            peer_wins = (
                peer_priority > float(own_priority)
                or (
                    abs(peer_priority - float(own_priority)) <= 1e-12
                    and int(peer_id) < self.robot_id
                )
            )
            if not peer_wins:
                continue
            time_reservation = self._peer_plan_at(packet, next_time)
            if time_reservation is None:
                continue
            peer_previous, peer_next = time_reservation
            # Same destination at the same time.
            if np.linalg.norm(candidate - peer_next) < self.safe_distance:
                return True
            # A -> B while peer reserves B -> A.
            if self._segment_swap(current, candidate, peer_previous, peer_next, self.safe_distance):
                return True
        return False

    # Purpose: compute a soft penalty for approaching a peer's recent trail.
    # English purpose: Compute a soft penalty near recently communicated peer trails.
    def _trail_penalty(self, candidate):
        """Compute a soft cost near recently communicated peer trails.

        English purpose: discourage, rather than forbid, motion near peer trails.
        English implementation: accumulates a linear distance penalty for every
        trail point inside the avoidance radius.
        """

        penalty = 0.0
        for packet in self._packets.values():
            for point in packet["trail"]:
                distance = float(np.linalg.norm(candidate - point))
                if distance < self.trail_avoid_radius:
                    penalty += 1.0 - distance / self.trail_avoid_radius
        return penalty

    # DARE 
    # English purpose: Filter hard conflicts and rerank DARE candidates by coordination cost.
    def filter_candidates(self, current_position, ordered_candidates, current_step, visible_peer_positions, coverage_manager=None, ignore_peer_reservations=False, ignore_soft_costs=False, own_priority=0.0):
        """Filter unsafe candidates and rerank the remaining DARE actions.

        DARE 

        English purpose: preserve DARE preference while applying communicated
        collision constraints and optional cooperation costs. English implementation:
        prunes stale packets, removes hard conflicts, adds soft costs to the original
        rank, and retains waiting as a high-cost fallback.
        """

        self.prune(current_step)
        current = np.asarray(current_position, dtype=np.float32)
        scored = []

        for rank, candidate in enumerate(ordered_candidates):
            candidate = np.asarray(candidate, dtype=np.float32)
            is_wait = np.allclose(candidate, current)

            # Waiting is always retained as a final fallback. Moving actions
            # must pass hard physical/reservation checks.
            if not is_wait and self._hard_conflict(
                current,
                candidate,
                current_step,
                visible_peer_positions,
                ignore_peer_reservations=ignore_peer_reservations,
                own_priority=own_priority,
            ):
                continue

            # Base rank keeps DARE as the primary planner. Coordination terms
            # only reorder candidates around DARE's preference.
            score = float(rank)
            if not ignore_soft_costs:
                score += self.trail_penalty_weight * self._trail_penalty(candidate)

                if coverage_manager is not None:
                    score += float(coverage_manager.coordination_cost(candidate, current_step))

            # Waiting should be possible but not selected unless alternatives
            # are worse or unsafe.
            if is_wait:
                score += 1e3

            scored.append((score, candidate))

        if not scored:
            return [current.copy()]

        scored.sort(key=lambda item: item[0])
        return [candidate for _, candidate in scored]