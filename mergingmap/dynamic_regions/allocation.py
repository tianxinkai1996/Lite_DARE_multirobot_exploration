"""Normalised graph-aware frontier allocation and claim exchange.

"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .extraction import (
    _region_matches_claim,
    _regions_equivalent,
    _world_distance,
)
from .models import FrontierRegion, PeerClaim

def _normalised(values):
    """Min-max normalise finite values and preserve infinity as infinity.

    English implementation: normalises finite values to [0, 1]; identical values
    map to zero, while unreachable infinities remain infinite.
    """

    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return [math.inf for _ in values]
    low, high = min(finite), max(finite)
    span = high - low
    result = []
    for value in values:
        value = float(value)
        if not math.isfinite(value):
            result.append(math.inf)
        elif span <= 1e-12:
            result.append(0.0)
        else:
            result.append((value - low) / span)
    return result

class RegionAllocationMixin:
    """Provide contact-limited claim exchange and graph-aware assignment."""

    def _graph_assignment_distance(self, robot_id, robot, position, region):
        """Return graph distance to a region, or Euclidean distance in lightweight tests.

        English implementation: invokes the routing mixin when a graph is supplied,
        otherwise uses a deterministic Euclidean fallback only for isolated tests.
        """

        if robot is None:
            return _world_distance(position, region.target_world)
        distances = self._graph_distance_map(robot_id, robot, region.target_world)
        return self._nearest_graph_distance(position, distances)

    def _resolve_component_conflicts(self, component, positions, step, robots=None, wait_steps=None):
        """Resolve duplicate leases by validity, graph distance, progress, wait, and ID.

        English implementation: ranks owners with the methodology's deterministic
        validity/distance/progress/wait/ID criteria and releases every loser.
        """

        active = [rid for rid in component if self.states[rid].lease is not None]
        wait_values = [0] * self.n_robots if wait_steps is None else list(wait_steps)

        def owner_key(robot_id):
            lease = self.states[robot_id].lease
            assert lease is not None
            robot = None if robots is None else robots[robot_id]
            distance = self._graph_assignment_distance(
                robot_id, robot, positions[robot_id], lease.region
            )
            valid_penalty = 0 if lease.expiry_step >= step else 1
            no_progress_age = max(0, int(step) - int(lease.last_progress_step))
            return (
                valid_penalty,
                distance,
                no_progress_age,
                -int(wait_values[robot_id]),
                int(robot_id),
            )

        active.sort(key=owner_key)
        kept: List[int] = []
        for robot_id in active:
            lease = self.states[robot_id].lease
            assert lease is not None
            winner = next(
                (
                    owner
                    for owner in kept
                    if _regions_equivalent(
                        lease.region,
                        self.states[owner].lease.region,
                        self.config,
                    )
                ),
                None,
            )
            if winner is None:
                kept.append(robot_id)
                continue
            self.claim_conflicts_resolved += 1
            self._emit(
                "claim_conflict_resolved",
                step=int(step),
                winner_robot_id=int(winner),
                loser_robot_id=int(robot_id),
                region_id=lease.region_id,
            )
            self._release(robot_id, step=step, reason="conflict")

    def _claim_packet_bytes(self, claim):
        """Estimate algorithm-layer payload bytes for one compact lease message."""

        return int(
            self.config.region_message_header_bytes
            + len(claim.region_id.encode("utf-8"))
            + 4  # owner
            + 8  # expiry
            + 16  # centroid
            + 16  # bbox
        )

    def _exchange_component_claims(self, component, step):
        """Exchange compact region ID and lease expiry inside one contact component."""

        if len(component) <= 1:
            return
        claims: List[PeerClaim] = []
        for owner_id in component:
            lease = self.states[owner_id].lease
            if lease is None:
                continue
            claims.append(
                PeerClaim(
                    owner_robot_id=owner_id,
                    region_id=lease.region_id,
                    centroid_world=lease.region.centroid_world,
                    bbox_cell=lease.region.bbox_cell,
                    expiry_step=step + self.config.claim_ttl_steps,
                )
            )

        for receiver_id in component:
            state = self.states[receiver_id]
            for claim in claims:
                if claim.owner_robot_id == receiver_id:
                    continue
                state.known_peer_claims[claim.region_id] = claim
                self.claim_messages += 1
                self.region_message_packets += 1
                self.region_message_bytes += self._claim_packet_bytes(claim)

    def _lease_penalty(self, robot_id, region, step):
        """Return one when a valid peer lease matches the candidate region."""

        return float(
            any(
                claim.expiry_step >= step
                and _region_matches_claim(region, claim, self.config)
                for claim in self.states[robot_id].known_peer_claims.values()
            )
        )

    def _assign_component(self, component, positions, step, robots=None):
        """Assign distinct feasible regions using normalised graph-aware costs.

        English implementation: computes all five methodology terms, normalises them
        jointly, and greedily selects the globally lowest feasible pair per iteration.
        """

        unassigned = {
            robot_id for robot_id in component if self.states[robot_id].lease is None
        }
        used_regions = [
            self.states[robot_id].lease.region
            for robot_id in component
            if self.states[robot_id].lease is not None
        ]

        while unassigned:
            rows: list[dict] = []
            for robot_id in sorted(unassigned):
                state = self.states[robot_id]
                robot = None if robots is None else robots[robot_id]
                for region in state.regions.values():
                    if any(
                        _regions_equivalent(region, used, self.config)
                        for used in used_regions
                    ):
                        continue
                    distance = self._graph_assignment_distance(
                        robot_id, robot, positions[robot_id], region
                    )
                    if not math.isfinite(distance):
                        continue
                    first_seen = state.first_seen_step.get(region.region_id, step)
                    age = max(0, int(step) - int(first_seen))
                    self.max_unassigned_region_age = max(
                        self.max_unassigned_region_age, age
                    )
                    rows.append(
                        {
                            "robot_id": robot_id,
                            "region": region,
                            "distance": distance,
                            "utility": float(region.utility),
                            "age": float(age),
                            "lease": self._lease_penalty(
                                robot_id, region, step=step
                            ),
                            "stall": float(
                                state.stall_steps
                                + state.region_failure_counts.get(region.region_id, 0)
                            ),
                        }
                    )
            if not rows:
                break

            for key in ("distance", "utility", "age", "lease", "stall"):
                normalised = _normalised([row[key] for row in rows])
                for row, value in zip(rows, normalised):
                    row[f"normalised_{key}"] = value

            for row in rows:
                row["cost"] = (
                    self.config.distance_weight * row["normalised_distance"]
                    - self.config.utility_weight * row["normalised_utility"]
                    - self.config.age_weight * row["normalised_age"]
                    + self.config.lease_penalty_weight * row["normalised_lease"]
                    + self.config.stall_penalty_weight * row["normalised_stall"]
                    + 1e-9 * row["robot_id"]
                )
            best = min(
                rows,
                key=lambda row: (
                    row["cost"],
                    row["robot_id"],
                    row["region"].region_id,
                ),
            )
            robot_id = int(best["robot_id"])
            region = best["region"]
            self._assign(robot_id, region, step=step)
            used_regions.append(region)
            unassigned.remove(robot_id)

