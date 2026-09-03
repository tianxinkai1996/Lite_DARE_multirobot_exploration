"""Per-robot sparse coverage memory and peer intention cache.

Records which tiles a robot has explored and which sparse coverage/goal
information was received from contacted peers. Supplies soft coordination
costs that discourage repeated exploration and duplicate goal pursuit
without modifying DARE's neural-policy input.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from parameter import UNKNOWN
from classes.multi_robot.trajectory_codec import decode_packet


Tile = Tuple[int, int]


def world_to_tile(position, tile_size):
    """Convert a continuous world position into an unbounded integer tile ID.

    This works even if the map size is unknown and coordinates are negative.
    Example with tile_size=4.0:
        ( 9.2, -3.1) -> ( 2, -1)
        (-0.5,  7.9) -> (-1,  1)
    """

    x, y = float(position[0]), float(position[1])
    s = float(tile_size)
    return (math.floor(x / s), math.floor(y / s))


def tile_to_center(tile, tile_size):
    """Return the world-coordinate centre of a tile.

    This is used only for scoring candidate movements against goal claims.
    """

    tx, ty = int(tile[0]), int(tile[1])
    s = float(tile_size)
    return np.asarray([(tx + 0.5) * s, (ty + 0.5) * s], dtype=np.float32)


class CoverageManager:
    """Per-robot sparse coverage memory and peer intention cache.

    A robot owns one CoverageManager. It records:
        1. local_covered_tiles: tiles this robot has locally explored;
        2. peer_covered_tiles: sparse tiles received from contacted peers;
        3. peer_goal_claims: short-term peer goal tiles, used to avoid duplicate
           movement towards the same frontier/region.

    It deliberately does not know the full map size or obstacle layout. The only
    coordinate assumption is that robots share a mission/world coordinate frame.
    """

    def __init__(self, robot_id, tile_size, coverage_threshold=0.5, max_delta_tiles=64, coverage_penalty_weight=3.0, goal_penalty_weight=10.0, goal_claim_radius_tiles=1, goal_claim_ttl_steps=8):
        self.robot_id = int(robot_id)
        self.tile_size = float(tile_size)
        self.coverage_threshold = float(coverage_threshold)
        self.max_delta_tiles = int(max_delta_tiles)
        self.coverage_penalty_weight = float(coverage_penalty_weight)
        self.goal_penalty_weight = float(goal_penalty_weight)
        self.goal_claim_radius_tiles = int(goal_claim_radius_tiles)
        self.goal_claim_ttl_steps = int(goal_claim_ttl_steps)

        # Tiles confirmed by this robot's own local map/sensing.
        self.local_covered_tiles: Set[Tile] = set()

        # For bandwidth saving: remember what has already been sent to each peer.
        self._sent_to_peer: Dict[int, Set[Tile]] = defaultdict(set)

        # Peer coverage and peer goal memories obtained only through contact.
        self.peer_covered_tiles: Dict[int, Set[Tile]] = defaultdict(set)
        self.peer_goal_claims: Dict[int, Tuple[Tile, int]] = {}

    # ------------------------------------------------------------------
    # Local coverage update
    # ------------------------------------------------------------------

    def update_current_tile(self, position):
        """Mark the tile containing the robot's current position as covered.

        This is the simplest and fastest coverage summary. It means:
            "I have physically visited and locally sensed around this node."
        """

        self.local_covered_tiles.add(world_to_tile(position, self.tile_size))

    def update_from_local_belief(self, local_belief, env, robot_position=None, sensor_range=None):
        """Update covered tiles from a robot's local belief map.

        A tile becomes covered when enough cells inside it are known, i.e. not
        UNKNOWN. This is more accurate than only recording the current tile.

        Args:
            local_belief:
                The robot's own local occupancy/belief map.
            env:
                MultiRobotEnv instance. Only coordinate conversion helpers are
                used here; this function does not expose env.ground_truth to DARE.
            robot_position:
                If provided, only a window around the robot is scanned. This is
                much faster than scanning the whole map every step.
            sensor_range:
                Radius of the scan window in metres. If omitted, env.sensor_range
                is used when available.
        """

        belief = np.asarray(local_belief)
        height, width = belief.shape

        # Limit the scan to the robot's sensor neighbourhood for speed. If no
        # position is supplied, fall back to scanning the full belief map.
        if robot_position is not None:
            radius = float(sensor_range if sensor_range is not None else getattr(env, "sensor_range", 0.0))
            cell_radius = int(math.ceil(radius / float(env.cell_size))) + 2
            centre = env.world_to_cell(robot_position)
            cx, cy = int(centre[0]), int(centre[1])
            x0 = max(0, cx - cell_radius)
            x1 = min(width - 1, cx + cell_radius)
            y0 = max(0, cy - cell_radius)
            y1 = min(height - 1, cy + cell_radius)
        else:
            x0, x1, y0, y1 = 0, width - 1, 0, height - 1

        # Count total and known cells per unbounded tile ID.
        total_by_tile: Dict[Tile, int] = defaultdict(int)
        known_by_tile: Dict[Tile, int] = defaultdict(int)

        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                world = env.cell_to_world((x, y))
                tile = world_to_tile(world, self.tile_size)
                total_by_tile[tile] += 1
                if int(belief[y, x]) != UNKNOWN:
                    known_by_tile[tile] += 1

        for tile, total in total_by_tile.items():
            if total <= 0:
                continue
            ratio = known_by_tile[tile] / float(total)
            if ratio >= self.coverage_threshold:
                self.local_covered_tiles.add(tile)

    # ------------------------------------------------------------------
    # Packet helpers
    # ------------------------------------------------------------------

    def coverage_delta_for_peer(self, peer_id):
        """Return unsent local covered tiles for a specific peer.

        The returned list is capped to max_delta_tiles to keep packets small.
        `mark_tiles_sent()` should be called only after the packet is actually
        created/sent.
        """

        already_sent = self._sent_to_peer[int(peer_id)]
        delta = sorted(self.local_covered_tiles - already_sent)
        return delta[: self.max_delta_tiles]

    def mark_tiles_sent(self, peer_id, tiles):
        """Record that these local tiles have been transmitted to peer_id."""

        self._sent_to_peer[int(peer_id)].update((int(x), int(y)) for x, y in tiles)

    def goal_claim_from_plan(self, plan):
        """Convert DARE's short future plan into a sparse goal claim.

        The last point of the short plan is treated as the currently intended
        exploration region. If DARE returns no plan, no goal is claimed.
        """

        plan_list = list(plan)
        if not plan_list:
            return None
        return world_to_tile(plan_list[-1], self.tile_size)

    def receive_packet(self, packet, current_step, node_resolution=None):
        """Receive sparse coverage and goal information from a contacted peer.

        The packet may be raw or compressed. Compressed packets are decoded here
        only to extract coverage tiles and goal claims. This does not alter the
        robot's local belief map.
        """

        if packet.get("mode") == "compressed":
            if node_resolution is None:
                raise ValueError("node_resolution is required to decode compressed coverage packets")
            decoded = decode_packet(packet, node_resolution)
        else:
            decoded = packet

        peer_id = int(decoded["sender_id"])
        if peer_id == self.robot_id:
            return

        for tile in decoded.get("coverage_tiles", []):
            self.peer_covered_tiles[peer_id].add((int(tile[0]), int(tile[1])))

        goal_tile = decoded.get("goal_tile")
        if goal_tile is not None:
            self.peer_goal_claims[peer_id] = (
                (int(goal_tile[0]), int(goal_tile[1])),
                int(decoded.get("step", current_step)),
            )

    def peer_covered_union(self):
        """Return the union of all sparse peer coverage tiles."""

        merged: Set[Tile] = set()
        for tiles in self.peer_covered_tiles.values():
            merged.update(tiles)
        return merged

    def active_goal_claims(self, current_step):
        """Return non-expired peer goal claims."""

        active: List[Tile] = []
        stale_peer_ids: List[int] = []
        for peer_id, (tile, step) in self.peer_goal_claims.items():
            if int(current_step) - int(step) <= self.goal_claim_ttl_steps:
                active.append(tile)
            else:
                stale_peer_ids.append(peer_id)
        for peer_id in stale_peer_ids:
            del self.peer_goal_claims[peer_id]
        return active

    # ------------------------------------------------------------------
    # Candidate scoring
    # ------------------------------------------------------------------

    def coverage_overlap_cost(self, candidate_position):
        """Soft cost for moving into a tile already covered by a peer."""

        tile = world_to_tile(candidate_position, self.tile_size)
        return 1.0 if tile in self.peer_covered_union() else 0.0

    def goal_claim_cost(self, candidate_position, current_step):
        """Strong cost for moving toward a tile already claimed by a peer.

        The cost is 1.0 when the candidate tile lies within a small Manhattan
        radius of a peer goal claim. This discourages two robots from pursuing
        the same frontier/region after they have exchanged intentions.
        """

        candidate_tile = world_to_tile(candidate_position, self.tile_size)
        radius = self.goal_claim_radius_tiles
        for goal_tile in self.active_goal_claims(current_step):
            manhattan = abs(candidate_tile[0] - goal_tile[0]) + abs(candidate_tile[1] - goal_tile[1])
            if manhattan <= radius:
                return 1.0
        return 0.0

    def coordination_cost(self, candidate_position, current_step):
        """Combined soft cost for reducing repeated exploration.

        ReservationManager uses this value in addition to DARE preference rank
        and hard collision reservations.
        """

        return (
            self.coverage_penalty_weight * self.coverage_overlap_cost(candidate_position)
            + self.goal_penalty_weight * self.goal_claim_cost(candidate_position, current_step)
        )