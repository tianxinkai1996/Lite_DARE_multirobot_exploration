from __future__ import annotations

import struct
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


Tile = Tuple[int, int]

# Path format: number_of_runs + start_grid_x + start_grid_y, then RLE deltas.
_HEADER = struct.Struct("<Hhh")  # number_of_runs, start_x, start_y
_RUN = struct.Struct("<bbB")     # dx, dy, repeat_count

# Tile-list format: number_of_runs + start_tile_x + start_tile_y, then RLE deltas.
_TILE_HEADER = struct.Struct("<Hhh")
_TILE_RUN = struct.Struct("<bbB")


def _points_array(points: Iterable[Sequence[float]]) -> np.ndarray:
    """Convert an iterable of 2-D points into an Nx2 float32 array."""

    arr = np.asarray(list(points), dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    return arr.reshape(-1, 2)


# ----------------------------------------------------------------------
# World-coordinate path compression
# ----------------------------------------------------------------------


def encode_path_rle(points: Iterable[Sequence[float]], node_resolution: float) -> bytes:
    """Encode node-aligned world points as start coordinate plus RLE deltas.

    This assumes paths are quantised to DARE's NODE_RESOLUTION. Each point is
    converted to integer grid coordinates by round(point / node_resolution).
    """

    path = _points_array(points)
    if len(path) == 0:
        return b""

    grid = np.rint(path / float(node_resolution)).astype(np.int16)
    if len(grid) == 1:
        return _HEADER.pack(0, int(grid[0, 0]), int(grid[0, 1]))

    deltas = np.diff(grid, axis=0)
    runs = []
    dx, dy = int(deltas[0, 0]), int(deltas[0, 1])
    repeat = 1

    for delta in deltas[1:]:
        ndx, ndy = int(delta[0]), int(delta[1])
        if ndx == dx and ndy == dy and repeat < 255:
            repeat += 1
        else:
            if not (-128 <= dx <= 127 and -128 <= dy <= 127):
                raise ValueError("Path delta is too large for compact codec")
            runs.append((dx, dy, repeat))
            dx, dy, repeat = ndx, ndy, 1

    if not (-128 <= dx <= 127 and -128 <= dy <= 127):
        raise ValueError("Path delta is too large for compact codec")
    runs.append((dx, dy, repeat))

    payload = bytearray(_HEADER.pack(len(runs), int(grid[0, 0]), int(grid[0, 1])))
    for run in runs:
        payload.extend(_RUN.pack(*run))
    return bytes(payload)


def decode_path_rle(payload: bytes, node_resolution: float) -> np.ndarray:
    """Decode a compressed path into world-coordinate points."""

    if not payload:
        return np.empty((0, 2), dtype=np.float32)
    if len(payload) < _HEADER.size:
        raise ValueError("Malformed compressed path payload")

    n_runs, x, y = _HEADER.unpack_from(payload, 0)
    expected_size = _HEADER.size + n_runs * _RUN.size
    if len(payload) != expected_size:
        raise ValueError("Compressed path payload has an unexpected size")

    points = [[x, y]]
    offset = _HEADER.size
    for _ in range(n_runs):
        dx, dy, repeat = _RUN.unpack_from(payload, offset)
        offset += _RUN.size
        for _ in range(repeat):
            x += dx
            y += dy
            points.append([x, y])

    return np.asarray(points, dtype=np.float32) * float(node_resolution)


# ----------------------------------------------------------------------
# Coverage tile compression
# ----------------------------------------------------------------------


def _normalise_tiles(tiles: Iterable[Sequence[int]]) -> List[Tile]:
    """Return sorted unique integer tile IDs."""

    unique = {(int(tile[0]), int(tile[1])) for tile in tiles}
    return sorted(unique)


def encode_tiles_delta_rle(tiles: Iterable[Sequence[int]]) -> bytes:
    """Encode sparse tile IDs as sorted coordinates plus RLE deltas.

    This is not a full map. It is a sparse list of explored tile IDs. Sorting
    gives deterministic deltas and good compression when nearby tiles are sent.
    """

    ordered = _normalise_tiles(tiles)
    if not ordered:
        return b""

    if len(ordered) == 1:
        return _TILE_HEADER.pack(0, ordered[0][0], ordered[0][1])

    runs = []
    prev_x, prev_y = ordered[0]
    dx = ordered[1][0] - prev_x
    dy = ordered[1][1] - prev_y
    repeat = 1
    prev_x, prev_y = ordered[1]

    for tile in ordered[2:]:
        ndx = tile[0] - prev_x
        ndy = tile[1] - prev_y
        if ndx == dx and ndy == dy and repeat < 255:
            repeat += 1
        else:
            if not (-128 <= dx <= 127 and -128 <= dy <= 127):
                # If sparse tiles are far apart, fall back to one-tile jumps in
                # separate runs by clipping is unsafe, so raise a clear error.
                raise ValueError("Coverage tile delta is too large for compact codec")
            runs.append((dx, dy, repeat))
            dx, dy, repeat = ndx, ndy, 1
        prev_x, prev_y = tile

    if not (-128 <= dx <= 127 and -128 <= dy <= 127):
        raise ValueError("Coverage tile delta is too large for compact codec")
    runs.append((dx, dy, repeat))

    payload = bytearray(_TILE_HEADER.pack(len(runs), ordered[0][0], ordered[0][1]))
    for run in runs:
        payload.extend(_TILE_RUN.pack(*run))
    return bytes(payload)


def decode_tiles_delta_rle(payload: bytes) -> List[Tile]:
    """Decode compressed sparse coverage tiles."""

    if not payload:
        return []
    if len(payload) < _TILE_HEADER.size:
        raise ValueError("Malformed compressed tile payload")

    n_runs, x, y = _TILE_HEADER.unpack_from(payload, 0)
    expected_size = _TILE_HEADER.size + n_runs * _TILE_RUN.size
    if len(payload) != expected_size:
        raise ValueError("Compressed tile payload has an unexpected size")

    tiles: List[Tile] = [(int(x), int(y))]
    offset = _TILE_HEADER.size
    for _ in range(n_runs):
        dx, dy, repeat = _TILE_RUN.unpack_from(payload, offset)
        offset += _TILE_RUN.size
        for _ in range(repeat):
            x += dx
            y += dy
            tiles.append((int(x), int(y)))
    return tiles


# ----------------------------------------------------------------------
# Packet construction
# ----------------------------------------------------------------------


def _trim_to_budget(
    trail: np.ndarray,
    plan: np.ndarray,
    coverage_tiles: List[Tile],
    goal_tile: Optional[Tile],
    node_resolution: float,
    budget: int,
) -> tuple[np.ndarray, np.ndarray, List[Tile], Optional[Tile], bytes, bytes, bytes]:
    """Prefer immediate safety and intention fields under a fixed byte budget.

    Priority order:
        1. current metadata + goal claim;
        2. short plan, because it avoids collisions;
        3. coverage delta, because it reduces repeated exploration;
        4. recent trail, useful but less precise than coverage tiles.
    """

    trail = trail.copy()
    plan = plan.copy()
    coverage_tiles = list(coverage_tiles)

    while True:
        trail_blob = encode_path_rle(trail, node_resolution) if len(trail) else b""
        plan_blob = encode_path_rle(plan, node_resolution) if len(plan) else b""
        coverage_blob = encode_tiles_delta_rle(coverage_tiles) if coverage_tiles else b""
        # Approximate metadata bytes: sender, step, current position, goal tile.
        metadata_bytes = 32 if goal_tile is not None else 28
        total = metadata_bytes + len(trail_blob) + len(plan_blob) + len(coverage_blob)
        if total <= budget:
            return trail, plan, coverage_tiles, goal_tile, trail_blob, plan_blob, coverage_blob

        # Drop oldest/lower-priority information first.
        if len(trail) > 1:
            trail = trail[1:]
        elif len(coverage_tiles) > 1:
            coverage_tiles = coverage_tiles[: max(1, len(coverage_tiles) // 2)]
        elif len(plan) > 1:
            plan = plan[:-1]
        elif goal_tile is not None:
            goal_tile = None
        else:
            return trail, plan, coverage_tiles, goal_tile, trail_blob, plan_blob, coverage_blob


def make_compressed_packet(
    *,
    sender_id: int,
    step: int,
    current_position: Sequence[float],
    trail: Iterable[Sequence[float]],
    plan: Iterable[Sequence[float]],
    node_resolution: float,
    trail_steps: int,
    plan_steps: int,
    packet_budget_bytes: int,
    coverage_tiles: Optional[Iterable[Sequence[int]]] = None,
    goal_tile: Optional[Sequence[int]] = None,
) -> dict:
    """Create a compressed direct one-hop message.

    coverage_tiles and goal_tile are optional for backwards compatibility. If
    they are omitted, this behaves like the original trail+plan packet.
    """

    trail_arr = _points_array(trail)[-int(trail_steps):]
    plan_arr = _points_array(plan)[:int(plan_steps)]
    cov = _normalise_tiles(coverage_tiles or [])
    requested_trail_points = int(len(trail_arr))
    requested_plan_points = int(len(plan_arr))
    requested_coverage_tiles = int(len(cov))
    goal = None if goal_tile is None else (int(goal_tile[0]), int(goal_tile[1]))
    requested_goal_claim = goal is not None

    trail_arr, plan_arr, cov, goal, trail_blob, plan_blob, coverage_blob = _trim_to_budget(
        trail_arr,
        plan_arr,
        cov,
        goal,
        node_resolution,
        int(packet_budget_bytes),
    )

    metadata_bytes = 28
    trail_bytes = len(trail_blob)
    plan_bytes = len(plan_blob)
    coverage_bytes = len(coverage_blob)
    goal_bytes = 4 if goal is not None else 0
    byte_count = metadata_bytes + trail_bytes + plan_bytes + coverage_bytes + goal_bytes
    raw_equivalent_byte_count = (
        metadata_bytes
        + int(trail_arr.nbytes)
        + int(plan_arr.nbytes)
        + int(len(cov) * 4)
        + goal_bytes
    )
    return {
        "mode": "compressed",
        "sender_id": int(sender_id),
        "step": int(step),
        "current": np.asarray(current_position, dtype=np.float32).copy(),
        "trail_blob": trail_blob,
        "plan_blob": plan_blob,
        "coverage_blob": coverage_blob,
        "goal_tile": goal,
        "byte_count": int(byte_count),
        "raw_equivalent_byte_count": int(raw_equivalent_byte_count),
        "compression_ratio_vs_same_content_raw": (
            1.0
            if raw_equivalent_byte_count <= 0
            else float(byte_count / raw_equivalent_byte_count)
        ),
        "metadata_bytes": int(metadata_bytes),
        "trail_payload_bytes": int(trail_bytes),
        "plan_payload_bytes": int(plan_bytes),
        "coverage_payload_bytes": int(coverage_bytes),
        "goal_payload_bytes": int(goal_bytes),
        "trail_point_count": int(len(trail_arr)),
        "plan_point_count": int(len(plan_arr)),
        "coverage_tile_count": int(len(cov)),
        "goal_claim_count": int(goal is not None),
        "packet_budget_truncated": bool(
            len(trail_arr) < requested_trail_points
            or len(plan_arr) < requested_plan_points
            or len(cov) < requested_coverage_tiles
            or (requested_goal_claim and goal is None)
        ),
        "dropped_trail_points": int(requested_trail_points - len(trail_arr)),
        "dropped_plan_points": int(requested_plan_points - len(plan_arr)),
        "dropped_coverage_tiles": int(requested_coverage_tiles - len(cov)),
        "dropped_goal_claims": int(requested_goal_claim and goal is None),
        # Store these for the caller so it can mark exactly what was sent.
        "sent_coverage_tiles": cov,
    }


def make_raw_packet(
    *,
    sender_id: int,
    step: int,
    current_position: Sequence[float],
    trail: Iterable[Sequence[float]],
    plan: Iterable[Sequence[float]],
    trail_steps: int,
    plan_steps: int,
    coverage_tiles: Optional[Iterable[Sequence[int]]] = None,
    goal_tile: Optional[Sequence[int]] = None,
) -> dict:
    """Uncompressed baseline message used for comparison experiments."""

    trail_arr = _points_array(trail)[-int(trail_steps):]
    plan_arr = _points_array(plan)[:int(plan_steps)]
    cov = _normalise_tiles(coverage_tiles or [])
    goal = None if goal_tile is None else (int(goal_tile[0]), int(goal_tile[1]))
    # Raw byte count estimates int16 tile pairs and float32 path coordinates.
    metadata_bytes = 28
    trail_bytes = int(trail_arr.nbytes)
    plan_bytes = int(plan_arr.nbytes)
    coverage_bytes = int(len(cov) * 4)
    goal_bytes = 4 if goal is not None else 0
    byte_count = metadata_bytes + trail_bytes + plan_bytes + coverage_bytes + goal_bytes
    return {
        "mode": "raw",
        "sender_id": int(sender_id),
        "step": int(step),
        "current": np.asarray(current_position, dtype=np.float32).copy(),
        "trail": trail_arr,
        "plan": plan_arr,
        "coverage_tiles": cov,
        "goal_tile": goal,
        "byte_count": int(byte_count),
        "raw_equivalent_byte_count": int(byte_count),
        "compression_ratio_vs_same_content_raw": 1.0,
        "metadata_bytes": int(metadata_bytes),
        "trail_payload_bytes": int(trail_bytes),
        "plan_payload_bytes": int(plan_bytes),
        "coverage_payload_bytes": int(coverage_bytes),
        "goal_payload_bytes": int(goal_bytes),
        "trail_point_count": int(len(trail_arr)),
        "plan_point_count": int(len(plan_arr)),
        "coverage_tile_count": int(len(cov)),
        "goal_claim_count": int(goal is not None),
        "packet_budget_truncated": False,
        "dropped_trail_points": 0,
        "dropped_plan_points": 0,
        "dropped_coverage_tiles": 0,
        "dropped_goal_claims": 0,
        "sent_coverage_tiles": cov,
    }


def decode_packet(packet: dict, node_resolution: float) -> dict:
    """Decode packet fields into arrays and sparse tile lists."""

    mode = packet.get("mode")
    if mode == "compressed":
        trail = decode_path_rle(packet.get("trail_blob", b""), node_resolution)
        plan = decode_path_rle(packet.get("plan_blob", b""), node_resolution)
        coverage_tiles = decode_tiles_delta_rle(packet.get("coverage_blob", b""))
        goal_tile = packet.get("goal_tile")
        if goal_tile is not None:
            goal_tile = (int(goal_tile[0]), int(goal_tile[1]))
    elif mode == "raw":
        trail = _points_array(packet.get("trail", []))
        plan = _points_array(packet.get("plan", []))
        coverage_tiles = _normalise_tiles(packet.get("coverage_tiles", []))
        goal_tile = packet.get("goal_tile")
        if goal_tile is not None:
            goal_tile = (int(goal_tile[0]), int(goal_tile[1]))
    else:
        raise ValueError(f"Unknown packet mode: {mode}")

    return {
        "sender_id": int(packet["sender_id"]),
        "step": int(packet["step"]),
        "current": np.asarray(packet["current"], dtype=np.float32).copy(),
        "trail": trail,
        "plan": plan,
        "coverage_tiles": coverage_tiles,
        "goal_tile": goal_tile,
        "byte_count": int(packet.get("byte_count", 0)),
    }