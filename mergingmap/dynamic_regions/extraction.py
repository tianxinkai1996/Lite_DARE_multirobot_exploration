"""Frontier extraction and region-matching helpers.

前沿提取与区域匹配辅助函数。
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import replace
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .models import (
    Cell,
    DynamicRegionConfig,
    FrontierRegion,
    PeerClaim,
    WorldPoint,
)

# 中文目的：把地图网格坐标转换为环境世界坐标。
# English purpose: Convert a map cell into an environment world coordinate.
def _cell_to_world(env: object, cell: Cell) -> WorldPoint:
    if not hasattr(env, "cell_to_world"):
        raise AttributeError(
            "The environment must provide cell_to_world() for region targets"
        )
    point = np.asarray(env.cell_to_world(cell), dtype=float).reshape(-1)
    if point.size < 2 or not np.all(np.isfinite(point[:2])):
        raise ValueError(f"Invalid world point for cell {cell}: {point}")
    return float(point[0]), float(point[1])


# 中文目的：识别与未知区域相邻的自由网格。
# English purpose: Identify free cells adjacent to unknown space.
def _frontier_mask(
    occupancy: np.ndarray,
    *,
    free_value: int,
    unknown_value: int,
) -> np.ndarray:
    """Return free cells that are 4-neighbour adjacent to unknown space."""
    free = occupancy == int(free_value)
    unknown = occupancy == int(unknown_value)
    adjacent_unknown = np.zeros_like(free, dtype=bool)

    adjacent_unknown[1:, :] |= unknown[:-1, :]
    adjacent_unknown[:-1, :] |= unknown[1:, :]
    adjacent_unknown[:, 1:] |= unknown[:, :-1]
    adjacent_unknown[:, :-1] |= unknown[:, 1:]

    return free & adjacent_unknown


# 中文目的：提取前沿掩码中的确定性八连通分量。
# English purpose: Extract deterministic 8-connected components from a frontier mask.
def _connected_components(mask: np.ndarray) -> List[List[Cell]]:
    """Extract deterministic 8-connected components as (x, y) cells."""
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: List[List[Cell]] = []
    neighbours = (
        (-1, -1), (0, -1), (1, -1),
        (-1, 0),             (1, 0),
        (-1, 1),  (0, 1),   (1, 1),
    )

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue

            queue = deque([(x, y)])
            visited[y, x] = True
            component: List[Cell] = []

            while queue:
                cx, cy = queue.popleft()
                component.append((cx, cy))
                for dx, dy in neighbours:
                    nx, ny = cx + dx, cy + dy
                    if (
                        0 <= nx < width
                        and 0 <= ny < height
                        and mask[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        queue.append((nx, ny))

            components.append(component)

    return components


# 中文目的：选择空间分离的确定性分区种子。
# English purpose: Choose deterministic spatially separated partition seeds.
def _farthest_seeds(cells: Sequence[Cell], count: int) -> List[Cell]:
    """Choose deterministic spatially separated seeds."""
    ordered = sorted(cells, key=lambda cell: (cell[1], cell[0]))
    seeds = [ordered[0]]

    while len(seeds) < count:
        best_cell = None
        best_distance = -1.0
        for cell in ordered:
            if cell in seeds:
                continue
            minimum = min(
                (cell[0] - seed[0]) ** 2 + (cell[1] - seed[1]) ** 2
                for seed in seeds
            )
            if minimum > best_distance:
                best_distance = float(minimum)
                best_cell = cell
        if best_cell is None:
            break
        seeds.append(best_cell)

    return seeds


# 中文目的：把过大的前沿连通分量拆分为较小区域。
# English purpose: Split an oversized frontier component into smaller regions.
def _partition_component(
    cells: Sequence[Cell],
    max_cells: int,
) -> List[List[Cell]]:
    """Split a large component by deterministic multi-source graph growth."""
    if len(cells) <= max_cells:
        return [list(cells)]

    partition_count = int(math.ceil(len(cells) / float(max_cells)))
    seeds = _farthest_seeds(cells, partition_count)
    cell_set = set(cells)
    labels: Dict[Cell, int] = {}
    queue: deque[Tuple[Cell, int]] = deque()

    for label, seed in enumerate(seeds):
        labels[seed] = label
        queue.append((seed, label))

    neighbours = (
        (-1, -1), (0, -1), (1, -1),
        (-1, 0),             (1, 0),
        (-1, 1),  (0, 1),   (1, 1),
    )

    while queue:
        (cx, cy), label = queue.popleft()
        for dx, dy in neighbours:
            neighbour = (cx + dx, cy + dy)
            if neighbour in cell_set and neighbour not in labels:
                labels[neighbour] = label
                queue.append((neighbour, label))

    clusters: List[List[Cell]] = [[] for _ in seeds]
    for cell in sorted(cells, key=lambda value: (value[1], value[0])):
        label = labels.get(cell)
        if label is None:
            label = min(
                range(len(seeds)),
                key=lambda index: (
                    (cell[0] - seeds[index][0]) ** 2
                    + (cell[1] - seeds[index][1]) ** 2
                ),
            )
        clusters[label].append(cell)

    return [cluster for cluster in clusters if cluster]


# 中文目的：依据量化位置与边界框生成稳定区域标识。
# English purpose: Create a stable region ID from quantised location and bounds.
def _make_region_id(
    cells: Sequence[Cell],
    *,
    quantization_cells: int,
) -> str:
    points = np.asarray(cells, dtype=float)
    centroid = points.mean(axis=0)
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    q = float(max(1, quantization_cells))

    values = (
        int(round(centroid[0] / q)),
        int(round(centroid[1] / q)),
        int(round(x0 / q)),
        int(round(y0 / q)),
        int(round(x1 / q)),
        int(round(y1 / q)),
    )
    return "region_" + "_".join(str(value) for value in values)


# 中文目的：从机器人当前融合占据图提取动态前沿任务。
# English purpose: Extract dynamic frontier tasks from a robot merged occupancy belief.
def extract_frontier_regions(
    occupancy: np.ndarray,
    *,
    env: object,
    config: DynamicRegionConfig,
) -> List[FrontierRegion]:
    """Extract current temporary frontier tasks from one merged belief."""
    grid = np.asarray(occupancy)
    if grid.ndim != 2:
        raise ValueError(f"Expected a 2-D occupancy grid, got {grid.shape}")

    mask = _frontier_mask(
        grid,
        free_value=config.free_value,
        unknown_value=config.unknown_value,
    )
    components = _connected_components(mask)
    regions: List[FrontierRegion] = []

    for component in components:
        if len(component) < config.min_frontier_cells:
            continue

        for partition in _partition_component(
            component,
            config.max_frontier_cells_per_region,
        ):
            if len(partition) < config.min_frontier_cells:
                continue

            points = np.asarray(partition, dtype=float)
            centroid = points.mean(axis=0)
            target = min(
                partition,
                key=lambda cell: (
                    (cell[0] - centroid[0]) ** 2
                    + (cell[1] - centroid[1]) ** 2,
                    cell[1],
                    cell[0],
                ),
            )
            x0, y0 = points.min(axis=0).astype(int)
            x1, y1 = points.max(axis=0).astype(int)
            centroid_cell = (float(centroid[0]), float(centroid[1]))
            centroid_world = _cell_to_world(
                env,
                (int(round(centroid[0])), int(round(centroid[1]))),
            )
            target_world = _cell_to_world(env, target)
            region_id = _make_region_id(
                partition,
                quantization_cells=config.region_id_quantization_cells,
            )

            regions.append(
                FrontierRegion(
                    region_id=region_id,
                    frontier_cells=frozenset(partition),
                    centroid_cell=centroid_cell,
                    centroid_world=centroid_world,
                    target_cell=target,
                    target_world=target_world,
                    bbox_cell=(int(x0), int(y0), int(x1), int(y1)),
                    utility=float(len(partition)),
                )
            )

    regions.sort(
        key=lambda region: (
            region.centroid_cell[1],
            region.centroid_cell[0],
            region.region_id,
        )
    )
    return regions


# 中文目的：在连续规划步之间保持动态前沿区域的稳定身份。
# English purpose: Preserve dynamic frontier identity across consecutive planning steps.
def track_frontier_regions(
    previous_regions: Sequence[FrontierRegion],
    current_regions: Sequence[FrontierRegion],
    config: DynamicRegionConfig,
) -> List[FrontierRegion]:
    """Greedily match new frontiers to old identities by IoU and centroid distance.

    中文实现：为所有满足阈值的旧/新区对计算
    ``w_I * IoU - w_mu * normalised_distance``，按分数从高到低做一对一匹配，
    匹配成功时保留旧 ``region_id``，未匹配区域使用新生成的量化标识。

    English implementation: scores every admissible old/new pair, performs a
    deterministic one-to-one greedy match, and copies the old identifier onto the
    matched new geometry so ages, leases, and progress records survive small changes.
    """

    previous = list(previous_regions)
    current = list(current_regions)
    if not previous or not current:
        return current

    candidates: List[Tuple[float, str, int, int]] = []
    scale = max(1.0, float(config.region_match_centroid_cells))
    for old_index, old_region in enumerate(previous):
        for new_index, new_region in enumerate(current):
            iou = _region_iou(old_region, new_region)
            distance = _centroid_cell_distance(old_region, new_region)
            if (
                iou < config.region_match_iou_threshold
                and distance > config.region_match_centroid_cells
            ):
                continue
            score = (
                float(config.tracking_iou_weight) * iou
                - float(config.tracking_centroid_weight) * distance / scale
            )
            candidates.append(
                (-score, old_region.region_id, old_index, new_index)
            )

    matched_old: set[int] = set()
    matched_new: set[int] = set()
    output = list(current)
    for _, _, old_index, new_index in sorted(candidates):
        if old_index in matched_old or new_index in matched_new:
            continue
        output[new_index] = replace(
            current[new_index],
            region_id=previous[old_index].region_id,
        )
        matched_old.add(old_index)
        matched_new.add(new_index)

    output.sort(
        key=lambda region: (
            region.centroid_cell[1],
            region.centroid_cell[0],
            region.region_id,
        )
    )
    return output


# 中文目的：计算两个区域前沿网格集合的交并比。
# English purpose: Compute frontier-cell intersection over union for two regions.
def _region_iou(first: FrontierRegion, second: FrontierRegion) -> float:
    union = first.frontier_cells | second.frontier_cells
    if not union:
        return 1.0
    return len(first.frontier_cells & second.frontier_cells) / float(len(union))


# 中文目的：计算两个区域质心的网格距离。
# English purpose: Compute cell-space distance between region centroids.
def _centroid_cell_distance(
    first: FrontierRegion,
    second: FrontierRegion,
) -> float:
    return float(
        np.linalg.norm(
            np.asarray(first.centroid_cell)
            - np.asarray(second.centroid_cell)
        )
    )


# 中文目的：计算两个世界坐标之间的欧氏距离。
# English purpose: Compute Euclidean distance between world coordinates.
def _world_distance(first: Sequence[float], second: Sequence[float]) -> float:
    return float(
        np.linalg.norm(
            np.asarray(first, dtype=float)
            - np.asarray(second, dtype=float)
        )
    )


# 中文目的：判定两个时刻的区域是否代表同一演化任务。
# English purpose: Determine whether two regions represent the same evolving task.
def _regions_equivalent(
    first: FrontierRegion,
    second: FrontierRegion,
    config: DynamicRegionConfig,
) -> bool:
    if first.region_id == second.region_id:
        return True
    if _region_iou(first, second) >= config.region_match_iou_threshold:
        return True
    return (
        _centroid_cell_distance(first, second)
        <= config.region_match_centroid_cells
    )


# 中文目的：判定本地区域是否与已知同伴声明冲突。
# English purpose: Determine whether a local region matches a known peer claim.
def _region_matches_claim(
    region: FrontierRegion,
    claim: PeerClaim,
    config: DynamicRegionConfig,
) -> bool:
    if region.region_id == claim.region_id:
        return True
    return (
        _world_distance(region.centroid_world, claim.centroid_world)
        <= config.region_conflict_distance
    )


# 中文目的：把接触边转换为机器人连通分量。
# English purpose: Convert contact edges into robot connected components.
def _contact_components(
    n_robots: int,
    contact_pairs: Sequence[Tuple[int, int]],
) -> List[List[int]]:
    parent = list(range(n_robots))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(first: int, second: int) -> None:
        root_first = find(first)
        root_second = find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    for first, second in contact_pairs:
        first = int(first)
        second = int(second)
        if 0 <= first < n_robots and 0 <= second < n_robots:
            union(first, second)

    groups: Dict[int, List[int]] = defaultdict(list)
    for robot_id in range(n_robots):
        groups[find(robot_id)].append(robot_id)
    return list(groups.values())





