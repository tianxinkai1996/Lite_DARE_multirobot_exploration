"""Independent-local-map multi-robot environment built around DARE's Env.

The original DARE Env owns one shared robot_belief. This wrapper retains the
original map loader and sensor model but creates one belief map per robot.
Ground truth and all true positions stay inside the simulator layer.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from parameter import CELL_SIZE, FREE, NODE_RESOLUTION, OCCUPIED, SENSOR_RANGE, UNKNOWN
from classes.env.env import Env
from classes.env.sensor import sensor_work
from classes.utils import MapInfo


NodeKey = Tuple[float, float]


class MultiRobotEnv(Env):
    """DARE-compatible environment with independent local belief maps."""

    def __init__(self, episode_index, n_agents, test=True, seed=0, min_start_separation=12.0, done_tolerance_cells=250, start_positions=None, allow_shared_start=False, start_clearance=0.0):
        # Use Env only for map loading, coordinate origin, and original map
        # preprocessing. Initialise it as one robot, then replace shared state.
        super().__init__(episode_index, n_agent=1, plot=False, test=test)

        self.n_agent = int(n_agents)
        if self.n_agent < 1:
            raise ValueError("n_agents must be at least 1")

        self.rng = np.random.default_rng(seed)
        self.cell_size = float(CELL_SIZE)
        self.sensor_range = float(SENSOR_RANGE)
        self.node_resolution = float(NODE_RESOLUTION)
        self.done_tolerance_cells = int(done_tolerance_cells)
        self.start_clearance = float(start_clearance)
        if self.start_clearance < 0.0:
            raise ValueError("start_clearance must be non-negative")

        self.hidden_graph = self._build_hidden_contact_graph()
        if start_positions is None:
            self.robot_locations = self._sample_start_locations(min_start_separation)
        else:
            self.robot_locations = self._validate_start_locations(
                start_positions, allow_shared_start=allow_shared_start
            )

        self.robot_beliefs: List[np.ndarray] = []
        self.robot_belief_infos: List[MapInfo] = []
        for robot_id in range(self.n_agent):
            belief = np.ones(self.ground_truth.shape, dtype=np.int32) * UNKNOWN
            cell = self.world_to_cell(self.robot_locations[robot_id])
            belief = sensor_work(
                cell,
                round(self.sensor_range / self.cell_size),
                belief,
                self.ground_truth,
            )
            self.robot_beliefs.append(belief)
            self.robot_belief_infos.append(
                MapInfo(belief, self.belief_origin_x, self.belief_origin_y, self.cell_size)
            )

        # Compatibility aliases. Multi-robot code must not use these aliases for
        # planning because they expose only robot 0's map.
        self.robot_belief = self.robot_beliefs[0]
        self.belief_info = self.robot_belief_infos[0]
        self.explored_rate = 0.0
        self.done = False
        self.evaluate_team_exploration_rate()

    # ------------------------------------------------------------------
    # Coordinate and static-map helpers
    # ------------------------------------------------------------------

    def world_to_cell(self, point):
        """Convert a world coordinate to the DARE occupancy-grid [x, y] cell."""
        point = np.asarray(point, dtype=np.float64)
        return np.array(
            [
                int(round((point[0] - self.belief_origin_x) / self.cell_size)),
                int(round((point[1] - self.belief_origin_y) / self.cell_size)),
            ],
            dtype=np.int32,
        )

    def cell_to_world(self, cell):
        cell = np.asarray(cell, dtype=np.float64)
        return np.array(
            [
                self.belief_origin_x + cell[0] * self.cell_size,
                self.belief_origin_y + cell[1] * self.cell_size,
            ],
            dtype=np.float32,
        )

    def _in_bounds(self, cell):
        x, y = int(cell[0]), int(cell[1])
        height, width = self.ground_truth.shape
        return 0 <= x < width and 0 <= y < height

    def is_free_world(self, point):
        cell = self.world_to_cell(point)
        return self._in_bounds(cell) and int(self.ground_truth[cell[1], cell[0]]) == FREE

    def is_start_clear_world(self, point, clearance=None):
        """Return True when a circular robot footprint is clear of walls.

        The centre must lie in a FREE ground-truth cell. Every non-FREE cell is
        treated as a closed square obstacle. The test uses exact circle-versus-
        axis-aligned-cell intersection, so it protects the complete footprint,
        not only the centre cell. Map boundaries are treated as walls.

        This method is simulator-side only and never exposes ground truth to the
        DARE policy.
        """
        centre = np.asarray(point, dtype=np.float64)
        radius = self.start_clearance if clearance is None else float(clearance)
        if radius < 0.0:
            raise ValueError("clearance must be non-negative")
        if not self.is_free_world(centre):
            return False
        if radius <= 0.0:
            return True

        height, width = self.ground_truth.shape
        half = 0.5 * self.cell_size
        min_world_x = self.belief_origin_x - half
        max_world_x = self.belief_origin_x + (width - 1) * self.cell_size + half
        min_world_y = self.belief_origin_y - half
        max_world_y = self.belief_origin_y + (height - 1) * self.cell_size + half

        # A footprint extending outside the known map is considered colliding.
        if (
            centre[0] - radius < min_world_x
            or centre[0] + radius > max_world_x
            or centre[1] - radius < min_world_y
            or centre[1] + radius > max_world_y
        ):
            return False

        centre_cell = self.world_to_cell(centre)
        # Include every cell whose square can intersect the circle.
        cell_radius = int(np.ceil((radius + half * np.sqrt(2.0)) / self.cell_size))
        cx, cy = int(centre_cell[0]), int(centre_cell[1])
        radius_sq = radius * radius

        for y in range(cy - cell_radius, cy + cell_radius + 1):
            for x in range(cx - cell_radius, cx + cell_radius + 1):
                if not self._in_bounds((x, y)):
                    return False
                if int(self.ground_truth[y, x]) == FREE:
                    continue

                obstacle_centre = self.cell_to_world((x, y)).astype(np.float64)
                dx = max(abs(centre[0] - obstacle_centre[0]) - half, 0.0)
                dy = max(abs(centre[1] - obstacle_centre[1]) - half, 0.0)
                if dx * dx + dy * dy <= radius_sq:
                    return False
        return True

    def line_is_clear_world(self, start, end):
        """Bresenham-style line-of-sight/static-collision check on ground truth.

        This function is simulator-only. Its result is used for sensor/contact
        simulation and final physical safety checks; DARE never receives the
        ground-truth map or this line trace.
        """
        x0, y0 = map(int, self.world_to_cell(start))
        x1, y1 = map(int, self.world_to_cell(end))

        if not self._in_bounds((x0, y0)) or not self._in_bounds((x1, y1)):
            return False

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0

        while True:
            if not self._in_bounds((x, y)):
                return False
            if int(self.ground_truth[y, x]) == OCCUPIED:
                return False
            if x == x1 and y == y1:
                return True
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    # ------------------------------------------------------------------
    # Hidden contact graph used only by the simulator
    # ------------------------------------------------------------------

    @staticmethod
    def _key(point):
        return (round(float(point[0]), 4), round(float(point[1]), 4))

    def _candidate_hidden_nodes(self):
        """Create 4 m-aligned free-space nodes in the hidden true map."""
        h, w = self.ground_truth.shape
        min_x = self.belief_origin_x
        min_y = self.belief_origin_y
        max_x = self.belief_origin_x + (w - 1) * self.cell_size
        max_y = self.belief_origin_y + (h - 1) * self.cell_size

        kx_min = int(np.ceil(min_x / self.node_resolution))
        kx_max = int(np.floor(max_x / self.node_resolution))
        ky_min = int(np.ceil(min_y / self.node_resolution))
        ky_max = int(np.floor(max_y / self.node_resolution))

        nodes: List[np.ndarray] = []
        for kx in range(kx_min, kx_max + 1):
            for ky in range(ky_min, ky_max + 1):
                point = np.array(
                    [kx * self.node_resolution, ky * self.node_resolution],
                    dtype=np.float32,
                )
                if self.is_start_clear_world(point):
                    nodes.append(point)
        return nodes

    def _build_hidden_contact_graph(self):
        """Build a hidden four-neighbour traversability graph from ground truth."""
        nodes = self._candidate_hidden_nodes()
        node_set = {self._key(node) for node in nodes}
        graph: Dict[NodeKey, set[NodeKey]] = {key: set() for key in node_set}
        offsets = (
            np.array([self.node_resolution, 0.0]),
            np.array([-self.node_resolution, 0.0]),
            np.array([0.0, self.node_resolution]),
            np.array([0.0, -self.node_resolution]),
        )

        for key in list(node_set):
            point = np.asarray(key, dtype=np.float32)
            for offset in offsets:
                neighbour = point + offset
                neighbour_key = self._key(neighbour)
                if neighbour_key not in node_set:
                    continue
                if self.line_is_clear_world(point, neighbour):
                    graph[key].add(neighbour_key)
        return graph

    def nearest_hidden_node(self, point):
        if not self.hidden_graph:
            return None
        point = np.asarray(point, dtype=np.float32)
        keys = np.asarray(list(self.hidden_graph.keys()), dtype=np.float32)
        index = int(np.argmin(np.linalg.norm(keys - point[None, :], axis=1)))
        return self._key(keys[index])

    def hidden_shortest_hops(self, start, goal, max_hops):
        """Return hidden shortest hop count up to max_hops, otherwise None."""
        start_key = self.nearest_hidden_node(start)
        goal_key = self.nearest_hidden_node(goal)
        if start_key is None or goal_key is None:
            return None
        if start_key == goal_key:
            return 0

        queue: deque[Tuple[NodeKey, int]] = deque([(start_key, 0)])
        seen = {start_key}
        while queue:
            node, hops = queue.popleft()
            if hops >= max_hops:
                continue
            for neighbour in self.hidden_graph.get(node, ()):
                if neighbour in seen:
                    continue
                next_hops = hops + 1
                if neighbour == goal_key:
                    return next_hops
                seen.add(neighbour)
                queue.append((neighbour, next_hops))
        return None

    # ------------------------------------------------------------------
    # Episode initialisation/local maps
    # ------------------------------------------------------------------

    def _sample_start_locations(self, min_start_separation):
        """Randomly sample distinct graph nodes using the episode seed."""
        startable_keys = [key for key, neighbours in self.hidden_graph.items() if neighbours]
        candidates = np.asarray(startable_keys, dtype=np.float32)
        if len(candidates) < self.n_agent:
            raise RuntimeError(
                f"Only {len(candidates)} hidden free-space nodes are available for "
                f"{self.n_agent} robots."
            )

        # Try requested separation first, then relax only if the particular map
        # cannot accommodate the requested team size.
        for scale in (1.0, 0.8, 0.6, 0.4, 0.0):
            order = self.rng.permutation(len(candidates))
            selected: List[np.ndarray] = []
            threshold = float(min_start_separation) * scale
            for idx in order:
                candidate = candidates[idx]
                if all(np.linalg.norm(candidate - old) >= threshold for old in selected):
                    selected.append(candidate)
                if len(selected) == self.n_agent:
                    return np.asarray(selected, dtype=np.float32)

        raise RuntimeError("Could not sample enough valid robot start locations.")


    def _validate_start_locations(self, start_positions, allow_shared_start=False):
        """Validate externally supplied starts.

        When ``allow_shared_start`` is true, every robot must use the same valid
        free graph node. Otherwise, all robot start nodes must be distinct.
        """

        starts = np.asarray(start_positions, dtype=np.float32)
        if starts.shape != (self.n_agent, 2):
            raise ValueError(
                f"start_positions must have shape ({self.n_agent}, 2), got {starts.shape}"
            )
        keys = [self._key(point) for point in starts]
        if allow_shared_start:
            if len(set(keys)) != 1:
                raise ValueError("Shared-start trials require all robots to use the same initial node")
        elif len(set(keys)) != self.n_agent:
            raise ValueError("Every robot must have a different initial position")
        for point in starts:
            key = self.nearest_hidden_node(point)
            if (
                key is None
                or key != self._key(point)
                or not self.is_start_clear_world(point)
                or not self.hidden_graph.get(key)
            ):
                raise ValueError(
                    "Invalid robot start position (not a clear, connected hidden graph node): "
                    f"{point.tolist()}"
                )
        return starts.copy()

    def get_local_belief_info(self, robot_id):
        return self.robot_belief_infos[int(robot_id)]

    def get_local_belief(self, robot_id):
        return self.robot_beliefs[int(robot_id)]

    def update_local_belief(self, robot_id):
        robot_id = int(robot_id)
        cell = self.world_to_cell(self.robot_locations[robot_id])
        updated = sensor_work(
            cell,
            round(self.sensor_range / self.cell_size),
            self.robot_beliefs[robot_id],
            self.ground_truth,
        )
        self.robot_beliefs[robot_id] = updated
        self.robot_belief_infos[robot_id] = MapInfo(
            updated,
            self.belief_origin_x,
            self.belief_origin_y,
            self.cell_size,
        )
        if robot_id == 0:
            self.robot_belief = updated
            self.belief_info = self.robot_belief_infos[0]

    # ------------------------------------------------------------------
    # Synchronous motion and team-only evaluation
    # ------------------------------------------------------------------

    def step_all(self, proposed_positions):
        """Commit already-resolved synchronous positions and update local maps.

        A static obstacle check remains as a final physical guard. It does not
        compute alternate global paths or disclose hidden state to DARE.
        """
        proposed = np.asarray(proposed_positions, dtype=np.float32)
        if proposed.shape != self.robot_locations.shape:
            raise ValueError(
                f"Expected proposed positions with shape {self.robot_locations.shape}, "
                f"received {proposed.shape}."
            )

        actual = proposed.copy()
        blocked: List[int] = []
        for robot_id in range(self.n_agent):
            if not self.line_is_clear_world(self.robot_locations[robot_id], proposed[robot_id]):
                actual[robot_id] = self.robot_locations[robot_id]
                blocked.append(robot_id)

        # Safety fallback: if any duplicate destination remains, keep later robot
        # in place. Properly resolved actions should never reach this branch.
        seen: Dict[NodeKey, int] = {}
        for robot_id, position in enumerate(actual):
            key = self._key(position)
            if key in seen:
                actual[robot_id] = self.robot_locations[robot_id]
                blocked.append(robot_id)
            else:
                seen[key] = robot_id

        self.robot_locations = actual
        for robot_id in range(self.n_agent):
            self.update_local_belief(robot_id)

        self.evaluate_team_exploration_rate()
        self.check_done()
        return actual, blocked

    def get_team_belief(self):
        """Union of local observations for metrics/visualisation only."""
        team = np.ones(self.ground_truth.shape, dtype=np.int32) * UNKNOWN
        for local in self.robot_beliefs:
            known = local != UNKNOWN
            team[known] = local[known]
        return team

    def evaluate_team_exploration_rate(self):
        free_truth = self.ground_truth == FREE
        observed_free = np.zeros_like(free_truth, dtype=bool)
        for local in self.robot_beliefs:
            observed_free |= local == FREE
        total_free = int(np.sum(free_truth))
        self.explored_rate = 0.0 if total_free == 0 else float(np.sum(observed_free & free_truth) / total_free)
        return self.explored_rate

    def check_done(self):
        free_truth = self.ground_truth == FREE
        observed_free = np.zeros_like(free_truth, dtype=bool)
        for local in self.robot_beliefs:
            observed_free |= local == FREE
        remaining = int(np.sum(free_truth & ~observed_free))
        self.done = remaining <= self.done_tolerance_cells
        return self.done