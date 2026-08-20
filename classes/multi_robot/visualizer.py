
"""PNG/GIF visualiser with a stable colour per robot and matching trail colour."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from parameter import FREE, OCCUPIED, UNKNOWN


class MultiRobotVisualizer:
    def __init__(
        self,
        *,
        output_root: str,
        episode_index: int,
        team_size: int,
        seed: int,
        communication_mode: str = "unknown",
        frame_stride: int = 1,
        background_mode: str = "ground_truth",
        dpi: int = 140,
    ) -> None:
        self.team_size = int(team_size)
        self.communication_mode = str(communication_mode)
        self.frame_stride = max(1, int(frame_stride))
        self.background_mode = background_mode
        self.dpi = int(dpi)

        self.episode_dir = (
            Path(output_root)
            / f"mode_{self.communication_mode}"
            / f"team_{self.team_size}"
            / f"episode_{int(episode_index):03d}_seed_{int(seed)}"
        )
        
        self.frame_dir = self.episode_dir / "frames"
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.gif_path = self.episode_dir / "trajectory.gif"
        colour_map = plt.get_cmap("tab10")
        self.robot_colours = {robot_id: colour_map(robot_id % 10) for robot_id in range(self.team_size)}

    @staticmethod
    def _display_map(env, mode: str) -> np.ndarray:
        occupancy = env.get_team_belief() if mode == "team_belief" else env.ground_truth
        image = np.full((*occupancy.shape, 3), 0.72, dtype=np.float32)
        image[occupancy == FREE] = [0.97, 0.97, 0.97]
        image[occupancy == OCCUPIED] = [0.10, 0.10, 0.10]
        image[occupancy == UNKNOWN] = [0.72, 0.72, 0.72]
        return image

    def save_frame(
        self,
        *,
        env,
        trajectories: Sequence[Iterable[Sequence[float]]],
        step: int,
        contact_pairs: Sequence[tuple[int, int]] = (),
        team_coverage: float | None = None,
    ) -> str | None:
        if step % self.frame_stride != 0:
            return None
        if team_coverage is None:
            team_coverage = float(env.explored_rate)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self._display_map(env, self.background_mode), origin="lower", interpolation="nearest")

        for robot_id, trajectory in enumerate(trajectories):
            points = np.asarray(list(trajectory), dtype=np.float32)
            if len(points) == 0:
                continue
            cells = np.asarray([env.world_to_cell(point) for point in points], dtype=np.float32)
            colour = self.robot_colours[robot_id]

            # Path, start, and current marker intentionally use the same colour.
            if len(cells) > 1:
                ax.plot(cells[:, 0], cells[:, 1], color=colour, linewidth=2.3, alpha=0.88, zorder=2)
            ax.scatter(cells[0, 0], cells[0, 1], s=42, color=colour, edgecolors="black", linewidths=0.7, alpha=0.85, zorder=3)
            ax.scatter(cells[-1, 0], cells[-1, 1], s=170, color=colour, edgecolors="black", linewidths=1.3, zorder=5, label=f"Robot {robot_id}")
            ax.text(cells[-1, 0], cells[-1, 1], str(robot_id), ha="center", va="center", color="white", fontsize=8, fontweight="bold", zorder=6)

        for robot_i, robot_j in contact_pairs:
            cell_i = env.world_to_cell(env.robot_locations[robot_i])
            cell_j = env.world_to_cell(env.robot_locations[robot_j])
            ax.plot([cell_i[0], cell_j[0]], [cell_i[1], cell_j[1]], "k--", linewidth=1.2, alpha=0.85, zorder=4)

        view_name = "Team Local Belief View" if self.background_mode == "team_belief" else "Ground-Truth Evaluation View"
        ax.set_title(
            f"Multi-Robot DARE | Robots: {self.team_size} | Step: {step} | "
            f"Coverage: {100.0 * team_coverage:.1f}%\n{view_name}"
        )
        ax.set_xlabel("Grid x")
        ax.set_ylabel("Grid y")
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.9, ncol=2)
        fig.tight_layout()

        frame_path = self.frame_dir / f"frame_{int(step):06d}.png"
        fig.savefig(frame_path, dpi=self.dpi)
        plt.close(fig)
        return str(frame_path)

    def build_gif(self, duration_seconds: float = 0.25) -> str | None:
        frames = sorted(self.frame_dir.glob("frame_*.png"))
        if not frames:
            return None
        with imageio.get_writer(self.gif_path, mode="I", duration=float(duration_seconds), loop=0) as writer:
            for frame_path in frames:
                writer.append_data(imageio.imread(frame_path))
        return str(self.gif_path)
