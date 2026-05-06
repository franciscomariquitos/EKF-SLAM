"""Live matplotlib view for the manual EKF-SLAM teleop simulator."""

from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ekf_slam import EKFSLAM


class LiveView:
    """Small live 2D viewer for the micro-simulator.

    It does not control the robot. The terminal/curses window still receives the
    keyboard commands. This viewer only displays the current trajectories.
    """

    def __init__(self, landmarks: np.ndarray, update_every: int = 2) -> None:
        self.landmarks = landmarks
        self.update_every = max(1, update_every)
        self.counter = 0

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 7))
        self.fig.canvas.manager.set_window_title("EKF-SLAM live micro-simulator")

        self.true_line, = self.ax.plot([], [], label="Ground truth", linewidth=2.5)
        self.odom_line, = self.ax.plot([], [], label="Odometry", linestyle="--", linewidth=1.8)
        self.ekf_line, = self.ax.plot([], [], label="EKF-SLAM", linewidth=2.2)

        self.robot_dot, = self.ax.plot([], [], marker="o", markersize=8, linestyle="None", label="Robot")
        self.heading_line, = self.ax.plot([], [], linewidth=2.0)
        self.est_tags_scatter = self.ax.scatter([], [], marker="x", s=90, linewidths=2.0, label="Estimated tags")

        for i, lm in enumerate(self.landmarks):
            tag_id, lx, ly = int(lm[0]), float(lm[1]), float(lm[2])
            self.ax.scatter(lx, ly, marker="s", s=90, alpha=0.5, label="True tags" if i == 0 else None)
            self.ax.text(lx + 0.08, ly + 0.08, f"T{tag_id}", fontsize=9)

        self.ax.set_title("Live EKF-SLAM manual teleop")
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.grid(True)
        self.ax.axis("equal")
        self.ax.legend(loc="best")
        self.ax.set_xlim(-2.0, 7.0)
        self.ax.set_ylim(-2.0, 7.0)
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def update(self, history: Dict[str, List[np.ndarray]], true_pose: np.ndarray, ekf: EKFSLAM) -> None:
        """Update trajectories and current robot pose."""
        self.counter += 1
        if self.counter % self.update_every != 0:
            return

        if len(history["true"]) == 0:
            return

        true_arr = np.asarray(history["true"], dtype=float)
        odom_arr = np.asarray(history["odom"], dtype=float)
        ekf_arr = np.asarray(history["ekf"], dtype=float)

        self.true_line.set_data(true_arr[:, 0], true_arr[:, 1])
        self.odom_line.set_data(odom_arr[:, 0], odom_arr[:, 1])
        self.ekf_line.set_data(ekf_arr[:, 0], ekf_arr[:, 1])

        x, y, theta = true_pose
        self.robot_dot.set_data([x], [y])
        heading_length = 0.35
        self.heading_line.set_data(
            [x, x + heading_length * np.cos(theta)],
            [y, y + heading_length * np.sin(theta)],
        )

        estimated = ekf.estimated_landmarks()
        if estimated:
            est_xy = np.asarray(list(estimated.values()), dtype=float)
            self.est_tags_scatter.set_offsets(est_xy)
        else:
            self.est_tags_scatter.set_offsets(np.empty((0, 2)))

        # Dynamically expand the visible region if the robot leaves the default map.
        all_xy = np.vstack([true_arr[:, 0:2], odom_arr[:, 0:2], ekf_arr[:, 0:2], self.landmarks[:, 1:3]])
        xmin, ymin = np.min(all_xy, axis=0) - 0.7
        xmax, ymax = np.max(all_xy, axis=0) + 0.7
        self.ax.set_xlim(float(xmin), float(xmax))
        self.ax.set_ylim(float(ymin), float(ymax))
        self.ax.set_aspect("equal", adjustable="box")

        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def close(self) -> None:
        """Close the live window."""
        plt.close(self.fig)
