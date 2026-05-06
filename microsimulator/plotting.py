"""Plotting functions for the EKF-SLAM micro-simulator."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ekf_slam import EKFSLAM


def plot_map(result: dict, outdir: Path, show: bool) -> None:
    """Plot ground truth, odometry, EKF trajectory, true tags and estimated tags."""
    history = result["history"]
    landmarks = result["landmarks"]
    ekf: EKFSLAM = result["ekf"]
    metrics = result["metrics"]

    plt.figure(figsize=(10, 8))
    plt.title(
        "EKF-SLAM micro-simulator: ground truth vs odometry vs EKF\n"
        f"RMSE odom={metrics['pose_rmse_odom_m']:.3f} m | "
        f"RMSE EKF={metrics['pose_rmse_ekf_m']:.3f} m"
    )

    plt.plot(history["true"][:, 0], history["true"][:, 1], label="Ground truth", linewidth=2.5)
    plt.plot(history["odom"][:, 0], history["odom"][:, 1], label="Odometry", linestyle="--", linewidth=1.8)
    plt.plot(history["ekf"][:, 0], history["ekf"][:, 1], label="EKF-SLAM", linewidth=2.2)

    for i, lm in enumerate(landmarks):
        tag_id, lx, ly = int(lm[0]), lm[1], lm[2]
        plt.scatter(lx, ly, marker="s", s=100, alpha=0.5, label="True tags" if i == 0 else None)
        plt.text(lx + 0.08, ly + 0.08, f"T{tag_id}", fontsize=9)

    estimated = ekf.estimated_landmarks()
    for i, (tag_id, (lx, ly)) in enumerate(estimated.items()):
        plt.scatter(lx, ly, marker="x", s=100, linewidths=2.5, label="Estimated tags" if i == 0 else None)
        plt.text(lx + 0.08, ly - 0.18, f"E{tag_id}", fontsize=9)

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "single_run_map.png", dpi=180)
    if show:
        plt.show()
    else:
        plt.close()


def plot_errors(result: dict, outdir: Path, show: bool) -> None:
    """Plot odometry and EKF position error over time."""
    history = result["history"]
    t = history["t"]

    odom_error = np.linalg.norm(history["true"][:, 0:2] - history["odom"][:, 0:2], axis=1)
    ekf_error = np.linalg.norm(history["true"][:, 0:2] - history["ekf"][:, 0:2], axis=1)

    plt.figure(figsize=(10, 5))
    plt.title("Pose error over time")
    plt.plot(t, odom_error, label="Odometry error")
    plt.plot(t, ekf_error, label="EKF-SLAM error")
    plt.xlabel("time [s]")
    plt.ylabel("position error [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "single_run_errors.png", dpi=180)
    if show:
        plt.show()
    else:
        plt.close()
