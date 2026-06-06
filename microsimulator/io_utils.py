"""CSV output and terminal reporting utilities."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import math
import numpy as np

def save_covariance_matrices_csv(result: dict, outdir: Path) -> None:
    """
    Save final EKF covariance blocks:
        - robot 3x3 covariance
        - one 2x2 covariance matrix per initialized landmark
    """
    ekf = result["ekf"]

    robot_path = outdir / "single_run_final_robot_covariance.csv"
    robot_cov = ekf.robot_covariance()

    with robot_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "col", "value"])

        for i in range(robot_cov.shape[0]):
            for j in range(robot_cov.shape[1]):
                writer.writerow([i, j, robot_cov[i, j]])

    landmark_path = outdir / "single_run_final_landmark_covariances.csv"

    with landmark_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tag_id",
            "cov_xx",
            "cov_xy",
            "cov_yx",
            "cov_yy",
            "sigma_x",
            "sigma_y",
            "trace",
            "determinant",
        ])

        for tag_id, cov in sorted(ekf.landmark_covariances().items()):
            sigma_x = math.sqrt(max(cov[0, 0], 0.0))
            sigma_y = math.sqrt(max(cov[1, 1], 0.0))

            writer.writerow([
                tag_id,
                cov[0, 0],
                cov[0, 1],
                cov[1, 0],
                cov[1, 1],
                sigma_x,
                sigma_y,
                float(np.trace(cov)),
                float(np.linalg.det(cov)),
            ])

            

def save_single_run_metrics_txt(result: dict, outdir: Path) -> None:
    """
    Save single-run micro-simulator metrics to a readable text file.
    """
    path = outdir / "single_run_metrics.txt"
    metrics = result["metrics"]

    with path.open("w") as f:
        f.write("Micro-simulator EKF-SLAM metrics\n")
        f.write("================================\n\n")

        width = max(len(str(key)) for key in metrics.keys())

        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key:<{width}} : {value:.6f}\n")
            else:
                f.write(f"{key:<{width}} : {value}\n")            

def save_history_csv(result: dict, outdir: Path) -> None:
    """Save the single-run trajectory history to CSV."""
    history = result["history"]
    path = outdir / "single_run_history.csv"

    optional_keys = [
        "robot_sigma_x",
        "robot_sigma_y",
        "robot_sigma_theta",
        "robot_cov_trace",
        "initialized_landmarks",
        "candidate_landmarks",
        "landmark_rmse_over_time",
        "landmark_mean_error_over_time",
        "n_landmarks_estimated_over_time",
    ]

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "t",
            "true_x", "true_y", "true_theta",
            "odom_x", "odom_y", "odom_theta",
            "ekf_x", "ekf_y", "ekf_theta",
            "n_measurements",
            "accepted_updates_cumulative",
            "rejected_outliers_cumulative",
            *optional_keys,
        ])

        for i in range(len(history["t"])):
            optional_values = [history[key][i] if key in history else "" for key in optional_keys]
            writer.writerow([
                history["t"][i],
                *history["true"][i],
                *history["odom"][i],
                *history["ekf"][i],
                history["n_measurements"][i],
                history["accepted_updates"][i],
                history["rejected_outliers"][i],
                *optional_values,
            ])


def save_diagnostics_csv(result: dict, outdir: Path) -> None:
    """Save per-measurement EKF diagnostic events to CSV."""
    ekf = result["ekf"]
    path = outdir / "single_run_measurement_diagnostics.csv"

    if not ekf.diagnostics:
        return

    fieldnames = sorted({key for event in ekf.diagnostics for key in event.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ekf.diagnostics)


def save_metrics_csv(metrics_rows: List[dict], outdir: Path) -> None:
    """Save Monte Carlo metrics to CSV."""
    if not metrics_rows:
        return

    path = outdir / "monte_carlo_metrics.csv"
    fieldnames = list(metrics_rows[0].keys())

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)


def print_metrics(title: str, metrics: dict) -> None:
    """Pretty-print a metrics dictionary."""
    print("\n" + title)
    print("=" * len(title))
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:40s}: {value:.4f}")
        else:
            print(f"{key:40s}: {value}")
