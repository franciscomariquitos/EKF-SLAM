"""Metrics for the EKF-SLAM micro-simulator."""

from __future__ import annotations

from typing import List

import numpy as np

from utils import rmse


def compute_metrics(history: dict, ekf, landmarks: np.ndarray, waypoints: np.ndarray) -> dict:
    """Compute pose, loop-closure and landmark errors."""
    true_xy = history["true"][:, 0:2]
    odom_xy = history["odom"][:, 0:2]
    ekf_xy = history["ekf"][:, 0:2]

    odom_pose_errors = np.linalg.norm(true_xy - odom_xy, axis=1)
    ekf_pose_errors = np.linalg.norm(true_xy - ekf_xy, axis=1)

    initial_xy = waypoints[0]
    odom_loop_error = float(np.linalg.norm(history["odom"][-1, 0:2] - initial_xy))
    ekf_loop_error = float(np.linalg.norm(history["ekf"][-1, 0:2] - initial_xy))
    true_loop_error = float(np.linalg.norm(history["true"][-1, 0:2] - initial_xy))

    true_landmarks = {int(row[0]): np.array([row[1], row[2]], dtype=float) for row in landmarks}
    est_landmarks = ekf.estimated_landmarks()

    landmark_errors = []
    for tag_id, est_xy_tuple in est_landmarks.items():
        if tag_id in true_landmarks:
            est_xy = np.array(est_xy_tuple, dtype=float)
            landmark_errors.append(float(np.linalg.norm(est_xy - true_landmarks[tag_id])))

    landmark_errors = np.asarray(landmark_errors, dtype=float)

    return {
        "pose_rmse_odom_m": rmse(odom_pose_errors),
        "pose_rmse_ekf_m": rmse(ekf_pose_errors),
        "pose_mean_odom_m": float(np.mean(odom_pose_errors)),
        "pose_mean_ekf_m": float(np.mean(ekf_pose_errors)),
        "final_loop_error_true_m": true_loop_error,
        "final_loop_error_odom_m": odom_loop_error,
        "final_loop_error_ekf_m": ekf_loop_error,
        "landmark_rmse_m": rmse(landmark_errors),
        "landmark_mean_error_m": float(np.mean(landmark_errors)) if landmark_errors.size else float("nan"),
        "n_landmarks_true": int(len(true_landmarks)),
        "n_landmarks_estimated": int(len(est_landmarks)),
        "total_measurements": int(ekf.stats["total_measurements"]),
        "accepted_updates": int(ekf.stats["accepted_updates"]),
        "rejected_outliers": int(ekf.stats["rejected_outliers"]),
        "initialized_landmarks": int(ekf.stats["initialized_landmarks"]),
        "candidate_rejections": int(ekf.stats["candidate_rejections"]),
        "mean_nis": float(np.mean(ekf.nis_values)) if ekf.nis_values else float("nan"),
    }


def summarize_monte_carlo(rows: List[dict]) -> dict:
    """Return mean and standard deviation of the most important Monte Carlo metrics."""
    keys = [
        "pose_rmse_odom_m",
        "pose_rmse_ekf_m",
        "final_loop_error_odom_m",
        "final_loop_error_ekf_m",
        "landmark_rmse_m",
        "accepted_updates",
        "rejected_outliers",
        "candidate_rejections",
    ]

    summary = {}
    for key in keys:
        values = np.array([row[key] for row in rows], dtype=float)
        summary[f"{key}_mean"] = float(np.nanmean(values))
        summary[f"{key}_std"] = float(np.nanstd(values))
    return summary
