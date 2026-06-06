"""Metrics and diagnostic helpers for the EKF-SLAM micro-simulator."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from utils import rmse


def pose_position_errors(history: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return odometry and EKF 2D position errors for every stored timestep."""
    true_xy = history["true"][:, 0:2]
    odom_xy = history["odom"][:, 0:2]
    ekf_xy = history["ekf"][:, 0:2]
    odom_pose_errors = np.linalg.norm(true_xy - odom_xy, axis=1)
    ekf_pose_errors = np.linalg.norm(true_xy - ekf_xy, axis=1)
    return odom_pose_errors, ekf_pose_errors


def true_landmark_dict(landmarks: np.ndarray) -> Dict[int, np.ndarray]:
    """Convert landmark array [id, x, y] into {id: np.array([x, y])}."""
    return {int(row[0]): np.array([row[1], row[2]], dtype=float) for row in landmarks}


def landmark_error_snapshot(ekf, landmarks: np.ndarray) -> dict:
    """
    Compute landmark errors at the current EKF state.

    Returns a dictionary with per-tag errors and aggregate RMSE/mean values. This is
    used during the simulation loop to build landmark convergence plots over time.
    """
    true_landmarks = true_landmark_dict(landmarks)
    est_landmarks = ekf.estimated_landmarks()

    per_tag = {}
    errors = []
    for tag_id, est_xy_tuple in est_landmarks.items():
        if tag_id not in true_landmarks:
            continue
        est_xy = np.array(est_xy_tuple, dtype=float)
        error = float(np.linalg.norm(est_xy - true_landmarks[tag_id]))
        per_tag[tag_id] = error
        errors.append(error)

    errors_arr = np.asarray(errors, dtype=float)
    return {
        "per_tag": per_tag,
        "rmse": rmse(errors_arr),
        "mean": float(np.mean(errors_arr)) if errors_arr.size else float("nan"),
        "n": int(errors_arr.size),
    }


def compute_metrics(history: dict, ekf, landmarks: np.ndarray, waypoints: np.ndarray) -> dict:
    """Compute pose, loop-closure, landmark and consistency metrics."""
    odom_pose_errors, ekf_pose_errors = pose_position_errors(history)

    initial_xy = waypoints[0]
    odom_loop_error = float(np.linalg.norm(history["odom"][-1, 0:2] - initial_xy))
    ekf_loop_error = float(np.linalg.norm(history["ekf"][-1, 0:2] - initial_xy))
    true_loop_error = float(np.linalg.norm(history["true"][-1, 0:2] - initial_xy))

    landmark_snapshot = landmark_error_snapshot(ekf, landmarks)
    true_landmarks = true_landmark_dict(landmarks)
    est_landmarks = ekf.estimated_landmarks()

    odom_rmse = rmse(odom_pose_errors)
    ekf_rmse = rmse(ekf_pose_errors)
    rmse_reduction = 100.0 * (odom_rmse - ekf_rmse) / odom_rmse if odom_rmse > 1e-12 else float("nan")
    loop_reduction = 100.0 * (odom_loop_error - ekf_loop_error) / odom_loop_error if odom_loop_error > 1e-12 else float("nan")

    nis_all = np.asarray([event["nis"] for event in ekf.diagnostics if np.isfinite(event.get("nis", np.nan))], dtype=float)
    nis_accepted = np.asarray([event["nis"] for event in ekf.diagnostics if event.get("accepted", False) and np.isfinite(event.get("nis", np.nan))], dtype=float)

    return {
        "pose_rmse_odom_m": odom_rmse,
        "pose_rmse_ekf_m": ekf_rmse,
        "pose_rmse_reduction_percent": float(rmse_reduction),
        "pose_mean_odom_m": float(np.mean(odom_pose_errors)),
        "pose_mean_ekf_m": float(np.mean(ekf_pose_errors)),
        "final_loop_error_true_m": true_loop_error,
        "final_loop_error_odom_m": odom_loop_error,
        "final_loop_error_ekf_m": ekf_loop_error,
        "loop_error_reduction_percent": float(loop_reduction),
        "landmark_rmse_m": float(landmark_snapshot["rmse"]),
        "landmark_mean_error_m": float(landmark_snapshot["mean"]),
        "n_landmarks_true": int(len(true_landmarks)),
        "n_landmarks_estimated": int(len(est_landmarks)),
        "total_measurements": int(ekf.stats["total_measurements"]),
        "accepted_updates": int(ekf.stats["accepted_updates"]),
        "rejected_outliers": int(ekf.stats["rejected_outliers"]),
        "initialized_landmarks": int(ekf.stats["initialized_landmarks"]),
        "candidate_rejections": int(ekf.stats["candidate_rejections"]),
        "mean_nis_all": float(np.mean(nis_all)) if nis_all.size else float("nan"),
        "mean_nis_accepted": float(np.mean(nis_accepted)) if nis_accepted.size else float("nan"),
        "median_nis_accepted": float(np.median(nis_accepted)) if nis_accepted.size else float("nan"),
    }


def summarize_monte_carlo(rows: List[dict]) -> dict:
    """Return mean and standard deviation of the most important Monte Carlo metrics."""
    keys = [
        "pose_rmse_odom_m",
        "pose_rmse_ekf_m",
        "pose_rmse_reduction_percent",
        "final_loop_error_odom_m",
        "final_loop_error_ekf_m",
        "loop_error_reduction_percent",
        "landmark_rmse_m",
        "accepted_updates",
        "rejected_outliers",
        "candidate_rejections",
        "mean_nis_accepted",
    ]

    summary = {}
    for key in keys:
        values = np.array([row.get(key, np.nan) for row in rows], dtype=float)
        summary[f"{key}_mean"] = float(np.nanmean(values))
        summary[f"{key}_std"] = float(np.nanstd(values))
    return summary
