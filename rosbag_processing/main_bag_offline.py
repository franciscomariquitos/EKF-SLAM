"""
main_bag_offline.py
-------------------

Offline EKF-SLAM processor for ROS 2 bags.

This script reads a ROS 2 bag directly, without requiring:

    ros2 bag play

It uses:
    /odom            -> EKF prediction
    /aruco_landmarks -> EKF correction

The /aruco_landmarks topic is expected as std_msgs/Float32MultiArray with format:

    [id, x_cam, y_cam, z_cam, range, bearing,
     id, x_cam, y_cam, z_cam, range, bearing,
     ...]

Only id, range and bearing are used by the EKF.

Main outputs:
    bag_history.csv
    bag_measurement_diagnostics.csv
    bag_map.png
    bag_nis.png
    bag_range_innovations.png
    bag_bearing_innovations.png
    bag_robot_uncertainty.png
    bag_robot_cov_trace.png
    bag_update_statistics.png
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message
from std_msgs.msg import Float32MultiArray

from config import SimConfig
from ekf_slam import EKFSLAM
from utils import normalize_angle, odometry_increment

from typing import Any, Dict, Iterable, List
from matplotlib.patches import Ellipse
from matplotlib.collections import LineCollection


# ---------------------------------------------------------------------
# Basic geometry helpers
# ---------------------------------------------------------------------


def yaw_from_quaternion(q) -> float:
    """
    Convert a ROS quaternion into planar yaw angle.

    ROS odometry stores orientation as a quaternion.
    The EKF only uses planar pose:

        [x, y, theta]
    """
    x = float(q.x)
    y = float(q.y)
    z = float(q.z)
    w = float(q.w)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)

    return math.atan2(siny_cosp, cosy_cosp)


def pose_from_odom(msg: Odometry) -> np.ndarray:
    """
    Convert nav_msgs/Odometry into planar pose [x, y, theta].
    """
    x = float(msg.pose.pose.position.x)
    y = float(msg.pose.pose.position.y)
    theta = yaw_from_quaternion(msg.pose.pose.orientation)

    return np.array([x, y, theta], dtype=float)


def angle_diff(a: float, b: float) -> float:
    """
    Smallest signed angular difference a - b in [-pi, pi].
    """
    return math.atan2(math.sin(a - b), math.cos(a - b))


def loop_closed(
    current_pose: np.ndarray,
    initial_pose: np.ndarray,
    elapsed_time: float,
    min_time: float,
    xy_threshold: float,
    yaw_threshold: float,
) -> bool:
    """
    Check whether the robot returned close to the initial odometry pose.

    This is useful when the bag has extra data before/after the actual loop.
    """
    if elapsed_time < min_time:
        return False

    xy_error = float(np.linalg.norm(current_pose[0:2] - initial_pose[0:2]))
    yaw_error = abs(angle_diff(current_pose[2], initial_pose[2]))

    return xy_error <= xy_threshold and yaw_error <= yaw_threshold


# ---------------------------------------------------------------------
# ArUco parsing
# ---------------------------------------------------------------------


def parse_aruco_landmarks(msg: Float32MultiArray) -> List[dict]:
    """
    Parse /aruco_landmarks.

    Your real topic format is:

        [id, x_cam, y_cam, z_cam, range, bearing,
         id, x_cam, y_cam, z_cam, range, bearing,
         ...]

    Example:
        [4.0, -0.204, -0.166, 0.696, 0.726, -0.286]

    Meaning:
        id      = 4
        x_cam   = -0.204
        y_cam   = -0.166
        z_cam   = 0.696
        range   = 0.726
        bearing = -0.286 rad

    EKF only uses:
        id, range, bearing
    """
    data = list(msg.data)

    if len(data) == 0:
        return []

    if len(data) % 6 != 0:
        print(f"[WARN] Invalid /aruco_landmarks length: {len(data)}. Expected multiple of 6.")
        return []

    measurements: List[dict] = []

    for i in range(0, len(data), 6):
        tag_id = int(data[i])

        # Not used directly by EKF, but kept here for clarity/debugging.
        x_cam = float(data[i + 1])
        y_cam = float(data[i + 2])
        z_cam = float(data[i + 3])

        rng = float(data[i + 4])
        # The detector bearing uses camera optical-frame convention:
        # x_cam > 0 is image/right side. The EKF uses planar robot convention:
        # positive bearing is counter-clockwise/left. Therefore the sign is inverted.
        bearing = normalize_angle(float(data[i + 5]))

        if rng <= 0.02:
            continue

        measurements.append({
            "id": tag_id,
            "range": rng,
            "bearing": bearing,
            "x_cam": x_cam,
            "y_cam": y_cam,
            "z_cam": z_cam,
        })

    return measurements


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


def configure_for_bag(cfg: SimConfig, args: argparse.Namespace) -> SimConfig:
    """
    Override EKF parameters from command-line arguments.

    Important:
        In bag processing, sim_* noise parameters do not generate data.
        The data already exists in the bag.

    Relevant parameters are:
        ekf_sigma_motion_*
        ekf_sigma_range
        ekf_sigma_bearing
        gates
        landmark initialization parameters
    """
    cfg.noise.ekf_sigma_motion_x = args.ekf_sigma_motion_x
    cfg.noise.ekf_sigma_motion_y = args.ekf_sigma_motion_y
    cfg.noise.ekf_sigma_motion_theta = math.radians(args.ekf_sigma_motion_theta_deg)

    cfg.noise.ekf_sigma_range = args.ekf_sigma_range
    cfg.noise.ekf_sigma_bearing = math.radians(args.ekf_sigma_bearing_deg)

    cfg.mahalanobis_gate = args.mahalanobis_gate
    cfg.min_observations_to_initialize = args.min_observations_to_initialize
    cfg.candidate_distance_gate = args.candidate_distance_gate

    cfg.max_raw_range_innovation = args.max_raw_range_innovation
    cfg.max_raw_bearing_innovation = math.radians(args.max_raw_bearing_innovation_deg)

    return cfg


# ---------------------------------------------------------------------
# Bag opening
# ---------------------------------------------------------------------


def open_bag_reader(bag_path: Path) -> SequentialReader:
    """
    Open a ROS 2 bag using rosbag2_py.
    """
    reader = SequentialReader()

    storage_options = StorageOptions(
        uri=str(bag_path),
        storage_id="sqlite3",
    )

    converter_options = ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader.open(storage_options, converter_options)
    return reader


# ---------------------------------------------------------------------
# CSV saving
# ---------------------------------------------------------------------


def save_history_csv(history: dict, outdir: Path) -> None:
    """
    Save trajectory and uncertainty history.
    """
    path = outdir / "bag_history.csv"

    with path.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "t",
            "odom_x", "odom_y", "odom_theta",
            "ekf_x", "ekf_y", "ekf_theta",
            "n_measurements",
            "accepted_updates_cumulative",
            "rejected_outliers_cumulative",
            "robot_sigma_x",
            "robot_sigma_y",
            "robot_sigma_theta",
            "robot_cov_trace",
            "initialized_landmarks",
            "candidate_landmarks",
        ])

        for i in range(len(history["t"])):
            odom = history["odom"][i]
            ekf = history["ekf"][i]

            writer.writerow([
                history["t"][i],
                odom[0], odom[1], odom[2],
                ekf[0], ekf[1], ekf[2],
                history["n_measurements"][i],
                history["accepted_updates"][i],
                history["rejected_outliers"][i],
                history["robot_sigma_x"][i],
                history["robot_sigma_y"][i],
                history["robot_sigma_theta"][i],
                history["robot_cov_trace"][i],
                history["initialized_landmarks"][i],
                history["candidate_landmarks"][i],
            ])


def save_diagnostics_csv(ekf: EKFSLAM, outdir: Path) -> None:
    """
    Save EKF measurement diagnostics.
    """
    path = outdir / "bag_measurement_diagnostics.csv"

    fieldnames = [
        "t",
        "tag_id",

        "landmark_status",
        "gate_type",

        "measured_range",
        "measured_bearing",
        "predicted_range",
        "predicted_bearing",
        "innovation_range",
        "innovation_bearing",
        "nis",

        "candidate_x",
        "candidate_y",
        "candidate_error",
        "candidate_count",
        "candidate_status",

        "accepted",
        "reason",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for event in ekf.diagnostics:
            writer.writerow({
                key: event.get(key, float("nan"))
                for key in fieldnames
            })

def save_covariance_matrices_csv(ekf: EKFSLAM, outdir: Path) -> None:
    """
    Save final EKF covariance blocks:
        - robot 3x3 covariance
        - one 2x2 covariance matrix per initialized landmark
    """

    robot_path = outdir / "bag_final_robot_covariance.csv"

    robot_cov = ekf.robot_covariance()

    with robot_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "col", "value"])

        for i in range(robot_cov.shape[0]):
            for j in range(robot_cov.shape[1]):
                writer.writerow([i, j, robot_cov[i, j]])

    landmark_path = outdir / "bag_final_landmark_covariances.csv"

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


# ---------------------------------------------------------------------
# Metrics text output
# ---------------------------------------------------------------------


def finite_stats(values: List[float] | np.ndarray, prefix: str) -> Dict[str, float]:
    """
    Return count, mean, median, percentile and max statistics for finite values.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return {
            f"{prefix}_count": 0.0,
            f"{prefix}_mean": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_p95": float("nan"),
            f"{prefix}_p99": float("nan"),
            f"{prefix}_max": float("nan"),
        }

    return {
        f"{prefix}_count": float(arr.size),
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p95": float(np.percentile(arr, 95)),
        f"{prefix}_p99": float(np.percentile(arr, 99)),
        f"{prefix}_max": float(np.max(arr)),
    }


def path_length_xy(poses: np.ndarray) -> float:
    """
    Compute planar path length from [x, y, theta] poses.
    """
    if poses.ndim != 2 or poses.shape[0] < 2:
        return float("nan")

    steps = np.linalg.norm(np.diff(poses[:, 0:2], axis=0), axis=1)
    return float(np.sum(steps))


def angle_change_stats(poses: np.ndarray, prefix: str) -> Dict[str, float]:
    """
    Compute angular step statistics in degrees.
    """
    if poses.ndim != 2 or poses.shape[0] < 2:
        return finite_stats(np.array([], dtype=float), prefix)

    dtheta = [
        abs(angle_diff(poses[i + 1, 2], poses[i, 2]))
        for i in range(poses.shape[0] - 1)
    ]

    return finite_stats(np.degrees(np.asarray(dtheta, dtype=float)), prefix)


def count_reasons(events: List[dict]) -> Dict[str, int]:
    """
    Count diagnostic reasons such as accepted_update, raw_range_gate, etc.
    """
    counts: Dict[str, int] = {}

    for event in events:
        reason = str(event.get("reason", "unknown"))
        counts[reason] = counts.get(reason, 0) + 1

    return counts


def compute_bag_metrics(
    history: dict,
    ekf: EKFSLAM,
    cfg: SimConfig,
    bag_path: Path,
    stop_reason: str,
    counters: Dict[str, int],
) -> Dict[str, Dict[str, Any]]:
    """
    Compute bag metrics similar to the micro-simulator.

    Important:
        A real bag does not have ground-truth pose or true landmark positions.
        Therefore true RMSE values cannot be computed unless an external reference is provided.
    """
    t = np.asarray(history["t"], dtype=float)
    odom = np.asarray(history["odom"], dtype=float)
    ekf_hist = np.asarray(history["ekf"], dtype=float)

    has_pose_history = (
        odom.ndim == 2
        and ekf_hist.ndim == 2
        and odom.shape[0] > 0
        and ekf_hist.shape[0] > 0
    )

    if has_pose_history:
        ekf_odom_gap = np.linalg.norm(ekf_hist[:, 0:2] - odom[:, 0:2], axis=1)

        odom_steps = (
            np.linalg.norm(np.diff(odom[:, 0:2], axis=0), axis=1)
            if odom.shape[0] > 1
            else np.array([])
        )

        ekf_steps = (
            np.linalg.norm(np.diff(ekf_hist[:, 0:2], axis=0), axis=1)
            if ekf_hist.shape[0] > 1
            else np.array([])
        )

        odom_loop_error = float(np.linalg.norm(odom[-1, 0:2] - odom[0, 0:2]))
        ekf_loop_error = float(np.linalg.norm(ekf_hist[-1, 0:2] - ekf_hist[0, 0:2]))

        loop_reduction = (
            100.0 * (odom_loop_error - ekf_loop_error) / odom_loop_error
            if odom_loop_error > 1e-12
            else float("nan")
        )
    else:
        ekf_odom_gap = np.array([])
        odom_steps = np.array([])
        ekf_steps = np.array([])
        odom_loop_error = float("nan")
        ekf_loop_error = float("nan")
        loop_reduction = float("nan")

    diagnostics = ekf.diagnostics

    accepted_events = [
        event for event in diagnostics
        if bool(event.get("accepted", False))
    ]

    rejected_events = [
        event for event in diagnostics
        if not bool(event.get("accepted", False))
    ]

    nis_all = np.asarray([
        float(event.get("nis", float("nan")))
        for event in diagnostics
    ])

    nis_accepted = np.asarray([
        float(event.get("nis", float("nan")))
        for event in accepted_events
    ])

    range_innov_accepted = np.asarray([
        abs(float(event.get("innovation_range", float("nan"))))
        for event in accepted_events
    ])

    bearing_innov_accepted_deg = np.asarray([
        abs(math.degrees(float(event.get("innovation_bearing", float("nan")))))
        for event in accepted_events
    ])

    valid_nis_accepted = nis_accepted[np.isfinite(nis_accepted)]

    nis_accepted_below_95 = (
        100.0 * float(np.mean(valid_nis_accepted <= 5.99))
        if valid_nis_accepted.size
        else float("nan")
    )

    nis_accepted_below_99 = (
        100.0 * float(np.mean(valid_nis_accepted <= 9.21))
        if valid_nis_accepted.size
        else float("nan")
    )

    robot_cov = ekf.robot_covariance()

    robot_sigma_x = float(math.sqrt(max(robot_cov[0, 0], 0.0)))
    robot_sigma_y = float(math.sqrt(max(robot_cov[1, 1], 0.0)))
    robot_sigma_theta_deg = float(math.degrees(math.sqrt(max(robot_cov[2, 2], 0.0))))

    n_updates_total = ekf.stats["accepted_updates"] + ekf.stats["rejected_outliers"]

    accepted_ratio = 100.0 * ekf.stats["accepted_updates"] / max(n_updates_total, 1)
    rejected_ratio = 100.0 * ekf.stats["rejected_outliers"] / max(n_updates_total, 1)

    estimated_landmarks = ekf.estimated_landmarks()

    landmark_lines = {}

    for tag_id, xy in sorted(estimated_landmarks.items()):
        landmark_lines[f"tag_{tag_id}_x_m"] = float(xy[0])
        landmark_lines[f"tag_{tag_id}_y_m"] = float(xy[1])

    metrics: Dict[str, Dict[str, Any]] = {
        "Run summary": {
            "bag_path": str(bag_path),
            "stop_reason": stop_reason,
            "duration_s": float(t[-1]) if t.size else float("nan"),
            "history_samples": int(len(history["t"])),
            **counters,
        },

        "Configuration used": {
            "ekf_sigma_motion_x_m_per_step": float(cfg.noise.ekf_sigma_motion_x),
            "ekf_sigma_motion_y_m_per_step": float(cfg.noise.ekf_sigma_motion_y),
            "ekf_sigma_motion_theta_deg_per_step": float(math.degrees(cfg.noise.ekf_sigma_motion_theta)),
            "ekf_sigma_range_m": float(cfg.noise.ekf_sigma_range),
            "ekf_sigma_bearing_deg": float(math.degrees(cfg.noise.ekf_sigma_bearing)),
            "mahalanobis_gate": float(cfg.mahalanobis_gate),
            "min_observations_to_initialize": int(cfg.min_observations_to_initialize),
            "candidate_distance_gate_m": float(cfg.candidate_distance_gate),
            "max_raw_range_innovation_m": float(cfg.max_raw_range_innovation),
            "max_raw_bearing_innovation_deg": float(math.degrees(cfg.max_raw_bearing_innovation)),
        },

        "Micro-simulator ground-truth metrics": {
            "note": "No ground truth pose/map is available in this bag, so true simulator RMSE metrics cannot be computed.",
            "pose_rmse_odom_m": float("nan"),
            "pose_rmse_ekf_m": float("nan"),
            "pose_rmse_reduction_percent": float("nan"),
            "pose_mean_odom_m": float("nan"),
            "pose_mean_ekf_m": float("nan"),
            "landmark_rmse_m": float("nan"),
            "landmark_mean_error_m": float("nan"),
        },

        "Bag trajectory consistency metrics": {
            "odom_path_length_m": path_length_xy(odom),
            "ekf_path_length_m": path_length_xy(ekf_hist),
            "final_loop_error_odom_m": odom_loop_error,
            "final_loop_error_ekf_m": ekf_loop_error,
            "loop_error_reduction_percent": loop_reduction,
            "final_ekf_odom_gap_m": float(ekf_odom_gap[-1]) if ekf_odom_gap.size else float("nan"),
            **finite_stats(ekf_odom_gap, "ekf_odom_gap_m"),
            **finite_stats(odom_steps, "odom_step_m"),
            **finite_stats(ekf_steps, "ekf_step_m"),
            **angle_change_stats(odom, "odom_step_yaw_deg"),
            **angle_change_stats(ekf_hist, "ekf_step_yaw_deg"),
        },

        "Landmark metrics": {
            "n_landmarks_estimated": int(len(estimated_landmarks)),
            "initialized_landmarks": int(ekf.stats["initialized_landmarks"]),
            "candidate_landmarks_remaining": int(len(ekf.candidate_landmarks)),
            "candidate_rejections": int(ekf.stats["candidate_rejections"]),
            **landmark_lines,
        },

        "Visual update metrics": {
            "total_measurements": int(ekf.stats["total_measurements"]),
            "accepted_updates": int(ekf.stats["accepted_updates"]),
            "rejected_outliers": int(ekf.stats["rejected_outliers"]),
            "accepted_ratio_percent": float(accepted_ratio),
            "rejected_ratio_percent": float(rejected_ratio),
            "diagnostic_events_total": int(len(diagnostics)),
            "diagnostic_events_accepted": int(len(accepted_events)),
            "diagnostic_events_rejected": int(len(rejected_events)),
        },

        "NIS consistency metrics": {
            **finite_stats(nis_all, "nis_all"),
            **finite_stats(nis_accepted, "nis_accepted"),
            "nis_accepted_below_95_gate_percent": nis_accepted_below_95,
            "nis_accepted_below_99_gate_percent": nis_accepted_below_99,
        },

        "Innovation metrics": {
            **finite_stats(range_innov_accepted, "abs_range_innovation_accepted_m"),
            **finite_stats(bearing_innov_accepted_deg, "abs_bearing_innovation_accepted_deg"),
        },

        "Final robot covariance": {
            "final_sigma_x_m": robot_sigma_x,
            "final_sigma_y_m": robot_sigma_y,
            "final_sigma_theta_deg": robot_sigma_theta_deg,
            "final_robot_cov_trace": float(np.trace(robot_cov)),
            "max_robot_cov_trace": float(np.nanmax(history["robot_cov_trace"])) if history["robot_cov_trace"] else float("nan"),
            "mean_robot_cov_trace": float(np.nanmean(history["robot_cov_trace"])) if history["robot_cov_trace"] else float("nan"),
        },

        "Diagnostic reason counts": count_reasons(diagnostics),
    }

    return metrics


def format_metric_value(value: Any) -> str:
    """
    Format one value for bag_metrics.txt.
    """
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        return f"{value:.6f}"

    return str(value)


def save_metrics_txt(metrics: Dict[str, Dict[str, Any]], outdir: Path) -> None:
    """
    Save bag EKF-SLAM metrics to a human-readable text file.
    """
    path = outdir / "bag_metrics.txt"

    with path.open("w") as f:
        f.write("Bag EKF-SLAM metrics\n")
        f.write("====================\n\n")

        for section, values in metrics.items():
            f.write(f"{section}\n")
            f.write("-" * len(section) + "\n")

            if not values:
                f.write("  none\n\n")
                continue

            width = max(len(str(key)) for key in values.keys())

            for key, value in values.items():
                f.write(f"{key:<{width}} : {format_metric_value(value)}\n")

            f.write("\n")

CHI2_2D_95 = 5.991  # chi-square 95% for 2D


def covariance_ellipse(
    mean_xy,
    cov_2x2,
    confidence=5.991,
    draw_scale=1.0,
    min_diameter=None,
    **kwargs
):
    cov_2x2 = np.asarray(cov_2x2, dtype=float)

    if cov_2x2.shape != (2, 2):
        return None

    if not np.all(np.isfinite(cov_2x2)):
        return None

    cov_2x2 = 0.5 * (cov_2x2 + cov_2x2.T)

    eigvals, eigvecs = np.linalg.eigh(cov_2x2)

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    eigvals = np.maximum(eigvals, 0.0)

    major_axis = eigvecs[:, 0]
    angle = math.degrees(math.atan2(major_axis[1], major_axis[0]))

    width = 2.0 * math.sqrt(confidence * eigvals[0])
    height = 2.0 * math.sqrt(confidence * eigvals[1])

    width *= draw_scale
    height *= draw_scale

    # Preserve ellipse aspect ratio.
    if min_diameter is not None:
        smallest = min(width, height)
        if smallest < min_diameter and smallest > 0.0:
            scale = min_diameter / smallest
            width *= scale
            height *= scale

    return Ellipse(
        xy=mean_xy,
        width=width,
        height=height,
        angle=angle,
        fill=False,
        **kwargs,
    )

def plot_bag_uncertainty_colored_path(history: dict, outdir: Path) -> None:
    """
    Plot EKF trajectory colored by robot covariance trace.
    """
    ekf_hist = np.asarray(history["ekf"], dtype=float)
    traces = np.asarray(history["robot_cov_trace"], dtype=float)

    if ekf_hist.ndim != 2 or ekf_hist.shape[0] < 2:
        return

    if traces.size < 2:
        return

    points = ekf_hist[:, 0:2]
    segments = np.stack([points[:-1], points[1:]], axis=1)

    lc = LineCollection(
        segments,
        array=traces[:-1],
        linewidth=2.5,
    )

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.add_collection(lc)

    plt.colorbar(lc, ax=ax, label="trace(robot covariance)")

    odom = np.asarray(history["odom"], dtype=float)
    if odom.ndim == 2 and odom.shape[0] > 0:
        plt.plot(
            odom[:, 0],
            odom[:, 1],
            "--",
            linewidth=1.2,
            label="Odometry",
        )

    ax.autoscale()

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Bag EKF trajectory colored by robot uncertainty")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "bag_uncertainty_colored_path.png", dpi=180)
    plt.close()         

# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------


def plot_map(history: dict, ekf: EKFSLAM, outdir: Path) -> None:
    """
    Plot odometry, EKF trajectory and estimated landmarks.

    There is no ground truth here unless you provide an external reference.
    """
    odom = np.asarray(history["odom"], dtype=float)
    ekf_hist = np.asarray(history["ekf"], dtype=float)

    if odom.size == 0 or ekf_hist.size == 0:
        return

    plt.figure(figsize=(9, 8))
    plt.title("Bag EKF-SLAM: odometry vs EKF trajectory")

    plt.plot(odom[:, 0], odom[:, 1], "--", linewidth=1.8, label="Odometry")
    plt.plot(ekf_hist[:, 0], ekf_hist[:, 1], linewidth=2.2, label="EKF-SLAM")

    estimated = ekf.estimated_landmarks()

    for i, (tag_id, (lx, ly)) in enumerate(estimated.items()):
        plt.scatter(
            lx,
            ly,
            marker="x",
            s=100,
            linewidths=2.5,
            label="Estimated ArUco tags" if i == 0 else None,
        )
        plt.text(lx + 0.05, ly + 0.05, f"E{tag_id}", fontsize=9)

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "bag_map.png", dpi=180)
    plt.close()


def plot_nis(ekf: EKFSLAM, outdir: Path) -> None:
    """
    Plot NIS consistency.
    """
    events = [
        e for e in ekf.diagnostics
        if not math.isnan(float(e.get("nis", float("nan"))))
    ]

    if not events:
        return

    t = np.asarray([float(e.get("t", np.nan)) for e in events])
    nis = np.asarray([float(e.get("nis", np.nan)) for e in events])

    plt.figure(figsize=(10, 4))
    plt.title("NIS consistency of visual measurements")
    plt.plot(t, nis, ".", label="NIS")
    plt.axhline(5.99, linestyle="--", label="95% chi-square gate")
    plt.axhline(9.21, linestyle=":", label="99% chi-square gate")
    plt.xlabel("time [s]")
    plt.ylabel("NIS")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "bag_nis.png", dpi=180)
    plt.close()


def plot_innovations(ekf: EKFSLAM, outdir: Path) -> None:
    """
    Plot range and bearing innovations.
    """
    events = [
        e for e in ekf.diagnostics
        if not math.isnan(float(e.get("innovation_range", float("nan"))))
    ]

    if not events:
        return

    t = np.asarray([float(e.get("t", np.nan)) for e in events])
    range_innov = np.asarray([float(e.get("innovation_range", np.nan)) for e in events])
    bearing_innov = np.asarray([float(e.get("innovation_bearing", np.nan)) for e in events])
    accepted = np.asarray([bool(e.get("accepted", False)) for e in events])

    plt.figure(figsize=(10, 4))
    plt.title("Range innovations")
    plt.plot(t[accepted], range_innov[accepted], ".", label="Accepted")
    plt.plot(t[~accepted], range_innov[~accepted], "x", label="Rejected")
    plt.xlabel("time [s]")
    plt.ylabel("range innovation [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "bag_range_innovations.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.title("Bearing innovations")
    plt.plot(t[accepted], np.degrees(bearing_innov[accepted]), ".", label="Accepted")
    plt.plot(t[~accepted], np.degrees(bearing_innov[~accepted]), "x", label="Rejected")
    plt.xlabel("time [s]")
    plt.ylabel("bearing innovation [deg]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "bag_bearing_innovations.png", dpi=180)
    plt.close()


def plot_robot_uncertainty(history: dict, outdir: Path) -> None:
    """
    Plot robot pose standard deviations.
    """
    if len(history["t"]) == 0:
        return

    t = np.asarray(history["t"], dtype=float)

    plt.figure(figsize=(10, 5))
    plt.title("Robot pose uncertainty")
    plt.plot(t, history["robot_sigma_x"], label="sigma x [m]")
    plt.plot(t, history["robot_sigma_y"], label="sigma y [m]")
    plt.plot(t, np.degrees(history["robot_sigma_theta"]), label="sigma theta [deg]")
    plt.xlabel("time [s]")
    plt.ylabel("standard deviation")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "bag_robot_uncertainty.png", dpi=180)
    plt.close()


def plot_cov_trace(history: dict, outdir: Path) -> None:
    """
    Plot trace of robot covariance.
    """
    if len(history["t"]) == 0:
        return

    t = np.asarray(history["t"], dtype=float)

    plt.figure(figsize=(10, 4))
    plt.title("Trace of robot pose covariance")
    plt.plot(t, history["robot_cov_trace"], label="trace(Sigma_robot)")
    plt.xlabel("time [s]")
    plt.ylabel("trace")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "bag_robot_cov_trace.png", dpi=180)
    plt.close()


def plot_update_statistics(history: dict, outdir: Path) -> None:
    """
    Plot cumulative accepted/rejected updates and landmark initialization.
    """
    if len(history["t"]) == 0:
        return

    t = np.asarray(history["t"], dtype=float)

    plt.figure(figsize=(10, 5))
    plt.title("EKF-SLAM update statistics")
    plt.plot(t, history["accepted_updates"], label="Accepted visual updates")
    plt.plot(t, history["rejected_outliers"], label="Rejected measurements")
    plt.plot(t, history["initialized_landmarks"], label="Initialized landmarks")
    plt.plot(t, history["candidate_landmarks"], label="Candidate landmarks")
    plt.xlabel("time [s]")
    plt.ylabel("count")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "bag_update_statistics.png", dpi=180)
    plt.close()


def plot_bag_covariance_map(history: dict, ekf: EKFSLAM, outdir: Path) -> None:
    """
    Plot bag trajectory with 95% covariance ellipses for the final robot pose
    and all initialized landmarks.

    Unlike the simulator, the bag has no true ground truth landmarks unless
    an external reference is provided.
    """
    odom = np.asarray(history["odom"], dtype=float)
    ekf_hist = np.asarray(history["ekf"], dtype=float)

    if odom.size == 0 or ekf_hist.size == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Bag EKF-SLAM map with 95% covariance ellipses")

    plt.plot(odom[:, 0], odom[:, 1], "--", label="Odometry")
    plt.plot(ekf_hist[:, 0], ekf_hist[:, 1], label="EKF-SLAM")

    estimated = ekf.estimated_landmarks()

    for tag_id, (lx, ly) in sorted(estimated.items()):
        plt.scatter(lx, ly, marker="x", s=100, linewidths=2)
        plt.text(lx + 0.05, ly + 0.05, f"E{tag_id}")

        cov = ekf.landmark_covariance(tag_id)

        if cov is not None:
            ell = covariance_ellipse(
                mean_xy=(lx, ly),
                cov_2x2=cov,
                edgecolor="red",
                linewidth=1.5,
                draw_scale=5.0,      # aumenta visualmente 5x
                min_diameter=0.20,   # opcional: nunca fica menor que 20 cm no plot
            )

            if ell is not None:
                ax.add_patch(ell)

    robot_xy = ekf.mu[0:2, 0]
    robot_cov_xy = ekf.robot_covariance()[0:2, 0:2]

    ell = covariance_ellipse(
        mean_xy=robot_xy,
        cov_2x2=robot_cov_xy,
        edgecolor="blue",
        linewidth=1.5,
        linestyle="--",
    )

    if ell is not None:
        ax.add_patch(ell)

    plt.scatter(robot_xy[0], robot_xy[1], marker="o", s=80, label="Final EKF pose")

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "bag_covariance_map.png", dpi=180)
    plt.close()

def save_all_results(history: dict, ekf: EKFSLAM, outdir: Path) -> None:
    """
    Save all CSV files and plots.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    save_covariance_matrices_csv(ekf, outdir)

    save_history_csv(history, outdir)
    save_diagnostics_csv(ekf, outdir)

    plot_map(history, ekf, outdir)
    plot_bag_covariance_map(history, ekf, outdir)
    plot_nis(ekf, outdir)
    plot_innovations(ekf, outdir)
    plot_robot_uncertainty(history, outdir)
    plot_cov_trace(history, outdir)
    plot_update_statistics(history, outdir)
    plot_bag_uncertainty_colored_path(history, outdir)
    
    


# ---------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    cfg0 = SimConfig()

    parser = argparse.ArgumentParser(
        description="Offline EKF-SLAM processor for ROS 2 bags."
    )

    parser.add_argument("--bag", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="bag_ekf_results")
    parser.add_argument("--clean", action="store_true")

    parser.add_argument("--max_duration", type=float, default=None)
    parser.add_argument("--stop_on_loop", action="store_true")
    parser.add_argument("--loop_min_time", type=float, default=20.0)
    parser.add_argument("--loop_xy_threshold", type=float, default=0.35)
    parser.add_argument("--loop_yaw_threshold_deg", type=float, default=45.0)

    # EKF prediction uncertainty.
    parser.add_argument("--ekf_sigma_motion_x", type=float, default=cfg0.noise.ekf_sigma_motion_x)
    parser.add_argument("--ekf_sigma_motion_y", type=float, default=cfg0.noise.ekf_sigma_motion_y)
    parser.add_argument(
        "--ekf_sigma_motion_theta_deg",
        type=float,
        default=math.degrees(cfg0.noise.ekf_sigma_motion_theta),
    )

    # EKF visual correction uncertainty.
    parser.add_argument("--ekf_sigma_range", type=float, default=cfg0.noise.ekf_sigma_range)
    parser.add_argument(
        "--ekf_sigma_bearing_deg",
        type=float,
        default=math.degrees(cfg0.noise.ekf_sigma_bearing),
    )

    # Gating and landmark initialization.
    parser.add_argument("--mahalanobis_gate", type=float, default=cfg0.mahalanobis_gate)
    parser.add_argument(
        "--min_observations_to_initialize",
        type=int,
        default=cfg0.min_observations_to_initialize,
    )
    parser.add_argument(
        "--candidate_distance_gate",
        type=float,
        default=cfg0.candidate_distance_gate,
    )

    parser.add_argument(
        "--max_raw_range_innovation",
        type=float,
        default=cfg0.max_raw_range_innovation,
    )
    parser.add_argument(
        "--max_raw_bearing_innovation_deg",
        type=float,
        default=math.degrees(cfg0.max_raw_bearing_innovation),
    )

    return parser

# ---------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------


def main() -> None:
    args = build_arg_parser().parse_args()

    bag_path = Path(args.bag).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()

    if not bag_path.exists():
        raise FileNotFoundError(f"Bag folder not found: {bag_path}")

    if args.clean and outdir.exists():
        shutil.rmtree(outdir)

    cfg = configure_for_bag(SimConfig(), args)
    ekf = EKFSLAM(cfg)

    reader = open_bag_reader(bag_path)

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}

    required_topics = ["/odom", "/aruco_landmarks"]
    for topic in required_topics:
        if topic not in type_map:
            raise RuntimeError(f"Required topic not found in bag: {topic}")

    msg_type_map = {
        topic: get_message(type_name)
        for topic, type_name in type_map.items()
    }

    prev_odom_pose: np.ndarray | None = None
    latest_odom_pose: np.ndarray | None = None
    initial_odom_pose: np.ndarray | None = None

    start_time_ns: int | None = None
    latest_time: float = 0.0

    history = {
        "t": [],
        "odom": [],
        "ekf": [],
        "n_measurements": [],
        "accepted_updates": [],
        "rejected_outliers": [],
        "robot_sigma_x": [],
        "robot_sigma_y": [],
        "robot_sigma_theta": [],
        "robot_cov_trace": [],
        "initialized_landmarks": [],
        "candidate_landmarks": [],
    }

    def store_history(n_measurements: int) -> None:
        if latest_odom_pose is None:
            return

        robot_cov = ekf.robot_covariance()

        history["t"].append(float(latest_time))
        history["odom"].append(latest_odom_pose.copy())
        history["ekf"].append(ekf.mu[0:3, 0].copy())

        history["n_measurements"].append(int(n_measurements))
        history["accepted_updates"].append(int(ekf.stats["accepted_updates"]))
        history["rejected_outliers"].append(int(ekf.stats["rejected_outliers"]))

        history["robot_sigma_x"].append(float(math.sqrt(max(robot_cov[0, 0], 0.0))))
        history["robot_sigma_y"].append(float(math.sqrt(max(robot_cov[1, 1], 0.0))))
        history["robot_sigma_theta"].append(float(math.sqrt(max(robot_cov[2, 2], 0.0))))
        history["robot_cov_trace"].append(float(np.trace(robot_cov)))

        history["initialized_landmarks"].append(int(ekf.stats["initialized_landmarks"]))
        history["candidate_landmarks"].append(int(len(ekf.candidate_landmarks)))

    n_messages_used = 0
    n_odom = 0
    n_aruco_msgs = 0
    n_aruco_measurements = 0
    n_total_bag_msgs = 0

    stop_reason = "bag ended"

    print(f"\nProcessing bag: {bag_path}")
    print("This processes the bag offline, as fast as possible.\n")

    while reader.has_next():
        topic, data, timestamp_ns = reader.read_next()
        n_total_bag_msgs += 1

        if start_time_ns is None:
            start_time_ns = timestamp_ns

        latest_time = 1e-9 * float(timestamp_ns - start_time_ns)

        # Optional max duration.
        if args.max_duration is not None and latest_time > args.max_duration:
            stop_reason = f"max_duration reached ({args.max_duration:.2f} s)"
            break

        # Ignore topics that are not needed by this EKF.
        if topic not in ["/odom", "/aruco_landmarks"]:
            continue

        msg_type = msg_type_map[topic]
        msg = deserialize_message(data, msg_type)

        n_messages_used += 1

        if topic == "/odom":
            if not isinstance(msg, Odometry):
                continue

            odom_pose = pose_from_odom(msg)
            latest_odom_pose = odom_pose.copy()
            n_odom += 1

            if initial_odom_pose is None:
                initial_odom_pose = odom_pose.copy()

            if prev_odom_pose is None:
                prev_odom_pose = odom_pose.copy()

                # EKF starts at the first odometry pose.
                ekf.mu[0:3, 0] = odom_pose

                store_history(n_measurements=0)
                continue

            d_rot1, d_trans, d_rot2 = odometry_increment(
                prev_odom_pose,
                odom_pose,
            )

            ekf.predict_from_odometry(d_rot1, d_trans, d_rot2)
            prev_odom_pose = odom_pose.copy()

            store_history(n_measurements=0)

            # Optional loop closure stop.
            if (
                args.stop_on_loop
                and initial_odom_pose is not None
                and loop_closed(
                    current_pose=odom_pose,
                    initial_pose=initial_odom_pose,
                    elapsed_time=latest_time,
                    min_time=args.loop_min_time,
                    xy_threshold=args.loop_xy_threshold,
                    yaw_threshold=math.radians(args.loop_yaw_threshold_deg),
                )
            ):
                stop_reason = (
                    "loop closure detected from odometry "
                    f"(t={latest_time:.2f}s)"
                )
                break

        elif topic == "/aruco_landmarks":
            if latest_odom_pose is None:
                continue

            if not isinstance(msg, Float32MultiArray):
                continue

            n_aruco_msgs += 1

            measurements_with_debug = parse_aruco_landmarks(msg)

            if len(measurements_with_debug) == 0:
                continue

            ekf_measurements = [
                {
                    "id": m["id"],
                    "range": m["range"],
                    "bearing": m["bearing"],
                }
                for m in measurements_with_debug
            ]

            n_aruco_measurements += len(ekf_measurements)

            before_events = len(ekf.diagnostics)
            ekf.update(ekf_measurements)

            # Add bag time to diagnostics generated by this update.
            for event in ekf.diagnostics[before_events:]:
                event["t"] = latest_time

            store_history(n_measurements=len(ekf_measurements))

        if n_messages_used % 500 == 0:
            print(
                f"used={n_messages_used} | "
                f"bag_msgs={n_total_bag_msgs} | "
                f"t={latest_time:.1f}s | "
                f"odom={n_odom} | "
                f"aruco_msgs={n_aruco_msgs} | "
                f"aruco_meas={n_aruco_measurements} | "
                f"landmarks={ekf.stats['initialized_landmarks']} | "
                f"accepted={ekf.stats['accepted_updates']} | "
                f"rejected={ekf.stats['rejected_outliers']}"
            )

    counters = {
    "total_bag_messages_read": int(n_total_bag_msgs),
    "used_messages_odom_plus_aruco": int(n_messages_used),
    "processed_odom_messages": int(n_odom),
    "processed_aruco_messages": int(n_aruco_msgs),
    "processed_aruco_measurements": int(n_aruco_measurements),
}

    save_all_results(history, ekf, outdir)

    metrics = compute_bag_metrics(
        history=history,
        ekf=ekf,
        cfg=cfg,
        bag_path=bag_path,
        stop_reason=stop_reason,
        counters=counters,
    )

    save_metrics_txt(metrics, outdir)

    print("\nDone.")
    print(f"Stop reason: {stop_reason}")
    print(f"Total bag messages read: {n_total_bag_msgs}")
    print(f"Used messages (/odom + /aruco_landmarks): {n_messages_used}")
    print(f"Processed odom messages: {n_odom}")
    print(f"Processed aruco messages: {n_aruco_msgs}")
    print(f"Processed aruco measurements: {n_aruco_measurements}")
    print(f"Initialized landmarks: {ekf.stats['initialized_landmarks']}")
    print(f"Accepted updates: {ekf.stats['accepted_updates']}")
    print(f"Rejected measurements: {ekf.stats['rejected_outliers']}")
    print(f"Metrics text file: {outdir / 'bag_metrics.txt'}")
    print(f"Results saved to: {outdir}")


if __name__ == "__main__":
    rclpy.init()
    try:
        main()
    finally:
        rclpy.shutdown()