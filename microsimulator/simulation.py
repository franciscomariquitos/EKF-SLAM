"""Simulation loop: ground truth, noisy odometry, visual measurements and EKF-SLAM."""

from __future__ import annotations

from typing import List

import numpy as np

from config import SimConfig
from ekf_slam import EKFSLAM
from evaluation import compute_metrics, landmark_error_snapshot
from utils import odometry_increment, pose_step
from world import default_landmarks, default_waypoints, generate_visual_measurements, waypoint_controller

import math


def _append_diagnostics(history: dict, ekf: EKFSLAM, landmarks: np.ndarray) -> None:
    """Append covariance and landmark-convergence diagnostics for the current EKF state."""
    robot_cov = ekf.robot_covariance()
    landmark_snapshot = landmark_error_snapshot(ekf, landmarks)

    history["robot_sigma_x"].append(float(np.sqrt(max(robot_cov[0, 0], 0.0))))
    history["robot_sigma_y"].append(float(np.sqrt(max(robot_cov[1, 1], 0.0))))
    history["robot_sigma_theta"].append(float(np.sqrt(max(robot_cov[2, 2], 0.0))))
    history["robot_cov_trace"].append(float(np.trace(robot_cov)))
    history["initialized_landmarks"].append(int(ekf.stats["initialized_landmarks"]))
    history["candidate_landmarks"].append(int(len(ekf.candidate_landmarks)))
    history["landmark_rmse_over_time"].append(float(landmark_snapshot["rmse"]))
    history["landmark_mean_error_over_time"].append(float(landmark_snapshot["mean"]))
    history["n_landmarks_estimated_over_time"].append(int(landmark_snapshot["n"]))
    history["landmark_errors_by_tag"].append(landmark_snapshot["per_tag"])


def run_single_simulation(cfg: SimConfig) -> dict:
    """Run one complete simulated trajectory."""
    rng = np.random.default_rng(cfg.seed)
    waypoints = default_waypoints()
    landmarks = default_landmarks()

    ekf = EKFSLAM(cfg)
    ekf.mu[2, 0] = math.pi

    true_pose = np.array([waypoints[0, 0], waypoints[0, 1], math.pi], dtype=float)
    odom_pose = true_pose.copy()
    prev_odom_pose = odom_pose.copy()

    camera_period_steps = max(1, int(round((1.0 / cfg.sensor.camera_rate_hz) / cfg.dt)))

    history = {
        "t": [],
        "true": [],
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
        "landmark_rmse_over_time": [],
        "landmark_mean_error_over_time": [],
        "n_landmarks_estimated_over_time": [],
        "landmark_errors_by_tag": [],
    }

    t = 0.0
    global_step = 0

    for wp_idx in range(1, len(waypoints)):
        target = waypoints[wp_idx]

        for _ in range(cfg.max_steps_per_segment):
            distance_to_target = np.linalg.norm(true_pose[0:2] - target)
            if distance_to_target <= cfg.goal_tolerance:
                break

            # 1) Ground truth motion.
            v_true, w_true = waypoint_controller(true_pose, target, cfg)
            true_pose = pose_step(true_pose, v_true, w_true, cfg.dt)

            # 2) Noisy odometry with systematic drift.
            noisy_v = v_true * (1.0 + cfg.noise.v_bias) + rng.normal(0.0, cfg.noise.sigma_v_odom)
            noisy_w = w_true * (1.0 + cfg.noise.w_bias) + rng.normal(0.0, cfg.noise.sigma_w_odom)
            odom_pose = pose_step(odom_pose, noisy_v, noisy_w, cfg.dt)

            # 3) EKF prediction from odometry increments, like later with ROS 2 /odom.
            d_rot1, d_trans, d_rot2 = odometry_increment(prev_odom_pose, odom_pose)
            ekf.predict_from_odometry(d_rot1, d_trans, d_rot2)
            prev_odom_pose = odom_pose.copy()

            # 4) Visual correction at camera frequency, not odometry frequency.
            measurements: List[dict] = []
            if global_step % camera_period_steps == 0:
                measurements = generate_visual_measurements(true_pose, landmarks, cfg, rng)
                before_events = len(ekf.diagnostics)
                ekf.update(measurements)
                # Attach simulation time to the diagnostic events created during this camera frame.
                for event in ekf.diagnostics[before_events:]:
                    event["t"] = t
                    event["step"] = global_step

            # 5) Store history.
            history["t"].append(t)
            history["true"].append(true_pose.copy())
            history["odom"].append(odom_pose.copy())
            history["ekf"].append(ekf.mu[0:3, 0].copy())
            history["n_measurements"].append(len(measurements))
            history["accepted_updates"].append(ekf.stats["accepted_updates"])
            history["rejected_outliers"].append(ekf.stats["rejected_outliers"])
            _append_diagnostics(history, ekf, landmarks)

            t += cfg.dt
            global_step += 1

    numeric_keys = [
        "true", "odom", "ekf", "t", "n_measurements", "accepted_updates", "rejected_outliers",
        "robot_sigma_x", "robot_sigma_y", "robot_sigma_theta", "robot_cov_trace",
        "initialized_landmarks", "candidate_landmarks", "landmark_rmse_over_time",
        "landmark_mean_error_over_time", "n_landmarks_estimated_over_time",
    ]
    for key in numeric_keys:
        history[key] = np.asarray(history[key], dtype=float)

    metrics = compute_metrics(history, ekf, landmarks, waypoints)

    return {
        "cfg": cfg,
        "history": history,
        "ekf": ekf,
        "landmarks": landmarks,
        "waypoints": waypoints,
        "metrics": metrics,
    }


def run_monte_carlo(base_cfg: SimConfig, runs: int) -> List[dict]:
    """Run several simulations with different seeds."""
    all_metrics = []
    for i in range(runs):
        cfg = SimConfig(
            dt=base_cfg.dt,
            max_v=base_cfg.max_v,
            max_w=base_cfg.max_w,
            goal_tolerance=base_cfg.goal_tolerance,
            max_steps_per_segment=base_cfg.max_steps_per_segment,
            mahalanobis_gate=base_cfg.mahalanobis_gate,
            min_observations_to_initialize=base_cfg.min_observations_to_initialize,
            candidate_distance_gate=base_cfg.candidate_distance_gate,
            seed=base_cfg.seed + i,
            noise=base_cfg.noise,
            sensor=base_cfg.sensor,
        )
        result = run_single_simulation(cfg)
        row = {"run": i + 1, "seed": cfg.seed}
        row.update(result["metrics"])
        all_metrics.append(row)
    return all_metrics
