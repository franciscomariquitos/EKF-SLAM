"""Plotting functions for the EKF-SLAM micro-simulator."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse

from ekf_slam import EKFSLAM

CHI2_2D_95 = 5.991
CHI2_2D_99 = 9.210


def _save_or_show(path: Path, show: bool) -> None:
    """Save current matplotlib figure and either show or close it."""
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    if show:
        plt.show()
    else:
        plt.close()


def _pose_errors(history: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return odometry and EKF 2D position errors over time."""
    odom_error = np.linalg.norm(history["true"][:, 0:2] - history["odom"][:, 0:2], axis=1)
    ekf_error = np.linalg.norm(history["true"][:, 0:2] - history["ekf"][:, 0:2], axis=1)
    return odom_error, ekf_error


def _covariance_ellipse(mean_xy: Iterable[float], cov_2x2: np.ndarray, confidence_scale: float = CHI2_2D_95, **kwargs) -> Ellipse | None:
    """Create a 2D covariance ellipse patch from mean and 2x2 covariance."""
    cov_2x2 = np.asarray(cov_2x2, dtype=float)
    if cov_2x2.shape != (2, 2) or not np.all(np.isfinite(cov_2x2)):
        return None

    eigvals, eigvecs = np.linalg.eigh(cov_2x2)
    eigvals = np.maximum(eigvals, 0.0)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    angle = math.degrees(math.atan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2.0 * np.sqrt(confidence_scale * eigvals)
    if not np.isfinite(width) or not np.isfinite(height):
        return None

    return Ellipse(xy=mean_xy, width=width, height=height, angle=angle, fill=False, **kwargs)


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
        f"RMSE EKF={metrics['pose_rmse_ekf_m']:.3f} m | "
        f"reduction={metrics['pose_rmse_reduction_percent']:.1f}%"
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
    _save_or_show(outdir / "single_run_map.png", show)


def plot_errors(result: dict, outdir: Path, show: bool) -> None:
    """Plot odometry and EKF position error over time."""
    history = result["history"]
    t = history["t"]
    odom_error, ekf_error = _pose_errors(history)

    plt.figure(figsize=(10, 5))
    plt.title("Pose error over time")
    plt.plot(t, odom_error, label="Odometry error")
    plt.plot(t, ekf_error, label="EKF-SLAM error")
    plt.xlabel("time [s]")
    plt.ylabel("position error [m]")
    plt.grid(True)
    plt.legend()
    _save_or_show(outdir / "single_run_errors.png", show)


def plot_error_improvement(result: dict, outdir: Path, show: bool) -> None:
    """Plot instantaneous EKF improvement over odometry in percent."""
    history = result["history"]
    t = history["t"]
    odom_error, ekf_error = _pose_errors(history)
    improvement = np.full_like(odom_error, np.nan, dtype=float)
    mask = odom_error > 1e-9
    improvement[mask] = 100.0 * (odom_error[mask] - ekf_error[mask]) / odom_error[mask]

    plt.figure(figsize=(10, 5))
    plt.title("Instantaneous EKF improvement over odometry")
    plt.plot(t, improvement, label="Improvement")
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("time [s]")
    plt.ylabel("improvement [%]")
    plt.grid(True)
    plt.legend()
    _save_or_show(outdir / "single_run_improvement.png", show)


def plot_covariance_map(result: dict, outdir: Path, show: bool) -> None:
    """Plot final map with 95% covariance ellipses for robot and landmarks."""
    history = result["history"]
    landmarks = result["landmarks"]
    ekf: EKFSLAM = result["ekf"]

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    plt.title("Final EKF-SLAM map with 95% covariance ellipses")
    plt.plot(history["true"][:, 0], history["true"][:, 1], label="Ground truth", linewidth=2.0)
    plt.plot(history["odom"][:, 0], history["odom"][:, 1], label="Odometry", linestyle="--", linewidth=1.5)
    plt.plot(history["ekf"][:, 0], history["ekf"][:, 1], label="EKF-SLAM", linewidth=2.0)

    for i, lm in enumerate(landmarks):
        tag_id, lx, ly = int(lm[0]), lm[1], lm[2]
        plt.scatter(lx, ly, marker="s", s=90, alpha=0.45, label="True tags" if i == 0 else None)
        plt.text(lx + 0.06, ly + 0.06, f"T{tag_id}", fontsize=8)

    for i, (tag_id, (lx, ly)) in enumerate(ekf.estimated_landmarks().items()):
        plt.scatter(lx, ly, marker="x", s=90, linewidths=2.0, label="Estimated tags" if i == 0 else None)
        cov = ekf.landmark_covariance(tag_id)
        if cov is not None:
            ell = _covariance_ellipse((lx, ly), cov, edgecolor="black", linewidth=1.2, alpha=0.75)
            if ell is not None:
                ax.add_patch(ell)

    robot_xy = history["ekf"][-1, 0:2]
    robot_cov = ekf.robot_covariance()[0:2, 0:2]
    ell = _covariance_ellipse(robot_xy, robot_cov, edgecolor="black", linewidth=2.0, alpha=0.9, linestyle="--")
    if ell is not None:
        ax.add_patch(ell)
    plt.scatter(robot_xy[0], robot_xy[1], marker="o", s=80, label="Final EKF pose")

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    _save_or_show(outdir / "single_run_covariance_map.png", show)


def plot_robot_covariance_snapshots(result: dict, outdir: Path, show: bool) -> None:
    """Plot EKF trajectory colored by robot covariance trace."""
    history = result["history"]
    points = history["ekf"][:, 0:2]
    traces = history.get("robot_cov_trace")
    if traces is None or len(points) < 2:
        return

    segments = np.stack([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, array=np.asarray(traces[:-1], dtype=float), linewidth=2.5)

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.add_collection(lc)
    plt.colorbar(lc, ax=ax, label="trace(robot covariance)")
    plt.plot(history["true"][:, 0], history["true"][:, 1], linestyle="--", linewidth=1.2, label="Ground truth")
    ax.autoscale()
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("EKF trajectory colored by robot uncertainty")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    _save_or_show(outdir / "single_run_uncertainty_colored_path.png", show)


def plot_error_heatmap(result: dict, outdir: Path, show: bool) -> None:
    """Plot EKF trajectory colored by instantaneous position error."""
    history = result["history"]
    _, ekf_error = _pose_errors(history)
    points = history["ekf"][:, 0:2]
    if len(points) < 2:
        return

    segments = np.stack([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, array=ekf_error[:-1], linewidth=3.0)

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.add_collection(lc)
    plt.colorbar(lc, ax=ax, label="EKF position error [m]")
    plt.plot(history["true"][:, 0], history["true"][:, 1], linestyle="--", linewidth=1.2, label="Ground truth")
    ax.autoscale()
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("EKF trajectory colored by position error")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    _save_or_show(outdir / "single_run_error_heatmap.png", show)


def plot_landmark_convergence(result: dict, outdir: Path, show: bool) -> None:
    """Plot global and per-landmark mapping error over time."""
    history = result["history"]
    t = history["t"]

    plt.figure(figsize=(10, 5))
    plt.title("Landmark map convergence")
    if "landmark_rmse_over_time" in history:
        plt.plot(t, history["landmark_rmse_over_time"], label="Landmark RMSE")
    if "landmark_mean_error_over_time" in history:
        plt.plot(t, history["landmark_mean_error_over_time"], label="Landmark mean error")
    plt.xlabel("time [s]")
    plt.ylabel("error [m]")
    plt.grid(True)
    plt.legend()
    _save_or_show(outdir / "single_run_landmark_convergence.png", show)

    # Per-tag errors.
    per_tag_history = history.get("landmark_errors_by_tag", [])
    tag_ids = sorted({tag_id for snapshot in per_tag_history for tag_id in snapshot.keys()})
    if not tag_ids:
        return

    plt.figure(figsize=(10, 5))
    plt.title("Per-landmark error over time")
    for tag_id in tag_ids:
        values = np.array([snapshot.get(tag_id, np.nan) for snapshot in per_tag_history], dtype=float)
        plt.plot(t, values, label=f"Tag {tag_id}")
    plt.xlabel("time [s]")
    plt.ylabel("landmark error [m]")
    plt.grid(True)
    plt.legend(ncol=2)
    _save_or_show(outdir / "single_run_landmark_errors_by_tag.png", show)


def plot_nis(result: dict, outdir: Path, show: bool) -> None:
    """Plot NIS values and chi-square gates for visual consistency analysis."""
    ekf: EKFSLAM = result["ekf"]
    events = [e for e in ekf.diagnostics if np.isfinite(e.get("nis", np.nan))]
    if not events:
        return

    t = np.array([e.get("t", i) for i, e in enumerate(events)], dtype=float)
    nis = np.array([e["nis"] for e in events], dtype=float)
    accepted = np.array([bool(e.get("accepted", False)) for e in events])

    plt.figure(figsize=(10, 5))
    plt.title("NIS consistency of visual measurements")
    plt.scatter(t[accepted], nis[accepted], s=18, label="accepted")
    plt.scatter(t[~accepted], nis[~accepted], s=18, marker="x", label="rejected")
    plt.axhline(CHI2_2D_95, linestyle="--", linewidth=1.2, label="95% gate (5.99)")
    plt.axhline(CHI2_2D_99, linestyle=":", linewidth=1.5, label="99% gate (9.21)")
    plt.xlabel("time [s]")
    plt.ylabel("NIS = innovationᵀ S⁻¹ innovation")
    plt.grid(True)
    plt.legend()
    _save_or_show(outdir / "single_run_nis.png", show)

    plt.figure(figsize=(8, 5))
    plt.title("NIS histogram")
    plt.hist(nis[np.isfinite(nis)], bins=25, alpha=0.8)
    plt.axvline(CHI2_2D_95, linestyle="--", linewidth=1.2, label="95% gate")
    plt.axvline(CHI2_2D_99, linestyle=":", linewidth=1.5, label="99% gate")
    plt.xlabel("NIS")
    plt.ylabel("count")
    plt.grid(True)
    plt.legend()
    _save_or_show(outdir / "single_run_nis_histogram.png", show)


def plot_innovations(result: dict, outdir: Path, show: bool) -> None:
    """Plot range and bearing innovations for known-landmark visual updates."""
    ekf: EKFSLAM = result["ekf"]
    events = [e for e in ekf.diagnostics if np.isfinite(e.get("innovation_range", np.nan))]
    if not events:
        return

    t = np.array([e.get("t", i) for i, e in enumerate(events)], dtype=float)
    rng_innov = np.array([e["innovation_range"] for e in events], dtype=float)
    bearing_innov_deg = np.degrees(np.array([e["innovation_bearing"] for e in events], dtype=float))
    accepted = np.array([bool(e.get("accepted", False)) for e in events])

    plt.figure(figsize=(10, 5))
    plt.title("Visual measurement innovations")
    plt.scatter(t[accepted], rng_innov[accepted], s=18, label="range innovation accepted [m]")
    plt.scatter(t[~accepted], rng_innov[~accepted], s=20, marker="x", label="range innovation rejected [m]")
    plt.xlabel("time [s]")
    plt.ylabel("range innovation [m]")
    plt.grid(True)
    plt.legend()
    _save_or_show(outdir / "single_run_range_innovations.png", show)

    plt.figure(figsize=(10, 5))
    plt.title("Bearing innovations")
    plt.scatter(t[accepted], bearing_innov_deg[accepted], s=18, label="accepted")
    plt.scatter(t[~accepted], bearing_innov_deg[~accepted], s=20, marker="x", label="rejected")
    plt.xlabel("time [s]")
    plt.ylabel("bearing innovation [deg]")
    plt.grid(True)
    plt.legend()
    _save_or_show(outdir / "single_run_bearing_innovations.png", show)


def plot_robot_uncertainty(result: dict, outdir: Path, show: bool) -> None:
    """Plot robot covariance standard deviations and trace over time."""
    history = result["history"]
    t = history["t"]
    if "robot_sigma_x" not in history:
        return

    plt.figure(figsize=(10, 5))
    plt.title("Robot pose uncertainty over time")
    plt.plot(t, history["robot_sigma_x"], label="σx [m]")
    plt.plot(t, history["robot_sigma_y"], label="σy [m]")
    plt.plot(t, np.degrees(history["robot_sigma_theta"]), label="σθ [deg]")
    plt.xlabel("time [s]")
    plt.ylabel("standard deviation")
    plt.grid(True)
    plt.legend()
    _save_or_show(outdir / "single_run_robot_uncertainty.png", show)

    plt.figure(figsize=(10, 5))
    plt.title("Trace of robot covariance")
    plt.plot(t, history["robot_cov_trace"], label="trace(Σ_robot)")
    plt.xlabel("time [s]")
    plt.ylabel("trace")
    plt.grid(True)
    plt.legend()
    _save_or_show(outdir / "single_run_robot_cov_trace.png", show)


def plot_update_statistics(result: dict, outdir: Path, show: bool) -> None:
    """Plot measurement, accepted-update and rejection statistics over time."""
    history = result["history"]
    t = history["t"]

    plt.figure(figsize=(10, 5))
    plt.title("Visual update statistics")
    plt.plot(t, history["n_measurements"], label="measurements per camera frame")
    plt.plot(t, history["initialized_landmarks"], label="initialized landmarks")
    plt.plot(t, history["candidate_landmarks"], label="candidate landmarks")
    plt.xlabel("time [s]")
    plt.ylabel("count")
    plt.grid(True)
    plt.legend()
    _save_or_show(outdir / "single_run_update_statistics.png", show)

    plt.figure(figsize=(10, 5))
    plt.title("Accepted updates and rejected outliers")
    plt.plot(t, history["accepted_updates"], label="accepted updates")
    plt.plot(t, history["rejected_outliers"], label="rejected outliers")
    plt.xlabel("time [s]")
    plt.ylabel("cumulative count")
    plt.grid(True)
    plt.legend()
    _save_or_show(outdir / "single_run_update_counters.png", show)


def plot_monte_carlo_boxplots(rows: list[dict], outdir: Path, show: bool) -> None:
    """Create boxplots for Monte Carlo metrics."""
    if not rows:
        return

    metrics = {
        "Odom RMSE [m]": [row.get("pose_rmse_odom_m", np.nan) for row in rows],
        "EKF RMSE [m]": [row.get("pose_rmse_ekf_m", np.nan) for row in rows],
        "Odom loop [m]": [row.get("final_loop_error_odom_m", np.nan) for row in rows],
        "EKF loop [m]": [row.get("final_loop_error_ekf_m", np.nan) for row in rows],
        "Landmark RMSE [m]": [row.get("landmark_rmse_m", np.nan) for row in rows],
    }

    labels = list(metrics.keys())
    values = [np.asarray(metrics[label], dtype=float) for label in labels]

    plt.figure(figsize=(12, 5))
    plt.title("Monte Carlo metric distributions")
    plt.boxplot(values, labels=labels, showmeans=True)
    plt.ylabel("metric value")
    plt.grid(True, axis="y")
    plt.xticks(rotation=20, ha="right")
    _save_or_show(outdir / "monte_carlo_boxplots.png", show)


def plot_all_diagnostics(result: dict, outdir: Path, show: bool) -> None:
    """Generate the full set of single-run diagnostic figures."""
    plot_map(result, outdir, show)
    plot_errors(result, outdir, show)
    plot_error_improvement(result, outdir, show)
    plot_covariance_map(result, outdir, show)
    plot_robot_covariance_snapshots(result, outdir, show)
    plot_error_heatmap(result, outdir, show)
    plot_landmark_convergence(result, outdir, show)
    plot_nis(result, outdir, show)
    plot_innovations(result, outdir, show)
    plot_robot_uncertainty(result, outdir, show)
    plot_update_statistics(result, outdir, show)
