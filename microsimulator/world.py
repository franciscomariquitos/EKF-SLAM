"""Simulated world: waypoints, landmarks, controller and visual sensor model."""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from config import SimConfig
from utils import normalize_angle


def default_waypoints() -> np.ndarray:
    """Closed trajectory used to test loop closure."""
    return np.array([
        [0.0, 0.0],
        [5.0, 0.0],
        [5.0, 5.0],
        [0.0, 5.0],
        [0.0, 0.0],
    ], dtype=float)


def default_landmarks() -> np.ndarray:
    """
    ArUco/AprilTag-like landmarks in the form [id, x, y].

    The geometry intentionally includes landmarks inside and outside the square to create
    regions with good and poor observation geometry.
    """
    return np.array([
        [1, 1.0, 1.0],
        [2, 4.2, -0.6],
        [3, 6.0, 2.2],
        [4, 4.7, 5.4],
        [5, 0.6, 4.4],
        [6, -0.8, 2.0],
    ], dtype=float)


def waypoint_controller(pose: np.ndarray, target: np.ndarray, cfg: SimConfig) -> Tuple[float, float]:
    """Proportional waypoint controller with speed saturation."""
    x, y, theta = pose
    dx = target[0] - x
    dy = target[1] - y
    distance = math.hypot(dx, dy)
    desired_heading = math.atan2(dy, dx)
    heading_error = normalize_angle(desired_heading - theta)

    k_v = 0.9
    k_w = 2.3

    v = min(cfg.max_v, k_v * distance)
    v *= max(0.0, math.cos(heading_error))

    w = np.clip(k_w * heading_error, -cfg.max_w, cfg.max_w)
    return float(v), float(w)


def generate_visual_measurements(
    true_pose: np.ndarray,
    landmarks: np.ndarray,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> List[dict]:
    """
    Simulate ArUco/AprilTag-like visual measurements.

    Returned measurements contain:
        id, range, bearing
    """
    x, y, theta = true_pose
    readings: List[dict] = []

    for lm in landmarks:
        tag_id = int(lm[0])
        lx, ly = float(lm[1]), float(lm[2])

        dx = lx - x
        dy = ly - y
        true_range = math.hypot(dx, dy)
        true_bearing = normalize_angle(math.atan2(dy, dx) - theta)

        visible = true_range <= cfg.sensor.max_range and abs(true_bearing) <= cfg.sensor.fov / 2.0
        if not visible:
            continue

        if rng.random() > cfg.sensor.detection_probability:
            continue

        measured_range = true_range + rng.normal(0.0, cfg.noise.sigma_range)
        measured_bearing = true_bearing + rng.normal(0.0, cfg.noise.sigma_bearing)

        # Occasional visual outlier: bad detection or unstable pose estimation.
        if rng.random() < cfg.sensor.outlier_probability:
            measured_range += rng.normal(0.7, 0.25)
            measured_bearing += rng.normal(math.radians(20.0), math.radians(8.0))

        readings.append({
            "id": tag_id,
            "range": max(0.01, float(measured_range)),
            "bearing": normalize_angle(float(measured_bearing)),
        })

    return readings
