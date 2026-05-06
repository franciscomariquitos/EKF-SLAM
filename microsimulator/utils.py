"""Small mathematical utilities used by the EKF-SLAM micro-simulator."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def normalize_angle(angle: float) -> float:
    """Normalize an angle to the interval [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def pose_step(pose: np.ndarray, v: float, w: float, dt: float) -> np.ndarray:
    """
    Integrate a simple unicycle model.

    pose = [x, y, theta]
    v    = linear velocity [m/s]
    w    = angular velocity [rad/s]
    dt   = time step [s]
    """
    x, y, theta = pose
    theta_new = normalize_angle(theta + w * dt)
    x_new = x + v * dt * math.cos(theta_new)
    y_new = y + v * dt * math.sin(theta_new)
    return np.array([x_new, y_new, theta_new], dtype=float)


def odometry_increment(prev_odom: np.ndarray, curr_odom: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert two consecutive odometry poses into the odometry motion model:

        u = (delta_rot1, delta_trans, delta_rot2)

    This is closer to the ROS 2 /odom workflow than directly giving v,w to the EKF.
    """
    x1, y1, th1 = prev_odom
    x2, y2, th2 = curr_odom

    dx = x2 - x1
    dy = y2 - y1
    delta_trans = math.hypot(dx, dy)

    if delta_trans < 1e-12:
        delta_rot1 = 0.0
    else:
        delta_rot1 = normalize_angle(math.atan2(dy, dx) - th1)

    delta_rot2 = normalize_angle(th2 - th1 - delta_rot1)
    return delta_rot1, delta_trans, delta_rot2


def rmse(values: np.ndarray) -> float:
    """Root mean square of a vector of errors."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values**2)))
