"""Configuration dataclasses for the EKF-SLAM micro-simulator."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class NoiseConfig:
    """Noise parameters used both to generate data and to configure the EKF."""

    # Noise used to generate simulated wheel odometry.
    sigma_v_odom: float = 0.025       # [m/s]
    sigma_w_odom: float = 0.025       # [rad/s]
    v_bias: float = 0.025             # systematic relative linear drift
    w_bias: float = -0.015            # systematic relative angular drift

    # Noise used to generate ArUco/AprilTag-like visual measurements.
    sigma_range: float = 0.04         # [m]
    sigma_bearing: float = math.radians(2.0)  # [rad]

    # Motion noise assumed by the EKF during prediction.
    sigma_motion_x: float = 0.015     # [m/step]
    sigma_motion_y: float = 0.015     # [m/step]
    sigma_motion_theta: float = math.radians(1.0)  # [rad/step]


@dataclass
class SensorConfig:
    """Visual sensor model parameters."""

    max_range: float = 5.5
    fov: float = math.radians(100.0)  # total field of view [rad]
    detection_probability: float = 0.90
    outlier_probability: float = 0.03
    camera_rate_hz: float = 5.0


@dataclass
class SimConfig:
    """Global simulator and EKF parameters."""

    dt: float = 0.05                  # 20 Hz odometry-like rate
    max_v: float = 0.45               # conservative TurtleBot3-like linear speed [m/s]
    max_w: float = 1.10               # angular velocity limit [rad/s]
    goal_tolerance: float = 0.12
    max_steps_per_segment: int = 4000

    # Mahalanobis gate for a 2D measurement [range, bearing].
    # 9.21 is approximately chi-square 2D at 99% confidence.
    mahalanobis_gate: float = 9.21

    # Avoid initializing landmarks from a single possibly bad visual detection.
    min_observations_to_initialize: int = 2
    candidate_distance_gate: float = 0.75  # [m]

    seed: int = 7
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
