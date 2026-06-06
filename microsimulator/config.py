"""
Configuration dataclasses for the EKF-SLAM micro-simulator.

This file defines:
    - simulated odometry noise;
    - simulated camera/ArUco noise;
    - EKF prediction uncertainty;
    - EKF visual correction uncertainty;
    - sensor and simulator parameters.

Important:
    sim_* parameters generate fake sensor data.
    ekf_* parameters define what the EKF assumes in Q and R.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class NoiseConfig:
    """
    Noise parameters used by the simulator and by the EKF.

    sim_*:
        noise/bias used to generate simulated measurements.

    ekf_*:
        noise assumed by the EKF when building Q_motion and R_meas.
    """

    # ------------------------------------------------------------------
    # 1) Simulated wheel odometry noise
    # ------------------------------------------------------------------
    # Translation is imperfect; rotation is good.
    #
    # The translational bias makes odometry slowly drift in position.
    # The angular noise and angular bias are kept low so heading remains reliable.

    sigma_v_odom: float = 0.02      # [m/s] translational random noise
    sigma_w_odom: float = 0.015      # [rad/s] small angular random noise

    v_bias: float = 0.05            # 5% systematic translational drift
    w_bias: float = 0.001            # almost no rotational drift

    # ------------------------------------------------------------------
    # 2) Simulated camera / ArUco noise
    # ------------------------------------------------------------------
    # Range is good; bearing is worse but still usable.
    #
    # Do not make bearing completely useless in the simulator, otherwise
    # landmark initialization becomes geometrically weak.

    sim_sigma_range: float = 0.010                 # [m] 2 cm range noise
    sim_sigma_bearing: float = math.radians(4.0)  # [rad] bearing noise

    sim_range_bias: float = 0.0
    sim_bearing_bias: float = math.radians(0.0)

    # ------------------------------------------------------------------
    # 3) EKF motion noise: prediction uncertainty
    # ------------------------------------------------------------------
    # EKF distrusts translational odometry moderately.
    # EKF still trusts heading because rotation is assumed good.

    ekf_sigma_motion_x: float = 0.020                  # [m/step]
    ekf_sigma_motion_y: float = 0.020                  # [m/step]
    ekf_sigma_motion_theta: float = math.radians(0.03) # [rad/step]

    # ------------------------------------------------------------------
    # 4) EKF camera / ArUco noise: correction uncertainty
    # ------------------------------------------------------------------
    # EKF trusts range more than bearing.
    # Bearing is worse than range, but not ignored.

    ekf_sigma_range: float = 0.010                   # [m]
    ekf_sigma_bearing: float = math.radians(4.0)    # [rad]


@dataclass
class SensorConfig:
    """Visual sensor model parameters."""

    # Maximum distance at which a tag can be detected.
    max_range: float = 2.5

    # Total horizontal field of view.
    # math.radians(100.0) means 100 degrees total, i.e. ±50 degrees.
    fov: float = math.radians(100.0)

    # Probability of detecting a tag that is inside range and FOV.
    # 1.0 means no missed detections.
    detection_probability: float = 1.0

    # Probability of generating an artificial bad visual measurement.
    # Keep 0.0 during debugging.
    outlier_probability: float = 0.0

    # Camera update rate.
    # If dt = 0.05, odometry is 20 Hz; camera_rate_hz = 5 gives one update every 4 steps.
    camera_rate_hz: float = 5.0


@dataclass
class SimConfig:
    """Global simulator and EKF parameters."""

    dt: float = 0.05

    max_v: float = 0.45
    max_w: float = 1.10

    goal_tolerance: float = 0.12
    max_steps_per_segment: int = 4000

    # Mahalanobis gate for initialized landmarks.
    # 9.21 is the 99% gate for 2D measurements.
    mahalanobis_gate: float = 9.21

    # New landmark initialization.
    # Make initialization stricter than before.
    min_observations_to_initialize: int = 3
    candidate_distance_gate: float = 0.7

    seed: int = 7

    # Raw innovation gates for already initialized landmarks.
    # Loose enough for loop closure, but not absurdly permissive.
    max_raw_range_innovation: float = 1.5
    max_raw_bearing_innovation: float = math.radians(50.0)

    noise: NoiseConfig = field(default_factory=NoiseConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)