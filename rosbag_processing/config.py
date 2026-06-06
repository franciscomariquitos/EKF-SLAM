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

    # ------------------------------------------------------------------
    # 3) EKF motion noise: prediction uncertainty
    # ------------------------------------------------------------------
    # These parameters ARE used in both the simulator and bag processing.
    #
    # In bag processing, /odom gives the motion increment. These values define
    # how uncertain the EKF believes that odometry prediction is.
    #
    # Larger x/y values:
    #     the EKF trusts translational odometry less;
    #     covariance grows faster while the robot moves.
    #
    # Smaller theta value:
    #     the EKF trusts rotational odometry more;
    #     heading changes are not allowed to drift too much.
    #
    # This matches the intended scenario:
    #     good rotation, poor translation.

    ekf_sigma_motion_x: float = 0.04                 # [m/step] EKF uncertainty in x prediction
    ekf_sigma_motion_y: float = 0.04                 # [m/step] EKF uncertainty in y prediction
    ekf_sigma_motion_theta: float = math.radians(0.3)  # [rad/step] EKF uncertainty in yaw prediction

    # ------------------------------------------------------------------
    # 4) EKF camera / ArUco noise: correction uncertainty
    # ------------------------------------------------------------------
    # These parameters ARE used in both the simulator and bag processing.
    #
    # They define how much the EKF trusts each ArUco measurement.
    #
    # Smaller range sigma:
    #     range is trusted more.
    #
    # Larger bearing sigma:
    #     bearing is trusted less.
    #
    # This matches the intended camera model:
    #     good range, poor bearing.

    ekf_sigma_range: float = 0.06                    # [m] EKF-assumed range uncertainty
    ekf_sigma_bearing: float = math.radians(30.0)    # [rad] EKF-assumed bearing uncertainty
    
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

    # ------------------------------------------------------------
    # Gates for NON-initialized landmarks
    # ------------------------------------------------------------
    # Used while the landmark is still only a candidate.
    # These gates should be tight.
    min_observations_to_initialize: int = 4
    candidate_distance_gate: float = 0.25

    # ------------------------------------------------------------
    # Gates for already initialized landmarks
    # ------------------------------------------------------------
    # Used after the landmark already exists in the EKF state.
    # These gates can be looser because loop closure can create
    # large innovations after odometry drift.
    max_raw_range_innovation: float = 2.5 
    max_raw_bearing_innovation: float = math.radians(60.0)

    mahalanobis_gate: float = 12.0

    noise: NoiseConfig = field(default_factory=NoiseConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)