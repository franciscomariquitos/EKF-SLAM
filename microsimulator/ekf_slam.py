"""EKF-SLAM core, independent of ROS/Gazebo/rosbags."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from config import SimConfig
from utils import normalize_angle


class EKFSLAM:
    """
    Landmark-based EKF-SLAM.

    State convention:
        mu = [x, y, theta, lm1_x, lm1_y, lm2_x, lm2_y, ...]^T

    Noise convention:
        Q_motion = process/motion covariance.
        R_meas   = visual measurement covariance for [range, bearing].
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg

        self.mu = np.zeros((3, 1), dtype=float)
        self.Sigma = np.diag([1e-6, 1e-6, 1e-6]).astype(float)

        # Maps a tag ID to the index of its x coordinate in mu.
        self.tag_dict: Dict[int, int] = {}

        # Temporary landmarks waiting for repeated consistent observations.
        self.candidate_landmarks: Dict[int, dict] = {}

        n = cfg.noise
        self.Q_motion = np.diag([
            n.sigma_motion_x**2,
            n.sigma_motion_y**2,
            n.sigma_motion_theta**2,
        ])

        self.R_meas = np.diag([
            n.sigma_range**2,
            n.sigma_bearing**2,
        ])

        self.stats = {
            "initialized_landmarks": 0,
            "candidate_rejections": 0,
            "accepted_updates": 0,
            "rejected_outliers": 0,
            "total_measurements": 0,
        }
        self.nis_values: List[float] = []

    def predict_from_odometry(self, delta_rot1: float, delta_trans: float, delta_rot2: float) -> None:
        """
        EKF prediction using the odometry increment model.

        u = (delta_rot1, delta_trans, delta_rot2)
        """
        n_state = self.mu.shape[0]
        theta = self.mu[2, 0]
        theta_mid = theta + delta_rot1

        self.mu[0, 0] += delta_trans * math.cos(theta_mid)
        self.mu[1, 0] += delta_trans * math.sin(theta_mid)
        self.mu[2, 0] = normalize_angle(theta + delta_rot1 + delta_rot2)

        G = np.eye(n_state)
        G[0, 2] = -delta_trans * math.sin(theta_mid)
        G[1, 2] = delta_trans * math.cos(theta_mid)

        Q_full = np.zeros((n_state, n_state), dtype=float)
        Q_full[0:3, 0:3] = self.Q_motion

        self.Sigma = G @ self.Sigma @ G.T + Q_full
        self._symmetrize_covariance()

    def landmark_position_from_measurement(self, rng: float, bearing: float) -> np.ndarray:
        """Convert relative [range, bearing] into global landmark position [mx, my]."""
        x, y, theta = self.mu[0, 0], self.mu[1, 0], self.mu[2, 0]
        alpha = theta + bearing
        return np.array([
            x + rng * math.cos(alpha),
            y + rng * math.sin(alpha),
        ], dtype=float)

    def initialize_landmark(self, tag_id: int, rng: float, bearing: float) -> None:
        """
        Initialize a new landmark through the inverse observation model:

            mx = x + r cos(theta + bearing)
            my = y + r sin(theta + bearing)

        The initial covariance is obtained by linear covariance propagation.
        """
        old_n = self.mu.shape[0]
        self.tag_dict[tag_id] = old_n

        theta = self.mu[2, 0]
        alpha = theta + bearing
        mx, my = self.landmark_position_from_measurement(rng, bearing)

        self.mu = np.vstack((self.mu, np.array([[mx], [my]], dtype=float)))

        # Jacobian of inverse observation model w.r.t. robot pose.
        Gx = np.array([
            [1.0, 0.0, -rng * math.sin(alpha)],
            [0.0, 1.0,  rng * math.cos(alpha)],
        ])

        # Jacobian of inverse observation model w.r.t. measurement [range, bearing].
        Gz = np.array([
            [math.cos(alpha), -rng * math.sin(alpha)],
            [math.sin(alpha),  rng * math.cos(alpha)],
        ])

        new_Sigma = np.zeros((old_n + 2, old_n + 2), dtype=float)
        new_Sigma[:old_n, :old_n] = self.Sigma

        # Cross-covariance between new landmark and previous state caused by robot pose uncertainty.
        cross = Gx @ self.Sigma[0:3, :]
        new_Sigma[old_n:old_n + 2, :old_n] = cross
        new_Sigma[:old_n, old_n:old_n + 2] = cross.T

        Sigma_rr = self.Sigma[0:3, 0:3]
        Sigma_mm = Gx @ Sigma_rr @ Gx.T + Gz @ self.R_meas @ Gz.T
        new_Sigma[old_n:old_n + 2, old_n:old_n + 2] = Sigma_mm

        self.Sigma = new_Sigma
        self._symmetrize_covariance()
        self.stats["initialized_landmarks"] += 1

    def update(self, measurements: List[dict]) -> None:
        """
        EKF correction using visual measurements with fields:
            {"id": int, "range": float, "bearing": float}

        Tag IDs give direct data association. Mahalanobis gating still rejects bad detections.
        """
        for meas in measurements:
            self.stats["total_measurements"] += 1

            tag_id = int(meas["id"])
            rng = float(meas["range"])
            bearing = normalize_angle(float(meas["bearing"]))

            if rng <= 0.02:
                self.stats["rejected_outliers"] += 1
                continue

            if tag_id not in self.tag_dict:
                self._handle_new_landmark_candidate(tag_id, rng, bearing)
                continue

            idx = self.tag_dict[tag_id]
            dx = self.mu[idx, 0] - self.mu[0, 0]
            dy = self.mu[idx + 1, 0] - self.mu[1, 0]
            q = dx**2 + dy**2

            if q < 1e-12:
                self.stats["rejected_outliers"] += 1
                continue

            sqrt_q = math.sqrt(q)

            z_hat = np.array([
                [sqrt_q],
                [normalize_angle(math.atan2(dy, dx) - self.mu[2, 0])],
            ])
            z = np.array([[rng], [bearing]])

            innovation = z - z_hat
            innovation[1, 0] = normalize_angle(innovation[1, 0])

            H = self._observation_jacobian(dx, dy, q, sqrt_q, idx)
            S = H @ self.Sigma @ H.T + self.R_meas

            try:
                S_inv_innovation = np.linalg.solve(S, innovation)
            except np.linalg.LinAlgError:
                self.stats["rejected_outliers"] += 1
                continue

            d2 = float((innovation.T @ S_inv_innovation)[0, 0])
            self.nis_values.append(d2)

            if d2 > self.cfg.mahalanobis_gate:
                self.stats["rejected_outliers"] += 1
                continue

            K = np.linalg.solve(S, H @ self.Sigma).T
            self.mu = self.mu + K @ innovation
            self.mu[2, 0] = normalize_angle(self.mu[2, 0])

            # Joseph covariance update: numerically safer than (I-KH)Sigma.
            I = np.eye(self.mu.shape[0])
            self.Sigma = (I - K @ H) @ self.Sigma @ (I - K @ H).T + K @ self.R_meas @ K.T
            self._symmetrize_covariance()
            self.stats["accepted_updates"] += 1

    def _handle_new_landmark_candidate(self, tag_id: int, rng: float, bearing: float) -> None:
        """Delay landmark initialization until enough consistent observations exist."""
        candidate_pos = self.landmark_position_from_measurement(rng, bearing)

        if self.cfg.min_observations_to_initialize <= 1:
            self.initialize_landmark(tag_id, rng, bearing)
            return

        candidate = self.candidate_landmarks.get(tag_id)
        if candidate is None:
            self.candidate_landmarks[tag_id] = {"pos": candidate_pos, "count": 1}
            return

        candidate_error = float(np.linalg.norm(candidate_pos - candidate["pos"]))
        if candidate_error > self.cfg.candidate_distance_gate:
            self.candidate_landmarks[tag_id] = {"pos": candidate_pos, "count": 1}
            self.stats["candidate_rejections"] += 1
            return

        old_count = int(candidate["count"])
        new_count = old_count + 1
        candidate["pos"] = (candidate["pos"] * old_count + candidate_pos) / new_count
        candidate["count"] = new_count

        if new_count >= self.cfg.min_observations_to_initialize:
            self.initialize_landmark(tag_id, rng, bearing)
            self.candidate_landmarks.pop(tag_id, None)

    def _observation_jacobian(self, dx: float, dy: float, q: float, sqrt_q: float, idx: int) -> np.ndarray:
        """Jacobian H of h(mu) = [range, bearing] for one observed landmark."""
        H = np.zeros((2, self.mu.shape[0]), dtype=float)

        # Derivatives w.r.t. robot pose.
        H[0, 0] = -dx / sqrt_q
        H[0, 1] = -dy / sqrt_q
        H[0, 2] = 0.0
        H[1, 0] = dy / q
        H[1, 1] = -dx / q
        H[1, 2] = -1.0

        # Derivatives w.r.t. observed landmark position.
        H[0, idx] = dx / sqrt_q
        H[0, idx + 1] = dy / sqrt_q
        H[1, idx] = -dy / q
        H[1, idx + 1] = dx / q
        return H

    def _symmetrize_covariance(self) -> None:
        self.Sigma = 0.5 * (self.Sigma + self.Sigma.T)

    def estimated_landmarks(self) -> Dict[int, Tuple[float, float]]:
        """Return estimated landmark map as {tag_id: (x, y)}."""
        result = {}
        for tag_id, idx in self.tag_dict.items():
            result[tag_id] = (float(self.mu[idx, 0]), float(self.mu[idx + 1, 0]))
        return result
