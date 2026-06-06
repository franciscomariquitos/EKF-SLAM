"""EKF-SLAM core, independent of ROS/Gazebo/rosbags."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from config import SimConfig
from utils import normalize_angle

from matplotlib.patches import Ellipse



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
            n.ekf_sigma_motion_x**2,
            n.ekf_sigma_motion_y**2,
            n.ekf_sigma_motion_theta**2,
        ])

        self.R_meas = np.diag([
            n.ekf_sigma_range**2,
            n.ekf_sigma_bearing**2,
        ])

        self.stats = {
            "initialized_landmarks": 0,
            "candidate_rejections": 0,
            "accepted_updates": 0,
            "rejected_outliers": 0,
            "total_measurements": 0,
        }
        self.nis_values: List[float] = []
        self.diagnostics: List[dict] = []

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

        The initial covariance is obtained by linear covariance propagation using the EKF-assumed visual measurement covariance R_meas.
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
        Every known-landmark measurement is logged in self.diagnostics, so plotting.py can later
        draw innovation and NIS consistency figures.
        """
        for meas in measurements:
            self.stats["total_measurements"] += 1

            tag_id = int(meas["id"])
            rng = float(meas["range"])
            bearing = normalize_angle(float(meas["bearing"]))

            base_event = {
                "tag_id": tag_id,
                "measured_range": rng,
                "measured_bearing": bearing,
                "innovation_range": float("nan"),
                "innovation_bearing": float("nan"),
                "nis": float("nan"),
                "accepted": False,
                "reason": "unknown",
            }

            if rng <= 0.02:
                self.stats["rejected_outliers"] += 1
                base_event["reason"] = "invalid_range"
                self.diagnostics.append(base_event)
                continue

            if tag_id not in self.tag_dict:
                candidate_info = self._handle_new_landmark_candidate(tag_id, rng, bearing)

                base_event.update(candidate_info)

                base_event["landmark_status"] = "not_initialized"
                base_event["gate_type"] = "candidate_distance_gate"
                base_event["reason"] = candidate_info["candidate_status"]
                base_event["accepted"] = bool(candidate_info["initialized"])

                self.diagnostics.append(base_event)
                continue

            idx = self.tag_dict[tag_id]
            dx = self.mu[idx, 0] - self.mu[0, 0]
            dy = self.mu[idx + 1, 0] - self.mu[1, 0]
            q = dx**2 + dy**2

            if q < 1e-12:
                self.stats["rejected_outliers"] += 1
                base_event["reason"] = "degenerate_geometry"
                self.diagnostics.append(base_event)
                continue

            sqrt_q = math.sqrt(q)

            z_hat = np.array([
                [sqrt_q],
                [normalize_angle(math.atan2(dy, dx) - self.mu[2, 0])],
            ])
            z = np.array([[rng], [bearing]])

            innovation = z - z_hat
            innovation[1, 0] = normalize_angle(innovation[1, 0])

            base_event["innovation_range"] = float(innovation[0, 0])
            base_event["innovation_bearing"] = float(innovation[1, 0])
            base_event["predicted_range"] = float(z_hat[0, 0])
            base_event["predicted_bearing"] = float(z_hat[1, 0])

            raw_range_error = abs(float(innovation[0, 0]))
            raw_bearing_error = abs(float(innovation[1, 0]))

            base_event["landmark_status"] = "initialized"
            base_event["gate_type"] = "known_landmark_gates"

            base_event["innovation_range"] = float(innovation[0, 0])
            base_event["innovation_bearing"] = float(innovation[1, 0])
            base_event["predicted_range"] = float(z_hat[0, 0])
            base_event["predicted_bearing"] = float(z_hat[1, 0])

            if raw_range_error > self.cfg.max_raw_range_innovation:
                self.stats["rejected_outliers"] += 1
                base_event["reason"] = "known_landmark_raw_range_gate"
                self.diagnostics.append(base_event)
                continue

            if raw_bearing_error > self.cfg.max_raw_bearing_innovation:
                self.stats["rejected_outliers"] += 1
                base_event["reason"] = "known_landmark_raw_bearing_gate"
                self.diagnostics.append(base_event)
                continue

            H = self._observation_jacobian(dx, dy, q, sqrt_q, idx)
            S = H @ self.Sigma @ H.T + self.R_meas

            try:
                S_inv_innovation = np.linalg.solve(S, innovation)
            except np.linalg.LinAlgError:
                self.stats["rejected_outliers"] += 1
                base_event["reason"] = "singular_innovation_covariance"
                self.diagnostics.append(base_event)
                continue

            d2 = float((innovation.T @ S_inv_innovation)[0, 0])
            self.nis_values.append(d2)
            base_event["nis"] = d2

            if d2 > self.cfg.mahalanobis_gate:
                self.stats["rejected_outliers"] += 1
                base_event["reason"] = "known_landmark_mahalanobis_rejected"
                self.diagnostics.append(base_event)
                continue

            K = np.linalg.solve(S, H @ self.Sigma).T

            self.mu = self.mu + K @ innovation
            self.mu[2, 0] = normalize_angle(self.mu[2, 0])

            I = np.eye(self.mu.shape[0])
            self.Sigma = (I - K @ H) @ self.Sigma @ (I - K @ H).T + K @ self.R_meas @ K.T
            self._symmetrize_covariance()

            self.stats["accepted_updates"] += 1

            base_event["accepted"] = True
            base_event["reason"] = "known_landmark_update"
            self.diagnostics.append(base_event)

    def _handle_new_landmark_candidate(self, tag_id: int, rng: float, bearing: float) -> dict:
        """
        Handle a landmark that is not initialized yet.

        For non-initialized landmarks there is no EKF innovation z - z_hat,
        because the landmark is not in the state vector yet.

        Instead, the consistency metric is:
            candidate_error = distance between repeated global candidate positions.

        This is the tight gate used before landmark initialization.
        """
        candidate_pos = self.landmark_position_from_measurement(rng, bearing)

        info = {
            "candidate_x": float(candidate_pos[0]),
            "candidate_y": float(candidate_pos[1]),
            "candidate_error": float("nan"),
            "candidate_count": 1,
            "candidate_status": "new_candidate_first_observation",
            "initialized": False,
        }

        if self.cfg.min_observations_to_initialize <= 1:
            self.initialize_landmark(tag_id, rng, bearing)
            info["candidate_status"] = "new_landmark_initialized"
            info["initialized"] = True
            return info

        candidate = self.candidate_landmarks.get(tag_id)

        if candidate is None:
            self.candidate_landmarks[tag_id] = {
                "pos": candidate_pos,
                "count": 1,
                "positions": [candidate_pos.copy()],
            }
            return info

        candidate_error = float(np.linalg.norm(candidate_pos - candidate["pos"]))

        info["candidate_error"] = candidate_error
        info["candidate_count"] = int(candidate["count"]) + 1

        if candidate_error > self.cfg.candidate_distance_gate:
            self.candidate_landmarks[tag_id] = {
                "pos": candidate_pos,
                "count": 1,
                "positions": [candidate_pos.copy()],
            }
            self.stats["candidate_rejections"] += 1

            info["candidate_status"] = "new_candidate_rejected"
            info["candidate_count"] = 1
            return info

        old_count = int(candidate["count"])
        new_count = old_count + 1

        candidate["positions"].append(candidate_pos.copy())
        candidate["pos"] = (candidate["pos"] * old_count + candidate_pos) / new_count
        candidate["count"] = new_count

        info["candidate_x"] = float(candidate["pos"][0])
        info["candidate_y"] = float(candidate["pos"][1])
        info["candidate_count"] = new_count
        info["candidate_status"] = "new_candidate_consistent"

        if new_count >= self.cfg.min_observations_to_initialize:
            self.initialize_landmark_from_candidate_positions(
                tag_id,
                candidate["positions"],
            )
            self.candidate_landmarks.pop(tag_id, None)

            info["candidate_status"] = "new_landmark_initialized"
            info["initialized"] = True

        return info
    
    def initialize_landmark_from_candidate_positions(self, tag_id: int, positions: list[np.ndarray]) -> None:
        """
        Initialize landmark from several consistent candidate positions.
        The initial landmark covariance is estimated from the spatial dispersion
        of those candidate positions.
        """
        old_n = self.mu.shape[0]
        self.tag_dict[tag_id] = old_n

        P = np.asarray(positions, dtype=float)
        mean_pos = np.mean(P, axis=0)

        mx = float(mean_pos[0])
        my = float(mean_pos[1])

        self.mu = np.vstack((self.mu, np.array([[mx], [my]], dtype=float)))

        new_Sigma = np.zeros((old_n + 2, old_n + 2), dtype=float)
        new_Sigma[:old_n, :old_n] = self.Sigma

        if P.shape[0] >= 2:
            cov = np.cov(P.T)
        else:
            cov = np.diag([0.25**2, 0.25**2])

        # Regularization: avoid singular / overconfident covariance.
        cov = np.asarray(cov, dtype=float)
        cov = 0.5 * (cov + cov.T)
        cov += np.eye(2) * (0.05**2)

        # Optional minimum uncertainty.
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 0.05**2)
        cov = eigvecs @ np.diag(eigvals) @ eigvecs.T

        new_Sigma[old_n:old_n + 2, old_n:old_n + 2] = cov

        self.Sigma = new_Sigma
        self._symmetrize_covariance()
        self.stats["initialized_landmarks"] += 1

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


    def robot_covariance(self) -> np.ndarray:
        """Return the 3x3 robot-pose covariance block."""
        return self.Sigma[0:3, 0:3].copy()

    def landmark_covariance(self, tag_id: int) -> np.ndarray | None:
        """Return the 2x2 covariance block of one landmark, if initialized."""
        idx = self.tag_dict.get(tag_id)
        if idx is None:
            return None
        return self.Sigma[idx:idx + 2, idx:idx + 2].copy()

    def landmark_covariances(self) -> Dict[int, np.ndarray]:
        """Return all initialized landmark covariance blocks as {tag_id: Sigma_2x2}."""
        return {tag_id: self.Sigma[idx:idx + 2, idx:idx + 2].copy() for tag_id, idx in self.tag_dict.items()}

    def estimated_landmarks(self) -> Dict[int, Tuple[float, float]]:
        """Return estimated landmark map as {tag_id: (x, y)}."""
        result = {}
        for tag_id, idx in self.tag_dict.items():
            result[tag_id] = (float(self.mu[idx, 0]), float(self.mu[idx + 1, 0]))
        return result
    def initialize_landmark_from_position(self, tag_id: int, position: np.ndarray) -> None:
        """
        Initialize a new landmark directly from an averaged global candidate position.
        This avoids initializing from a single last measurement.
        """
        old_n = self.mu.shape[0]

        self.tag_dict[tag_id] = old_n

        mx = float(position[0])
        my = float(position[1])

        self.mu = np.vstack((self.mu, np.array([[mx], [my]], dtype=float)))

        new_Sigma = np.zeros((old_n + 2, old_n + 2), dtype=float)
        new_Sigma[:old_n, :old_n] = self.Sigma

        # Give the new landmark enough uncertainty.
        # Do not make it overconfident at birth.
        init_var = 1.0**2
        new_Sigma[old_n:old_n + 2, old_n:old_n + 2] = np.diag([init_var, init_var])

        self.Sigma = new_Sigma
        self._symmetrize_covariance()
        self.stats["initialized_landmarks"] += 1
