"""Interactive keyboard teleoperation simulation for the EKF-SLAM micro-simulator.

This module lets the user drive the simulated robot manually from the terminal,
close to how TurtleBot3 teleop is used with /cmd_vel. It keeps the same EKF core,
visual sensor model and evaluation pipeline used by the automatic waypoint
simulation.
"""

from __future__ import annotations

import curses
import time
from typing import List, Tuple

import numpy as np

from config import SimConfig
from ekf_slam import EKFSLAM
from evaluation import compute_metrics
from utils import odometry_increment, pose_step
from world import default_landmarks, generate_visual_measurements
from live_view import LiveView


HELP_TEXT = [
    "EKF-SLAM manual teleop micro-simulator",
    "",
    "Controls:",
    "  W / Up      : set forward velocity",
    "  S / Down    : set reverse velocity",
    "  A / Left    : set positive angular velocity (turn left)",
    "  D / Right   : set negative angular velocity (turn right)",
    "  Z           : stop rotation only",
    "  X or Space  : stop all motion",
    "  + / -       : increase/decrease linear speed scale",
    "  ] / [       : increase/decrease angular speed scale",
    "  Q           : finish simulation and save results",
    "",
    "The command is persistent: press X or Space to stop, like stopping /cmd_vel.",
]


def _handle_key(
    key: int,
    v_cmd: float,
    w_cmd: float,
    linear_scale: float,
    angular_scale: float,
    cfg: SimConfig,
) -> Tuple[float, float, float, float, bool]:
    """Convert a keyboard key into teleoperation commands.

    Returns:
        v_cmd, w_cmd, linear_scale, angular_scale, should_quit
    """
    should_quit = False

    if key in (ord("q"), ord("Q")):
        should_quit = True

    elif key in (ord("w"), ord("W"), curses.KEY_UP):
        v_cmd = cfg.max_v * linear_scale

    elif key in (ord("s"), ord("S"), curses.KEY_DOWN):
        # Reverse is intentionally slower than forward, like a cautious real robot.
        v_cmd = -0.55 * cfg.max_v * linear_scale

    elif key in (ord("a"), ord("A"), curses.KEY_LEFT):
        w_cmd = cfg.max_w * angular_scale

    elif key in (ord("d"), ord("D"), curses.KEY_RIGHT):
        w_cmd = -cfg.max_w * angular_scale

    elif key in (ord("z"), ord("Z")):
        w_cmd = 0.0

    elif key in (ord("x"), ord("X"), ord(" ")):
        v_cmd = 0.0
        w_cmd = 0.0

    elif key in (ord("+"), ord("=")):
        linear_scale = min(1.0, linear_scale + 0.10)

    elif key in (ord("-"), ord("_")):
        linear_scale = max(0.10, linear_scale - 0.10)

    elif key in (ord("]"), ord("}")):
        angular_scale = min(1.0, angular_scale + 0.10)

    elif key in (ord("["), ord("{")):
        angular_scale = max(0.10, angular_scale - 0.10)

    return v_cmd, w_cmd, linear_scale, angular_scale, should_quit


def _safe_addstr(stdscr, row: int, col: int, text: str) -> None:
    """Write text without crashing if the terminal is small.

    curses.addstr raises _curses.error when the requested row/column is outside
    the terminal window, or when the text reaches the lower-right corner. This
    helper clips the text to the available width and silently skips rows that do
    not fit.
    """
    max_y, max_x = stdscr.getmaxyx()

    if row < 0 or row >= max_y or col < 0 or col >= max_x:
        return

    # Leave one column free because some terminals fail when writing exactly in
    # the last column.
    available_width = max(0, max_x - col - 1)
    if available_width <= 0:
        return

    try:
        stdscr.addstr(row, col, text[:available_width])
    except curses.error:
        # Avoid killing the simulator because of terminal drawing limitations.
        pass


def _draw_screen(
    stdscr,
    t: float,
    v_cmd: float,
    w_cmd: float,
    linear_scale: float,
    angular_scale: float,
    true_pose: np.ndarray,
    odom_pose: np.ndarray,
    ekf: EKFSLAM,
    n_measurements: int,
    max_duration_s: float,
) -> None:
    """Draw the live text interface."""
    stdscr.erase()
    max_y, max_x = stdscr.getmaxyx()

    if max_y < 12 or max_x < 50:
        _safe_addstr(stdscr, 0, 0, "Terminal too small. Enlarge it or reduce font size.")
        _safe_addstr(stdscr, 1, 0, f"Current size: {max_x} columns x {max_y} rows")
        _safe_addstr(stdscr, 2, 0, "Press Q to finish.")
        stdscr.refresh()
        return

    row = 0
    for line in HELP_TEXT:
        _safe_addstr(stdscr, row, 0, line)
        row += 1
        if row >= max_y - 1:
            stdscr.refresh()
            return

    row += 1
    _safe_addstr(stdscr, row, 0, f"Time: {t:6.2f} / {max_duration_s:.1f} s")
    row += 1
    _safe_addstr(stdscr, row, 0, f"Command: v = {v_cmd:+.3f} m/s, w = {w_cmd:+.3f} rad/s")
    row += 1
    _safe_addstr(stdscr, row, 0, f"Speed scales: linear = {linear_scale:.2f}, angular = {angular_scale:.2f}")
    row += 2

    _safe_addstr(stdscr, row, 0, f"True pose: x={true_pose[0]:+.2f}, y={true_pose[1]:+.2f}, theta={true_pose[2]:+.2f}")
    row += 1
    _safe_addstr(stdscr, row, 0, f"Odom pose: x={odom_pose[0]:+.2f}, y={odom_pose[1]:+.2f}, theta={odom_pose[2]:+.2f}")
    row += 1
    ekf_pose = ekf.mu[0:3, 0]
    _safe_addstr(stdscr, row, 0, f"EKF pose : x={ekf_pose[0]:+.2f}, y={ekf_pose[1]:+.2f}, theta={ekf_pose[2]:+.2f}")
    row += 2

    _safe_addstr(stdscr, row, 0, f"Last camera detections: {n_measurements}")
    row += 1
    _safe_addstr(stdscr, row, 0, f"Initialized landmarks: {len(ekf.tag_dict)}")
    row += 1
    _safe_addstr(stdscr, row, 0, f"Accepted updates: {ekf.stats['accepted_updates']}")
    row += 1
    _safe_addstr(stdscr, row, 0, f"Rejected outliers: {ekf.stats['rejected_outliers']}")
    row += 2
    _safe_addstr(stdscr, row, 0, "Drive a loop and return near the start to evaluate loop-closure error.")
    stdscr.refresh()


def run_manual_teleop_simulation(cfg: SimConfig, max_duration_s: float = 120.0, live_plot: bool = False) -> dict:
    """Run an interactive manual teleoperation simulation.

    The user drives the simulated robot with the keyboard. The simulator still
    generates noisy odometry and visual landmark observations, then runs the same
    EKF-SLAM prediction/correction pipeline as the automatic simulator.
    """
    return curses.wrapper(_run_manual_teleop_curses, cfg, max_duration_s, live_plot)


def _run_manual_teleop_curses(stdscr, cfg: SimConfig, max_duration_s: float, live_plot: bool) -> dict:
    curses.curs_set(0)
    stdscr.keypad(True)
    stdscr.nodelay(False)
    stdscr.timeout(max(1, int(cfg.dt * 1000)))

    rng = np.random.default_rng(cfg.seed)
    landmarks = default_landmarks()
    viewer = LiveView(landmarks) if live_plot else None
    waypoints = np.array([[0.0, 0.0]], dtype=float)  # used only as the reference start point for metrics

    ekf = EKFSLAM(cfg)
    true_pose = np.array([0.0, 0.0, 0.0], dtype=float)
    odom_pose = true_pose.copy()
    prev_odom_pose = odom_pose.copy()

    history = {
        "t": [],
        "true": [],
        "odom": [],
        "ekf": [],
        "n_measurements": [],
        "accepted_updates": [],
        "rejected_outliers": [],
    }

    camera_period_steps = max(1, int(round((1.0 / cfg.sensor.camera_rate_hz) / cfg.dt)))

    v_cmd = 0.0
    w_cmd = 0.0
    linear_scale = 0.55
    angular_scale = 0.65
    t = 0.0
    global_step = 0
    should_quit = False

    last_time = time.monotonic()

    while t < max_duration_s and not should_quit:
        key = stdscr.getch()
        if key != -1:
            v_cmd, w_cmd, linear_scale, angular_scale, should_quit = _handle_key(
                key, v_cmd, w_cmd, linear_scale, angular_scale, cfg
            )

        now = time.monotonic()
        elapsed = now - last_time
        if elapsed < cfg.dt:
            # Keep the real-time rhythm stable even if the terminal returns early.
            time.sleep(cfg.dt - elapsed)
        last_time = time.monotonic()

        # 1) Ground-truth motion follows the command velocity.
        true_pose = pose_step(true_pose, v_cmd, w_cmd, cfg.dt)

        # 2) Simulated noisy odometry, including systematic drift.
        noisy_v = v_cmd * (1.0 + cfg.noise.v_bias) + rng.normal(0.0, cfg.noise.sigma_v_odom)
        noisy_w = w_cmd * (1.0 + cfg.noise.w_bias) + rng.normal(0.0, cfg.noise.sigma_w_odom)
        odom_pose = pose_step(odom_pose, noisy_v, noisy_w, cfg.dt)

        # 3) EKF prediction from odometry increments.
        d_rot1, d_trans, d_rot2 = odometry_increment(prev_odom_pose, odom_pose)
        ekf.predict_from_odometry(d_rot1, d_trans, d_rot2)
        prev_odom_pose = odom_pose.copy()

        # 4) EKF correction from ArUco/AprilTag-like visual measurements.
        measurements: List[dict] = []
        if global_step % camera_period_steps == 0:
            measurements = generate_visual_measurements(true_pose, landmarks, cfg, rng)
            ekf.update(measurements)

        # 5) Store history.
        history["t"].append(t)
        history["true"].append(true_pose.copy())
        history["odom"].append(odom_pose.copy())
        history["ekf"].append(ekf.mu[0:3, 0].copy())
        history["n_measurements"].append(len(measurements))
        history["accepted_updates"].append(ekf.stats["accepted_updates"])
        history["rejected_outliers"].append(ekf.stats["rejected_outliers"])

        _draw_screen(
            stdscr,
            t,
            v_cmd,
            w_cmd,
            linear_scale,
            angular_scale,
            true_pose,
            odom_pose,
            ekf,
            len(measurements),
            max_duration_s,
        )

        if viewer is not None:
            viewer.update(history, true_pose, ekf)

        t += cfg.dt
        global_step += 1

    if viewer is not None:
        viewer.close()

    for key in ["true", "odom", "ekf"]:
        history[key] = np.asarray(history[key], dtype=float)
    history["t"] = np.asarray(history["t"], dtype=float)

    metrics = compute_metrics(history, ekf, landmarks, waypoints)

    return {
        "cfg": cfg,
        "history": history,
        "ekf": ekf,
        "landmarks": landmarks,
        "waypoints": waypoints,
        "metrics": metrics,
    }
