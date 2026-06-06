"""Manual keyboard-mode entry point for the EKF-SLAM micro-simulator."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import SimConfig
from io_utils import (
    print_metrics,
    save_covariance_matrices_csv,
    save_diagnostics_csv,
    save_history_csv,
    save_metrics_csv,
    save_single_run_metrics_txt,
)
from manual_teleop import run_manual_teleop_simulation
from plotting import plot_all_diagnostics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive EKF-SLAM micro-simulator with keyboard teleop.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--duration", type=float, default=120.0, help="Maximum teleop duration [s].")
    parser.add_argument("--outdir", type=str, default="microsim_manual_results", help="Output folder.")
    parser.add_argument("--no-show", action="store_true", help="Only save plots; do not open matplotlib windows after finishing.")
    parser.add_argument("--live-plot", action="store_true", help="Open a live matplotlib window showing the robot moving during teleop.")
    parser.add_argument("--gate", type=float, default=9.21, help="Mahalanobis gate for visual measurements.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = SimConfig(seed=args.seed, mahalanobis_gate=args.gate)

    result = run_manual_teleop_simulation(cfg, max_duration_s=args.duration, live_plot=args.live_plot)
    print_metrics("Manual teleop simulation metrics", result["metrics"])
    save_history_csv(result, outdir)
    save_diagnostics_csv(result, outdir)
    save_covariance_matrices_csv(result, outdir)
    save_single_run_metrics_txt(result, outdir)
    plot_all_diagnostics(result, outdir, show=not args.no_show)

    print(f"\nSaved results to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
