#!/usr/bin/env python3
"""
main.py


"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import SimConfig
from evaluation import summarize_monte_carlo
from io_utils import print_metrics, save_history_csv, save_metrics_csv
from plotting import plot_errors, plot_map
from simulation import run_monte_carlo, run_single_simulation


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Micro-simulador EKF-SLAM para TurtleBot3 + ArUco-like landmarks")
    parser.add_argument("--runs", type=int, default=30, help="número de runs Monte Carlo")
    parser.add_argument("--seed", type=int, default=7, help="seed da primeira simulação")
    parser.add_argument("--outdir", type=str, default="microsim_results", help="pasta de resultados")
    parser.add_argument("--no-show", action="store_true", help="não abrir janelas matplotlib; só guardar PNGs")
    parser.add_argument("--gate", type=float, default=9.21, help="limiar Mahalanobis; 9.21 ≈ chi-square 2D a 99%")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = SimConfig(seed=args.seed, mahalanobis_gate=args.gate)

    # One run for visual debugging and figures.
    single = run_single_simulation(cfg)
    print_metrics("Single run metrics", single["metrics"])
    save_history_csv(single, outdir)
    plot_map(single, outdir, show=not args.no_show)
    plot_errors(single, outdir, show=not args.no_show)

    # Monte Carlo for statistical evidence.
    if args.runs > 0:
        rows = run_monte_carlo(cfg, args.runs)
        save_metrics_csv(rows, outdir)
        summary = summarize_monte_carlo(rows)
        print_metrics(f"Monte Carlo summary ({args.runs} runs)", summary)

    print(f"\nResultados guardados em: {outdir.resolve()}")
    print("Ficheiros principais:")
    print(f"  - {outdir / 'single_run_map.png'}")
    print(f"  - {outdir / 'single_run_errors.png'}")
    print(f"  - {outdir / 'single_run_history.csv'}")
    print(f"  - {outdir / 'monte_carlo_metrics.csv'}")


if __name__ == "__main__":
    main()
