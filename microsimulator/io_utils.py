"""CSV output and terminal reporting utilities."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List


def save_history_csv(result: dict, outdir: Path) -> None:
    """Save the single-run trajectory history to CSV."""
    history = result["history"]
    path = outdir / "single_run_history.csv"

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "t",
            "true_x", "true_y", "true_theta",
            "odom_x", "odom_y", "odom_theta",
            "ekf_x", "ekf_y", "ekf_theta",
            "n_measurements",
            "accepted_updates_cumulative",
            "rejected_outliers_cumulative",
        ])

        for i in range(len(history["t"])):
            writer.writerow([
                history["t"][i],
                *history["true"][i],
                *history["odom"][i],
                *history["ekf"][i],
                history["n_measurements"][i],
                history["accepted_updates"][i],
                history["rejected_outliers"][i],
            ])


def save_metrics_csv(metrics_rows: List[dict], outdir: Path) -> None:
    """Save Monte Carlo metrics to CSV."""
    if not metrics_rows:
        return

    path = outdir / "monte_carlo_metrics.csv"
    fieldnames = list(metrics_rows[0].keys())

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)


def print_metrics(title: str, metrics: dict) -> None:
    """Pretty-print a metrics dictionary."""
    print("\n" + title)
    print("=" * len(title))
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:32s}: {value:.4f}")
        else:
            print(f"{key:32s}: {value}")
