"""
scripts/run_full_analysis.py
============================
Run the full analysis pipeline on a SINGLE .h5 file and save results.

Usage
-----
    python scripts/run_full_analysis.py \\
        --slice data/processed/slice_01_baseline.h5 \\
        --condition baseline

    python scripts/run_full_analysis.py \\
        --slice data/processed/slice_01_stimulation.h5 \\
        --condition stimulation \\
        --config path/to/config.yaml \\
        --save-figures

This script is useful for:
- Checking that one recording processes correctly before batch running.
- Inspecting intermediate results (PC1 distribution, state detection).
- Quick QC plots.

Outputs
-------
- results/<slice_id>_summary.csv   (per-epoch metrics)
- results/figures/<slice_id>_*.pdf (if --save-figures is passed)
"""

import sys
import os
import argparse
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.core.experiment import Experiment
from src.data_io import save_results
from src import visualization as viz


def run_single(h5_path: str, condition: str, cfg: dict, save_figures: bool = False):
    """
    Run the full analysis for one recording and print a summary.

    Parameters
    ----------
    h5_path : str
    condition : str
    cfg : dict
    save_figures : bool
        If True, save diagnostic figures to the figures directory.
    """
    # ---- Run pipeline ----
    exp = Experiment(h5_path, condition=condition, cfg=cfg)
    exp.run()

    # ---- Print summary ----
    stats = exp.duration_stats
    print("\n" + "=" * 50)
    print(f"  Results: {exp.slice_id} | {condition}")
    print("=" * 50)
    print(f"  Up states   : {stats['n_up']}")
    print(f"  Down states : {stats['n_down']}")
    print(f"  Mean Up duration   : {stats['mean_up_s']:.3f} ± {stats['sem_up_s']:.3f} s")
    print(f"  Mean Down duration : {stats['mean_down_s']:.3f} ± {stats['sem_down_s']:.3f} s")
    print(f"  SO frequency       : {stats['so_frequency']:.4f} Hz")
    print("=" * 50 + "\n")

    # ---- Save summary CSV ----
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"{exp.slice_id}_{condition}_summary.csv")
    save_results(exp.summary_row, out_path)

    # ---- Optional figures ----
    if save_figures:
        _save_diagnostic_figures(exp, cfg)

    return exp


def _save_diagnostic_figures(exp: Experiment, cfg: dict):
    """Save a standard set of QC figures for this recording."""
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for scripts

    sid = f"{exp.slice_id}_{exp.condition}"

    # 1. PC1 distribution
    fig, _ = viz.plot_pc1_distribution(exp.pc1, exp.threshold, cfg)
    viz.save_figure(fig, f"{sid}_pc1_distribution", cfg)
    import matplotlib.pyplot as plt
    plt.close(fig)

    # 2. States overlaid on PC1 (first 60 seconds)
    fig, _ = viz.plot_states_overlay(
        exp.t_ds, exp.pc1, exp.states, exp.threshold, cfg,
        t_range_s=(0, min(60, exp.t_ds[-1]))
    )
    viz.save_figure(fig, f"{sid}_states_overlay", cfg)
    plt.close(fig)

    # 3. Feature signals (first 60 seconds)
    fig, _ = viz.plot_features(
        exp.features, exp.t_ds, cfg,
        t_range_s=(0, min(60, exp.t_ds[-1]))
    )
    viz.save_figure(fig, f"{sid}_features", cfg)
    plt.close(fig)

    # 4. Aligned firing rates
    if exp.up_traces is not None:
        fig, _ = viz.plot_aligned_firing_rates(
            exp.up_traces, exp.down_traces, exp.t_win, exp.condition, cfg
        )
        viz.save_figure(fig, f"{sid}_aligned_firing_rates", cfg)
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run the full analysis on a single .h5 recording."
    )
    parser.add_argument(
        "--slice", required=True,
        help="Path to the .h5 file to analyse."
    )
    parser.add_argument(
        "--condition", required=True,
        choices=None,   # validated at runtime via config
        help="Condition label (must be in config.yaml conditions list)."
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to config.yaml (default: project root)."
    )
    parser.add_argument(
        "--save-figures", action="store_true",
        help="Save diagnostic figures to the figures directory."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_single(args.slice, args.condition, cfg, save_figures=args.save_figures)


if __name__ == "__main__":
    main()
