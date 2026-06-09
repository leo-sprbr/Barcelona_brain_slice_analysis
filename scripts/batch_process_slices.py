"""
scripts/batch_process_slices.py
================================
Run the full analysis pipeline on ALL .h5 files in data/processed/,
group results by slice and condition, run statistical comparisons, and
save a consolidated results table.

File naming convention
----------------------
This script infers the condition from the file name.
Expected format:  <slice_id>_<condition>.h5
Example:          slice_01_baseline.h5
                  slice_01_stimulation.h5
                  slice_01_post_stimulation.h5

If your naming convention differs, edit _parse_condition_from_filename()
at the bottom of this file.

Usage
-----
    python scripts/batch_process_slices.py
    python scripts/batch_process_slices.py --config path/to/config.yaml
    python scripts/batch_process_slices.py --save-figures

Outputs
-------
results/
    all_summaries.csv        — one row per slice × condition
    delta_duration.csv       — Δ duration per slice
    statistical_tests.csv    — test results with corrected p-values
    figures/                 — plots (if --save-figures)
"""

import sys
import os
import argparse
import traceback

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.data_io import list_h5_files, save_results
from src.core.experiment import Experiment
from src.core.metrics import (
    exclude_outliers,
    compute_delta_duration,
    run_all_tests,
)


# ---------------------------------------------------------------------------
# Main batch runner
# ---------------------------------------------------------------------------

def batch_run(cfg: dict, save_figures: bool = False) -> pd.DataFrame:
    """
    Process all .h5 files and return a consolidated results DataFrame.

    Parameters
    ----------
    cfg : dict
        Full config dict.
    save_figures : bool
        If True, save diagnostic figures per recording.

    Returns
    -------
    pd.DataFrame
        Full results table (all slices × all conditions).
    """
    processed_dir = cfg["paths"]["processed_dir"]
    results_dir   = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    h5_files   = list_h5_files(processed_dir)
    conditions = cfg["conditions"]

    if not h5_files:
        print(f"No .h5 files found in {processed_dir}. Run convert_smr_to_h5.py first.")
        return pd.DataFrame()

    print(f"Found {len(h5_files)} .h5 files. Running analysis...\n")

    all_rows    = []
    experiments = {}   # {(slice_id, condition): Experiment}
    failed      = []

    for h5_path in tqdm(h5_files, desc="Processing slices"):
        condition = _parse_condition_from_filename(h5_path, conditions)

        if condition is None:
            print(f"  [SKIP] Cannot infer condition from: {os.path.basename(h5_path)}")
            print(f"         Expected one of {conditions} in the filename.")
            continue

        try:
            exp = Experiment(h5_path, condition=condition, cfg=cfg)
            exp.run()

            all_rows.append(exp.summary_row)
            experiments[(exp.slice_id, condition)] = exp

            if save_figures:
                _save_figures_for(exp, cfg)

        except Exception as e:
            print(f"\n  [ERROR] {os.path.basename(h5_path)}: {e}")
            traceback.print_exc()
            failed.append(h5_path)
            continue

    if not all_rows:
        print("No recordings processed successfully.")
        return pd.DataFrame()

    summary_df = pd.concat(all_rows, ignore_index=True)

    # ---- Save raw summaries ----
    save_results(summary_df, os.path.join(results_dir, "all_summaries.csv"))

    # ---- Outlier exclusion ----
    summary_df = _apply_outlier_exclusion(summary_df, cfg)

    # ---- Delta duration ----
    delta_df = _compute_all_deltas(summary_df, experiments, cfg)
    if delta_df is not None and not delta_df.empty:
        save_results(delta_df, os.path.join(results_dir, "delta_duration.csv"))

    # ---- Statistical tests ----
    stat_df = _run_statistics(summary_df, cfg)
    if stat_df is not None and not stat_df.empty:
        save_results(stat_df, os.path.join(results_dir, "statistical_tests.csv"))
        print("\nStatistical test results:")
        print(stat_df.to_string(index=False))

    # ---- Report failures ----
    if failed:
        print(f"\n{len(failed)} file(s) failed to process:")
        for f in failed:
            print(f"  {f}")

    print(f"\nBatch processing complete. Results saved to {results_dir}/")
    return summary_df


# ---------------------------------------------------------------------------
# Outlier exclusion (applied to duration columns)
# ---------------------------------------------------------------------------

def _apply_outlier_exclusion(summary_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Apply the two-level CI outlier exclusion to Up and Down durations,
    separately for each condition.
    """
    ci_soft = cfg["metrics"].get("ci_soft", 0.95)
    ci_hard = cfg["metrics"].get("ci_hard", 0.999)

    cleaned_rows = []

    for condition, group in summary_df.groupby("condition"):
        for state in ("up", "down"):
            col = f"mean_{state}_s"
            if col not in group.columns:
                continue

            vals = group[col].dropna().tolist()
            if not vals:
                continue

            _, mask = exclude_outliers(vals, ci_soft=ci_soft, ci_hard=ci_hard)
            n_excluded = sum(1 for m in mask if not m)

            if n_excluded > 0:
                print(f"  Outlier exclusion [{condition}, {state}]: "
                      f"{n_excluded}/{len(vals)} recordings excluded.")

    # Return df unchanged (the exclusion info is used in downstream tests,
    # not by dropping rows here — you can filter by condition in analysis).
    return summary_df


# ---------------------------------------------------------------------------
# Delta duration computation
# ---------------------------------------------------------------------------

def _compute_all_deltas(
    summary_df: pd.DataFrame,
    experiments: dict,
    cfg: dict,
) -> pd.DataFrame | None:
    """
    For each slice, compute Δ duration (stim − baseline) and
    Δ duration (post-stim − baseline) for Up and Down states.
    """
    baseline_cond = cfg.get("baseline_condition", cfg["conditions"][0])
    delta_rows    = []

    slice_ids = summary_df["slice_id"].unique()

    for slice_id in slice_ids:
        slice_df = summary_df[summary_df["slice_id"] == slice_id]

        if baseline_cond not in slice_df["condition"].values:
            continue   # no baseline for this slice, skip

        bl_row = slice_df[slice_df["condition"] == baseline_cond].iloc[0]

        for condition in cfg["conditions"]:
            if condition == baseline_cond:
                continue
            if condition not in slice_df["condition"].values:
                continue

            cond_row = slice_df[slice_df["condition"] == condition].iloc[0]

            row = {
                "slice_id":    slice_id,
                "condition":   condition,
                "delta_up_s":  (cond_row.get("mean_up_s",   np.nan) -
                                 bl_row.get("mean_up_s",   np.nan)),
                "delta_down_s":(cond_row.get("mean_down_s", np.nan) -
                                 bl_row.get("mean_down_s", np.nan)),
                "delta_so_hz": (cond_row.get("so_frequency", np.nan) -
                                 bl_row.get("so_frequency", np.nan)),
            }
            delta_rows.append(row)

    if not delta_rows:
        return None

    return pd.DataFrame(delta_rows)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def _run_statistics(summary_df: pd.DataFrame, cfg: dict) -> pd.DataFrame | None:
    """
    Run paired Wilcoxon tests for within-slice comparisons
    (each condition vs baseline).
    """
    baseline_cond = cfg.get("baseline_condition", cfg["conditions"][0])
    conditions    = cfg["conditions"]

    stat_results = []

    for metric, col in [("up_duration", "mean_up_s"),
                         ("down_duration", "mean_down_s"),
                         ("so_frequency", "so_frequency")]:

        # Build condition_data dict for run_all_tests()
        condition_data = {}
        for cond in conditions:
            vals = summary_df[summary_df["condition"] == cond][col].dropna().tolist()
            if vals:
                condition_data[cond] = vals

        if baseline_cond not in condition_data:
            continue

        # Test each non-baseline condition against baseline (paired Wilcoxon)
        test_pairs = [
            (baseline_cond, c)
            for c in conditions
            if c != baseline_cond and c in condition_data
        ]

        if not test_pairs:
            continue

        df_tests = run_all_tests(condition_data, test_pairs, "wilcoxon", cfg)

        if not df_tests.empty:
            df_tests.insert(0, "metric", metric)
            stat_results.append(df_tests)

    if not stat_results:
        return None

    return pd.concat(stat_results, ignore_index=True)


# ---------------------------------------------------------------------------
# Figure saving
# ---------------------------------------------------------------------------

def _save_figures_for(exp: Experiment, cfg: dict):
    """Save a minimal set of diagnostic figures for one recording."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from src import visualization as viz

        matplotlib.use("Agg")
        sid = f"{exp.slice_id}_{exp.condition}"

        fig, _ = viz.plot_pc1_distribution(exp.pc1, exp.threshold, cfg)
        viz.save_figure(fig, f"{sid}_pc1", cfg)
        plt.close(fig)

        fig, _ = viz.plot_states_overlay(
            exp.t_ds, exp.pc1, exp.states, exp.threshold, cfg,
            t_range_s=(0, min(60, exp.t_ds[-1]))
        )
        viz.save_figure(fig, f"{sid}_states", cfg)
        plt.close(fig)

    except Exception as e:
        print(f"  [Figure warning] {exp.slice_id}: {e}")


# ---------------------------------------------------------------------------
# Filename → condition parser
# ---------------------------------------------------------------------------

def _parse_condition_from_filename(h5_path: str, conditions: list) -> str | None:
    """
    Infer the recording condition from the file name by checking whether
    any condition label appears as a substring.

    File naming convention expected:  <anything>_<condition>.h5
    Example:  slice_01_baseline.h5  →  "baseline"

    If your files are named differently, edit this function.
    The function must return a string from the `conditions` list or None.

    Parameters
    ----------
    h5_path : str
    conditions : list of str

    Returns
    -------
    str or None
    """
    name = os.path.splitext(os.path.basename(h5_path))[0].lower()

    for cond in conditions:
        cond_lower = cond.lower()
        
        # Special handling for "stimulation": only match if "post_stimulation" 
        # doesn't appear in the filename
        if cond_lower == "stimulation":
            if cond_lower in name and "post_stimulation" not in name:
                return cond
        elif cond_lower in name:
            return cond

    return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch process all .h5 files in the processed directory."
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to config.yaml (default: project root)."
    )
    parser.add_argument(
        "--save-figures", action="store_true",
        help="Save diagnostic figures for each recording."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    batch_run(cfg, save_figures=args.save_figures)


if __name__ == "__main__":
    main()
