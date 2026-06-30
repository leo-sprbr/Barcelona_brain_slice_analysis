"""
src/core/metrics.py
===================
All quantitative metrics derived from Up/Down state detection, and the
statistical tests used to compare conditions.

Contents
--------
1. compute_state_durations  — Up/Down mean durations and SO frequency
2. exclude_outliers         — CI-based outlier removal (95% / 99.9%)
3. compute_delta_duration   — Δ duration (stim − baseline) per slice
4. paired_wilcoxon_test     — within-slice comparison (stim vs baseline)
5. mannwhitney_test         — between-group comparison
6. build_summary_table      — assemble all metrics into a DataFrame

All statistical tests are non-parametric, as per the paper.
Bonferroni correction is applied for multiple comparisons (configurable).

Usage
-----
    from src.core.metrics import (
        compute_state_durations,
        exclude_outliers,
        compute_delta_duration,
        paired_wilcoxon_test,
        mannwhitney_test,
    )
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu
from src.core.firing_rates import (
    compute_epoch_firing_rates,
    compute_aligned_traces,
)
from src.utils import confidence_interval, apply_multiple_correction, significance_label

import sys, os

# from src.core.metrics import compute_firing_rates
sys.path.insert(0, os.path.abspath('..'))

# ---------------------------------------------------------------------------
# 1. State duration summary
# ---------------------------------------------------------------------------
def compute_firing_rates(epochs: dict, log_mua_raw: np.ndarray, t_ds: np.ndarray, cfg: dict) -> None:
    """
    Compute mean firing rates per epoch and transition-aligned traces.
    Uses the raw (un-z-scored) logMUA signal.
    """
    if epochs is None:
        raise RuntimeError("Call detect_states() before compute_firing_rates().")

    up_frs, down_frs = compute_epoch_firing_rates(
        log_mua_raw, t_ds, epochs, cfg
    )

    up_traces, down_traces, t_win = compute_aligned_traces(
        log_mua_raw, t_ds, epochs, cfg
    )

    print(f"  Firing rates: mean Up FR = {np.nanmean(up_frs):.3f}, "
            f"mean Down FR = {np.nanmean(down_frs):.3f}")
    
def compute_state_durations(epochs: dict) -> dict:
    """
    Compute summary duration statistics from a set of Up/Down epochs.

    Parameters
    ----------
    epochs : dict
        {"up": [(start_s, end_s, duration_s), ...], "down": [...]}
        Output of up_down.detect_states().

    Returns
    -------
    dict with keys:
        up_durations   : list of float — individual Up durations (s)
        down_durations : list of float — individual Down durations (s)
        mean_up_s      : float — mean Up duration (s)
        mean_down_s    : float — mean Down duration (s)
        sem_up_s       : float — SEM of Up durations
        sem_down_s     : float — SEM of Down durations
        so_frequency   : float — mean SO frequency (Hz) = 1 / (Up + Down cycle)
        n_up           : int   — number of Up states detected
        n_down         : int   — number of Down states detected
    """
    up_durs   = [d for _, _, d in epochs["up"]]
    down_durs = [d for _, _, d in epochs["down"]]

    # SO frequency = inverse of mean cycle duration (Up + Down)
    # Use the shorter of the two lists (need matched pairs)
    n_cycles = min(len(up_durs), len(down_durs))
    if n_cycles > 0:
        cycle_durs = [up_durs[i] + down_durs[i] for i in range(n_cycles)]
        so_freq    = float(np.mean([1.0 / c for c in cycle_durs if c > 0]))
    else:
        so_freq = np.nan

    def _mean(x): return float(np.mean(x)) if x else np.nan
    def _sem(x):  return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan

    return {
        "up_durations":   up_durs,
        "down_durations": down_durs,
        "mean_up_s":      _mean(up_durs),
        "mean_down_s":    _mean(down_durs),
        "sem_up_s":       _sem(up_durs),
        "sem_down_s":     _sem(down_durs),
        "so_frequency":   so_freq,
        "n_up":           len(up_durs),
        "n_down":         len(down_durs),
    }


# ---------------------------------------------------------------------------
# 2. Outlier exclusion
# ---------------------------------------------------------------------------

def exclude_outliers(
    duration_values: list[float],
    ci_soft: float = 0.95,
    ci_hard: float = 0.999,
) -> tuple[list[float], list[bool]]:
    """
    Remove duration outliers using the two-level CI approach from the paper.

    - Events outside the 95% CI are soft-excluded from duration analyses.
    - Events outside the 99.9% CI (recordings with very few Up states)
      are hard-excluded entirely.

    The mask returned can be used to filter epoch lists and firing rate
    arrays consistently.

    Parameters
    ----------
    duration_values : list of float
        Up or Down state durations (seconds) for one condition.
    ci_soft : float
        Soft CI threshold (default 0.95).
    ci_hard : float
        Hard CI threshold (default 0.999).

    Returns
    -------
    filtered : list of float
        Durations with outliers removed.
    mask : list of bool
        True for values that were kept (False = excluded).
    """
    arr = np.array(duration_values, dtype=float)

    if len(arr) < 3:
        # Not enough data to compute CI — keep everything
        return duration_values, [True] * len(duration_values)

    # Hard exclusion first (99.9% CI)
    lo_hard, hi_hard = confidence_interval(arr, ci=ci_hard)
    hard_mask = (arr >= lo_hard) & (arr <= hi_hard)
    arr_hard  = arr[hard_mask]

    if len(arr_hard) < 3:
        hard_mask = np.ones(len(arr), dtype=bool)  # keep all if too few survive
        arr_hard  = arr

    # Soft exclusion (95% CI) on the hard-filtered set
    lo_soft, hi_soft = confidence_interval(arr_hard, ci=ci_soft)
    soft_mask_on_hard = (arr_hard >= lo_soft) & (arr_hard <= hi_soft)

    # Reconstruct full mask
    full_mask = np.zeros(len(arr), dtype=bool)
    hard_indices = np.where(hard_mask)[0]
    for i, kept in zip(hard_indices, soft_mask_on_hard):
        full_mask[i] = kept

    filtered = arr[full_mask].tolist()
    return filtered, full_mask.tolist()


# ---------------------------------------------------------------------------
# 3. Delta duration
# ---------------------------------------------------------------------------

def compute_delta_duration(
    baseline_durations: list[float],
    comparison_durations: list[float],
) -> list[float]:
    """
    Compute the change in state duration between a comparison condition
    and baseline.

    Δ duration = duration_comparison − duration_baseline   (seconds)

    Epochs are assumed to be paired (same slice, matched ordering).
    If the lists differ in length, only the overlapping portion is used.

    Parameters
    ----------
    baseline_durations : list of float
    comparison_durations : list of float

    Returns
    -------
    list of float
        Δ duration values (seconds), positive = longer in comparison.
    """
    n = min(len(baseline_durations), len(comparison_durations))
    if n == 0:
        return []

    return [
        comparison_durations[i] - baseline_durations[i]
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# 4. Paired Wilcoxon signed-rank test
# ---------------------------------------------------------------------------

def paired_wilcoxon_test(
    baseline_values: list[float],
    comparison_values: list[float],
    cfg: dict,
) -> dict:
    """
    Paired Wilcoxon signed-rank test (within-slice comparison).

    Used when comparing the SAME slice across conditions
    (e.g. baseline vs stimulation, baseline vs post-stimulation).
    This is a non-parametric alternative to the paired t-test.

    Parameters
    ----------
    baseline_values : list of float
        Metric values (e.g. Up durations) under baseline condition.
    comparison_values : list of float
        Paired metric values under comparison condition.
    cfg : dict
        Full config dict. Uses cfg["statistics"].

    Returns
    -------
    dict with keys:
        statistic : float
        p_value   : float (raw, uncorrected)
        n_pairs   : int
        label     : str  (significance stars, e.g. "**")
        test      : str  ("wilcoxon")
    """
    n = min(len(baseline_values), len(comparison_values))

    if n < 5:
        return _empty_result("wilcoxon", n, reason="too few pairs")

    bl   = np.array(baseline_values[:n], dtype=float)
    comp = np.array(comparison_values[:n], dtype=float)

    # Drop pairs with NaN
    valid = ~(np.isnan(bl) | np.isnan(comp))
    bl, comp = bl[valid], comp[valid]

    if valid.sum() < 5:
        return _empty_result("wilcoxon", int(valid.sum()), reason="too few valid pairs")

    stat, p = wilcoxon(bl, comp, alternative="two-sided")

    return {
        "statistic": float(stat),
        "p_value":   float(p),
        "n_pairs":   int(valid.sum()),
        "label":     significance_label(p, cfg["statistics"]),
        "test":      "wilcoxon",
    }


# ---------------------------------------------------------------------------
# 5. Mann-Whitney U test (independent groups)
# ---------------------------------------------------------------------------

def mannwhitney_test(
    group_a: list[float],
    group_b: list[float],
    cfg: dict,
) -> dict:
    """
    Two-sided Mann-Whitney U test for independent groups.

    Used when comparing DIFFERENT slices or conditions that are not
    paired (e.g. baseline activity vs. sham / inactive nanoparticles).

    Parameters
    ----------
    group_a : list of float
        Metric values for group A.
    group_b : list of float
        Metric values for group B.
    cfg : dict
        Full config dict. Uses cfg["statistics"].

    Returns
    -------
    dict with keys:
        statistic : float
        p_value   : float (raw, uncorrected)
        n_a, n_b  : int
        label     : str  (significance stars)
        test      : str  ("mannwhitney")
    """
    a = np.array([v for v in group_a if not np.isnan(v)], dtype=float)
    b = np.array([v for v in group_b if not np.isnan(v)], dtype=float)

    if len(a) < 3 or len(b) < 3:
        return _empty_result("mannwhitney", 0, reason="too few observations")

    stat, p = mannwhitneyu(a, b, alternative="two-sided")

    return {
        "statistic": float(stat),
        "p_value":   float(p),
        "n_a":       int(len(a)),
        "n_b":       int(len(b)),
        "label":     significance_label(p, cfg["statistics"]),
        "test":      "mannwhitney",
    }


# ---------------------------------------------------------------------------
# 6. Run all pairwise tests with correction
# ---------------------------------------------------------------------------

def run_all_tests(
    condition_data: dict[str, list[float]],
    test_pairs: list[tuple[str, str]],
    test_type: str,
    cfg: dict,
) -> pd.DataFrame:
    """
    Run a battery of statistical tests and apply multiple comparison
    correction (Bonferroni by default, configurable in config.yaml).

    Parameters
    ----------
    condition_data : dict
        {condition_name: [values, ...]}
        e.g. {"baseline": [0.3, 0.4, ...], "stimulation": [0.5, 0.6, ...]}
    test_pairs : list of (str, str)
        Pairs of condition names to compare.
        e.g. [("baseline", "stimulation"), ("baseline", "post_stimulation")]
    test_type : str
        "wilcoxon" for paired tests, "mannwhitney" for independent.
    cfg : dict
        Full config dict.

    Returns
    -------
    pd.DataFrame with columns:
        condition_a, condition_b, statistic, p_raw, p_corrected, label, test
    """
    results = []

    for cond_a, cond_b in test_pairs:
        if cond_a not in condition_data or cond_b not in condition_data:
            continue

        if test_type == "wilcoxon":
            res = paired_wilcoxon_test(
                condition_data[cond_a], condition_data[cond_b], cfg
            )
        else:
            res = mannwhitney_test(
                condition_data[cond_a], condition_data[cond_b], cfg
            )

        results.append({
            "condition_a": cond_a,
            "condition_b": cond_b,
            **res,
        })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Apply multiple comparison correction to the p-values
    correction = cfg["statistics"].get("correction", "bonferroni")
    if "p_value" in df.columns and len(df) > 1:
        corrected = apply_multiple_correction(df["p_value"].tolist(), method=correction)
        df["p_corrected"] = corrected
        # Update significance labels using corrected p
        df["label"] = df["p_corrected"].apply(
            lambda p: significance_label(p, cfg["statistics"])
        )
    else:
        df["p_corrected"] = df.get("p_value", np.nan)

    return df


# ---------------------------------------------------------------------------
# 7. Summary table
# ---------------------------------------------------------------------------

def build_summary_table(
    slice_id: str,
    condition: str,
    duration_stats: dict,
    up_frs: list[float],
    down_frs: list[float],
) -> pd.DataFrame:
    """
    Assemble per-condition summary metrics into a one-row DataFrame.
    Concatenate across slices and conditions to build the full results table.

    Parameters
    ----------
    slice_id : str
    condition : str
    duration_stats : dict
        Output of compute_state_durations().
    up_frs : list of float
        Mean logMUA per Up epoch.
    down_frs : list of float
        Mean logMUA per Down epoch.

    Returns
    -------
    pd.DataFrame, one row per slice × condition.
    """
    return pd.DataFrame([{
        "slice_id":       slice_id,
        "condition":      condition,
        "n_up":           duration_stats["n_up"],
        "n_down":         duration_stats["n_down"],
        "mean_up_s":      duration_stats["mean_up_s"],
        "sem_up_s":       duration_stats["sem_up_s"],
        "mean_down_s":    duration_stats["mean_down_s"],
        "sem_down_s":     duration_stats["sem_down_s"],
        "so_frequency":   duration_stats["so_frequency"],
        "mean_fr_up":     float(np.nanmean(up_frs))   if up_frs   else np.nan,
        "mean_fr_down":   float(np.nanmean(down_frs)) if down_frs else np.nan,
    }])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _empty_result(test: str, n: int, reason: str = "") -> dict:
    """Return a placeholder result dict when a test cannot be run."""
    result = {
        "statistic": np.nan,
        "p_value":   np.nan,
        "label":     "n/a",
        "test":      test,
        "note":      reason,
    }
    if test == "wilcoxon":
        result["n_pairs"] = n
    else:
        result["n_a"] = n
        result["n_b"] = 0
    return result

def print_channel_metrics_summary(metrics_data: dict, states_data: dict, cfg: dict) -> None:
    """
    Accepts the pipeline metrics and state structures to print a beautifully
    formatted, human-readable statistical summary table for all channels.
    Safely unpacks data metrics regardless of whether they are formatted as
    standard Python dictionaries, Pandas Series, or DataFrames.
    """
    # 1. Grab the total number of processed channels
    n_processed_channels = len(metrics_data["summaries"])
    
    # 2. Grab the configured channel labels
    channel_list = cfg["preprocessing"].get("channel", None)
    
    if channel_list is None or channel_list == []:
        channel_list = list(range(1, n_processed_channels + 1))
        
    # Helper to safely extract a clean float out of potential Pandas objects
    def _to_float(item, key_name):
        if item is None:
            return 0.0
        
        # Extract the value depending on whether it's a DataFrame, Series, or dict
        try:
            if hasattr(item, "get"):
                val = item.get(key_name, 0.0)
            else:
                val = item[key_name]
        except (KeyError, IndexError, ValueError):
            return 0.0

        # If the extracted value itself is a Pandas Series, pull the first value
        if hasattr(val, "iloc"):
            if val.empty:
                return 0.0
            val = val.iloc[0]
            
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    # 3. Collect all lines of text in a tracking list
    lines = []
    lines.append("="*50)
    lines.append("         METRICS SUMMARY PER CHANNEL         ")
    lines.append("="*50)
    
    # 4. Iterate through every channel summary row stored in metrics_data
    for idx, summary in enumerate(metrics_data["summaries"]):
        real_ch = channel_list[idx] if idx < len(channel_list) else (idx + 1)
        
        lines.append(f"\n▶ Hardware Channel {real_ch}:")
        
        # Check if this channel threw a KDE bimodal error and was skipped
        if isinstance(summary, dict) and "status" in summary:
            lines.append(f"  ❌ Status: SKIPPED (KDE non-bimodal distribution)")
            continue
        elif np.isnan(states_data["thresholds"][idx]):
            lines.append(f"  ❌ Status: SKIPPED (KDE non-bimodal distribution)")
            continue
            
        # 5. Extract and format values cleanly using our adaptive unpacking tool
        n_up      = int(_to_float(summary, 'n_up'))
        n_down    = int(_to_float(summary, 'n_down'))
        mean_up   = _to_float(summary, 'mean_up_s')
        sem_up    = _to_float(summary, 'sem_up_s')
        mean_down = _to_float(summary, 'mean_down_s')
        sem_down  = _to_float(summary, 'sem_down_s')
        so_freq   = _to_float(summary, 'so_frequency')
        
        # 6. Add the cleaned text metrics to our list
        lines.append(f"  • Total States Detected : Up: {n_up} | Down: {n_down}")
        lines.append(f"  • Mean Up Duration      : {mean_up:.3f} seconds (± {sem_up:.3f} SEM)")
        lines.append(f"  • Mean Down Duration    : {mean_down:.3f} seconds (± {sem_down:.3f} SEM)")
        lines.append(f"  • Slow Oscillation Freq : {so_freq:.3f} Hz")
        
        # 7. Unpack and print your multi-unit firing rate intensities (mean logMUA)
        mean_up_fr   = _to_float(summary, 'mean_fr_up')
        mean_down_fr = _to_float(summary, 'mean_fr_down')
        
        lines.append(f"  • Mean Up Firing Rate   : {mean_up_fr:.3f} (logMUA strength)")
        lines.append(f"  • Mean Down Firing Rate : {mean_down_fr:.3f} (logMUA strength)")
            
    lines.append("\n" + "="*50)
    
    # 8. Join the lines together into a single block of string text
    full_output_text = "\n".join(lines)
    
    # 9. Write out to the console terminal screen
    print(full_output_text)
    

    processed_dir = os.path.dirname(cfg["paths"]["processed_dir"])
    fig_dir = os.path.join(processed_dir, "results", "figures")
    t_start = cfg['visualization']['t_start']
    t_end = cfg['visualization']['t_end']

    # 2. Build the name of the dynamic subfolder (e.g., "3280-3340")
    time_folder = f"{t_start}-{t_end}"
    fig_dir = os.path.join(fig_dir, time_folder)
    os.makedirs(fig_dir, exist_ok=True)

    metric_summary_path = os.path.join(fig_dir, f"Metric_summary_{time_folder}.txt")
    
    try:
        with open(metric_summary_path, "w", encoding="utf-8") as f:
            f.write(full_output_text)
        print(f"\n💾 [Saved Successfully]: Full report text written to '{os.path.abspath(metric_summary_path)}'\n")
    except Exception as e:
        print(f"\n⚠️ Warning: Could not write metrics to text file due to error: {e}\n")
# ---------------------------------------------------------------------------
# Extension points
# ---------------------------------------------------------------------------
# - To add a new metric (e.g. Up/Down duration ratio, CV of durations):
#   Add a function compute_<metric>(epochs) → dict here and call it
#   from Experiment.run_analysis() in experiment.py.
#
# - To support more than 2 conditions in omnibus testing (e.g. Kruskal-Wallis
#   followed by post-hoc Dunn test): add run_kruskal_wallis() here.
#
# - To add effect size metrics (e.g. Cohen's d, rank-biserial r):
#   Extend the result dict returned by each test function.
