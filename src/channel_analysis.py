"""
src/channel_analysis.py
=======================
Per-channel logMUA statistics and saturation-based channel classification.

Purpose
-------
During magnetoelectric stimulation some channels receive much higher energy
than others (proximity to the nanoparticles / electrode geometry). These
"saturated" / high-logMUA channels should be analysed separately from
"low" channels so that the stimulation effect on Up/Down state dynamics
can be compared between the two groups.

This module handles:
  1. Computing mean logMUA and logMUA variance per channel, pooled across
     all stimulation files for a given recording set.
  2. Finding the KDE valley threshold that separates high from low channels.
  3. Returning the channel group assignments.

The channel indices here are always 0-based (as stored in the data).
Add +1 for display/printing — this is handled in the notebook and plots.

Usage
-----
    from src.channel_analysis import compute_channel_stats, classify_channels

    stats = compute_channel_stats(stim_h5_paths, cfg)
    # stats: DataFrame with columns [channel_0based, mean_logmua, var_logmua]

    high_ch, low_ch, threshold = classify_channels(stats, cfg)
    # high_ch, low_ch: lists of 0-based channel indices
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_io import load_h5
from src.preprocessing import (
    bandpass_filter,
    _compute_log_mua,
    _downsample,
    _compute_raw_mua,
)
from src.utils import compute_kde, find_kde_valley


# ---------------------------------------------------------------------------
# Step 1: Per-channel logMUA statistics across stimulation files
# ---------------------------------------------------------------------------

def compute_channel_stats(
    h5_paths: list[str],
    cfg: dict,
) -> pd.DataFrame:
    """
    For each channel, compute the mean and variance of the logMUA signal
    pooled across all provided .h5 files (typically all stimulation files).

    These two statistics capture:
      - mean_logmua  : overall firing level (elevated in saturated channels)
      - var_logmua   : temporal variability (also elevated when a channel
                       shows bursting / saturation artefacts)

    Parameters
    ----------
    h5_paths : list of str
        Paths to .h5 files to include (should be stimulation files only).
    cfg : dict
        Full config dict. Uses cfg["preprocessing"] for filter parameters.

    Returns
    -------
    pd.DataFrame with columns:
        channel_0based  : int  (0–31, as stored in the data)
        channel_1based  : int  (1–32, for display)
        mean_logmua     : float  (mean across time AND files)
        var_logmua      : float  (variance across time AND files)
        mean_mua        : float
        var_mua         : float
    """
    preproc_cfg = cfg["preprocessing"]
    mua_band    = preproc_cfg["mua_band"]
    mua_smooth  = preproc_cfg.get("mua_smooth_ms",  80.0)
    target_fs   = preproc_cfg.get("downsample_to_hz", None)

    if not h5_paths:
        raise ValueError("No .h5 files provided to compute_channel_stats().")

    # Load one file to get channel count
    _, meta0 = load_h5(h5_paths[0])
    n_channels = meta0["n_channels"]

    # Accumulators: list of arrays per channel
    # We collect all logMUA values (after downsampling) and compute
    # stats at the end to handle files of different lengths cleanly.
    all_means_log = np.zeros(n_channels)
    all_vars_log  = np.zeros(n_channels)
    all_means_raw = np.zeros(n_channels)
    all_vars_raw = np.zeros(n_channels)
    n_files   = 0

    for h5_path in tqdm(h5_paths, desc="Computing per-channel logMUA stats"):
        try:
            data, meta = load_h5(h5_path)
            fs = meta["fs"]
        except Exception as e:
            print(f"  [SKIP] {h5_path}: {e}")
            continue

        file_means_log = np.zeros(n_channels)
        file_vars_log  = np.zeros(n_channels)

        file_means_raw = np.zeros(n_channels)
        file_vars_raw = np.zeros(n_channels)

        for ch in range(n_channels):
            sig = data[:, ch].astype(np.float64)

            # Compute logMUA for this channel
            log_mua = _compute_log_mua(
                sig, fs, mua_band, mua_smooth
            )
            mua = _compute_raw_mua(
                sig, fs, mua_band, mua_smooth
            )

            # Optionally downsample to match the pipeline
            if target_fs is not None and target_fs < fs:
                log_mua = _downsample(log_mua, fs, target_fs)
                mua = _downsample(mua, fs, target_fs)

            file_means_log[ch] = float(np.nanmean(log_mua))
            file_vars_log[ch]  = float(np.nanvar(log_mua))

            file_means_raw[ch] = float(np.nanmean(mua))
            file_vars_raw[ch] = float(np.nanvar(mua))


        all_means_log += file_means_log
        all_vars_log  += file_vars_log
        all_means_raw += file_means_raw
        all_vars_raw += file_vars_raw
        n_files   += 1

    if n_files == 0:
        raise RuntimeError("No files were successfully processed.")

    # Average across files
    all_means_log /= n_files
    all_vars_log  /= n_files
    all_means_raw /= n_files
    all_vars_raw /= n_files

    all_log_vars_raw = np.log(all_vars_raw)

    df = pd.DataFrame({
        "channel_0based": np.arange(n_channels),
        "channel_1based": np.arange(1, n_channels + 1),   # +1 offset for display
        "mean_logmua":    all_means_log,
        "var_logmua":     all_vars_log,
        "mean_mua":       all_means_raw,
        "var_mua":        all_vars_raw,
        "log_var_mua":    all_log_vars_raw
    })

    return df


# ---------------------------------------------------------------------------
# Step 2: KDE valley threshold and group assignment
# ---------------------------------------------------------------------------

def classify_channels(
    stats_df: pd.DataFrame,
    cfg: dict,
    metric: str = "mean_logmua",
    bandwidth_factor: float = 1.0,
) -> tuple[list[int], list[int], float]:
    """
    Find the KDE valley in the distribution of `metric` across channels,
    and split channels into high (above threshold) and low (below threshold)
    groups.

    Parameters
    ----------
    stats_df : pd.DataFrame
        Output of compute_channel_stats().
    cfg : dict
        Full config dict. Uses cfg["preprocessing"]["high_channels"] and
        cfg["preprocessing"]["low_channels"] if already manually filled —
        in that case those values take priority and this function just
        validates them.
    metric : str
        Which column to threshold on. "mean_logmua" (default) or "var_logmua".
    bandwidth_factor : float
        KDE bandwidth scaling (passed to compute_kde).

    Returns
    -------
    high_channels : list of int
        0-based channel indices above the threshold (saturated / high-FR).
    low_channels : list of int
        0-based channel indices at or below the threshold.
    threshold : float
        The KDE valley value used for classification.
    """
    preproc_cfg = cfg.get("preprocessing", {})

    # If already manually set in config, use those directly
    high_manual = preproc_cfg.get("high_channels", [])
    low_manual  = preproc_cfg.get("low_channels",  [])

    if high_manual and low_manual:
        print("  Using channel groups from config.yaml (manual override).")
        print(f"  High channels (0-based): {high_manual}")
        print(f"  Low  channels (0-based): {low_manual}")
        # Compute threshold as midpoint between group means for reporting
        vals = stats_df.set_index("channel_0based")[metric]
        high_vals = [vals[c] for c in high_manual if c in vals.index]
        low_vals  = [vals[c] for c in low_manual  if c in vals.index]
        threshold = (np.mean(high_vals) + np.mean(low_vals)) / 2.0 if (high_vals and low_vals) else 0.0
        return list(high_manual), list(low_manual), threshold

    # Auto-detect via KDE valley
    values = stats_df[metric].values

    try:
        x, density = compute_kde(values, bandwidth_factor=bandwidth_factor)
        threshold   = find_kde_valley(x, density)
    except ValueError:
        # Distribution not bimodal — fall back to median split
        threshold = float(np.median(values))
        print(f"  Warning: KDE valley not found in {metric} distribution. "
              f"Falling back to median split (threshold = {threshold:.4f}).")

    high_channels = stats_df.loc[
        stats_df[metric] > threshold, "channel_0based"
    ].tolist()
    low_channels  = stats_df.loc[
        stats_df[metric] <= threshold, "channel_0based"
    ].tolist()

    return high_channels, low_channels, threshold


# ---------------------------------------------------------------------------
# Step 3: Build per-group config overrides
# ---------------------------------------------------------------------------

def make_group_cfg(base_cfg: dict, channel_indices: list[int]) -> dict:
    """
    Return a copy of base_cfg with preprocessing.channel overridden to
    average the specified channel indices. All other settings unchanged.

    This is the key function that plugs into the existing Experiment class:
    Experiment(h5_path, condition, cfg=make_group_cfg(cfg, high_channels))
    runs the full pipeline on just the high-channel average.

    Parameters
    ----------
    base_cfg : dict
        Full config dict (will not be mutated).
    channel_indices : list of int
        0-based channel indices to average.

    Returns
    -------
    dict
        Modified config with preprocessing.channel set to channel_indices.
    """
    import copy
    group_cfg = copy.deepcopy(base_cfg)
    group_cfg["preprocessing"]["channel"] = list(channel_indices)
    return group_cfg
