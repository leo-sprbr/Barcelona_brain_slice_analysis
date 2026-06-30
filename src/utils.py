"""
src/utils.py
============
Shared utility functions used across the pipeline.

Contents
--------
- gaussian_smooth     : 1-D Gaussian convolution
- bin_signal          : downsample by averaging non-overlapping bins
- compute_kde         : kernel density estimate on a 1-D array
- find_kde_valley     : find the valley between two modes in a KDE
- zscore              : re-exported from preprocessing for convenience
- confidence_interval : parametric or bootstrap CI on a 1-D array
- significance_label  : convert a p-value to a star-string
- get_condition_color : map condition names to matplotlib colors

These functions are intentionally stateless (pure functions with no
side effects) so they are easy to test in isolation.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def gaussian_smooth(signal: np.ndarray, sigma_samples: float) -> np.ndarray:
    """
    Apply a 1-D Gaussian smoothing kernel to a signal.

    Parameters
    ----------
    signal : np.ndarray, shape (n,)
    sigma_samples : float
        Standard deviation of the Gaussian kernel in samples.
        Convert from ms: sigma_samples = sigma_ms * fs / 1000

    Returns
    -------
    np.ndarray, shape (n,)
        Smoothed signal.
    """
    if sigma_samples <= 0:
        return signal.copy()
    return gaussian_filter1d(signal.astype(float), sigma=sigma_samples)


def bin_signal(
    signal: np.ndarray, fs: float, bin_size_ms: float
) -> tuple[np.ndarray, float]:
    """
    Downsample a signal by averaging non-overlapping bins.

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples,)
    fs : float
        Original sampling rate (Hz).
    bin_size_ms : float
        Bin size in milliseconds.

    Returns
    -------
    binned : np.ndarray
        Binned signal.
    new_fs : float
        Effective sampling rate after binning (Hz).
    """
    bin_samples = max(1, int(round(bin_size_ms * 1e-3 * fs)))
    n_bins      = len(signal) // bin_samples
    binned      = signal[: n_bins * bin_samples].reshape(n_bins, bin_samples).mean(axis=1)
    new_fs      = fs / bin_samples
    return binned, new_fs


# ---------------------------------------------------------------------------
# KDE and threshold helpers
# ---------------------------------------------------------------------------

def compute_kde(
    data: np.ndarray, n_points: int = 512, bandwidth_factor: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a kernel density estimate over a 1-D data array.

    Parameters
    ----------
    data : np.ndarray
        1-D data (e.g. PC1 values).
    n_points : int
        Number of evaluation points.
    bandwidth_factor : float
        Scaling factor applied to Scott's rule bandwidth.

    Returns
    -------
    x : np.ndarray, shape (n_points,)
        Evaluation grid.
    density : np.ndarray, shape (n_points,)
        Estimated density at each grid point.
    """
    kde = gaussian_kde(data, bw_method="scott")
    kde.set_bandwidth(kde.factor * bandwidth_factor)

    x = np.linspace(data.min(), data.max(), n_points)
    density = kde(x)
    return x, density


def find_kde_valley(
    x: np.ndarray, density: np.ndarray
) -> float:
    """
    Find the minimum (valley) between the two dominant modes of a
    bimodal KDE. This minimum is used as the Up/Down threshold.

    Strategy: find all local minima and return the one closest to the
    midpoint between the two largest local maxima.

    Parameters
    ----------
    x : np.ndarray
        Evaluation grid from compute_kde.
    density : np.ndarray
        KDE density values from compute_kde.

    Returns
    -------
    float
        The x-value of the valley (threshold).

    Raises
    ------
    ValueError
        If fewer than two local maxima are found (distribution not bimodal).
    """
    from scipy.signal import argrelextrema

    # Find local maxima and minima
    maxima_idx = argrelextrema(density, np.greater, order=5)[0]
    minima_idx = argrelextrema(density, np.less,    order=5)[0]

    if len(maxima_idx) < 2:
        raise ValueError(
            "PC1 distribution does not appear bimodal — fewer than 2 local "
            "maxima found in the KDE. Try adjusting kde_bandwidth_factor in "
            "config.yaml, or switch to threshold_method: 'manual'."
        )

    # Take the two tallest peaks
    top2 = maxima_idx[np.argsort(density[maxima_idx])[-2:]]
    peak_lo, peak_hi = sorted(x[top2])

    # Among all minima, pick the one between the two peaks
    midpoint = (peak_lo + peak_hi) / 2.0
    between  = minima_idx[
        (x[minima_idx] > peak_lo) & (x[minima_idx] < peak_hi)
    ]

    if len(between) == 0:
        # Fall back to the midpoint itself
        return float(midpoint)

    # Pick the minimum closest to the midpoint
    closest = between[np.argmin(np.abs(x[between] - midpoint))]
    return float(x[closest])


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def confidence_interval(
    data: np.ndarray, ci: float = 0.95
) -> tuple[float, float]:
    """
    Compute a parametric (normal-distribution) confidence interval.

    Used for outlier exclusion of Up/Down duration distributions
    (95% CI and 99.9% CI per the paper).

    Parameters
    ----------
    data : np.ndarray
        1-D array of values.
    ci : float
        Confidence level (0–1). Default 0.95.

    Returns
    -------
    (lower, upper) : tuple of float
    """
    from scipy.stats import t as t_dist

    n    = len(data)
    mean = np.mean(data)
    sem  = np.std(data, ddof=1) / np.sqrt(n)
    dof  = n - 1
    alpha = 1.0 - ci
    t_crit = t_dist.ppf(1 - alpha / 2, dof)
    margin = t_crit * sem
    return float(mean - margin), float(mean + margin)


def significance_label(p: float, stat_cfg: dict) -> str:
    """
    Convert a p-value to a significance star string.

    Parameters
    ----------
    p : float
        p-value.
    stat_cfg : dict
        The "statistics" section of config.yaml, which contains
        significance_labels as a list of [threshold, label] pairs.

    Returns
    -------
    str
        e.g. "****", "***", "**", "*", or "ns" (not significant).
    """
    labels = stat_cfg.get("significance_labels", [
        [0.05,   "*"],
        [0.01,   "**"],
        [0.001,  "***"],
        [0.0001, "****"],
    ])
    # Sort from most to least stringent
    labels_sorted = sorted(labels, key=lambda x: x[0])
    for threshold, label in reversed(labels_sorted):
        if p <= threshold:
            return label
    return "ns"


def apply_multiple_correction(
    p_values: list[float], method: str = "bonferroni"
) -> list[float]:
    """
    Apply multiple comparison correction to a list of p-values.

    Parameters
    ----------
    p_values : list of float
    method : str
        "bonferroni" or "fdr_bh" (Benjamini-Hochberg).

    Returns
    -------
    list of float
        Corrected p-values.
    """
    from statsmodels.stats.multitest import multipletests

    if method == "none":
        return list(p_values)

    method_map = {"bonferroni": "bonferroni", "fdr_bh": "fdr_bh"}
    if method not in method_map:
        raise ValueError(
            f"Unknown correction method: {method}. "
            f"Choose from: {list(method_map.keys())}"
        )

    _, corrected, _, _ = multipletests(p_values, method=method_map[method])
    return list(corrected)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def get_condition_color(condition: str, conditions_list: list, colormap: str = "tab10") -> tuple:
    """
    Map a condition name to a consistent matplotlib color.

    Parameters
    ----------
    condition : str
        Condition name (must be in conditions_list).
    conditions_list : list of str
        Full ordered list of conditions (from config.yaml).
    colormap : str
        Matplotlib colormap name.

    Returns
    -------
    tuple
        RGBA color tuple.
    """
    import matplotlib.cm as cm

    cmap = cm.get_cmap(colormap)
    idx  = conditions_list.index(condition) if condition in conditions_list else 0
    return cmap(idx / max(len(conditions_list) - 1, 1))


from scipy import signal

def spectrogram(lfp_signal, fs):
    # 1. Use a shorter segment window for sharper time precision (e.g., 1 second)
    nperseg = int(fs * 1.0) 
    shape = lfp_signal.shape
    print(f"Spectrogram input signal shape: {shape}, fs: {fs}, nperseg: {nperseg}")
    # 2. Drastically increase overlap (90%) for smooth, continuous time transitions
    noverlap = int(nperseg * 0.90)
    
    nfft = int(nperseg * 4)
    # Compute spectrogram (1-30 Hz) on the original LFP
    freqs, times, Sxx = signal.spectrogram(
        lfp_signal,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
    )
    # Filter to 1-20 Hz
    freq_mask = (freqs >= 1) & (freqs <= 20)
    freqs = freqs[freq_mask]
    Sxx = Sxx[freq_mask, :]

    # Convert to dB
    Sxx_log = np.log10(Sxx + 1e-10)
    # FIX: Explicitly clip the returned 'times' array to never exceed 
    # the true length of the input data signal
    true_duration = len(lfp_signal) / fs
    valid_mask = times <= true_duration
    
    return freqs, times[valid_mask], Sxx_log[:, valid_mask]

import re

def parse_trial_id(slice_id: str, conditions: list) -> str:
    """
    Remove the trailing condition label from a slice_id to obtain a
    trial-level grouping key.

    Example
    -------
    slice_id  = '20260414_SLICE3_CONTROL_Nathalia_vitaly_trial_01_baseline'
    conditions = ['baseline', 'stimulation', 'post_stimulation']
    → returns  '20260414_SLICE3_CONTROL_Nathalia_vitaly_trial_01'

    Works by removing the longest matching condition suffix
    (with an optional leading underscore or hyphen).
    If no condition suffix is found the original slice_id is returned,
    which is fine — it just won't pair with anything.
    """
    # Sort longest-first so 'post_stimulation' is tried before 'stimulation'
    for cond in sorted(conditions, key=len, reverse=True):
        # Match the condition at the end, preceded by a separator
        pattern = r'[_\-]' + re.escape(cond) + r'$'
        stripped = re.sub(pattern, '', slice_id, flags=re.IGNORECASE)
        if stripped != slice_id:
            return stripped
    return slice_id   # no condition suffix found — return as-is


def get_stimulation_files(
    h5_paths: list[str],
    conditions: list[str],
    stim_condition: str = "stimulation",
) -> list[str]:
    """
    Filter a list of .h5 paths to keep only those whose filename contains
    the stimulation condition label (identified via parse_trial_id logic).
 
    Parameters
    ----------
    h5_paths : list of str
    conditions : list of str
        All condition labels from config.yaml.
    stim_condition : str
        The condition label that marks stimulation files.
 
    Returns
    -------
    list of str
        Filtered list of paths.
    """
    import os
    result = []
    for path in h5_paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        # The condition label should appear as a suffix in the stem
        trial_id = parse_trial_id(stem, conditions)
        if trial_id != stem:
            # A condition suffix was found — check which one
            suffix = stem[len(trial_id):].lstrip('_-').lower()
            if suffix == stim_condition.lower():
                result.append(path)
    return result


def shared_ylims(*value_lists, padding: float = 0.1) -> tuple[float, float]:
    """
    Compute a shared y-axis range across multiple lists of values,
    with a relative padding on each side.

    Use this to synchronise y-axes between two side-by-side plots so
    that the same metric is visually comparable across groups.

    Parameters
    ----------
    *value_lists : list of float
        Any number of value lists (NaNs are ignored).
    padding : float
        Fractional padding added to each side of the range.
        Default 0.1 = 10% on each side.

    Returns
    -------
    (ymin, ymax) : tuple of float

    Example
    -------
        ymin, ymax = shared_ylims(delta_high, delta_low)
        for ax in axes:
            ax.set_ylim(ymin, ymax)
    """
    all_vals = np.array([
        v for lst in value_lists
        for v in lst
        if v is not None and not np.isnan(v)
    ])

    if len(all_vals) == 0:
        return (-1.0, 1.0)

    lo, hi = float(all_vals.min()), float(all_vals.max())
    span   = hi - lo if hi != lo else 1.0
    return (lo - padding * span, hi + padding * span)
# ---------------------------------------------------------------------------
# Extension points
# ---------------------------------------------------------------------------
# - To add a new smoothing kernel (e.g. boxcar, Hann window): add a function
#   smooth_<name>(signal, params) and call it instead of gaussian_smooth()
#   in firing_rates.py.
#
# - To use a bootstrap CI instead of parametric: replace the body of
#   confidence_interval() or add a separate bootstrap_ci() function.
#
# - To support additional multiple-testing corrections: add the key to
#   method_map in apply_multiple_correction().
