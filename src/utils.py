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
