"""
src/core/firing_rates.py
========================
Firing rate computation from the logMUA signal.

Two types of output are produced:

1. **Mean firing rate per epoch** — for each detected Up or Down state
   epoch, the mean of the (smoothed) logMUA within that epoch.
   Stored as a list of floats aligned with the epoch list from up_down.py.

2. **Transition-aligned traces** — the logMUA signal is sliced around
   every Up-onset and Down-onset, smoothed with a Gaussian kernel, and
   averaged across events. This gives the mean ± SEM peri-event firing
   rate for each condition.

Usage
-----
    from src.core.firing_rates import (
        compute_epoch_firing_rates,
        compute_aligned_traces,
    )

    # Mean FR per epoch
    up_frs, down_frs = compute_epoch_firing_rates(
        log_mua, t_ds, epochs, cfg
    )

    # Transition-aligned traces
    up_traces, down_traces, t_win = compute_aligned_traces(
        log_mua, t_ds, epochs, cfg
    )
    # up_traces : np.ndarray, shape (n_up_events, n_window_samples)
    # t_win     : np.ndarray, shape (n_window_samples,)  — time relative to onset
"""

import numpy as np
from src.utils import gaussian_smooth


# ---------------------------------------------------------------------------
# 1. Mean firing rate per epoch
# ---------------------------------------------------------------------------

def compute_epoch_firing_rates(
    log_mua: np.ndarray,
    t_ds: np.ndarray,
    epochs: dict,
    cfg: dict,
) -> tuple[list[float], list[float]]:
    """
    Compute the mean logMUA within each Up and Down state epoch.

    Note: logMUA is used as a proxy for network firing rate. Taking the
    mean within an epoch gives a single scalar 'firing rate' per state.

    Parameters
    ----------
    log_mua : np.ndarray, shape (n_samples,)
        The logMUA signal (column 2 of the feature matrix, BEFORE z-scoring,
        OR the z-scored version — be consistent within your analysis).
        Pass the raw logMUA for absolute comparisons, or z-scored for
        relative comparisons.
    t_ds : np.ndarray, shape (n_samples,)
        Time vector (seconds) matching log_mua.
    epochs : dict
        Output of up_down.detect_states():
        {"up": [(start_s, end_s, duration_s), ...], "down": [...]}
    cfg : dict
        Full configuration dict. Uses cfg["firing_rates"]["smoothing_sigma_ms"].

    Returns
    -------
    up_frs : list of float
        Mean logMUA for each Up-state epoch, in epoch order.
    down_frs : list of float
        Mean logMUA for each Down-state epoch, in epoch order.
    """
    fs_ds = _infer_fs(t_ds)

    # Optionally smooth the logMUA before averaging
    sigma_ms = cfg["firing_rates"].get("smoothing_sigma_ms", 50)
    sigma_samples = sigma_ms * 1e-3 * fs_ds
    smoothed = gaussian_smooth(log_mua, sigma_samples)

    up_frs   = _mean_in_epochs(smoothed, t_ds, epochs["up"])
    down_frs = _mean_in_epochs(smoothed, t_ds, epochs["down"])

    return up_frs, down_frs


def _mean_in_epochs(
    signal: np.ndarray,
    t_ds: np.ndarray,
    epoch_list: list[tuple],
) -> list[float]:
    """
    For each (start_s, end_s, duration_s) in epoch_list, return the
    mean of `signal` within [start_s, end_s].
    """
    fs = _infer_fs(t_ds)
    means = []

    for start_s, end_s, _ in epoch_list:
        lo = int(round(start_s * fs))
        hi = int(round(end_s   * fs))
        lo = max(0, lo)
        hi = min(len(signal), hi)

        if hi > lo:
            means.append(float(signal[lo:hi].mean()))
        else:
            means.append(np.nan)   # epoch too short after rounding

    return means


# ---------------------------------------------------------------------------
# 2. Transition-aligned firing rate traces
# ---------------------------------------------------------------------------

def compute_aligned_traces(
    log_mua: np.ndarray,
    t_ds: np.ndarray,
    epochs: dict,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract logMUA snippets aligned to Up-onset and Down-onset events,
    smooth each snippet, and return the full matrix of traces.

    Parameters
    ----------
    log_mua : np.ndarray, shape (n_samples,)
        The logMUA signal.
    t_ds : np.ndarray, shape (n_samples,)
        Time vector (seconds).
    epochs : dict
        Output of up_down.detect_states().
    cfg : dict
        Full config dict. Uses:
            cfg["firing_rates"]["smoothing_sigma_ms"]
            cfg["firing_rates"]["alignment_window_s"]

    Returns
    -------
    up_traces : np.ndarray, shape (n_up_events, n_window_samples)
        Each row is one Up-onset aligned trace.
        Rows with insufficient data are NaN-padded.
    down_traces : np.ndarray, shape (n_down_events, n_window_samples)
        Same for Down-onset.
    t_win : np.ndarray, shape (n_window_samples,)
        Time axis in seconds, centred on 0 (= state onset).
    """
    fs_ds      = _infer_fs(t_ds)
    sigma_ms   = cfg["firing_rates"].get("smoothing_sigma_ms", 50)
    win_s      = cfg["firing_rates"].get("alignment_window_s", 1.0)

    sigma_samples = sigma_ms * 1e-3 * fs_ds
    smoothed      = gaussian_smooth(log_mua, sigma_samples)

    win_samples = int(round(win_s * fs_ds))
    n_window    = 2 * win_samples + 1
    t_win       = np.linspace(-win_s, win_s, n_window)

    # Up-onset = start_s of each Up epoch
    up_onsets   = [start_s for start_s, _, _ in epochs["up"]]
    # Down-onset = start_s of each Down epoch
    down_onsets = [start_s for start_s, _, _ in epochs["down"]]

    up_traces   = _extract_snippets(smoothed, up_onsets,   fs_ds, win_samples)
    down_traces = _extract_snippets(smoothed, down_onsets, fs_ds, win_samples)

    return up_traces, down_traces, t_win


def _extract_snippets(
    signal: np.ndarray,
    onset_times_s: list[float],
    fs: float,
    win_samples: int,
) -> np.ndarray:
    """
    Extract signal snippets centred on each onset time.

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples,)
    onset_times_s : list of float
        Event onset times in seconds.
    fs : float
        Sampling rate.
    win_samples : int
        Half-window size in samples (total window = 2*win_samples + 1).

    Returns
    -------
    np.ndarray, shape (n_events, 2*win_samples + 1)
        Rows with insufficient signal are filled with NaN.
    """
    n_events = len(onset_times_s)
    n_window = 2 * win_samples + 1
    traces   = np.full((n_events, n_window), np.nan)

    for i, t_onset in enumerate(onset_times_s):
        centre = int(round(t_onset * fs))
        lo     = centre - win_samples
        hi     = centre + win_samples + 1

        # Only extract if fully within signal bounds
        if lo >= 0 and hi <= len(signal):
            traces[i] = signal[lo:hi]

    return traces


# ---------------------------------------------------------------------------
# 3. Delta firing rate
# ---------------------------------------------------------------------------

def compute_delta_firing_rate(
    baseline_frs: list[float],
    comparison_frs: list[float],
) -> list[float]:
    """
    Compute the normalised change in firing rate between a comparison
    condition and the baseline.

    Δ FR = (FR_comparison - FR_baseline) / |FR_baseline|

    This gives a unitless fractional change. Values > 0 mean increased
    firing; values < 0 mean decreased firing.

    Parameters
    ----------
    baseline_frs : list of float
        Mean firing rates for each epoch under baseline condition.
    comparison_frs : list of float
        Mean firing rates for matched epochs under comparison condition.
        Must be the same length as baseline_frs.

    Returns
    -------
    list of float
        Δ FR values, one per epoch.
    """
    if len(baseline_frs) != len(comparison_frs):
        raise ValueError(
            f"baseline_frs and comparison_frs must have the same length. "
            f"Got {len(baseline_frs)} and {len(comparison_frs)}."
        )

    deltas = []
    for bl, comp in zip(baseline_frs, comparison_frs):
        if np.isnan(bl) or np.isnan(comp) or bl == 0:
            deltas.append(np.nan)
        else:
            deltas.append((comp - bl) / abs(bl))

    return deltas


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _infer_fs(t_ds: np.ndarray) -> float:
    """Infer the sampling rate from the time vector."""
    if len(t_ds) < 2:
        return 1.0
    return 1.0 / float(t_ds[1] - t_ds[0])


# ---------------------------------------------------------------------------
# Extension points
# ---------------------------------------------------------------------------
# - To compute firing rates from spike times rather than logMUA:
#   Add a compute_spike_epoch_rates(spike_times, epochs) function that
#   counts spikes per epoch and normalises by duration.
#
# - To use a different smoothing kernel for aligned traces:
#   Replace the gaussian_smooth() call with your preferred function
#   from src/utils.py.
#
# - To align to state OFFSET rather than onset:
#   Replace start_s → end_s in the onset_times_s list comprehensions.
#
# - To baseline-normalise each trace (z-score relative to pre-onset):
#   Add a _baseline_normalise_traces(traces, t_win, baseline_window_s)
#   function that z-scores each row relative to the pre-event window.
