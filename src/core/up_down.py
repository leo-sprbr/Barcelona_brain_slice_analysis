"""
src/core/up_down.py
===================
Up/Down state detection using PCA on three LFP-derived feature signals,
as described in Tort-Colet et al. (2021).

Algorithm summary
-----------------
1. Receive the (n_samples, 3) feature matrix from preprocessing.py
   (columns: z-scored SO, gamma envelope, logMUA).
2. Run PCA → take PC1 (which captures the bimodal Up/Down variance).
3. Threshold the PC1 distribution to separate Up and Down states:
   - "kde" mode : find the valley of the bimodal KDE automatically.
   - "manual"   : use a fixed threshold from config.yaml.
4. Build a binary state vector (1 = Up, 0 = Down).
5. Enforce minimum state durations (merge short events into neighbours).
6. Extract event epochs: list of (start_idx, end_idx, duration_s) for
   each Up and Down state.

Usage
-----
    from src.core.up_down import detect_states

    states, epochs, pc1, threshold = detect_states(features, t_ds, cfg)

    # states    : np.ndarray of 0/1, same length as t_ds
    # epochs    : dict with keys "up" and "down", each a list of
    #             (start_s, end_s, duration_s) tuples
    # pc1       : np.ndarray — the raw PC1 values (useful for QC plots)
    # threshold : float — the threshold used
"""

import numpy as np
from sklearn.decomposition import PCA

from src.utils import compute_kde, find_kde_valley


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def detect_states(
    features: np.ndarray,
    t_ds: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, dict, np.ndarray, float]:
    """
    Detect Up and Down states from the PCA of the feature matrix.

    Parameters
    ----------
    features : np.ndarray, shape (n_samples, 3)
        Z-scored feature matrix: [SO, gamma_env, logMUA].
        Output of preprocessing.extract_features().
    t_ds : np.ndarray, shape (n_samples,)
        Time vector (seconds) at the downsampled rate.
    cfg : dict
        Full configuration dict (from src.config.load_config()).
        Uses cfg["up_down"] and cfg["preprocessing"]["downsample_to_hz"].

    Returns
    -------
    states : np.ndarray, shape (n_samples,), dtype int
        Binary state vector: 1 = Up state, 0 = Down state.
    epochs : dict
        {"up":   [(start_s, end_s, duration_s), ...],
         "down": [(start_s, end_s, duration_s), ...]}
    pc1 : np.ndarray, shape (n_samples,)
        PC1 score values (for diagnostic plots).
    threshold : float
        The threshold value applied on PC1.
    """
    ud_cfg = cfg["up_down"]

    # 1. PCA — keep only PC1
    pc1 = _run_pca(features)

    # 2. Determine threshold
    threshold = _compute_threshold(pc1, ud_cfg)

    # 3. Binary state vector (Up = 1 where PC1 > threshold)
    raw_states = (pc1 > threshold).astype(int)

    # 4. Enforce minimum durations
    fs_ds = _infer_fs(t_ds)
    states = _enforce_min_duration(raw_states, fs_ds, ud_cfg)

    # 5. Extract epochs
    epochs = _extract_epochs(states, t_ds)

    return states, epochs, pc1, threshold


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def _run_pca(features: np.ndarray) -> np.ndarray:
    """
    Fit a PCA on the feature matrix and return the first component scores.

    The sign of PC1 is set so that higher values correspond to Up states
    (more gamma / MUA activity → higher PC1).

    Parameters
    ----------
    features : np.ndarray, shape (n_samples, n_features)

    Returns
    -------
    pc1 : np.ndarray, shape (n_samples,)
    """
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(features).ravel()

    # Convention: Up states should have HIGHER PC1 values.
    # The gamma and logMUA columns (indices 1, 2) should have positive loadings.
    # If the mean loading on those columns is negative, flip the sign.
    loadings = pca.components_[0]           # shape (n_features,)
    activity_loadings = loadings[1:].mean() # gamma + logMUA loadings
    if activity_loadings < 0:
        pc1 = -pc1

    return pc1


# ---------------------------------------------------------------------------
# Thresholding
# ---------------------------------------------------------------------------

def _compute_threshold(pc1: np.ndarray, ud_cfg: dict) -> float:
    """
    Compute the Up/Down threshold from the PC1 distribution.

    Parameters
    ----------
    pc1 : np.ndarray
    ud_cfg : dict
        The "up_down" section of config.yaml.

    Returns
    -------
    float
        Threshold value.
    """
    method = ud_cfg.get("threshold_method", "kde")

    if method == "manual":
        return float(ud_cfg.get("manual_threshold", 0.0))

    elif method == "kde":
        bw_factor = ud_cfg.get("kde_bandwidth_factor", 1.0)
        x, density = compute_kde(pc1, bandwidth_factor=bw_factor)
        return find_kde_valley(x, density)

    else:
        raise ValueError(
            f"Unknown threshold_method: '{method}'. "
            f"Choose 'kde' or 'manual' in config.yaml."
        )


# ---------------------------------------------------------------------------
# Minimum duration enforcement
# ---------------------------------------------------------------------------

def _enforce_min_duration(
    states: np.ndarray, fs: float, ud_cfg: dict
) -> np.ndarray:
    """
    Remove states shorter than the configured minimum durations by
    merging them with the neighbouring state.

    Short Up  states → set to 0 (Down)
    Short Down states → set to 1 (Up)

    This is applied iteratively until no more short states remain.

    Parameters
    ----------
    states : np.ndarray, shape (n_samples,), dtype int
    fs : float
        Sampling rate of the states vector.
    ud_cfg : dict

    Returns
    -------
    np.ndarray, shape (n_samples,), dtype int
        States with short events merged.
    """
    min_up_samples   = int(ud_cfg.get("min_up_duration_s",   0.05) * fs)
    min_down_samples = int(ud_cfg.get("min_down_duration_s", 0.05) * fs)

    states = states.copy()
    changed = True

    while changed:
        changed = False
        runs = _get_runs(states)

        for state_val, start, end in runs:
            length = end - start
            min_len = min_up_samples if state_val == 1 else min_down_samples

            if length < min_len:
                # Flip this short run to the opposite state
                states[start:end] = 1 - state_val
                changed = True
                break   # Restart after any change (runs are now stale)

    return states


# ---------------------------------------------------------------------------
# Epoch extraction
# ---------------------------------------------------------------------------

def _extract_epochs(
    states: np.ndarray, t_ds: np.ndarray
) -> dict:
    """
    Convert the binary state vector into a list of (start, end, duration)
    epoch tuples for Up and Down states.

    Parameters
    ----------
    states : np.ndarray, shape (n_samples,), dtype int
    t_ds : np.ndarray, shape (n_samples,)
        Time vector in seconds.

    Returns
    -------
    dict
        {"up":   [(start_s, end_s, duration_s), ...],
         "down": [(start_s, end_s, duration_s), ...]}
    """
    up_epochs   = []
    down_epochs = []

    for state_val, start_idx, end_idx in _get_runs(states):
        start_s   = float(t_ds[start_idx])
        end_s     = float(t_ds[min(end_idx, len(t_ds) - 1)])
        duration  = end_s - start_s

        epoch = (start_s, end_s, duration)
        if state_val == 1:
            up_epochs.append(epoch)
        else:
            down_epochs.append(epoch)

    return {"up": up_epochs, "down": down_epochs}


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _get_runs(states: np.ndarray) -> list[tuple[int, int, int]]:
    """
    Find contiguous runs of identical values in a binary array.

    Returns
    -------
    list of (value, start_idx, end_idx)
        end_idx is exclusive (like Python slice notation).
    """
    runs   = []
    n      = len(states)
    i      = 0

    while i < n:
        val   = states[i]
        j     = i + 1
        while j < n and states[j] == val:
            j += 1
        runs.append((int(val), i, j))
        i = j

    return runs


def _infer_fs(t_ds: np.ndarray) -> float:
    """Infer the sampling rate from the time vector."""
    if len(t_ds) < 2:
        return 1.0
    return 1.0 / (t_ds[1] - t_ds[0])


# ---------------------------------------------------------------------------
# Extension points
# ---------------------------------------------------------------------------
# - To use a different dimensionality reduction (e.g. ICA, UMAP):
#   Replace _run_pca() with your function; keep the same output shape.
#
# - To use a Gaussian Mixture Model for thresholding:
#   Add a _compute_threshold_gmm() function and add a new option
#   threshold_method: "gmm" in config.yaml / _compute_threshold().
#
# - To use more than 3 feature signals: just add columns to the
#   feature matrix in preprocessing.py; this module uses all columns.
