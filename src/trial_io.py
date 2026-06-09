"""
src/trial_io.py
===============
Data structures and I/O helpers for trial-based recordings.

This module contains everything related to cutting a long recording into
trial windows and handling artifact rejection within those windows.

It is the ONLY new module required by the trial-windowing workflow.
All other modules (experiment.py, data_io.py, preprocessing.py, etc.)
are unchanged — they simply receive smaller, cleaner arrays.

Contents
--------
  TrialDefinition   — dataclass describing one trial and its sub-windows
  ConditionWindow   — dataclass describing one condition window
  detect_artifacts  — find samples exceeding a noise threshold
  interpolate_artifacts — replace artifact samples via linear interpolation
  save_trial_h5     — write a clipped window to a new .h5 file
  load_trial_metadata — read the extra trial metadata from a windowed .h5 file

Artifact rejection strategy
----------------------------
We use a two-stage approach:

  1. Amplitude threshold: any sample whose |value| exceeds
     `z_threshold × running_std` (computed in non-overlapping short windows)
     is flagged as an artifact.

  2. Padding: flagged samples are expanded by `pad_ms` on each side to
     capture the ringing / filter edge effects surrounding a spike.

  3. Interpolation: flagged regions are replaced by linear interpolation
     between the last clean sample before and the first clean sample after.
     This is gentler than zeroing and avoids introducing a step discontinuity.

All parameters are read from config.yaml → "artifact_rejection" section.
"""

import numpy as np
import h5py
import os
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ConditionWindow:
    """
    A sub-window within a trial, corresponding to one analysis condition.

    Attributes
    ----------
    name : str
        Condition label (must match config.yaml "conditions" list).
    start_s : float
        Start time of this window in seconds (in the ORIGINAL recording).
    end_s : float
        End time of this window in seconds (in the ORIGINAL recording).
    """
    name:    str
    start_s: float
    end_s:   float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class TrialDefinition:
    """
    One stimulation trial within a long recording.

    A trial has an overall time window (start_s, end_s) and zero or more
    condition sub-windows (e.g. baseline, stimulation, post_stimulation).

    If no condition sub-windows are specified, the full trial window is
    saved as a single file with condition label "recording".

    Attributes
    ----------
    label : str
        Short unique name for this trial (e.g. "trial_01").
    start_s : float
        Trial start in seconds (in the ORIGINAL recording).
    end_s : float
        Trial end in seconds (in the ORIGINAL recording).
    conditions : list of ConditionWindow
        Sub-windows for each condition. May be empty.
    """
    label:      str
    start_s:    float
    end_s:      float
    conditions: list = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


# ---------------------------------------------------------------------------
# Artifact detection
# ---------------------------------------------------------------------------

def detect_artifacts(
    lfp: np.ndarray,
    fs: float,
    artifact_cfg: dict,
) -> np.ndarray:
    """
    Detect artifact samples in a 1-D LFP signal.

    Strategy: compute the running standard deviation in non-overlapping
    short windows, then flag any sample whose absolute value exceeds
    `z_threshold × local_std`. Flagged regions are padded by `pad_ms`
    on both sides.

    Parameters
    ----------
    lfp : np.ndarray, shape (n_samples,)
        Single-channel signal (or channel average). Used only for detection;
        interpolation is applied channel-by-channel in the calling code.
    fs : float
        Sampling rate (Hz).
    artifact_cfg : dict
        The "artifact_rejection" section of config.yaml. Keys:
            z_threshold   (float) — how many SDs above local noise = artifact
                                    Default 8.0. Lower = more aggressive.
            window_ms     (float) — local std window size (ms). Default 1000.
            pad_ms        (float) — samples to pad around each artifact (ms).
                                    Default 50.
            min_gap_ms    (float) — merge artifact regions closer than this.
                                    Default 100.

    Returns
    -------
    np.ndarray, shape (n_samples,), dtype bool
        True at samples identified as artifacts.
    """
    z_threshold = artifact_cfg.get("z_threshold",  8.0)
    window_ms   = artifact_cfg.get("window_ms",  1000.0)
    pad_ms      = artifact_cfg.get("pad_ms",       50.0)
    min_gap_ms  = artifact_cfg.get("min_gap_ms",  100.0)

    win_samples = max(1, int(round(window_ms  * 1e-3 * fs)))
    pad_samples = max(0, int(round(pad_ms     * 1e-3 * fs)))
    gap_samples = max(0, int(round(min_gap_ms * 1e-3 * fs)))

    n        = len(lfp)
    local_std = np.empty(n)

    # Compute local std in non-overlapping windows, broadcast back to samples
    n_wins = max(1, n // win_samples)
    for i in range(n_wins):
        lo = i * win_samples
        hi = min(lo + win_samples, n)
        std = lfp[lo:hi].std()
        local_std[lo:hi] = std if std > 0 else 1.0

    # Handle any trailing samples not covered by full windows
    if n_wins * win_samples < n:
        local_std[n_wins * win_samples :] = local_std[n_wins * win_samples - 1]

    # Flag samples exceeding the threshold
    mask = np.abs(lfp) > z_threshold * local_std

    # Pad around flagged regions
    if pad_samples > 0 and mask.any():
        from scipy.ndimage import binary_dilation
        struct = np.ones(2 * pad_samples + 1, dtype=bool)
        mask   = binary_dilation(mask, structure=struct)

    # Merge gaps smaller than min_gap_ms
    if gap_samples > 0 and mask.any():
        mask = _merge_short_gaps(mask, gap_samples)

    return mask


def _merge_short_gaps(mask: np.ndarray, gap_samples: int) -> np.ndarray:
    """
    Fill gaps between artifact regions that are shorter than gap_samples.
    Prevents leaving tiny clean islands between two artifact bursts.
    """
    from scipy.ndimage import binary_closing
    struct = np.ones(gap_samples, dtype=bool)
    return binary_closing(mask, structure=struct)


# ---------------------------------------------------------------------------
# Artifact interpolation
# ---------------------------------------------------------------------------

def interpolate_artifacts(
    signal: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Replace artifact samples with linear interpolation between the nearest
    clean neighbours on each side.

    If the artifact extends to the very start or end of the signal, the
    boundary value is used (constant extrapolation).

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples,)
        One channel of data.
    mask : np.ndarray, shape (n_samples,), dtype bool
        True at artifact samples.

    Returns
    -------
    np.ndarray, shape (n_samples,)
        Signal with artifact regions replaced.
    """
    out   = signal.copy()
    n     = len(out)
    clean = np.where(~mask)[0]

    if len(clean) == 0:
        # Entire signal is flagged — return zeros (nothing to interpolate from)
        return np.zeros_like(signal)

    if not mask.any():
        return out

    # Linear interpolation using clean samples as control points
    art_idx = np.where(mask)[0]
    out[art_idx] = np.interp(art_idx, clean, signal[clean])

    return out


# ---------------------------------------------------------------------------
# Saving trial windows
# ---------------------------------------------------------------------------

def save_trial_h5(
    path: str,
    data: np.ndarray,
    original_meta: dict,
    extra_meta: dict,
) -> None:
    """
    Save a clipped, artifact-cleaned trial window to a new .h5 file.

    The output layout is identical to the files written by convert_smr_to_h5.py,
    so the rest of the pipeline (data_io.load_h5, Experiment, etc.) reads it
    without any modifications.

    Extra trial-level metadata is stored as additional attributes in the
    /metadata group.

    Parameters
    ----------
    path : str
        Full path to the output .h5 file.
    data : np.ndarray, shape (n_samples, n_channels)
        Clipped and cleaned LFP data.
    original_meta : dict
        Metadata dict from load_h5() on the source file.
    extra_meta : dict
        Additional key → value pairs to store (e.g. trial_label,
        trial_window_start_s, artifact_fraction, source_file).
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    n_samples, n_channels = data.shape

    with h5py.File(path, "w") as f:
        # --- Metadata group ---
        meta_grp = f.create_group("metadata")

        # Copy original metadata
        meta_grp.attrs["original_file"]    = original_meta.get("original_file", "")
        meta_grp.attrs["fs"]               = float(original_meta["fs"])
        meta_grp.attrs["units"]            = str(original_meta.get("units", "a.u."))
        meta_grp.attrs["duration_seconds"] = float(n_samples / original_meta["fs"])

        # Write extra trial-level metadata
        for key, value in extra_meta.items():
            meta_grp.attrs[key] = value

        # --- Raw data dataset ---
        f.create_dataset(
            "raw_data",
            data=data.astype(np.float32),   # float32 saves ~50% space vs float64
            compression="gzip",
            compression_opts=4,
            chunks=True,
        )


# ---------------------------------------------------------------------------
# Loading trial metadata
# ---------------------------------------------------------------------------

def load_trial_metadata(h5_path: str) -> dict:
    """
    Read the trial-specific metadata attributes from a windowed .h5 file.

    Returns an empty dict if the file was not created by define_trials.py
    (i.e. it is a full-session file without trial metadata).

    Parameters
    ----------
    h5_path : str

    Returns
    -------
    dict
        May include keys: trial_label, trial_window_start_s,
        trial_window_end_s, artifact_fraction, source_file.
    """
    trial_keys = {
        "trial_label", "trial_window_start_s", "trial_window_end_s",
        "artifact_fraction", "source_file",
    }
    result = {}

    with h5py.File(h5_path, "r") as f:
        meta_grp = f["metadata"]
        for key in trial_keys:
            if key in meta_grp.attrs:
                result[key] = meta_grp.attrs[key]

    return result


# ---------------------------------------------------------------------------
# Extension points
# ---------------------------------------------------------------------------
# - To use a different artifact detection method (e.g. based on derivatives
#   rather than amplitude, or using a global rather than local threshold):
#   Replace or extend detect_artifacts(). The mask it returns must be a
#   boolean array of the same length as the input signal.
#
# - To replace linear interpolation with something smoother (e.g. cubic
#   spline, or setting artifacts to NaN for downstream exclusion):
#   Replace or extend interpolate_artifacts().
#
# - To support a different output format (e.g. .nwb, .mat):
#   Add a save_trial_<format>() function here following the same interface.
