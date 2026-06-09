"""
src/core/experiment.py
======================
The Experiment class is the central object of the pipeline.

It represents one recording file (one .h5 file = one slice × one condition)
and provides a clean interface to run the full analysis chain:

    load → preprocess → detect Up/Down states → compute firing rates → metrics

Design principle
----------------
Each step stores its results as instance attributes, so you can inspect
intermediate results at any point in the chain, and can re-run individual
steps without re-running everything.

Typical usage
-------------
    from src.config import load_config
    from src.core.experiment import Experiment

    cfg = load_config()

    exp = Experiment("data/processed/slice_01_baseline.h5", condition="baseline", cfg=cfg)
    exp.run()

    # Access results
    print(exp.duration_stats)
    print(exp.up_epochs[:5])

    # Or run step by step for debugging:
    exp.load_data()
    exp.preprocess()
    exp.detect_states()
    exp.compute_firing_rates()
    exp.compute_metrics()
"""

import os
import numpy as np
import pandas as pd

from src.data_io import load_h5, get_slice_id
from src.preprocessing import select_channel, extract_features
from src.core.up_down import detect_states
from src.core.firing_rates import (
    compute_epoch_firing_rates,
    compute_aligned_traces,
)
from src.core.metrics import compute_state_durations, build_summary_table


class Experiment:
    """
    Container for one recording (one .h5 file), its analysis results,
    and methods to run each analysis step.

    Parameters
    ----------
    h5_path : str
        Path to the .h5 file for this recording.
    condition : str
        Condition label for this recording (must match config.yaml
        "conditions" list, e.g. "baseline", "stimulation", "post_stimulation").
    cfg : dict
        Full configuration dict loaded by src.config.load_config().

    Attributes (populated after run())
    ------------------------------------
    raw_data        : np.ndarray, shape (n_samples, n_channels) — raw LFP
    meta            : dict — recording metadata (fs, units, duration, etc.)
    lfp             : np.ndarray, shape (n_samples,) — selected/averaged channel
    features        : np.ndarray, shape (n_ds_samples, 3) — z-scored feature matrix
    t_ds            : np.ndarray, shape (n_ds_samples,) — downsampled time vector
    log_mua_raw     : np.ndarray — logMUA before z-scoring (for firing rate calc)
    states          : np.ndarray, shape (n_ds_samples,) — binary Up/Down vector
    epochs          : dict — {"up": [...], "down": [...]} epoch tuples
    pc1             : np.ndarray — PC1 values (for QC plots)
    threshold       : float — threshold used for state detection
    up_frs          : list of float — mean FR per Up epoch
    down_frs        : list of float — mean FR per Down epoch
    up_traces       : np.ndarray — transition-aligned FR traces (Up onset)
    down_traces     : np.ndarray — transition-aligned FR traces (Down onset)
    t_win           : np.ndarray — time axis for aligned traces
    duration_stats  : dict — output of compute_state_durations()
    summary_row     : pd.DataFrame — one-row summary for this recording
    """

    def __init__(self, h5_path: str, condition: str, cfg: dict):
        self.h5_path   = h5_path
        self.condition = condition
        self.cfg       = cfg
        self.slice_id  = get_slice_id(h5_path)

        # Validate that condition is registered in config
        if condition not in cfg["conditions"]:
            raise ValueError(
                f"Condition '{condition}' not found in config.yaml conditions list: "
                f"{cfg['conditions']}. Add it to config.yaml or fix the typo."
            )

        # Result attributes — all None until the corresponding step runs
        self.raw_data       = None
        self.meta           = None
        self.lfp            = None
        self.features       = None
        self.t_ds           = None
        self.log_mua_raw    = None
        self.states         = None
        self.epochs         = None
        self.pc1            = None
        self.threshold      = None
        self.up_frs         = None
        self.down_frs       = None
        self.up_traces      = None
        self.down_traces    = None
        self.t_win          = None
        self.duration_stats = None
        self.summary_row    = None

    # -----------------------------------------------------------------------
    # Full pipeline: run all steps in order
    # -----------------------------------------------------------------------

    def run(self) -> "Experiment":
        """
        Run the complete analysis chain for this recording.

        Steps run in order:
            1. load_data()
            2. preprocess()
            3. detect_states()
            4. compute_firing_rates()
            5. compute_metrics()

        Returns self so you can chain: exp = Experiment(...).run()
        """
        print(f"[{self.slice_id} | {self.condition}] Running full analysis...")
        self.load_data()
        self.preprocess()
        self.detect_states()
        self.compute_firing_rates()
        self.compute_metrics()
        print(f"[{self.slice_id} | {self.condition}] Done. "
              f"Detected {self.duration_stats['n_up']} Up states, "
              f"{self.duration_stats['n_down']} Down states.")
        return self

    # -----------------------------------------------------------------------
    # Step 1: Load
    # -----------------------------------------------------------------------

    def load_data(self) -> None:
        """Load raw data and metadata from the .h5 file."""
        self.raw_data, self.meta = load_h5(self.h5_path)
        print(f"  Loaded {self.meta['n_samples']} samples × "
              f"{self.meta['n_channels']} channels @ {self.meta['fs']} Hz")

    # -----------------------------------------------------------------------
    # Step 2: Preprocess
    # -----------------------------------------------------------------------

    def preprocess(self) -> None:
        """
        Extract the LFP channel(s) and compute the three feature signals
        (SO, gamma envelope, logMUA) used for state detection.

        Also stores the raw (unscored) logMUA for firing rate calculations.
        """
        if self.raw_data is None:
            raise RuntimeError("Call load_data() before preprocess().")

        preproc_cfg = self.cfg["preprocessing"]
        fs          = self.meta["fs"]

        # Select/average channel(s)
        self.lfp = select_channel(self.raw_data, preproc_cfg)

        # Extract z-scored features and downsampled time vector
        self.features, self.t_ds = extract_features(self.lfp, fs, preproc_cfg)

        # Also keep the raw (un-z-scored) logMUA for firing rate calculation.
        # We re-extract it here so firing rates are in original signal units.
        # (features[:, 2] is already z-scored; we need the unscaled version)
        self.log_mua_raw = self._extract_raw_log_mua(fs, preproc_cfg)

        print(f"  Features extracted: shape {self.features.shape}, "
              f"downsampled to {1.0/(self.t_ds[1]-self.t_ds[0]):.1f} Hz")

    def _extract_raw_log_mua(self, fs: float, preproc_cfg: dict) -> np.ndarray:
        """
        Re-run just the logMUA extraction without z-scoring, and downsample
        to match t_ds. Used for firing rate calculations in physical units.
        """
        from src.preprocessing import bandpass_filter, _sliding_log_power, _downsample

        mua_band   = preproc_cfg["mua_band"]
        mua_win_ms = preproc_cfg["mua_window_ms"]
        order      = preproc_cfg.get("filter_order", 4)
        target_fs  = preproc_cfg.get("downsample_to_hz", None)

        mua_filt = bandpass_filter(self.lfp, fs, mua_band[0], mua_band[1], order=order)
        log_mua  = _sliding_log_power(mua_filt, fs, mua_win_ms)

        if target_fs is not None and target_fs < fs:
            log_mua = _downsample(log_mua, fs, target_fs)

        return log_mua

    # -----------------------------------------------------------------------
    # Step 3: Up/Down state detection
    # -----------------------------------------------------------------------

    def detect_states(self) -> None:
        """
        Run PCA-based Up/Down state detection on the feature matrix.
        Stores states, epochs, PC1 values, and the detection threshold.
        """
        if self.features is None:
            raise RuntimeError("Call preprocess() before detect_states().")

        self.states, self.epochs, self.pc1, self.threshold = detect_states(
            self.features, self.t_ds, self.cfg
        )

        n_up   = len(self.epochs["up"])
        n_down = len(self.epochs["down"])
        print(f"  State detection: {n_up} Up states, {n_down} Down states "
              f"(threshold = {self.threshold:.3f})")

    # -----------------------------------------------------------------------
    # Step 4: Firing rates
    # -----------------------------------------------------------------------

    def compute_firing_rates(self) -> None:
        """
        Compute mean firing rates per epoch and transition-aligned traces.
        Uses the raw (un-z-scored) logMUA signal.
        """
        if self.epochs is None:
            raise RuntimeError("Call detect_states() before compute_firing_rates().")

        self.up_frs, self.down_frs = compute_epoch_firing_rates(
            self.log_mua_raw, self.t_ds, self.epochs, self.cfg
        )

        self.up_traces, self.down_traces, self.t_win = compute_aligned_traces(
            self.log_mua_raw, self.t_ds, self.epochs, self.cfg
        )

        print(f"  Firing rates: mean Up FR = {np.nanmean(self.up_frs):.3f}, "
              f"mean Down FR = {np.nanmean(self.down_frs):.3f}")

    # -----------------------------------------------------------------------
    # Step 5: Metrics summary
    # -----------------------------------------------------------------------

    def compute_metrics(self) -> None:
        """
        Compute duration statistics and assemble the one-row summary DataFrame.
        """
        if self.epochs is None:
            raise RuntimeError("Call detect_states() before compute_metrics().")

        self.duration_stats = compute_state_durations(self.epochs)

        self.summary_row = build_summary_table(
            slice_id      = self.slice_id,
            condition     = self.condition,
            duration_stats = self.duration_stats,
            up_frs        = self.up_frs   or [],
            down_frs      = self.down_frs or [],
        )

    # -----------------------------------------------------------------------
    # Convenience properties
    # -----------------------------------------------------------------------

    @property
    def fs_ds(self) -> float:
        """Downsampled sampling rate (Hz), inferred from t_ds."""
        if self.t_ds is None or len(self.t_ds) < 2:
            return None
        return 1.0 / float(self.t_ds[1] - self.t_ds[0])

    @property
    def n_up(self) -> int:
        """Number of detected Up states."""
        return len(self.epochs["up"]) if self.epochs else 0

    @property
    def n_down(self) -> int:
        """Number of detected Down states."""
        return len(self.epochs["down"]) if self.epochs else 0

    def __repr__(self) -> str:
        status = "ready" if self.summary_row is not None else "not yet run"
        return (
            f"Experiment(slice={self.slice_id}, condition={self.condition}, "
            f"status={status})"
        )


# ---------------------------------------------------------------------------
# Extension points
# ---------------------------------------------------------------------------
# - To add a new analysis step (e.g. coherence between channels):
#   1. Add a method compute_<thing>(self) here.
#   2. Add it to the run() chain.
#   3. Store the result as self.<thing>.
#
# - To support multi-segment recordings (multiple trials per file):
#   Modify load_data() to loop over segments, running detect_states()
#   per segment and concatenating the epoch lists.
#
# - To enable saving/loading the Experiment object itself:
#   Add save(path) / classmethod load(path) using pickle or h5py.
