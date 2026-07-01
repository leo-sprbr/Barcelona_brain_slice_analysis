import os
import numpy as np
import pandas as pd

from src.data_io import get_slice_id, load_h5_time_window
from src.preprocessing import select_channel, extract_features, select_channel_raw, extract_features_all_channels
from src.core.firing_rates import (
    compute_epoch_firing_rates,
    compute_aligned_traces,
)
from src.core.metrics import compute_state_durations, build_summary_table
from src.core.up_down import detect_states, _run_pca
from src.preprocessing import bandpass_filter, _sliding_log_power, _downsample, highpass_filter, notch_filter
class Activity_analysis:
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
        
        self.meta           = None
        self.fs             = None
        
        self.raw_data       = None
        self.lfp            = None

        self.so_matrix      = None
        self.gamma_matrix   = None
        self.mua_matrix     = None


        self.detected_states = None
        self.channel_metrics = None

        self.metrics_data   = None
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

    def run(self):
        """
        Run the complete analysis chain for this recording.

        Steps run in order:
            1. load_data()
            2. preprocess()
            3. detect_states()
            4. compute_firing_rates()
            5. compute_metrics()

        Returns self so you can chain: exp = Activity_analysis(...).run()
        """
        print(f"[{self.slice_id} | {self.condition}] Running full analysis...")
        self.load_data()
        self.preprocess()
        self.extract_features_all_channels()


    # -----------------------------------------------------------------------
    # Step 1: Load
    # -----------------------------------------------------------------------

    def load_data(self) -> None:
        """Load raw data and metadata from the .h5 file."""
        self.raw_data, self.meta = load_h5_time_window(self.h5_path, self.cfg)
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
        self.fs          = self.meta["fs"]

        # Select channel(s) listed in the configuration file
        self.lfp = select_channel_raw(self.raw_data, preproc_cfg)

        

    def extract_features_all_channels(self):
        """
        Loops through all channels, extracts features, and groups them by feature type.
        """

        # --- FIX: Force 1D arrays into a 2D matrix shape of (n_samples, 1) ---
        if self.lfp.ndim == 1:
            self.lfp = self.lfp[:, np.newaxis]
            
        n_samples, n_channels = self.lfp.shape
        preproc_cfg = self.cfg['preprocessing']
        
        # 1. Run a quick dry-run on channel 0 to find out the downsampled length (n_downsampled)
        first_feat, self.t_ds = extract_features(self.lfp[:, 0], self.fs, preproc_cfg)
        n_ds = first_feat.shape[0]
        
        # 2. Pre-allocate matrices to hold data for all channels (shape: n_samples x n_channels)
        self.so_matrix   = np.zeros((n_ds, n_channels))
        self.gamma_matrix = np.zeros((n_ds, n_channels))
        self.mua_matrix   = np.zeros((n_ds, n_channels))

        # 3. Pre-allocate matrices for States & PCA Data
        states_matrix = np.zeros((n_ds, n_channels), dtype=int)
        pc1_matrix    = np.zeros((n_ds, n_channels))
        thresholds    = np.zeros(n_channels)

        # 4. Storage lists for statistical metrics and snippet arrays
        epochs_list     = [] # To hold the custom dictionary lists per channel
        metrics_list    = [] # To hold your compute_metrics summary rows
        up_traces_list  = [] # To hold trace snippets
        down_traces_list = []
        t_win           = None # Will be populated by the first successful channel

        # 5. Master Loop
        for ch in range(n_channels):
            print(f"Extracting features for Channel {ch + 1}/{n_channels}...")
            
            # Extract z-scored features
            feats, _ = extract_features(self.lfp[:, ch], self.fs, preproc_cfg)
            self.so_matrix[:, ch]   = feats[:, 0]
            self.gamma_matrix[:, ch] = feats[:, 1]
            self.mua_matrix[:, ch]   = feats[:, 2]
            pc1_matrix[:, ch] = _run_pca(feats)

            try:
                # Detect states immediately using those freshly minted features
                states, epochs, _, threshold = detect_states(feats, self.t_ds, self.cfg)
                # Store state variables
                states_matrix[:, ch] = states
                thresholds[ch]       = threshold
                epochs_list.append(epochs)
                # --- SAFE METRIC CALCULATION FOR HEALTHY CHANNELS ---
                # Re-extract raw logMUA for true physical scale firing calculations
                log_mua_raw = self._extract_raw_log_mua(self.lfp[:, ch], self.fs, preproc_cfg)
                # Extract both rates and aligned snippet window traces
                up_frs, down_frs, up_traces, down_traces, t_win_ch = self.compute_firing_rates(log_mua_raw, self.t_ds, epochs, self.cfg)
                t_win = t_win_ch # Keep global tracking reference
                # Compute final statistics row
                summary_row = self.compute_metrics(epochs, up_frs, down_frs)
                summary_row["channel"] = ch + 1  # Tag it with its real hardware number
                # print("AVAILABLE COLUMNS:", list(summary_row.keys()))  # <--- ADD THIS LINE
                metrics_list.append(summary_row)
                up_traces_list.append(up_traces)
                down_traces_list.append(down_traces)


            except ValueError as e:
                # Check if this is our specific KDE bimodal error
                if "bimodal" in str(e):
                    print(f"⚠️ [Skipping Channel {ch + 1}]: Distribution is not bimodal. Filling with placeholders.")
                    # Store safe placeholder zero arrays / None so your structure does not break
                    states_matrix[:, ch] = 0 
                    thresholds[ch]       = np.nan  # Use NaN to easily spot it on later analysis steps
                    empty_epochs         = {"up": [], "down": []}
                    epochs_list.append(empty_epochs)

                    # Append empty placeholders to metric trackers so index alignments match channel orders
                    metrics_list.append({"channel": ch + 1, "status": "KDE Failure / Skipped"})
                    up_traces_list.append(None)
                    down_traces_list.append(None)

                else:
                    # If it's a completely different ValueError, raise it normally
                    raise e
                    # Also keep the raw (un-z-scored) logMUA for firing rate calculation.

        self.detected_states = {
            "states": states_matrix,
            "pc1": pc1_matrix,
            "thresholds": thresholds,
            "epochs": epochs_list
        }

        self.channel_metrics = {
            "summaries": metrics_list,      # Access via: channel_metrics["summaries"][ch]
            "up_traces": up_traces_list,
            "down_traces": down_traces_list,
            "t_win": t_win
        }



    def _extract_raw_log_mua(self,lfp: np.ndarray, fs: float, preproc_cfg: dict) -> np.ndarray:
        mua_band   = preproc_cfg["mua_band"]
        mua_win_ms = preproc_cfg["mua_window_ms"]
        order      = preproc_cfg.get("filter_order", 4)
        target_fs  = preproc_cfg.get("downsample_to_hz", None)

        # Apply the same cleanup used by extract_features so firing rates
        # come from a drift-free, mains-free signal.
        hp_cutoff = preproc_cfg.get("hp_cutoff_hz", 0.1)
        if hp_cutoff and hp_cutoff > 0:
            lfp = highpass_filter(lfp, fs, hp_cutoff, order=order)
        notch_hz = preproc_cfg.get("notch_hz", 50.0)
        notch_q  = preproc_cfg.get("notch_quality", 30.0)
        if notch_hz and notch_hz > 0:
            lfp = notch_filter(lfp, fs, freq_hz=notch_hz, quality=notch_q)

        mua_filt = bandpass_filter(lfp, fs, mua_band[0], mua_band[1], order=order)
        log_mua  = _sliding_log_power(mua_filt, fs, mua_win_ms)

        if target_fs is not None and target_fs < fs:
            log_mua = _downsample(log_mua, fs, target_fs)
        return log_mua

    # -----------------------------------------------------------------------
    # Step 4: Firing rates
    # -----------------------------------------------------------------------

    def compute_firing_rates(self,log_mua_raw: np.ndarray, t_ds: float, epochs: dict, cfg: dict) -> tuple:
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

        return up_frs, down_frs, up_traces, down_traces, t_win

    # -----------------------------------------------------------------------
    # Step 5: Metrics summary
    # -----------------------------------------------------------------------

    def compute_metrics(self, epochs: dict, up_frs: list, down_frs: list) -> None:
        """
        Compute duration statistics and assemble the one-row summary DataFrame.
        """
        if epochs is None:
            raise RuntimeError("Call detect_states() before compute_metrics().")

        duration_stats = compute_state_durations(epochs)

        summary_row = build_summary_table(
            slice_id      = "S1",
            condition     = "Baseline",
            duration_stats = duration_stats,
            up_frs        = up_frs,
            down_frs      = down_frs,
        )
        return summary_row
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
