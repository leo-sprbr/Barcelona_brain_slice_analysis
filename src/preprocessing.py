
"""
src/preprocessing.py
====================
Signal processing functions that transform raw LFP into the three
feature signals used for Up/Down state detection.

METHOD — what the paper actually says
--------------------------------------
The Tort-Colet / Sanchez-Vives group uses three signals, but the exact
implementations matter a lot. Based on the primary literature (Mattia et al.,
Ruiz-Mejias et al., the Cobrawap pipeline paper by Gutzen et al. 2022, and
the Tort-Colet 2021 paper):

  1. SO deflection
     → the DECIMATED RAW SIGNAL (not bandpass-filtered).
       The raw LFP is simply low-pass filtered (or just downsampled) to
       remove high-frequency noise, then used directly. This preserves the
       full slow-oscillation waveform, including the sharp Up-state onset.
       (The paper says "decimated raw signal", not "bandpass SO signal".)

  2. Gamma envelope
     → RMS (root-mean-square) of the gamma-band (30–80 Hz) filtered signal
       in a LONG sliding window (500 ms default), then smoothed with a
       Gaussian. Using variance in a 50 ms window (original implementation)
       gives a noisy, rapidly-fluctuating estimate that doesn't track state
       well. The longer window + RMS is far more stable.

  3. logMUA
     → Power in the 200–1500 Hz band computed via Welch PSD in sliding
       windows, normalised by the FULL-SPECTRUM power in each window
       (so the metric captures the RELATIVE high-frequency content, not
       absolute power). This normalisation is the key step described in
       Mattia et al. (2010) and the Cobrawap paper (section 4.3.2), and is
       what makes logMUA state-dependent rather than tracking overall signal
       amplitude. After log-scaling, an 80 ms moving-average smooth is
       applied (Mattia & Del Giudice 2002).

After extraction each signal is z-scored, then they are stacked
column-wise into a (n_samples, 3) matrix ready for PCA.

All parameters come from config.yaml → "preprocessing" section.

Usage
-----
    from src.preprocessing import select_channel, extract_features

    lfp = select_channel(raw_data, cfg["preprocessing"])
    features, t_ds = extract_features(lfp, fs, cfg["preprocessing"])
    # features.shape → (n_downsampled_samples, 3)
    # t_ds           → time vector at the downsampled rate
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch, filtfilt
from scipy.signal import resample_poly
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from fractions import Fraction
from src.core.up_down import detect_states, _run_pca
from src.core.firing_rates import (
    compute_epoch_firing_rates,
    compute_aligned_traces,
)

from src.core.metrics import compute_state_durations, build_summary_table
# ---------------------------------------------------------------------------
# Channel selection
# ---------------------------------------------------------------------------

def select_channel(data: np.ndarray, preproc_cfg: dict) -> np.ndarray:
    """
    Select or average channels from the raw data array.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_channels)
        Raw LFP data.
    preproc_cfg : dict
        The "preprocessing" section of config.yaml.
        Key: "channel" — int, list of ints, or null (→ average all).

    Returns
    -------
    lfp : np.ndarray, shape (n_samples,)
        Single-channel (or averaged) LFP signal.
    """
    channel = preproc_cfg.get("channel", None)
    print(f"Selecting channel(s): {channel}")

    # So it is either one single channel or the average of multiple channels
    if channel is None:
        lfp = data.mean(axis=1)
    elif isinstance(channel, list):
        lfp = data[:, channel].mean(axis=1)
    elif isinstance(channel, int):
        lfp = data[:, channel]
    else:
        raise ValueError(
            f"config.yaml preprocessing.channel must be null, an int, "
            f"or a list of ints. Got: {channel}"
        )

    return lfp.astype(np.float64)

def select_channel_raw(data: np.ndarray, preproc_cfg: dict) -> np.ndarray:
    """
    Select channels from the raw data array.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_channels)
        Raw LFP data.
    preproc_cfg : dict
        The "preprocessing" section of config.yaml.
        Key: "channel" — int, list of ints, or null (→ select all).

    Returns
    -------
    lfp : np.ndarray, shape (n_samples,)
        Single-channel LFP signal.
    """
    channel = preproc_cfg.get("channel", None)
    print(f"Selecting channel(s): {channel}")
    
    if channel is None:
        print("All channels are selected.")
        lfp = data  # Returns the full 2D matrix (n_samples, n_channels)
        
    elif isinstance(channel, int):
        idx = channel - 1
        lfp = data[:, idx]  # Returns a 1D vector (n_samples,)
        
    elif isinstance(channel, list):
        if len(channel) == 1:
            idx = channel[0] - 1
            print(f"Single channel list detected. Unpacking channel {channel[0]}")
            lfp = data[:, idx]  # Forces it to be a clean 1D vector (n_samples,)
        else:
            idx = [c - 1 for c in channel]
            lfp = data[:, np.array(idx)]  # Returns a 2D matrix subset (n_samples, len(channel))
            
    else:
        raise ValueError(
            f"config.yaml preprocessing.channel must be null, an int, "
            f"or a list of ints. Got: {channel}"
        )

    return lfp.astype(np.float64)


# ---------------------------------------------------------------------------
# Bandpass filtering
# ---------------------------------------------------------------------------

def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass filter (forward-backward, sosfiltfilt).

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples,)
    fs : float
    low_hz, high_hz : float
    order : int

    Returns
    -------
    np.ndarray, shape (n_samples,)
    """
    nyq = fs / 2.0
    if high_hz >= nyq:
        raise ValueError(
            f"High cutoff {high_hz} Hz ≥ Nyquist {nyq} Hz. "
            f"Lower the cutoff or raise the sampling rate."
        )
    sos = butter(order, [low_hz / nyq, high_hz / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, signal)


def lowpass_filter(
    signal: np.ndarray,
    fs: float,
    cutoff_hz: float,
    order: int = 4,
) -> np.ndarray:
    """
    Zero-phase Butterworth low-pass filter.
    Used to produce the decimated raw signal (SO feature).
    """
    nyq = fs / 2.0
    if cutoff_hz >= nyq:
        return signal.copy()
    sos = butter(order, cutoff_hz / nyq, btype="low", output="sos")
    return sosfiltfilt(sos, signal)

def highpass_filter(
    signal: np.ndarray,
    fs: float,
    cutoff_hz: float,
    order: int = 4,
) -> np.ndarray:
    """
    Zero-phase Butterworth low-pass filter.
    Used to produce the decimated raw signal (SO feature).
    """
    nyq = fs / 2.0
    if cutoff_hz >= nyq:
        return signal.copy()
    sos = butter(order, cutoff_hz / nyq, btype="high", output="sos")
    return sosfiltfilt(sos, signal)

# ---------------------------------------------------------------------------
# Feature extraction — main entry point
# ---------------------------------------------------------------------------

def extract_features(
    lfp: np.ndarray,
    fs: float,
    preproc_cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the three feature signals (SO, gamma envelope, logMUA),
    z-score each, and return them stacked as a matrix.

    Parameters
    ----------
    lfp : np.ndarray, shape (n_samples)
        Single-channel LFP signal at original sampling rate.
    fs : float
        Sampling rate in Hz.
    preproc_cfg : dict
        The "preprocessing" section of config.yaml.

    Returns
    -------
    features : np.ndarray, shape (n_downsampled, 3)
        Column 0 — z-scored SO (decimated raw LFP)
        Column 1 — z-scored gamma RMS envelope
        Column 2 — z-scored logMUA (normalised, smoothed)
    t_ds : np.ndarray, shape (n_downsampled,)
        Time vector (seconds) at the downsampled sampling rate.
    """

    order     = preproc_cfg.get("filter_order", 4)
    target_fs = preproc_cfg.get("downsample_to_hz", None)

    # ── Cleaning: remove slow drift + mains noise BEFORE any feature extraction
    hp_cutoff = preproc_cfg.get("hp_cutoff_hz", 0.1)
    if hp_cutoff and hp_cutoff > 0:
        lfp = highpass_filter(lfp, fs, hp_cutoff, order=order)

    notch_hz = preproc_cfg.get("notch_hz", 50.0)
    notch_q  = preproc_cfg.get("notch_quality", 30.0)
    if notch_hz and notch_hz > 0:
        lfp = notch_filter(lfp, fs, freq_hz=notch_hz, quality=notch_q)
    # ────────────────────────────────────────────────────────────────────────

    # ── NEW: remove slow drift before any feature extraction ──────────────
    hp_cutoff = preproc_cfg.get("hp_cutoff_hz", 0.1)
    if hp_cutoff and hp_cutoff > 0:
        lfp = highpass_filter(lfp, fs, hp_cutoff, order=order)

    # ------------------------------------------------------------------
    # 1. SO signal — decimated raw LFP
    #    Low-pass filter at so_lowpass_hz to remove high-frequency noise,
    #    then downsample. This is what "decimated raw signal" means:
    #    the raw trace at a lower sample rate, preserving slow waveform.
    #    DO NOT bandpass here — keeping the DC / infra-slow components
    #    is what gives the strong Up (negative deflection) / Down
    #    (return to baseline) contrast in the final PC1.
    # ------------------------------------------------------------------
    so_lowpass = preproc_cfg.get("so_lowpass_hz", 10.0)
    so_signal  = lowpass_filter(lfp, fs, so_lowpass, order=order)

    # ------------------------------------------------------------------
    # 2. Gamma envelope — RMS in a long sliding window, then smoothed
    #    A 500 ms window (default) captures several gamma cycles and
    #    gives a stable estimate that tracks state (Up = high gamma).
    #    RMS is preferred over variance: it's more numerically stable and
    #    directly interpretable as signal amplitude.
    # ------------------------------------------------------------------
    gamma_band   = preproc_cfg["gamma_band"]
    gamma_win_ms = preproc_cfg.get("gamma_window_ms", 200.0)

    gamma_filt = bandpass_filter(lfp, fs, gamma_band[0], gamma_band[1], order=order)
    gamma_env  = _sliding_variance(gamma_filt, fs, gamma_win_ms)

    # ------------------------------------------------------------------
    # 3. logMUA — normalised high-frequency power, log-scaled + smoothed
    #    Key step: divide the MUA-band power by the TOTAL power in each
    #    window. This normalisation (from Mattia et al. 2010) makes the
    #    logMUA track the RELATIVE high-frequency content, not just the
    #    overall signal amplitude.
    #    After log-scaling, apply an 80 ms moving-average smooth
    #    (Mattia & Del Giudice 2002) to reduce fast fluctuations.
    # ------------------------------------------------------------------
    mua_band   = preproc_cfg["mua_band"]
    mua_win_ms = preproc_cfg.get("mua_window_ms", 5.0)

    mua_filt = bandpass_filter(lfp, fs, mua_band[0], mua_band[1], order=order)
    log_mua  = _sliding_log_power(mua_filt, fs, mua_win_ms)
    
    # ------------------------------------------------------------------
    # Align lengths: logMUA may be shorter after windowed computation.
    # After downsampling, trim all three to the shortest.
    # ------------------------------------------------------------------
    if target_fs is not None and target_fs < fs:
        so_signal = _downsample(so_signal, fs, target_fs)
        gamma_env = _downsample(gamma_env, fs, target_fs)
        log_mua   = _downsample(log_mua,   fs, target_fs)
        fs_out    = float(target_fs)
    else:
        fs_out = float(fs)

    # Trim to common length
    n = min(len(so_signal), len(gamma_env), len(log_mua))
    so_signal = so_signal[:n]
    gamma_env = gamma_env[:n]
    log_mua   = log_mua[:n]

    # Trim filtering artifacts
    so_signal = so_signal[100:-100]
    gamma_env = gamma_env[100:-100]
    log_mua = log_mua[100:-100]

    # Z-score each feature independently
    so_z    = _zscore(so_signal)
    gamma_z = _zscore(gamma_env)
    mua_z   = _zscore(log_mua)

    n = min(len(so_signal), len(gamma_env), len(log_mua))

    features = np.column_stack([so_z, gamma_z, mua_z])
    t_ds     = np.arange(n) / fs_out

    return features, t_ds

def extract_features_all_channels(lfp_all: np.ndarray, fs: float, cfg: dict) -> tuple[dict, np.ndarray]:
    """
    Loops through all channels, extracts features, and groups them by feature type.
    
    Returns
    -------
    grouped_features : dict
        Contains 'SO', 'gamma', and 'logMUA' arrays of shape (n_downsampled, n_channels)
    t_ds : np.ndarray
        The downsampled time vector.
    """

    # --- FIX: Force 1D arrays into a 2D matrix shape of (n_samples, 1) ---
    if lfp_all.ndim == 1:
        lfp_all = lfp_all[:, np.newaxis]
        
    n_samples, n_channels = lfp_all.shape
    preproc_cfg = cfg['preprocessing']
    
    # 1. Run a quick dry-run on channel 0 to find out the downsampled length (n_downsampled)
    first_feat, t_ds = extract_features(lfp_all[:, 0], fs, preproc_cfg)
    n_ds = first_feat.shape[0]
    
    # 2. Pre-allocate matrices to hold data for all channels (shape: n_samples x n_channels)
    so_matrix   = np.zeros((n_ds, n_channels))
    gamma_matrix = np.zeros((n_ds, n_channels))
    mua_matrix   = np.zeros((n_ds, n_channels))

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
        feats, _ = extract_features(lfp_all[:, ch], fs, preproc_cfg)
        so_matrix[:, ch]   = feats[:, 0]
        gamma_matrix[:, ch] = feats[:, 1]
        mua_matrix[:, ch]   = feats[:, 2]
        pc1_matrix[:, ch] = _run_pca(feats)

        try:
            # Detect states immediately using those freshly minted features
            states, epochs, _, threshold = detect_states(feats, t_ds, cfg)
            # Store state variables
            states_matrix[:, ch] = states
            thresholds[ch]       = threshold
            epochs_list.append(epochs)
            # --- SAFE METRIC CALCULATION FOR HEALTHY CHANNELS ---
            # Re-extract raw logMUA for true physical scale firing calculations
            log_mua_raw = _extract_raw_log_mua(lfp_all[:, ch], fs, preproc_cfg)
            # Extract both rates and aligned snippet window traces
            up_frs, down_frs, up_traces, down_traces, t_win_ch = compute_firing_rates(log_mua_raw, t_ds, epochs, cfg)
            t_win = t_win_ch # Keep global tracking reference
            # Compute final statistics row
            summary_row = compute_metrics(epochs, up_frs, down_frs)
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

    grouped_features = {
        "SO": so_matrix,
        "gamma": gamma_matrix,
        "logMUA": mua_matrix
    }

    detected_states = {
        "states": states_matrix,
        "pc1": pc1_matrix,
        "thresholds": thresholds,
        "epochs": epochs_list
    }

    channel_metrics = {
        "summaries": metrics_list,      # Access via: channel_metrics["summaries"][ch]
        "up_traces": up_traces_list,
        "down_traces": down_traces_list,
        "t_win": t_win
    }

    return grouped_features, detected_states, channel_metrics,t_ds
# ---------------------------------------------------------------------------
# Feature-specific helpers
# ---------------------------------------------------------------------------

def _sliding_rms(signal: np.ndarray, fs: float, window_ms: float) -> np.ndarray:
    """
    Compute the RMS (root-mean-square) of `signal` in a sliding window.

    Uses scipy's uniform_filter1d for O(n) efficiency.
    RMS(x) = sqrt( mean(x^2) )

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples,)
    fs : float
    window_ms : float
        Window length in milliseconds.

    Returns
    -------
    np.ndarray, shape (n_samples,)
    """
    win_samples = max(1, int(round(window_ms * 1e-3 * fs)))
    # uniform_filter1d on x² then sqrt — equivalent to sliding RMS
    power = uniform_filter1d(signal ** 2, size=win_samples, mode="nearest")
    return np.sqrt(np.maximum(power, 0.0))


def _compute_log_mua(
    lfp: np.ndarray,
    fs: float,
    mua_band: list,
    smooth_ms: float,
) -> np.ndarray:
    """
    Compute logMUA - standard approach in slow-wave/up-down literature.
    
    Much cleaner and higher contrast than full-PSD Welch normalization.
    """
    if len(lfp) < fs * 1:  # at least 1 second
        raise ValueError("Signal too short")

    # 1. Bandpass filter in MUA range (4th order Butterworth, zero-phase)
    nyq = fs / 2
    low = mua_band[0] / nyq
    high = min(mua_band[1] / nyq, 0.99)  # avoid Nyquist issues
    
    b, a = butter(N=4, Wn=[low, high], btype='bandpass')
    filtered = filtfilt(b, a, lfp.astype(np.float64))  # zero-phase

    # 2. Instantaneous power
    power = filtered ** 2

    # 3. Log transform (with small offset)
    log_mua = np.log10(power + 1e-10)

    # 4. Smooth
    smooth_samples = max(5, int(round(smooth_ms * 1e-3 * fs)))
    log_mua = uniform_filter1d(log_mua, size=smooth_samples, mode='nearest')

    return log_mua

def _compute_raw_mua(
    lfp: np.ndarray,
    fs: float,
    mua_band: list,
    smooth_ms: float,
) -> np.ndarray:
    """
    Compute raw MUA (envelope of high-frequency power).
    
    This is the standard non-logged version used as a biological marker
    of local population spiking activity.
    """
    if len(lfp) < fs * 1:  # at least 1 second
        raise ValueError("Signal too short")

    # 1. Bandpass filter in MUA range (4th order Butterworth, zero-phase)
    nyq = fs / 2
    low = mua_band[0] / nyq
    high = min(mua_band[1] / nyq, 0.99)  # avoid Nyquist issues
    
    b, a = butter(N=4, Wn=[low, high], btype='bandpass')
    filtered = filtfilt(b, a, lfp.astype(np.float64))  # zero-phase

    # 2. Instantaneous power (rectified power)
    power = filtered ** 2

    # 3. Smooth (this is the core of raw MUA envelope)
    smooth_samples = max(5, int(round(smooth_ms * 1e-3 * fs)))
    raw_mua = uniform_filter1d(power, size=smooth_samples, mode='nearest')

    return raw_mua

def _extract_raw_log_mua(lfp: np.ndarray, fs: float, preproc_cfg: dict) -> np.ndarray:
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
# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _downsample(signal: np.ndarray, fs_orig: float, fs_target: float) -> np.ndarray:
    """Downsample signal from fs_orig to fs_target using polyphase resampling."""
    ratio = Fraction(int(fs_target), int(fs_orig)).limit_denominator(1000)
    up, down = ratio.numerator, ratio.denominator
    return resample_poly(signal, up, down)


def _zscore(signal: np.ndarray) -> np.ndarray:
    """
    Z-score a 1-D signal: subtract mean, divide by std.
    If std == 0 (constant signal), return zeros.
    """
    std = signal.std()
    if std == 0:
        return np.zeros_like(signal)
    return (signal - signal.mean()) / std


def _sliding_variance(signal: np.ndarray, fs: float, window_ms: float) -> np.ndarray:
    """
    Compute the variance of `signal` in a sliding rectangular window.

    Returns an array of the same length (edges handled by partial windows).

    Parameters
    ----------
    signal : np.ndarray
    fs : float
    window_ms : float
        Window length in milliseconds.
    """
    win_samples = max(1, int(round(window_ms * 1e-3 * fs)))
    n           = len(signal)
    variance    = np.empty(n)

    # Precompute cumulative sums for O(n) sliding variance
    # Var(x) = E[x²] - E[x]²
    cs  = np.cumsum(signal)
    cs2 = np.cumsum(signal ** 2)

    for i in range(n):
        lo = max(0, i - win_samples // 2)
        hi = min(n, i + win_samples // 2 + 1)
        w  = hi - lo
        s  = cs[hi - 1]  - (cs[lo - 1] if lo > 0 else 0)
        s2 = cs2[hi - 1] - (cs2[lo - 1] if lo > 0 else 0)
        variance[i] = s2 / w - (s / w) ** 2

    return variance


def _sliding_log_power(signal: np.ndarray, fs: float, window_ms: float) -> np.ndarray:
    """
    Compute the log-scaled power of `signal` in a sliding rectangular window.

    power = mean(signal²) in window
    log_power = log(power + epsilon)   [epsilon avoids log(0)]
    """
    win_samples = max(1, int(round(window_ms * 1e-3 * fs)))
    n           = len(signal)
    eps         = 1e-12   # small constant to prevent log(0)

    cs2 = np.cumsum(signal ** 2)

    log_power = np.empty(n)
    for i in range(n):
        lo = max(0, i - win_samples // 2)
        hi = min(n, i + win_samples // 2 + 1)
        w  = hi - lo
        s2 = cs2[hi - 1] - (cs2[lo - 1] if lo > 0 else 0)
        log_power[i] = np.log(s2 / w + eps)

    return log_power

def compute_firing_rates(log_mua_raw: np.ndarray, t_ds: float, epochs: dict, cfg: dict) -> tuple:
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


def compute_metrics(epochs: dict, up_frs: list, down_frs: list) -> None:
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


from scipy.signal import iirnotch, filtfilt

def notch_filter(
    signal: np.ndarray,
    fs: float,
    freq_hz: float = 50.0,
    quality: float = 30.0,
) -> np.ndarray:
    """
    Zero-phase IIR notch filter to remove mains noise.

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples,)
    fs : float
        Sampling rate (Hz).
    freq_hz : float
        Notch frequency to remove (50 Hz for Europe, 60 Hz for the US).
    quality : float
        Q factor. Higher = narrower notch. 30 is a standard value that
        removes 50 Hz cleanly without eating into nearby biological signals
        (which is why the gamma band starts safely above at, say, 30 Hz
        but we still want the notch to be surgical around 50 Hz).

    Returns
    -------
    np.ndarray, shape (n_samples,)
    """
    if freq_hz <= 0 or freq_hz >= fs / 2:
        return signal.copy()
    b, a = iirnotch(w0=freq_hz, Q=quality, fs=fs)
    return filtfilt(b, a, signal)

# ---------------------------------------------------------------------------
# Extension points
# ---------------------------------------------------------------------------
# - To add a new feature signal (e.g. spindle band envelope):
#     1. Compute it here and z-score it.
#     2. Add it as a new column in the np.column_stack() call.
#     3. Add corresponding config keys in config.yaml → preprocessing.
#     PCA in up_down.py will automatically use all columns.
#
# - The logMUA computation (_compute_log_mua) is the most expensive step.
#   For very long recordings you can speed it up by increasing mua_step_ms
#   in config.yaml (e.g. 10 ms instead of 5 ms). The interpolation handles
#   the sparser sampling transparently.
#
# - If your sampling rate is lower than ~4 kHz, the 200–1500 Hz MUA band
#   may not be fully captured. Adjust mua_band in config.yaml accordingly.

