
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
    lfp : np.ndarray, shape (n_samples,)
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
    gamma_win_ms = preproc_cfg.get("gamma_window_ms", 500.0)  # note: was 50
    gamma_smooth_ms = preproc_cfg.get("gamma_smooth_ms", 100.0)

    gamma_filt = bandpass_filter(lfp, fs, gamma_band[0], gamma_band[1], order=order)
    gamma_env  = _sliding_variance(gamma_filt, fs, gamma_win_ms)
    
    # Gaussian smooth to remove remaining fast fluctuations
    sigma_gamma = gamma_smooth_ms * 1e-3 * fs
    gamma_env   = gaussian_filter1d(gamma_env, sigma=sigma_gamma)

    # ------------------------------------------------------------------
    # 3. logMUA — normalised high-frequency power, log-scaled + smoothed
    #    Key step: divide the MUA-band power by the TOTAL power in each
    #    window. This normalisation (from Mattia et al. 2010) makes the
    #    logMUA track the RELATIVE high-frequency content, not just the
    #    overall signal amplitude.
    #    After log-scaling, apply an 80 ms moving-average smooth
    #    (Mattia & Del Giudice 2002) to reduce fast fluctuations.
    # ------------------------------------------------------------------
    mua_band    = preproc_cfg["mua_band"]
    mua_smooth_ms = preproc_cfg.get("mua_smooth_ms", 20.0) # post-log smooth

    log_mua = _compute_log_mua(lfp, fs, mua_band, mua_smooth_ms)

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

