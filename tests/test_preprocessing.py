"""
tests/test_preprocessing.py
============================
Unit tests for src/preprocessing.py

Run with:  pytest tests/test_preprocessing.py -v
"""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import (
    select_channel,
    bandpass_filter,
    extract_features,
    _zscore,
    _sliding_variance,
    _sliding_log_power,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FS = 10_000          # 10 kHz — typical for brain slice recordings
DURATION_S = 10
N_SAMPLES  = FS * DURATION_S
N_CHANNELS = 32


@pytest.fixture
def white_noise_data():
    """White noise LFP data: (n_samples, n_channels)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((N_SAMPLES, N_CHANNELS))


@pytest.fixture
def minimal_cfg():
    """Minimal preprocessing config matching config.yaml structure."""
    return {
        "channel": None,
        "so_band": [0.1, 1.5],
        "gamma_band": [30, 80],
        "gamma_window_ms": 50,
        "mua_band": [200, 1500],
        "mua_window_ms": 5,
        "filter_order": 4,
        "downsample_to_hz": 1000,
    }


# ---------------------------------------------------------------------------
# select_channel
# ---------------------------------------------------------------------------

class TestSelectChannel:
    def test_average_all(self, white_noise_data):
        cfg = {"channel": None}
        lfp = select_channel(white_noise_data, cfg)
        assert lfp.shape == (N_SAMPLES,)
        # Average of all channels should be close to mean of input
        np.testing.assert_allclose(
            lfp, white_noise_data.mean(axis=1), rtol=1e-10
        )

    def test_single_channel(self, white_noise_data):
        cfg = {"channel": 5}
        lfp = select_channel(white_noise_data, cfg)
        assert lfp.shape == (N_SAMPLES,)
        np.testing.assert_array_equal(lfp, white_noise_data[:, 5])

    def test_channel_list(self, white_noise_data):
        cfg = {"channel": [0, 1, 2]}
        lfp = select_channel(white_noise_data, cfg)
        assert lfp.shape == (N_SAMPLES,)
        np.testing.assert_allclose(
            lfp, white_noise_data[:, [0, 1, 2]].mean(axis=1)
        )

    def test_invalid_channel(self, white_noise_data):
        cfg = {"channel": "bad"}
        with pytest.raises(ValueError):
            select_channel(white_noise_data, cfg)


# ---------------------------------------------------------------------------
# bandpass_filter
# ---------------------------------------------------------------------------

class TestBandpassFilter:
    def test_output_shape(self):
        sig = np.random.randn(N_SAMPLES)
        out = bandpass_filter(sig, FS, 30, 80)
        assert out.shape == sig.shape

    def test_attenuates_out_of_band(self):
        """A pure 1 Hz tone should be largely attenuated by a 30-80 Hz filter."""
        t   = np.arange(N_SAMPLES) / FS
        sig = np.sin(2 * np.pi * 1.0 * t)      # 1 Hz tone
        out = bandpass_filter(sig, FS, 30, 80)
        # After filtering, power should be << original
        assert out.std() < 0.01 * sig.std()

    def test_passes_in_band(self):
        """A 50 Hz tone should pass through a 30-80 Hz filter mostly intact."""
        t   = np.arange(N_SAMPLES) / FS
        sig = np.sin(2 * np.pi * 50.0 * t)
        out = bandpass_filter(sig, FS, 30, 80)
        assert out.std() > 0.5 * sig.std()

    def test_high_cutoff_above_nyquist_raises(self):
        sig = np.random.randn(100)
        with pytest.raises(ValueError):
            bandpass_filter(sig, fs=100, low_hz=10, high_hz=60)  # 60 >= 50 Hz Nyquist


# ---------------------------------------------------------------------------
# _zscore
# ---------------------------------------------------------------------------

class TestZscore:
    def test_mean_zero(self):
        sig = np.random.randn(1000) * 5 + 3
        z   = _zscore(sig)
        assert abs(z.mean()) < 1e-10

    def test_std_one(self):
        sig = np.random.randn(1000) * 5 + 3
        z   = _zscore(sig)
        assert abs(z.std() - 1.0) < 1e-10

    def test_constant_signal(self):
        sig = np.ones(100)
        z   = _zscore(sig)
        np.testing.assert_array_equal(z, np.zeros(100))


# ---------------------------------------------------------------------------
# extract_features
# ---------------------------------------------------------------------------

class TestExtractFeatures:
    def test_output_shape(self, minimal_cfg):
        rng = np.random.default_rng(0)
        lfp = rng.standard_normal(N_SAMPLES)
        features, t_ds = extract_features(lfp, FS, minimal_cfg)

        expected_n = int(N_SAMPLES * minimal_cfg["downsample_to_hz"] / FS)
        assert features.shape[1] == 3
        assert abs(features.shape[0] - expected_n) <= 2   # allow rounding
        assert len(t_ds) == features.shape[0]

    def test_features_zscored(self, minimal_cfg):
        """Each feature column should have mean ≈ 0 and std ≈ 1."""
        rng = np.random.default_rng(1)
        lfp = rng.standard_normal(N_SAMPLES)
        features, _ = extract_features(lfp, FS, minimal_cfg)

        for col in range(3):
            assert abs(features[:, col].mean()) < 0.1
            assert abs(features[:, col].std() - 1.0) < 0.1

    def test_no_downsample(self, minimal_cfg):
        """With downsample_to_hz=None the output length should match input."""
        cfg = minimal_cfg.copy()
        cfg["downsample_to_hz"] = None
        rng = np.random.default_rng(2)
        lfp = rng.standard_normal(N_SAMPLES)
        features, t_ds = extract_features(lfp, FS, cfg)
        assert features.shape[0] == N_SAMPLES
