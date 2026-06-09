"""
tests/test_trial_io.py
======================
Unit tests for src/trial_io.py

Run with:  pytest tests/test_trial_io.py -v
"""

import numpy as np
import pytest
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trial_io import (
    detect_artifacts,
    interpolate_artifacts,
    save_trial_h5,
    load_trial_metadata,
    TrialDefinition,
    ConditionWindow,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FS = 10_000   # 10 kHz

@pytest.fixture
def clean_signal():
    """Low-amplitude white noise with no spikes."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 1.0, FS * 10)   # 10 seconds


@pytest.fixture
def noisy_signal(clean_signal):
    """Clean signal with two synthetic large spikes inserted."""
    sig = clean_signal.copy()
    # Spike 1 at t=1.0 s
    sig[FS * 1 : FS * 1 + 50] += 1000.0
    # Spike 2 at t=7.0 s
    sig[FS * 7 : FS * 7 + 30] += -800.0
    return sig


@pytest.fixture
def artifact_cfg():
    return {
        "z_threshold":  8.0,
        "window_ms":  1000.0,
        "pad_ms":       50.0,
        "min_gap_ms":  100.0,
    }


# ---------------------------------------------------------------------------
# detect_artifacts
# ---------------------------------------------------------------------------

class TestDetectArtifacts:
    def test_clean_signal_no_artifacts(self, clean_signal, artifact_cfg):
        mask = detect_artifacts(clean_signal, FS, artifact_cfg)
        # Very few or no samples should be flagged in clean noise
        assert mask.mean() < 0.01

    def test_detects_large_spike(self, noisy_signal, artifact_cfg):
        mask = detect_artifacts(noisy_signal, FS, artifact_cfg)
        # Both spike regions should be flagged
        assert mask[FS * 1 : FS * 1 + 50].any(), "Spike 1 not detected"
        assert mask[FS * 7 : FS * 7 + 30].any(), "Spike 2 not detected"

    def test_output_same_length(self, noisy_signal, artifact_cfg):
        mask = detect_artifacts(noisy_signal, FS, artifact_cfg)
        assert len(mask) == len(noisy_signal)

    def test_output_dtype_bool(self, noisy_signal, artifact_cfg):
        mask = detect_artifacts(noisy_signal, FS, artifact_cfg)
        assert mask.dtype == bool

    def test_lower_threshold_more_aggressive(self, noisy_signal):
        """A lower z-threshold should flag more samples."""
        cfg_loose  = {"z_threshold": 3.0, "window_ms": 1000, "pad_ms": 0, "min_gap_ms": 0}
        cfg_strict = {"z_threshold": 20.0, "window_ms": 1000, "pad_ms": 0, "min_gap_ms": 0}
        mask_loose  = detect_artifacts(noisy_signal, FS, cfg_loose)
        mask_strict = detect_artifacts(noisy_signal, FS, cfg_strict)
        assert mask_loose.sum() >= mask_strict.sum()

    def test_padding_expands_mask(self, noisy_signal):
        """Padding should increase the flagged region around each spike."""
        cfg_no_pad  = {"z_threshold": 8.0, "window_ms": 1000, "pad_ms": 0,   "min_gap_ms": 0}
        cfg_with_pad = {"z_threshold": 8.0, "window_ms": 1000, "pad_ms": 50.0, "min_gap_ms": 0}
        mask_no_pad   = detect_artifacts(noisy_signal, FS, cfg_no_pad)
        mask_with_pad = detect_artifacts(noisy_signal, FS, cfg_with_pad)
        assert mask_with_pad.sum() >= mask_no_pad.sum()


# ---------------------------------------------------------------------------
# interpolate_artifacts
# ---------------------------------------------------------------------------

class TestInterpolateArtifacts:
    def test_no_mask_unchanged(self, clean_signal):
        mask = np.zeros(len(clean_signal), dtype=bool)
        out  = interpolate_artifacts(clean_signal, mask)
        np.testing.assert_array_equal(out, clean_signal)

    def test_artifact_region_replaced(self, noisy_signal, artifact_cfg):
        mask = detect_artifacts(noisy_signal, FS, artifact_cfg)
        out  = interpolate_artifacts(noisy_signal, mask)
        # The spike at t=1 s should now have reasonable values
        spike_region = out[FS * 1 : FS * 1 + 50]
        assert np.abs(spike_region).max() < 100.0, \
            "Spike region not sufficiently attenuated after interpolation"

    def test_clean_regions_unchanged(self, noisy_signal, artifact_cfg):
        """Samples outside artifact mask must not be modified."""
        mask = detect_artifacts(noisy_signal, FS, artifact_cfg)
        out  = interpolate_artifacts(noisy_signal, mask)
        clean_idx = np.where(~mask)[0]
        np.testing.assert_array_almost_equal(
            out[clean_idx], noisy_signal[clean_idx], decimal=10
        )

    def test_all_flagged_returns_zeros(self):
        """Edge case: entire signal flagged → return zeros."""
        sig  = np.ones(100) * 500.0
        mask = np.ones(100, dtype=bool)
        out  = interpolate_artifacts(sig, mask)
        np.testing.assert_array_equal(out, np.zeros(100))

    def test_output_same_length(self, noisy_signal, artifact_cfg):
        mask = detect_artifacts(noisy_signal, FS, artifact_cfg)
        out  = interpolate_artifacts(noisy_signal, mask)
        assert len(out) == len(noisy_signal)


# ---------------------------------------------------------------------------
# save_trial_h5 + load_trial_metadata
# ---------------------------------------------------------------------------

class TestSaveLoadTrialH5:
    def test_roundtrip(self, tmp_path):
        """Data written by save_trial_h5 should be readable by load_h5."""
        from src.data_io import load_h5

        rng  = np.random.default_rng(7)
        data = rng.standard_normal((5000, 4)).astype(np.float32)
        meta = {
            "fs": 1000.0, "units": "mV",
            "duration_seconds": 5.0, "original_file": "test.smr"
        }
        extra = {
            "trial_label": "trial_01",
            "trial_window_start_s": 10.0,
            "trial_window_end_s":   15.0,
            "artifact_fraction":    0.02,
            "source_file":          "test.h5",
        }

        out_path = str(tmp_path / "trial_01_baseline.h5")
        save_trial_h5(out_path, data, meta, extra)

        loaded_data, loaded_meta = load_h5(out_path)
        assert loaded_data.shape == data.shape
        assert abs(loaded_meta["fs"] - 1000.0) < 1e-6

    def test_trial_metadata_roundtrip(self, tmp_path):
        rng  = np.random.default_rng(8)
        data = rng.standard_normal((1000, 2)).astype(np.float32)
        meta = {"fs": 500.0, "units": "uV", "duration_seconds": 2.0, "original_file": "x.smr"}
        extra = {
            "trial_label": "trial_02",
            "trial_window_start_s": 20.0,
            "trial_window_end_s":   22.0,
            "artifact_fraction":    0.005,
            "source_file":          "x.h5",
        }

        out_path = str(tmp_path / "x_trial_02_stimulation.h5")
        save_trial_h5(out_path, data, meta, extra)

        trial_meta = load_trial_metadata(out_path)
        assert trial_meta["trial_label"] == "trial_02"
        assert abs(trial_meta["trial_window_start_s"] - 20.0) < 1e-6
        assert abs(trial_meta["artifact_fraction"] - 0.005) < 1e-6


# ---------------------------------------------------------------------------
# TrialDefinition + ConditionWindow
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_duration(self):
        cw = ConditionWindow(name="baseline", start_s=10.0, end_s=50.0)
        assert cw.duration_s == pytest.approx(40.0)

    def test_trial_duration(self):
        t = TrialDefinition(label="t1", start_s=0.0, end_s=300.0)
        assert t.duration_s == pytest.approx(300.0)

    def test_trial_default_empty_conditions(self):
        t = TrialDefinition(label="t1", start_s=0.0, end_s=100.0)
        assert t.conditions == []
