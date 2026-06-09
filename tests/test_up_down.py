"""
tests/test_up_down.py
=====================
Unit tests for src/core/up_down.py

Run with:  pytest tests/test_up_down.py -v
"""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.up_down import (
    detect_states,
    _run_pca,
    _compute_threshold,
    _enforce_min_duration,
    _extract_epochs,
    _get_runs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_cfg(method="kde", threshold=0.0,
                         min_up=0.05, min_down=0.05):
    return {
        "up_down": {
            "threshold_method":    method,
            "manual_threshold":    threshold,
            "min_up_duration_s":   min_up,
            "min_down_duration_s": min_down,
            "kde_bandwidth_factor": 1.0,
        },
        "preprocessing": {
            "downsample_to_hz": 1000,
        },
    }


def _make_bimodal_features(n_samples=10_000, fs=1000.0, seed=42):
    """
    Create a synthetic feature matrix where PC1 should be bimodal.
    Up states: high gamma + logMUA. Down states: low.
    """
    rng = np.random.default_rng(seed)

    # 50% Up, 50% Down — alternating blocks of 500 samples
    states_true = np.zeros(n_samples, dtype=int)
    block       = 500
    for i in range(0, n_samples, block * 2):
        states_true[i : i + block] = 1   # Up

    # Features: Up → higher values on all 3 signals
    f = np.zeros((n_samples, 3))
    f[states_true == 1] = rng.normal(loc=1.5, scale=0.3, size=(states_true.sum(), 3))
    f[states_true == 0] = rng.normal(loc=-1.5, scale=0.3, size=((1 - states_true).sum(), 3))

    t_ds = np.arange(n_samples) / fs
    return f, t_ds, states_true


# ---------------------------------------------------------------------------
# _get_runs
# ---------------------------------------------------------------------------

class TestGetRuns:
    def test_single_run(self):
        arr  = np.ones(5, dtype=int)
        runs = _get_runs(arr)
        assert runs == [(1, 0, 5)]

    def test_two_runs(self):
        arr  = np.array([0, 0, 1, 1, 1])
        runs = _get_runs(arr)
        assert runs == [(0, 0, 2), (1, 2, 5)]

    def test_alternating(self):
        arr  = np.array([0, 1, 0, 1])
        runs = _get_runs(arr)
        assert len(runs) == 4
        assert all(end - start == 1 for _, start, end in runs)


# ---------------------------------------------------------------------------
# _enforce_min_duration
# ---------------------------------------------------------------------------

class TestEnforceMinDuration:
    def test_short_up_state_removed(self):
        """A 3-sample Up state with min_up=10 samples should be removed."""
        fs = 1000.0
        # Down(100) - Up(3) - Down(100)
        states = np.array([0] * 100 + [1] * 3 + [0] * 100)
        cfg    = _make_synthetic_cfg(min_up=0.01)  # 10 ms = 10 samples at 1 kHz
        out    = _enforce_min_duration(states, fs, cfg["up_down"])
        # Short Up state should be merged → all Down
        assert out.sum() == 0 or out[100:103].sum() == 0

    def test_long_states_preserved(self):
        """States longer than min duration must not be changed."""
        fs  = 1000.0
        cfg = _make_synthetic_cfg(min_up=0.05, min_down=0.05)
        states = np.array([0] * 500 + [1] * 500)
        out    = _enforce_min_duration(states, fs, cfg["up_down"])
        # Both halves still present
        assert out[:500].sum()  == 0
        assert out[500:].sum()  == 500


# ---------------------------------------------------------------------------
# _extract_epochs
# ---------------------------------------------------------------------------

class TestExtractEpochs:
    def test_basic(self):
        states = np.array([0, 0, 1, 1, 1, 0, 0])
        t_ds   = np.arange(7) / 1000.0
        epochs = _extract_epochs(states, t_ds)

        assert len(epochs["up"])   == 1
        assert len(epochs["down"]) == 2
        # Check that durations are positive
        for state_key in ("up", "down"):
            for start, end, dur in epochs[state_key]:
                assert dur > 0
                assert end > start

    def test_empty(self):
        states = np.zeros(100, dtype=int)
        t_ds   = np.arange(100) / 1000.0
        epochs = _extract_epochs(states, t_ds)
        assert len(epochs["up"])   == 0
        assert len(epochs["down"]) == 1


# ---------------------------------------------------------------------------
# detect_states (integration)
# ---------------------------------------------------------------------------

class TestDetectStates:
    def test_returns_correct_types(self):
        features, t_ds, _ = _make_bimodal_features()
        cfg = _make_synthetic_cfg(method="kde")
        states, epochs, pc1, threshold = detect_states(features, t_ds, cfg)

        assert isinstance(states, np.ndarray)
        assert isinstance(epochs, dict)
        assert isinstance(pc1, np.ndarray)
        assert isinstance(threshold, float)

    def test_state_values_binary(self):
        features, t_ds, _ = _make_bimodal_features()
        cfg = _make_synthetic_cfg(method="kde")
        states, _, _, _ = detect_states(features, t_ds, cfg)
        assert set(np.unique(states)).issubset({0, 1})

    def test_state_length_matches_input(self):
        features, t_ds, _ = _make_bimodal_features(n_samples=5000)
        cfg = _make_synthetic_cfg(method="kde")
        states, _, pc1, _ = detect_states(features, t_ds, cfg)
        assert len(states) == len(t_ds)
        assert len(pc1)    == len(t_ds)

    def test_detects_up_and_down(self):
        """With synthetic bimodal data both state types should be detected."""
        features, t_ds, _ = _make_bimodal_features()
        cfg = _make_synthetic_cfg(method="kde")
        _, epochs, _, _ = detect_states(features, t_ds, cfg)
        assert len(epochs["up"])   > 0
        assert len(epochs["down"]) > 0

    def test_manual_threshold(self):
        """Manual threshold above all PC1 → no Up states."""
        features, t_ds, _ = _make_bimodal_features()
        cfg = _make_synthetic_cfg(method="manual", threshold=999.0)
        states, epochs, _, _ = detect_states(features, t_ds, cfg)
        assert len(epochs["up"]) == 0

    def test_accuracy_with_clean_bimodal(self):
        """
        With clean bimodal synthetic data the detected states should
        agree with the ground truth at least 80% of the time.
        """
        features, t_ds, states_true = _make_bimodal_features(seed=0)
        cfg = _make_synthetic_cfg(method="kde")
        states_detected, _, _, _ = detect_states(features, t_ds, cfg)

        accuracy = (states_detected == states_true).mean()
        assert accuracy > 0.80, f"Detection accuracy too low: {accuracy:.2%}"
