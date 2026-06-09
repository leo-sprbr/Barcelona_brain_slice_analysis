"""
tests/test_metrics.py
=====================
Unit tests for src/core/metrics.py

Run with:  pytest tests/test_metrics.py -v
"""

import numpy as np
import pandas as pd
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.metrics import (
    compute_state_durations,
    exclude_outliers,
    compute_delta_duration,
    paired_wilcoxon_test,
    mannwhitney_test,
    build_summary_table,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_epochs():
    """Simple epochs dict with 3 Up states and 3 Down states."""
    up   = [(0.0, 0.5, 0.5), (1.0, 1.4, 0.4), (2.0, 2.6, 0.6)]
    down = [(0.5, 1.0, 0.5), (1.4, 2.0, 0.6), (2.6, 3.0, 0.4)]
    return {"up": up, "down": down}


@pytest.fixture
def minimal_stat_cfg():
    return {
        "statistics": {
            "alpha": 0.05,
            "correction": "bonferroni",
            "significance_labels": [
                [0.05, "*"], [0.01, "**"], [0.001, "***"], [0.0001, "****"]
            ],
        }
    }


# ---------------------------------------------------------------------------
# compute_state_durations
# ---------------------------------------------------------------------------

class TestComputeStateDurations:
    def test_mean_durations(self, simple_epochs):
        result = compute_state_durations(simple_epochs)
        np.testing.assert_allclose(result["mean_up_s"],   0.5, atol=0.01)
        np.testing.assert_allclose(result["mean_down_s"], 0.5, atol=0.01)

    def test_n_counts(self, simple_epochs):
        result = compute_state_durations(simple_epochs)
        assert result["n_up"]   == 3
        assert result["n_down"] == 3

    def test_so_frequency_positive(self, simple_epochs):
        result = compute_state_durations(simple_epochs)
        assert result["so_frequency"] > 0

    def test_so_frequency_value(self, simple_epochs):
        """SO freq = 1 / mean(Up + Down cycle). Cycles are 0.5+0.5=1.0, etc."""
        result = compute_state_durations(simple_epochs)
        # Mean cycle duration ≈ 1.0 s → SO freq ≈ 1.0 Hz
        np.testing.assert_allclose(result["so_frequency"], 1.0, atol=0.1)

    def test_empty_epochs(self):
        result = compute_state_durations({"up": [], "down": []})
        assert np.isnan(result["mean_up_s"])
        assert result["n_up"] == 0

    def test_sem(self, simple_epochs):
        result = compute_state_durations(simple_epochs)
        assert result["sem_up_s"] >= 0
        assert result["sem_down_s"] >= 0


# ---------------------------------------------------------------------------
# exclude_outliers
# ---------------------------------------------------------------------------

class TestExcludeOutliers:
    def test_no_outliers_passes_all(self):
        vals = [0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6]
        filtered, mask = exclude_outliers(vals, ci_soft=0.95, ci_hard=0.999)
        assert all(mask)

    def test_extreme_outlier_excluded(self):
        vals = [0.5] * 20 + [100.0]   # last is extreme outlier
        filtered, mask = exclude_outliers(vals, ci_soft=0.95, ci_hard=0.999)
        assert not mask[-1]            # outlier excluded
        assert all(mask[:-1])          # others kept

    def test_too_few_returns_all(self):
        vals = [0.5, 0.6]
        filtered, mask = exclude_outliers(vals)
        assert all(mask)

    def test_filtered_length_matches_mask(self):
        vals = [0.4, 0.5, 0.6, 0.7, 100.0]
        filtered, mask = exclude_outliers(vals)
        assert len(filtered) == sum(mask)


# ---------------------------------------------------------------------------
# compute_delta_duration
# ---------------------------------------------------------------------------

class TestComputeDeltaDuration:
    def test_basic(self):
        bl   = [0.4, 0.5, 0.6]
        comp = [0.5, 0.6, 0.7]
        delta = compute_delta_duration(bl, comp)
        np.testing.assert_allclose(delta, [0.1, 0.1, 0.1], atol=1e-10)

    def test_negative_delta(self):
        bl   = [0.6, 0.7]
        comp = [0.4, 0.5]
        delta = compute_delta_duration(bl, comp)
        assert all(d < 0 for d in delta)

    def test_length_mismatch_uses_shorter(self):
        bl   = [0.5, 0.5, 0.5]
        comp = [0.6, 0.6]
        delta = compute_delta_duration(bl, comp)
        assert len(delta) == 2

    def test_empty_returns_empty(self):
        assert compute_delta_duration([], []) == []


# ---------------------------------------------------------------------------
# paired_wilcoxon_test
# ---------------------------------------------------------------------------

class TestPairedWilcoxon:
    def test_significant_difference(self, minimal_stat_cfg):
        rng = np.random.default_rng(42)
        bl   = rng.normal(0.4, 0.05, 20).tolist()
        comp = (np.array(bl) + 0.2 + rng.normal(0, 0.01, 20)).tolist()
        result = paired_wilcoxon_test(bl, comp, minimal_stat_cfg)
        assert result["p_value"] < 0.05
        assert result["test"] == "wilcoxon"

    def test_no_difference(self, minimal_stat_cfg):
        rng = np.random.default_rng(0)
        bl   = rng.normal(0.5, 0.05, 20).tolist()
        comp = rng.normal(0.5, 0.05, 20).tolist()
        result = paired_wilcoxon_test(bl, comp, minimal_stat_cfg)
        assert result["p_value"] > 0.01   # not strongly significant

    def test_too_few_pairs(self, minimal_stat_cfg):
        result = paired_wilcoxon_test([0.5, 0.6], [0.6, 0.7], minimal_stat_cfg)
        assert np.isnan(result["p_value"])
        assert "note" in result


# ---------------------------------------------------------------------------
# mannwhitney_test
# ---------------------------------------------------------------------------

class TestMannWhitney:
    def test_significant(self, minimal_stat_cfg):
        rng = np.random.default_rng(1)
        a = rng.normal(0.4, 0.05, 20).tolist()
        b = rng.normal(0.7, 0.05, 20).tolist()
        result = mannwhitney_test(a, b, minimal_stat_cfg)
        assert result["p_value"] < 0.05

    def test_n_counts(self, minimal_stat_cfg):
        a = [0.5] * 10
        b = [0.6] * 12
        result = mannwhitney_test(a, b, minimal_stat_cfg)
        assert result["n_a"] == 10
        assert result["n_b"] == 12


# ---------------------------------------------------------------------------
# build_summary_table
# ---------------------------------------------------------------------------

class TestBuildSummaryTable:
    def test_returns_dataframe(self, simple_epochs):
        stats = compute_state_durations(simple_epochs)
        df = build_summary_table("slice_01", "baseline", stats, [1.0, 2.0], [0.5, 0.6])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_columns_present(self, simple_epochs):
        stats = compute_state_durations(simple_epochs)
        df = build_summary_table("slice_01", "baseline", stats, [], [])
        expected = ["slice_id", "condition", "n_up", "n_down",
                    "mean_up_s", "mean_down_s", "so_frequency"]
        for col in expected:
            assert col in df.columns, f"Column '{col}' missing from summary table"
