"""Tests for Module 3 frequentist testing."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from module3_ab_engine.frequentist import (
    proportion_test,
    means_test,
    required_sample_size,
    ABGroups,
)


def test_proportion_test_detects_true_effect():
    rng = np.random.default_rng(0)
    ctrl = pd.Series(rng.binomial(1, 0.10, 5000).astype(float))
    trt = pd.Series(rng.binomial(1, 0.13, 5000).astype(float))  # 3pp lift
    groups = ABGroups(control=ctrl, treatment=trt, metric_name="ctr", is_binary=True)
    result = proportion_test(groups, alpha=0.05)
    assert result.significant, "Should detect a 3pp lift at n=5000"
    assert result.absolute_lift > 0


def test_proportion_test_no_false_positive():
    rng = np.random.default_rng(1)
    ctrl = pd.Series(rng.binomial(1, 0.10, 5000).astype(float))
    trt = pd.Series(rng.binomial(1, 0.10, 5000).astype(float))  # no lift
    groups = ABGroups(control=ctrl, treatment=trt, metric_name="ctr", is_binary=True)
    result = proportion_test(groups, alpha=0.05)
    # Not guaranteed, but p-value should generally be > 0.05 with equal rates
    assert result.p_value > 0.001, "Should not reject with equal rates (mostly)"


def test_means_test():
    rng = np.random.default_rng(2)
    ctrl = pd.Series(rng.normal(100, 20, 1000))
    trt = pd.Series(rng.normal(110, 20, 1000))  # 10-unit lift
    groups = ABGroups(control=ctrl, treatment=trt, metric_name="dwell", is_binary=False)
    result = means_test(groups)
    assert result.significant
    assert result.absolute_lift > 0


def test_sample_size_calculator():
    info = required_sample_size(baseline_rate=0.10, mde=0.02, alpha=0.05, power=0.80)
    assert info["n_per_arm"] > 0
    assert info["n_total"] == info["n_per_arm"] * 2
    # Standard result should be around 1800 per arm for these params
    assert 1000 < info["n_per_arm"] < 5000
