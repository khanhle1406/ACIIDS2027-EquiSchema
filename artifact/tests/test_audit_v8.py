"""Statistical regression tests independent of fitted models."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_v8 import esd, jaccard


def test_identity_has_exact_zero_excess_and_interval() -> None:
    predictions = np.array([[0.1, 0.9], [0.8, 0.2], [0.6, 0.7]])
    result = esd(predictions, predictions, nboot=50)
    assert result["delta_seed_ind"] > 0
    assert result["esd_ind"] == result["esd_ci_lo"] == result["esd_ci_hi"] == 0


def test_regression_translation_is_excess_drift() -> None:
    a = np.zeros((5, 6))
    result = esd(a, a + 4, classification=False, scale=2, nboot=50)
    assert result["esd_ind"] == result["esd_ci_lo"] == 2


def test_cluster_alignment_and_invalid_inputs() -> None:
    a = np.ones((5, 6))
    with pytest.raises(ValueError):
        esd(a, a, clusters=np.arange(5))
    with pytest.raises(ValueError):
        esd(a, a, scale=0)
    assert esd(a, a, clusters=np.array([1, 1, 2, 2, 3, 3]), nboot=20)["esd_ind"] == 0


def test_logical_sets_are_not_fraction_of_full_context() -> None:
    assert jaccard({("row", "a")}, {("row", "a"), ("row", "b")}) == 0.5
    assert jaccard(set(), set()) == 1
