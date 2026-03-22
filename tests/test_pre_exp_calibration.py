from __future__ import annotations

import pytest

from forecasting.evaluation import compute_ece


def test_compute_ece_matches_manual_binned_example():
    predictions = [0.05, 0.15, 0.15, 0.85, 0.95]
    labels = [0, 0, 1, 1, 1]

    ece, bins = compute_ece(predictions, labels, bins=10)

    assert pytest.approx(ece, rel=1e-9) == 0.19
    assert [row["bin_index"] for row in bins] == [0, 1, 8, 9]
    assert pytest.approx(bins[1]["avg_pred"], rel=1e-9) == 0.15
    assert pytest.approx(bins[1]["avg_label"], rel=1e-9) == 0.5
