"""Tests for the Combined Risk Parity weights-over-time report exhibit."""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from src.report_figures import (
    WEIGHT_CATEGORY_ORDER,
    combined_risk_parity_weight_summary,
    save_combined_risk_parity_weights_figure,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_combined_risk_parity_weights_aggregate_and_export(tmp_path) -> None:
    weights = pd.read_csv(PROJECT_ROOT / "results/data/fund_weights.csv")
    aggregated, summary = combined_risk_parity_weight_summary(weights)
    assert len(aggregated) == 36
    assert list(aggregated.columns) == WEIGHT_CATEGORY_ORDER
    assert np.allclose(aggregated.sum(axis=1), 1.0, atol=1e-8)
    assert summary["weights_sum_to_one"]
    assert summary["average_active_holdings"] > 0

    path, exported_summary = save_combined_risk_parity_weights_figure(
        weights, tmp_path / "weights.png"
    )
    assert exported_summary == summary
    with Image.open(path) as image:
        assert image.size == (3000, 1800)
