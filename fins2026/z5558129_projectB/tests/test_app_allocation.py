"""Tests for display-only adjusted allocation analytics."""

from pathlib import Path

import numpy as np
import pandas as pd
from src.app_allocation import (
    allocation_total_is_valid,
    compare_allocations,
    fee_adjusted_growth,
    lookthrough_crypto,
)
from src.app_data import load_artifacts, recommendation_rows

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = load_artifacts(PROJECT_ROOT / "results")


def _weights(profile: str, preference: str) -> tuple[pd.DataFrame, pd.Series]:
    rows = recommendation_rows(ARTIFACTS.recommendations, profile, preference)
    rows = rows.loc[rows["scenario_key"] != "higher_25_alternative"]
    return rows, rows.set_index("fund")["recommended_weight"]


def test_original_and_adjusted_use_identical_calendar_and_start():
    _, original = _weights("Balanced", "higher")
    adjusted = original.copy()
    adjusted.iloc[0] -= 0.05
    adjusted.iloc[1] += 0.05
    result = compare_allocations(ARTIFACTS.returns, original, adjusted)
    assert list(result.returns) == ["Original recommendation", "User-adjusted"]
    assert result.returns.notna().all().all()
    assert result.growth.index.equals(result.returns.index)
    assert result.start_date == result.returns.index.min()
    assert result.annualization == 365


def test_equity_only_allocation_uses_252_day_annualization():
    _, original = _weights("Balanced", "none")
    result = compare_allocations(ARTIFACTS.returns, original, original)
    assert result.annualization == 252
    np.testing.assert_allclose(
        result.metrics.iloc[0, 1:].astype(float),
        result.metrics.iloc[1, 1:].astype(float),
    )


def test_lookthrough_crypto_uses_precomputed_fund_exposures():
    rows, weights = _weights("Growth", "higher")
    exposures = rows.set_index("fund")["crypto_exposure"]
    expected = rows["crypto_contribution"].sum()
    assert np.isclose(lookthrough_crypto(weights, exposures), expected)


def test_display_rounding_tolerance_accepts_a_100_percent_total():
    displayed_as_full = pd.Series([0.38615, 0.35102, 0.08742, 0.17540])
    assert displayed_as_full.sum() != 1.0
    assert np.isclose(displayed_as_full.sum(), 0.99999)
    assert allocation_total_is_valid(displayed_as_full)
    outside_tolerance = pd.Series([0.50, 0.30, 0.1998])
    assert not allocation_total_is_valid(outside_tolerance)


def test_zero_fee_produces_identical_gross_and_net_growth():
    returns = pd.Series([0.01, -0.02, 0.03])
    growth = fee_adjusted_growth(returns, annual_fee=0.0, annualization=252)
    np.testing.assert_allclose(growth.iloc[:, 0], growth.iloc[:, 1])


def test_positive_fee_reduces_adjusted_portfolio_final_value():
    returns = pd.Series([0.001] * 252)
    growth = fee_adjusted_growth(returns, annual_fee=0.005, annualization=252)
    assert growth.iloc[-1, 1] < growth.iloc[-1, 0]


def test_fee_uses_requested_allocation_calendar_consistently():
    zero_252 = pd.Series(np.zeros(252))
    zero_365 = pd.Series(np.zeros(365))
    net_252 = fee_adjusted_growth(zero_252, annual_fee=0.01, annualization=252).iloc[-1, 1]
    net_365 = fee_adjusted_growth(zero_365, annual_fee=0.01, annualization=365).iloc[-1, 1]
    assert np.isclose(net_252, 0.99)
    assert np.isclose(net_365, 0.99)
    assert np.isclose(net_252, net_365)
