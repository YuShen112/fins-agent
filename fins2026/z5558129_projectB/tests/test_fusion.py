"""Tests for the baseline sector-sentiment tilt and capped normalization."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src import fusion


def base_weight_fixture() -> pd.DataFrame:
    date = pd.Timestamp("2021-02-01")
    return pd.DataFrame(
        {
            "rebalance_date": [date] * 4,
            "fund": ["Equity Equal Weight"] * 4,
            "ticker": ["A", "B", "C", "D"],
            "asset_class": ["Equity"] * 4,
            "target_weight": [0.25] * 4,
        }
    )


def sector_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trading_date": pd.to_datetime(["2021-02-01", "2021-02-01"]),
            "signal_source_date": pd.to_datetime(["2021-01-29", "2021-01-29"]),
            "sector": ["One", "Two"],
            "sector_sentiment_index": [0.2, -0.5],
            "covered_ticker_count": [2, 3],
            "covered_ticker_proportion": [0.4, 0.6],
        }
    )


def universe_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {"ticker": ["A", "B", "C", "D"], "sector": ["One", "One", "Two", "Two"]}
    )


def test_capped_normalization_redistributes_excess_proportionally() -> None:
    weights, bound = fusion.capped_proportional_weights(
        np.array([0.90, 0.05, 0.05]), 0.60
    )
    assert bound
    assert np.allclose(weights, [0.60, 0.20, 0.20])
    assert np.isclose(weights.sum(), 1.0)

    subtotal, subtotal_bound = fusion.capped_proportional_weights(
        np.array([0.90, 0.10]), 0.40, target_total=0.60
    )
    assert subtotal_bound
    assert np.allclose(subtotal, [0.40, 0.20])


def test_coverage_rule_blocks_only_low_coverage_sector() -> None:
    weights, sectors, rebalances = fusion.build_sector_tilted_targets(
        base_weight_fixture(),
        sector_fixture(),
        universe_fixture(),
        individual_cap=0.60,
    )
    sector_details = sectors.set_index("sector")
    assert bool(sector_details.loc["One", "coverage_rule_blocked"])
    assert sector_details.loc["One", "applied_multiplier"] == 1.0
    assert not bool(sector_details.loc["Two", "coverage_rule_blocked"])
    assert sector_details.loc["Two", "applied_multiplier"] == 0.5
    result = weights.set_index("ticker")["target_weight"]
    assert np.allclose(result.loc[["A", "B"]], 1 / 3)
    assert np.allclose(result.loc[["C", "D"]], 1 / 6)
    assert not bool(rebalances.loc[0, "cap_bound"])


def test_nonlagged_signal_is_rejected() -> None:
    signals = sector_fixture()
    signals["signal_source_date"] = signals["trading_date"]
    with pytest.raises(ValueError, match="sector signal is not lagged"):
        fusion.build_sector_tilted_targets(
            base_weight_fixture(),
            signals,
            universe_fixture(),
            individual_cap=0.60,
        )


def test_tilted_backtest_uses_target_on_rebalance_date() -> None:
    dates = pd.bdate_range("2021-02-01", periods=3)
    returns = pd.DataFrame(
        {
            "date": dates,
            "A": [0.10, 0.0, 0.0],
            "B": [0.0, 0.0, 0.0],
            "C": [0.0, 0.0, 0.0],
            "D": [0.0, 0.0, 0.0],
        }
    )
    weights, _, _ = fusion.build_sector_tilted_targets(
        base_weight_fixture(),
        sector_fixture(),
        universe_fixture(),
        individual_cap=0.60,
    )
    daily, metrics = fusion.backtest_tilted_targets(returns, weights)
    assert np.isclose(daily.loc[0, "daily_return"], (1 / 3) * 0.10)
    assert metrics.loc[0, "observations"] == 3


def test_combined_tilt_preserves_crypto_and_reallocates_only_equities() -> None:
    base = base_weight_fixture().copy()
    base["fund"] = "Combined Maximum Sharpe"
    base["target_weight"] = 0.15
    crypto = pd.DataFrame(
        {
            "rebalance_date": [pd.Timestamp("2021-02-01")] * 2,
            "fund": ["Combined Maximum Sharpe"] * 2,
            "ticker": ["BTC-USD", "ETH-USD"],
            "asset_class": ["Crypto", "Crypto"],
            "target_weight": [0.20, 0.20],
        }
    )
    base = pd.concat([base, crypto], ignore_index=True)
    weights, _, diagnostics = fusion.build_sector_tilted_targets(
        base,
        sector_fixture(),
        universe_fixture(),
        base_fund="Combined Maximum Sharpe",
        tilted_fund="Combined Maximum Sharpe + Sentiment",
        individual_cap=0.25,
        preserve_crypto=True,
    )
    result = weights.set_index("ticker")
    assert result.loc["BTC-USD", "target_weight"] == 0.20
    assert result.loc["ETH-USD", "target_weight"] == 0.20
    assert np.isclose(weights["target_weight"].sum(), 1.0)
    assert np.isclose(
        weights.loc[weights["asset_class"] == "Equity", "target_weight"].sum(),
        0.60,
    )
    assert diagnostics.loc[0, "crypto_preservation_error"] == 0.0
