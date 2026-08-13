"""Tests for constrained combined-fund portfolio construction and timing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src import portfolios


def synthetic_returns(periods: int = 90) -> pd.DataFrame:
    """Create deterministic returns with distinct asset risk/return profiles."""
    dates = pd.bdate_range("2020-01-01", periods=periods)
    angle = np.arange(periods)
    return pd.DataFrame(
        {
            "date": dates,
            "AAA": 0.0004 + 0.004 * np.sin(angle / 4),
            "BBB": 0.0001 + 0.001 * np.cos(angle / 5),
            "BTC-USD": 0.0008 + 0.009 * np.sin(angle / 3 + 1),
        }
    )


def test_equal_weight_and_optimized_weights_satisfy_constraints() -> None:
    data = synthetic_returns(60).drop(columns="date")
    crypto_mask = np.array([False, False, True])
    for method in portfolios.METHOD_LABELS:
        weights, diagnostic = portfolios.optimize_weights(
            data,
            method,
            crypto_mask,
            pd.Timestamp("2020-04-01"),
            individual_cap=0.60,
            crypto_cap=0.40,
        )
        assert diagnostic["solver_success"]
        assert not portfolios.validate_weights(
            weights, crypto_mask, individual_cap=0.60, crypto_cap=0.40
        )
    equal, _ = portfolios.optimize_weights(
        data,
        "equal_weight",
        crypto_mask,
        pd.Timestamp("2020-04-01"),
        individual_cap=0.60,
        crypto_cap=0.40,
    )
    assert np.allclose(equal, 1 / 3)


def test_backtest_uses_prior_window_and_monthly_rebalances() -> None:
    result = portfolios.oos_backtest(
        synthetic_returns(),
        window=20,
        individual_cap=0.60,
        crypto_cap=0.40,
    )
    diagnostics = result.diagnostics
    assert (diagnostics["estimation_observations"] == 20).all()
    assert (diagnostics["estimation_end"] < diagnostics["rebalance_date"]).all()
    rebalance_dates = diagnostics["rebalance_date"].drop_duplicates()
    source = synthetic_returns()
    monthly_first = source.groupby(source["date"].dt.to_period("M")).head(1)
    expected = monthly_first.loc[monthly_first.index >= 20, "date"]
    assert rebalance_dates.tolist() == expected.tolist()
    assert result.daily["date"].min() == rebalance_dates.min()


def test_backtest_constraints_and_method_differences() -> None:
    result = portfolios.oos_backtest(
        synthetic_returns(),
        window=20,
        individual_cap=0.60,
        crypto_cap=0.40,
    )
    grouped = result.weights.groupby(["rebalance_date", "fund"])
    assert np.allclose(grouped["target_weight"].sum(), 1.0)
    assert result.weights["target_weight"].ge(-1e-8).all()
    assert result.weights["target_weight"].le(0.60 + 1e-7).all()
    crypto = result.weights.loc[result.weights["asset_class"] == "Crypto"]
    assert crypto.groupby(["rebalance_date", "fund"])["target_weight"].sum().le(
        0.40 + 1e-7
    ).all()
    assert portfolios.method_weight_distances(result.weights)[
        "meaningfully_different"
    ].all()


def test_weights_drift_between_rebalances() -> None:
    result = portfolios.oos_backtest(
        synthetic_returns(),
        methods=("equal_weight",),
        window=20,
        individual_cap=0.60,
        crypto_cap=0.40,
    )
    fund_daily = result.daily.sort_values("date")
    first_date = fund_daily.iloc[0]["date"]
    first_returns = synthetic_returns().set_index("date").loc[first_date]
    expected = first_returns.mean()
    assert np.isclose(fund_daily.iloc[0]["daily_return"], expected)
    second_date = fund_daily.iloc[1]["date"]
    second_returns = synthetic_returns().set_index("date").loc[second_date].to_numpy()
    drifted = (np.full(3, 1 / 3) * (1 + first_returns.to_numpy())) / (1 + expected)
    assert np.isclose(fund_daily.iloc[1]["daily_return"], drifted @ second_returns)


def test_optimizer_error_reports_date_method_and_reason() -> None:
    with pytest.raises(
        portfolios.PortfolioOptimizationError,
        match=r"2020-04-01 min_variance: estimation window contains",
    ):
        portfolios.optimize_weights(
            pd.DataFrame({"AAA": [0.01, np.nan], "BTC-USD": [0.02, 0.01]}),
            "min_variance",
            np.array([False, True]),
            pd.Timestamp("2020-04-01"),
            individual_cap=0.60,
            crypto_cap=0.50,
        )


def test_metrics_and_input_validation() -> None:
    metrics = portfolios.performance_metrics(pd.Series([0.10, -0.05, 0.02]), 3)
    growth = (1.10 * 0.95 * 1.02)
    assert np.isclose(metrics["final_growth_of_1"], growth)
    assert metrics["maximum_drawdown"] < 0

    bad = synthetic_returns().iloc[[1, 0]].copy()
    with pytest.raises(ValueError, match="strictly increasing"):
        portfolios.oos_backtest(
            bad, window=2, individual_cap=0.60, crypto_cap=0.40
        )


def test_crypto_risk_parity_can_differ_under_twenty_five_percent_cap() -> None:
    periods = 400
    dates = pd.date_range("2020-01-01", periods=periods, freq="D")
    angle = np.arange(periods)
    data = pd.DataFrame(
        {
            f"C{i}-USD": (
                0.0003
                + (0.002 + i * 0.001) * np.sin(angle / (3 + i) + i)
            )
            for i in range(10)
        }
    )
    crypto_mask = np.ones(10, dtype=bool)
    equal, _ = portfolios.optimize_weights(
        data,
        "equal_weight",
        crypto_mask,
        dates[-1],
        individual_cap=0.25,
        crypto_cap=None,
        periods_per_year=365,
    )
    risk_parity, _ = portfolios.optimize_weights(
        data,
        "risk_parity",
        crypto_mask,
        dates[-1],
        individual_cap=0.25,
        crypto_cap=None,
        periods_per_year=365,
    )
    assert not portfolios.validate_weights(
        risk_parity, crypto_mask, individual_cap=0.25, crypto_cap=None
    )
    assert 0.5 * np.abs(equal - risk_parity).sum() >= 0.01


def test_native_daily_calendar_and_365_day_annualisation() -> None:
    dates = pd.date_range("2020-01-01", periods=800, freq="D")
    angle = np.arange(len(dates))
    panel = pd.DataFrame(
        {
            "date": dates,
            "BTC-USD": 0.0004 + 0.01 * np.sin(angle / 8),
            "ETH-USD": 0.0003 + 0.008 * np.cos(angle / 7),
        }
    )
    result = portfolios.oos_backtest(
        panel,
        methods=("equal_weight",),
        window=365,
        individual_cap=0.60,
        crypto_cap=None,
        periods_per_year=365,
        method_labels={"equal_weight": "Crypto Equal Weight"},
    )
    first_live = result.daily["date"].min()
    assert first_live == pd.Timestamp("2021-01-01")
    assert result.daily["date"].dt.dayofweek.ge(5).any()
    assert result.diagnostics["estimation_observations"].eq(365).all()
    assert result.diagnostics["periods_per_year"].eq(365).all()
