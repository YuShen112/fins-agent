"""Tests for transparent fact-sheet risk, recovery, and diversification metrics."""

import numpy as np
import pandas as pd
from src.app_fund_analysis import (
    downside_risk,
    fund_ranks,
    latest_diversification,
    recovery_diagnostics,
)


def test_downside_risk_uses_zero_mar_and_requested_annualization() -> None:
    returns = pd.Series([0.02, -0.01, -0.03, 0.01])
    downside, sortino = downside_risk(returns, 0.12, 252, mar=0.0)
    expected = np.sqrt((0.01**2 + 0.03**2) / 4) * np.sqrt(252)
    assert np.isclose(downside, expected)
    assert np.isclose(sortino, 0.12 / expected)


def test_recovery_metrics_are_calendar_days() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2021-01-01", "2021-01-04", "2021-01-10", "2021-01-20", "2021-01-25"]
            ),
            "growth_of_1": [1.00, 1.10, 0.80, 1.10, 1.20],
        }
    )
    result = recovery_diagnostics(frame)
    assert result.longest_drawdown_days == 16
    assert result.maximum_drawdown_recovery_days == 10
    assert result.maximum_drawdown_recovered


def test_unrecovered_maximum_drawdown_reports_elapsed_calendar_days() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-01", "2021-01-05", "2021-01-20"]),
            "growth_of_1": [1.00, 0.70, 0.80],
        }
    )
    result = recovery_diagnostics(frame)
    assert result.longest_drawdown_days == 19
    assert result.maximum_drawdown_recovery_days is None
    assert not result.maximum_drawdown_recovered
    assert result.maximum_drawdown_elapsed_days == 15


def test_latest_diversification_uses_only_latest_rebalance() -> None:
    weights = pd.DataFrame(
        {
            "rebalance_date": pd.to_datetime(
                ["2022-01-01", "2022-01-01", "2022-02-01", "2022-02-01"]
            ),
            "target_weight": [0.5, 0.5, 0.75, 0.25],
        }
    )
    effective, top_five, as_of = latest_diversification(weights)
    assert np.isclose(effective, 1 / (0.75**2 + 0.25**2))
    assert np.isclose(top_five, 1.0)
    assert as_of == pd.Timestamp("2022-02-01")


def test_fund_ranks_apply_investor_friendly_direction() -> None:
    metrics = pd.DataFrame(
        {
            "fund": ["A", "B"],
            "annualized_return": [0.10, 0.20],
            "annualized_volatility": [0.10, 0.30],
            "maximum_drawdown": [-0.10, -0.40],
            "sharpe_ratio": [1.0, 0.7],
        }
    )
    ranks = fund_ranks(metrics, "A").set_index("Measure")["Rank among 8 funds"]
    assert ranks.to_dict() == {
        "Annual return": 2,
        "Volatility": 1,
        "Maximum drawdown": 1,
        "Sharpe ratio": 1,
    }
