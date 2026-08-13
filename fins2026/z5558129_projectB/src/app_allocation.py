"""Pure historical analytics for user-adjusted fund allocations."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AllocationComparison:
    returns: pd.DataFrame
    growth: pd.DataFrame
    metrics: pd.DataFrame
    annualization: int
    start_date: pd.Timestamp


def allocation_total_is_valid(weights: pd.Series) -> bool:
    """Accept totals within 0.01 percentage points of 100%."""
    return math.isclose(float(weights.sum() * 100), 100.0, abs_tol=0.01)


def _is_crypto_only(fund: str) -> bool:
    return fund.startswith("Crypto ")


def compare_allocations(
    fund_returns: pd.DataFrame,
    original_weights: pd.Series,
    adjusted_weights: pd.Series,
) -> AllocationComparison:
    """Compare two allocations on one calendar and common live start date."""
    if not allocation_total_is_valid(original_weights):
        raise ValueError("original weights must sum to 1")
    if not allocation_total_is_valid(adjusted_weights):
        raise ValueError("adjusted weights must sum to 1")
    if set(original_weights.index) != set(adjusted_weights.index):
        raise ValueError("original and adjusted allocations must contain the same funds")

    active = [
        fund
        for fund in original_weights.index
        if original_weights[fund] > 1e-12 or adjusted_weights[fund] > 1e-12
    ]
    selected = fund_returns.loc[fund_returns["fund"].isin(active), ["date", "fund", "daily_return"]]
    missing = set(active).difference(selected["fund"].unique())
    if missing:
        raise ValueError(f"missing returns for funds: {sorted(missing)}")

    first_live = selected.groupby("fund")["date"].min()
    common_start = first_live.max()
    selected = selected.loc[selected["date"] >= common_start]
    matrix = selected.pivot(index="date", columns="fund", values="daily_return").sort_index()
    uses_crypto_calendar = any(
        _is_crypto_only(fund) and (original_weights[fund] > 1e-12 or adjusted_weights[fund] > 1e-12)
        for fund in active
    )
    if uses_crypto_calendar:
        matrix = matrix.asfreq("D")
        annualization = 365
    else:
        annualization = 252
    matrix = matrix.reindex(columns=active).fillna(0.0)

    portfolio_returns = pd.DataFrame(
        {
            "Original recommendation": matrix.dot(original_weights.reindex(active)),
            "User-adjusted": matrix.dot(adjusted_weights.reindex(active)),
        }
    )
    growth = (1.0 + portfolio_returns).cumprod()
    metric_rows = []
    for name in portfolio_returns:
        values = portfolio_returns[name]
        path = growth[name]
        volatility = values.std(ddof=1) * np.sqrt(annualization)
        metric_rows.append(
            {
                "Allocation": name,
                "Annual return": path.iloc[-1] ** (annualization / len(values)) - 1,
                "Volatility": volatility,
                "Sharpe ratio": (
                    values.mean() / values.std(ddof=1) * np.sqrt(annualization)
                    if values.std(ddof=1) > 0
                    else np.nan
                ),
                "Maximum drawdown": (path / path.cummax() - 1).min(),
                "Final value of $1": path.iloc[-1],
            }
        )
    return AllocationComparison(
        portfolio_returns,
        growth,
        pd.DataFrame(metric_rows),
        annualization,
        common_start,
    )


def lookthrough_crypto(weights: pd.Series, fund_crypto_exposure: pd.Series) -> float:
    """Calculate allocation-level crypto exposure from precomputed fund exposures."""
    return float(weights.dot(fund_crypto_exposure.reindex(weights.index)))


def fee_adjusted_growth(
    adjusted_returns: pd.Series,
    annual_fee: float,
    annualization: int,
) -> pd.DataFrame:
    """Compare adjusted-allocation growth before and after an annual asset charge."""
    if not 0.0 <= annual_fee < 1.0:
        raise ValueError("annual_fee must be between zero and one")
    if annualization not in {252, 365}:
        raise ValueError("annualization must be 252 or 365")
    returns = pd.to_numeric(adjusted_returns, errors="raise").astype(float)
    daily_fee = 1.0 - (1.0 - annual_fee) ** (1.0 / annualization)
    gross_growth = (1.0 + returns).cumprod()
    net_returns = (1.0 + returns) * (1.0 - daily_fee) - 1.0
    net_growth = (1.0 + net_returns).cumprod()
    return pd.DataFrame(
        {
            "Gross historical growth of $1": gross_growth,
            "Net-of-fee historical growth of $1": net_growth,
        },
        index=returns.index,
    )
