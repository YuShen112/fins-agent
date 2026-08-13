"""Transparent, presentation-ready diagnostics for fund fact sheets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RecoveryDiagnostics:
    """Calendar-day drawdown and recovery measures."""

    longest_drawdown_days: int
    maximum_drawdown_recovery_days: int | None
    maximum_drawdown_recovered: bool
    maximum_drawdown_elapsed_days: int


@dataclass(frozen=True)
class FundFactSheetAnalysis:
    """Additional diagnostics derived only from precomputed fund artifacts."""

    downside_deviation: float
    sortino_ratio: float
    annualization: int
    recovery: RecoveryDiagnostics
    effective_holdings: float
    top_five_concentration: float
    holdings_as_of: pd.Timestamp
    ranks: pd.DataFrame


def annualization_for_fund(fund: str) -> int:
    """Return the established native-calendar annualization for a fund."""
    return 365 if fund.startswith("Crypto ") else 252


def downside_risk(
    daily_returns: pd.Series,
    annualized_return: float,
    annualization: int,
    mar: float = 0.0,
) -> tuple[float, float]:
    """Return annualized downside deviation and Sortino ratio.

    Daily returns below the daily MAR contribute their squared shortfall; all
    other observations contribute zero. The Sortino numerator uses the
    precomputed annualized fund return and an annual MAR of zero by default.
    """
    values = pd.to_numeric(daily_returns, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError("daily_returns must contain at least one finite observation")
    daily_mar = (1.0 + mar) ** (1.0 / annualization) - 1.0
    shortfalls = np.minimum(values - daily_mar, 0.0)
    downside_deviation = float(np.sqrt(np.mean(shortfalls**2)) * np.sqrt(annualization))
    sortino = (
        float((annualized_return - mar) / downside_deviation) if downside_deviation > 0 else np.nan
    )
    return downside_deviation, sortino


def recovery_diagnostics(fund_returns: pd.DataFrame) -> RecoveryDiagnostics:
    """Measure complete and ongoing drawdowns in calendar days."""
    required = {"date", "growth_of_1"}
    if missing := required.difference(fund_returns.columns):
        raise ValueError(f"fund returns are missing columns: {sorted(missing)}")
    data = fund_returns.loc[:, ["date", "growth_of_1"]].copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").dropna().drop_duplicates("date", keep="last")
    if data.empty:
        raise ValueError("fund returns must contain at least one observation")

    dates = data["date"].reset_index(drop=True)
    growth = data["growth_of_1"].astype(float).reset_index(drop=True)
    running_peak = growth.cummax()
    drawdown = growth / running_peak - 1.0
    tolerance = 1e-12

    longest_days = 0
    drawdown_start: pd.Timestamp | None = None
    latest_peak_date = dates.iloc[0]
    latest_peak_value = growth.iloc[0]
    for date, value in zip(dates, growth, strict=True):
        if value >= latest_peak_value - tolerance:
            if drawdown_start is not None:
                longest_days = max(longest_days, int((date - drawdown_start).days))
                drawdown_start = None
            latest_peak_date = date
            latest_peak_value = max(latest_peak_value, value)
        elif drawdown_start is None:
            drawdown_start = latest_peak_date
    if drawdown_start is not None:
        longest_days = max(longest_days, int((dates.iloc[-1] - drawdown_start).days))

    trough_position = int(drawdown.to_numpy().argmin())
    trough_date = dates.iloc[trough_position]
    peak_value = running_peak.iloc[trough_position]
    prior = growth.iloc[: trough_position + 1]
    peak_positions = np.flatnonzero(np.isclose(prior, peak_value, rtol=0, atol=tolerance))
    peak_position = int(peak_positions[-1])
    recoveries = np.flatnonzero(
        growth.iloc[trough_position + 1 :].to_numpy() >= peak_value - tolerance
    )
    if recoveries.size:
        recovery_position = trough_position + 1 + int(recoveries[0])
        recovery_days = int((dates.iloc[recovery_position] - trough_date).days)
        recovered = True
        elapsed_days = recovery_days
    else:
        recovery_days = None
        recovered = bool(np.isclose(growth.iloc[trough_position], peak_value, atol=tolerance))
        elapsed_days = 0 if recovered else int((dates.iloc[-1] - trough_date).days)
        if recovered:
            recovery_days = 0

    if trough_position == peak_position:
        recovery_days = 0
        recovered = True
        elapsed_days = 0

    return RecoveryDiagnostics(longest_days, recovery_days, recovered, elapsed_days)


def latest_diversification(fund_weights: pd.DataFrame) -> tuple[float, float, pd.Timestamp]:
    """Calculate effective holdings and top-five concentration at latest rebalance."""
    required = {"rebalance_date", "target_weight"}
    if missing := required.difference(fund_weights.columns):
        raise ValueError(f"fund weights are missing columns: {sorted(missing)}")
    weights = fund_weights.copy()
    weights["rebalance_date"] = pd.to_datetime(weights["rebalance_date"])
    latest_date = weights["rebalance_date"].max()
    latest = weights.loc[weights["rebalance_date"] == latest_date, "target_weight"].astype(float)
    if latest.empty or not np.isclose(latest.sum(), 1.0, atol=1e-8):
        raise ValueError("latest target weights must exist and sum to one")
    effective_holdings = float(1.0 / np.square(latest).sum())
    top_five = float(latest.nlargest(5).sum())
    return effective_holdings, top_five, latest_date


def fund_ranks(metrics: pd.DataFrame, fund: str) -> pd.DataFrame:
    """Rank a fund against the full eight-fund table using investor-friendly direction."""
    specifications = (
        ("Annual return", "annualized_return", False),
        ("Volatility", "annualized_volatility", True),
        ("Maximum drawdown", "maximum_drawdown", False),
        ("Sharpe ratio", "sharpe_ratio", False),
    )
    if fund not in set(metrics["fund"]):
        raise ValueError(f"Unknown fund: {fund}")
    rows = []
    for label, column, ascending in specifications:
        ranks = metrics[column].rank(method="min", ascending=ascending).astype(int)
        position = int(ranks.loc[metrics["fund"] == fund].iloc[0])
        rows.append({"Measure": label, "Rank among 8 funds": position})
    return pd.DataFrame(rows)


def build_fact_sheet_analysis(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    metrics: pd.DataFrame,
    fund: str,
) -> FundFactSheetAnalysis:
    """Build all supplementary fact-sheet diagnostics for one selected fund."""
    fund_returns = returns.loc[returns["fund"] == fund].copy()
    fund_weights = weights.loc[weights["fund"] == fund].copy()
    metric = metrics.loc[metrics["fund"] == fund]
    if fund_returns.empty or fund_weights.empty or metric.empty:
        raise ValueError(f"Precomputed artifacts are incomplete for {fund}")
    annualization = annualization_for_fund(fund)
    downside_deviation, sortino = downside_risk(
        fund_returns["daily_return"],
        float(metric.iloc[0]["annualized_return"]),
        annualization,
    )
    effective, top_five, as_of = latest_diversification(fund_weights)
    return FundFactSheetAnalysis(
        downside_deviation=downside_deviation,
        sortino_ratio=sortino,
        annualization=annualization,
        recovery=recovery_diagnostics(fund_returns),
        effective_holdings=effective,
        top_five_concentration=top_five,
        holdings_as_of=as_of,
        ranks=fund_ranks(metrics, fund),
    )
