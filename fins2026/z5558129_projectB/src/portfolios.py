"""Constrained portfolio construction and walk-forward multi-asset backtests."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

METHOD_LABELS = {
    "equal_weight": "Combined Equal Weight",
    "min_variance": "Combined Minimum Variance",
    "max_sharpe": "Combined Maximum Sharpe",
}
SUPPORTED_METHODS = {"equal_weight", "min_variance", "max_sharpe", "risk_parity"}


class PortfolioOptimizationError(RuntimeError):
    """Raised when a dated optimization fails or returns infeasible weights."""


@dataclass(frozen=True)
class BacktestResult:
    """Daily fund history, rebalance targets, metrics, and diagnostics."""

    daily: pd.DataFrame
    weights: pd.DataFrame
    metrics: pd.DataFrame
    diagnostics: pd.DataFrame


def _validate_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a wide date-by-asset return panel."""
    if "date" not in returns.columns:
        raise ValueError("returns is missing required column: date")
    panel = returns.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    if panel["date"].isna().any():
        raise ValueError("returns contains invalid dates")
    if panel["date"].duplicated().any():
        raise ValueError("returns contains duplicate dates")
    if not panel["date"].is_monotonic_increasing:
        raise ValueError("returns dates must be strictly increasing")
    asset_columns = panel.columns.drop("date")
    if asset_columns.empty:
        raise ValueError("returns contains no asset columns")
    panel[asset_columns] = panel[asset_columns].apply(pd.to_numeric, errors="coerce")
    return panel


def validate_weights(
    weights: np.ndarray,
    crypto_mask: np.ndarray,
    *,
    individual_cap: float = 0.10,
    crypto_cap: float | None = 0.25,
    tolerance: float = 1e-7,
) -> list[str]:
    """Return descriptions of any portfolio-constraint violations."""
    issues: list[str] = []
    if not np.isfinite(weights).all():
        issues.append("weights contain non-finite values")
        return issues
    if abs(float(weights.sum()) - 1.0) > tolerance:
        issues.append(f"weights sum to {weights.sum():.12f}, not 1")
    if float(weights.min()) < -tolerance:
        issues.append(f"minimum weight {weights.min():.12f} is negative")
    if float(weights.max()) > individual_cap + tolerance:
        issues.append(
            f"maximum weight {weights.max():.12f} exceeds {individual_cap:.2%}"
        )
    crypto_weight = float(weights[crypto_mask].sum())
    if crypto_cap is not None and crypto_weight > crypto_cap + tolerance:
        issues.append(
            f"crypto weight {crypto_weight:.12f} exceeds {crypto_cap:.2%}"
        )
    return issues


def optimize_weights(
    estimation_returns: pd.DataFrame,
    method: str,
    crypto_mask: np.ndarray,
    rebalance_date: pd.Timestamp,
    *,
    individual_cap: float = 0.10,
    crypto_cap: float | None = 0.25,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> tuple[np.ndarray, dict[str, object]]:
    """Construct one constrained portfolio from a historical return window."""
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"unsupported portfolio method: {method}")
    values = estimation_returns.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise PortfolioOptimizationError(
            f"{rebalance_date.date()} {method}: estimation window contains missing "
            "or non-finite returns"
        )
    observations, asset_count = values.shape
    if len(crypto_mask) != asset_count:
        raise ValueError("crypto_mask length does not match asset count")

    initial = np.full(asset_count, 1.0 / asset_count)
    initial_issues = validate_weights(
        initial,
        crypto_mask,
        individual_cap=individual_cap,
        crypto_cap=crypto_cap,
    )
    if initial_issues:
        raise PortfolioOptimizationError(
            f"{rebalance_date.date()} {method}: equal-weight starting point is "
            f"infeasible: {'; '.join(initial_issues)}"
        )
    if method == "equal_weight":
        return initial, {
            "solver_success": True,
            "solver_status": 0,
            "solver_message": "closed-form equal weight",
            "objective_value": np.nan,
            "estimation_observations": observations,
        }

    mean_returns = values.mean(axis=0)
    covariance = np.cov(values, rowvar=False, ddof=1)

    def variance_objective(weights: np.ndarray) -> float:
        return float(weights @ covariance @ weights)

    def negative_sharpe(weights: np.ndarray) -> float:
        variance = variance_objective(weights)
        if variance <= 0 or not np.isfinite(variance):
            return 1e12
        excess_daily = float(weights @ mean_returns) - risk_free_rate / periods_per_year
        return -excess_daily / np.sqrt(variance)

    def risk_parity_objective(weights: np.ndarray) -> float:
        portfolio_variance = variance_objective(weights)
        if portfolio_variance <= 0 or not np.isfinite(portfolio_variance):
            return 1e12
        contributions = weights * (covariance @ weights) / portfolio_variance
        target = np.full(asset_count, 1.0 / asset_count)
        return float(np.square(contributions - target).sum())

    objectives = {
        "min_variance": variance_objective,
        "max_sharpe": negative_sharpe,
        "risk_parity": risk_parity_objective,
    }
    objective = objectives[method]
    constraints = [
        {"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
    ]
    if crypto_cap is not None:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda weights: crypto_cap - weights[crypto_mask].sum(),
            }
        )
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, individual_cap)] * asset_count,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2_000, "disp": False},
    )
    if not result.success:
        raise PortfolioOptimizationError(
            f"{rebalance_date.date()} {method}: SLSQP failed with status "
            f"{result.status}: {result.message}"
        )
    issues = validate_weights(
        result.x,
        crypto_mask,
        individual_cap=individual_cap,
        crypto_cap=crypto_cap,
    )
    if issues:
        raise PortfolioOptimizationError(
            f"{rebalance_date.date()} {method}: SLSQP returned infeasible weights: "
            f"{'; '.join(issues)}"
        )
    return result.x, {
        "solver_success": bool(result.success),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "objective_value": float(result.fun),
        "estimation_observations": observations,
    }


def performance_metrics(
    daily_returns: pd.Series,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> dict[str, float | int]:
    """Calculate standard performance statistics from out-of-sample returns."""
    values = pd.to_numeric(daily_returns, errors="coerce").dropna()
    if values.empty:
        raise ValueError("daily_returns must contain at least one valid return")
    growth = (1.0 + values).cumprod()
    annualized_return = float(growth.iloc[-1] ** (periods_per_year / len(values)) - 1)
    annualized_volatility = float(values.std(ddof=1) * np.sqrt(periods_per_year))
    excess_mean = float(values.mean() - risk_free_rate / periods_per_year)
    sharpe = (
        float(excess_mean / values.std(ddof=1) * np.sqrt(periods_per_year))
        if values.std(ddof=1) > 0
        else np.nan
    )
    drawdown = growth / growth.cummax() - 1.0
    return {
        "observations": len(values),
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe,
        "maximum_drawdown": float(drawdown.min()),
        "final_growth_of_1": float(growth.iloc[-1]),
    }


def oos_backtest(
    returns: pd.DataFrame,
    methods: tuple[str, ...] = ("equal_weight", "min_variance", "max_sharpe"),
    *,
    window: int = 252,
    individual_cap: float = 0.10,
    crypto_cap: float | None = 0.25,
    risk_free_rate: float = 0.0,
    transaction_cost: float = 0.0,
    periods_per_year: int = 252,
    method_labels: dict[str, str] | None = None,
) -> BacktestResult:
    """Run a monthly, no-look-ahead backtest on the supplied native calendar."""
    if window <= 1:
        raise ValueError("window must exceed one observation")
    if transaction_cost != 0:
        raise NotImplementedError("only zero transaction costs are currently supported")
    labels = METHOD_LABELS if method_labels is None else method_labels
    missing_labels = set(methods).difference(labels)
    if missing_labels:
        raise ValueError(f"missing fund labels for methods: {sorted(missing_labels)}")

    panel = _validate_returns(returns)
    assets = panel.columns.drop("date").tolist()
    crypto_mask = np.array([ticker.endswith("-USD") for ticker in assets])
    complete = panel.dropna(subset=assets).reset_index(drop=True)
    if len(complete) <= window:
        raise ValueError(f"fewer than {window + 1} complete return dates are available")

    month = complete["date"].dt.to_period("M")
    candidate_indices = complete.groupby(month, sort=True).head(1).index
    rebalance_indices = [index for index in candidate_indices if index >= window]
    if not rebalance_indices:
        raise ValueError("no monthly rebalance date has a complete estimation window")

    first_live_index = rebalance_indices[0]
    live = complete.loc[first_live_index:].reset_index(drop=True)
    original_to_live = {
        original_index: original_index - first_live_index
        for original_index in rebalance_indices
    }
    targets: dict[str, dict[int, np.ndarray]] = {method: {} for method in methods}
    weight_records: list[dict[str, object]] = []
    diagnostic_records: list[dict[str, object]] = []

    for original_index in rebalance_indices:
        rebalance_date = complete.loc[original_index, "date"]
        estimation = complete.loc[
            original_index - window : original_index - 1, assets
        ]
        if len(estimation) != window:
            raise AssertionError("estimation window length is not exact")
        for method in methods:
            weights, diagnostic = optimize_weights(
                estimation,
                method,
                crypto_mask,
                rebalance_date,
                individual_cap=individual_cap,
                crypto_cap=crypto_cap,
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year,
            )
            live_index = original_to_live[original_index]
            targets[method][live_index] = weights
            label = labels[method]
            for ticker, weight, is_crypto in zip(assets, weights, crypto_mask):
                weight_records.append(
                    {
                        "rebalance_date": rebalance_date,
                        "fund": label,
                        "ticker": ticker,
                        "asset_class": "Crypto" if is_crypto else "Equity",
                        "target_weight": float(weight),
                    }
                )
            diagnostic_records.append(
                {
                    "rebalance_date": rebalance_date,
                    "fund": label,
                    "estimation_start": complete.loc[original_index - window, "date"],
                    "estimation_end": complete.loc[original_index - 1, "date"],
                    "individual_max": float(weights.max()),
                    "crypto_weight": float(weights[crypto_mask].sum()),
                    "periods_per_year": periods_per_year,
                    **diagnostic,
                }
            )

    daily_records: list[dict[str, object]] = []
    for method in methods:
        label = labels[method]
        current_weights: np.ndarray | None = None
        growth = 1.0
        for live_index, row in live.iterrows():
            if live_index in targets[method]:
                current_weights = targets[method][live_index].copy()
            if current_weights is None:
                raise AssertionError("live period began without target weights")
            asset_returns = row[assets].to_numpy(dtype=float)
            portfolio_return = float(current_weights @ asset_returns)
            growth *= 1.0 + portfolio_return
            daily_records.append(
                {
                    "date": row["date"],
                    "fund": label,
                    "daily_return": portfolio_return,
                    "growth_of_1": growth,
                }
            )
            denominator = 1.0 + portfolio_return
            if denominator <= 0:
                raise ValueError(f"{row['date'].date()} {method}: portfolio value is nonpositive")
            current_weights = current_weights * (1.0 + asset_returns) / denominator

    daily = pd.DataFrame(daily_records).sort_values(["date", "fund"]).reset_index(drop=True)
    weights = pd.DataFrame(weight_records).sort_values(
        ["rebalance_date", "fund", "ticker"]
    ).reset_index(drop=True)
    diagnostics = pd.DataFrame(diagnostic_records).sort_values(
        ["rebalance_date", "fund"]
    ).reset_index(drop=True)
    metrics = pd.DataFrame(
        [
            {
                "fund": fund,
                **performance_metrics(
                    group["daily_return"],
                    periods_per_year=periods_per_year,
                    risk_free_rate=risk_free_rate,
                ),
            }
            for fund, group in daily.groupby("fund", sort=True)
        ]
    )
    return BacktestResult(daily, weights, metrics, diagnostics)


def method_weight_distances(weights: pd.DataFrame) -> pd.DataFrame:
    """Summarize pairwise target-weight differences across fund methods."""
    wide = weights.pivot(
        index=["rebalance_date", "ticker"], columns="fund", values="target_weight"
    )
    records = []
    for left, right in combinations(wide.columns, 2):
        distance = 0.5 * (wide[left] - wide[right]).abs().groupby("rebalance_date").sum()
        records.append(
            {
                "fund_a": left,
                "fund_b": right,
                "mean_half_l1_distance": float(distance.mean()),
                "minimum_half_l1_distance": float(distance.min()),
                "maximum_half_l1_distance": float(distance.max()),
                "meaningfully_different": bool(distance.mean() >= 0.01),
            }
        )
    return pd.DataFrame(records)
