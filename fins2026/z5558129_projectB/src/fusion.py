"""Transparent, look-ahead-safe sentiment tilt for Equity Equal Weight."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import portfolios


@dataclass(frozen=True)
class FusionResult:
    """Tilted targets, daily performance, metrics, and audit diagnostics."""

    daily: pd.DataFrame
    weights: pd.DataFrame
    metrics: pd.DataFrame
    sector_diagnostics: pd.DataFrame
    rebalance_diagnostics: pd.DataFrame


def capped_proportional_weights(
    raw_weights: np.ndarray,
    individual_cap: float,
    *,
    target_total: float = 1.0,
    tolerance: float = 1e-12,
) -> tuple[np.ndarray, bool]:
    """Normalize non-negative weights while redistributing above-cap excess."""
    raw = np.asarray(raw_weights, dtype=float)
    if raw.ndim != 1 or not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("raw_weights must be a finite non-negative vector")
    if raw.sum() <= 0:
        raise ValueError("raw_weights must contain positive mass")
    if target_total <= 0 or target_total > 1 + tolerance:
        raise ValueError("target_total must be in (0, 1]")
    if individual_cap <= 0 or individual_cap * len(raw) < target_total - tolerance:
        raise ValueError("individual cap is infeasible for the number of assets")

    weights = np.zeros_like(raw)
    remaining = np.ones(len(raw), dtype=bool)
    remaining_mass = target_total
    cap_bound = False
    while remaining.any():
        remaining_raw = raw[remaining]
        if remaining_raw.sum() <= tolerance:
            proposals = np.full(remaining.sum(), remaining_mass / remaining.sum())
        else:
            proposals = remaining_mass * remaining_raw / remaining_raw.sum()
        above_cap = proposals > individual_cap + tolerance
        remaining_indices = np.flatnonzero(remaining)
        if not above_cap.any():
            weights[remaining_indices] = proposals
            break
        capped_indices = remaining_indices[above_cap]
        weights[capped_indices] = individual_cap
        remaining[capped_indices] = False
        remaining_mass -= individual_cap * len(capped_indices)
        cap_bound = True

    if not np.isclose(weights.sum(), target_total, atol=1e-9):
        raise ValueError(
            f"capped weights sum to {weights.sum():.12f}, not {target_total:.12f}"
        )
    if weights.max() > individual_cap + 1e-9 or weights.min() < -1e-12:
        raise ValueError("capped normalization returned infeasible weights")
    return weights, cap_bound


def build_sector_tilted_targets(
    base_weights: pd.DataFrame,
    sector_index: pd.DataFrame,
    ticker_universe: pd.DataFrame,
    *,
    base_fund: str = "Equity Equal Weight",
    tilted_fund: str = "Equity Equal Weight + Sentiment",
    coverage_threshold: float = 0.60,
    individual_cap: float = 0.10,
    preserve_crypto: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply ``1 + lagged sector sentiment`` to eligible sector targets."""
    base = base_weights.loc[base_weights["fund"] == base_fund].copy()
    if base.empty:
        raise ValueError(f"base weights contain no rows for {base_fund}")
    base["rebalance_date"] = pd.to_datetime(base["rebalance_date"])
    universe = ticker_universe[["ticker", "sector"]].drop_duplicates()
    if universe["ticker"].duplicated().any():
        raise ValueError("ticker_universe maps a ticker to multiple sectors")
    base = base.drop(columns="sector", errors="ignore").merge(
        universe, on="ticker", how="left", validate="many_to_one"
    )
    is_crypto = base["asset_class"].eq("Crypto") | base["ticker"].str.endswith("-USD")
    if not preserve_crypto and is_crypto.any():
        raise ValueError("base fund contains crypto but preserve_crypto is False")
    if base.loc[~is_crypto, "sector"].isna().any():
        raise ValueError("one or more base assets have no sector mapping")

    signals = sector_index.copy()
    signals["trading_date"] = pd.to_datetime(signals["trading_date"])
    signals["signal_source_date"] = pd.to_datetime(signals["signal_source_date"])
    if signals.duplicated(["trading_date", "sector"]).any():
        raise ValueError("sector index contains duplicate trading-date-sector rows")

    target_records: list[dict[str, object]] = []
    sector_records: list[dict[str, object]] = []
    rebalance_records: list[dict[str, object]] = []
    for rebalance_date, group in base.groupby("rebalance_date", sort=True):
        group = group.copy()
        group["is_crypto"] = (
            group["asset_class"].eq("Crypto") | group["ticker"].str.endswith("-USD")
        )
        equity_group = group.loc[~group["is_crypto"]].copy()
        crypto_group = group.loc[group["is_crypto"]].copy()
        date_signals = signals.loc[signals["trading_date"] == rebalance_date]
        required_sectors = set(equity_group["sector"])
        if set(date_signals["sector"]) != required_sectors:
            raise ValueError(f"{rebalance_date.date()}: incomplete sector signals")
        if (date_signals["signal_source_date"] >= rebalance_date).any():
            raise ValueError(f"{rebalance_date.date()}: sector signal is not lagged")

        signal_lookup = date_signals.set_index("sector")
        applied_multipliers: dict[str, float] = {}
        for sector in sorted(required_sectors):
            signal_row = signal_lookup.loc[sector]
            coverage = float(signal_row["covered_ticker_proportion"])
            blocked = coverage < coverage_threshold
            sentiment_score = float(signal_row["sector_sentiment_index"])
            multiplier = 1.0 if blocked else 1.0 + sentiment_score
            if multiplier < 0:
                raise ValueError(
                    f"{rebalance_date.date()} {sector}: negative tilt multiplier"
                )
            applied_multipliers[sector] = multiplier
            sector_records.append(
                {
                    "rebalance_date": rebalance_date,
                    "signal_source_date": signal_row["signal_source_date"],
                    "sector": sector,
                    "sector_sentiment_index": sentiment_score,
                    "covered_ticker_count": int(signal_row["covered_ticker_count"]),
                    "covered_ticker_proportion": coverage,
                    "coverage_rule_blocked": blocked,
                    "applied_multiplier": multiplier,
                }
            )

        raw = equity_group["target_weight"].to_numpy(dtype=float) * equity_group[
            "sector"
        ].map(applied_multipliers).to_numpy(dtype=float)
        equity_total = float(equity_group["target_weight"].sum())
        tilted_equity, redistribution_cap_bound = capped_proportional_weights(
            raw,
            individual_cap,
            target_total=equity_total,
        )
        equity_group["tilted_target"] = tilted_equity
        crypto_group["tilted_target"] = crypto_group["target_weight"]
        crypto_group["sector"] = "Crypto"
        final_group = pd.concat([equity_group, crypto_group], ignore_index=True)
        final_targets = final_group["tilted_target"].to_numpy(dtype=float)
        base_targets = final_group["target_weight"].to_numpy(dtype=float)
        changes = final_targets - base_targets
        equity_changes = (
            final_group.loc[~final_group["is_crypto"], "tilted_target"]
            - final_group.loc[~final_group["is_crypto"], "target_weight"]
        )
        crypto_preservation_error = float(
            (
                final_group.loc[final_group["is_crypto"], "tilted_target"]
                - final_group.loc[final_group["is_crypto"], "target_weight"]
            ).abs().max()
            if final_group["is_crypto"].any()
            else 0.0
        )
        rebalance_records.append(
            {
                "rebalance_date": rebalance_date,
                "cap_bound": redistribution_cap_bound,
                "any_final_weight_at_cap": bool(
                    (final_targets >= individual_cap - 1e-9).any()
                ),
                "maximum_target_weight": float(final_targets.max()),
                "mean_absolute_asset_weight_change": float(np.abs(changes).mean()),
                "mean_absolute_equity_weight_change": float(
                    equity_changes.abs().mean()
                ),
                "half_l1_weight_distance": float(0.5 * np.abs(changes).sum()),
                "base_crypto_weight": float(crypto_group["target_weight"].sum()),
                "tilted_crypto_weight": float(crypto_group["tilted_target"].sum()),
                "crypto_preservation_error": crypto_preservation_error,
            }
        )
        for row, target, change in zip(
            final_group.itertuples(index=False), final_targets, changes
        ):
            multiplier = (
                1.0 if row.is_crypto else applied_multipliers[row.sector]
            )
            target_records.append(
                {
                    "rebalance_date": rebalance_date,
                    "fund": tilted_fund,
                    "ticker": row.ticker,
                    "asset_class": "Crypto" if row.is_crypto else "Equity",
                    "sector": row.sector,
                    "base_target_weight": float(row.target_weight),
                    "applied_multiplier": multiplier,
                    "target_weight": float(target),
                    "weight_change": float(change),
                }
            )
    return (
        pd.DataFrame(target_records),
        pd.DataFrame(sector_records),
        pd.DataFrame(rebalance_records),
    )


def backtest_tilted_targets(
    equity_returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    *,
    fund_name: str = "Equity Equal Weight + Sentiment",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply monthly tilt targets before same-day returns and drift between them."""
    returns = equity_returns.copy()
    returns["date"] = pd.to_datetime(returns["date"])
    assets = returns.columns.drop("date").tolist()
    returns = returns.dropna(subset=assets).set_index("date").sort_index()
    targets = target_weights.pivot(
        index="rebalance_date", columns="ticker", values="target_weight"
    ).reindex(columns=assets)
    targets.index = pd.to_datetime(targets.index)
    if targets.isna().any().any():
        raise ValueError("tilted targets do not cover every equity asset")
    if not targets.index.isin(returns.index).all():
        raise ValueError("one or more tilted rebalance dates are absent from returns")

    daily_records: list[dict[str, object]] = []
    current: np.ndarray | None = None
    growth = 1.0
    for date, asset_returns in returns.loc[targets.index.min() :].iterrows():
        if date in targets.index:
            current = targets.loc[date].to_numpy(dtype=float)
        if current is None:
            raise AssertionError("tilted live period began without target weights")
        values = asset_returns.to_numpy(dtype=float)
        daily_return = float(current @ values)
        growth *= 1.0 + daily_return
        daily_records.append(
            {
                "date": date,
                "fund": fund_name,
                "daily_return": daily_return,
                "growth_of_1": growth,
            }
        )
        current = current * (1.0 + values) / (1.0 + daily_return)
    daily = pd.DataFrame(daily_records)
    metrics = pd.DataFrame(
        [
            {
                "fund": fund_name,
                **portfolios.performance_metrics(
                    daily["daily_return"], periods_per_year=252
                ),
            }
        ]
    )
    return daily, metrics


def run_baseline_fusion(
    equity_returns: pd.DataFrame,
    base_weights: pd.DataFrame,
    sector_index: pd.DataFrame,
    ticker_universe: pd.DataFrame,
) -> FusionResult:
    """Build and backtest the single approved baseline sentiment fusion."""
    weights, sector_diagnostics, rebalance_diagnostics = (
        build_sector_tilted_targets(
            base_weights,
            sector_index,
            ticker_universe,
        )
    )
    daily, metrics = backtest_tilted_targets(equity_returns, weights)
    return FusionResult(
        daily=daily,
        weights=weights,
        metrics=metrics,
        sector_diagnostics=sector_diagnostics,
        rebalance_diagnostics=rebalance_diagnostics,
    )


def run_combined_max_sharpe_fusion(
    equity_calendar_returns: pd.DataFrame,
    base_weights: pd.DataFrame,
    sector_index: pd.DataFrame,
    ticker_universe: pd.DataFrame,
) -> FusionResult:
    """Apply the unchanged raw sector tilt only to combined-fund equities."""
    fund_name = "Combined Maximum Sharpe + Sentiment"
    weights, sector_diagnostics, rebalance_diagnostics = (
        build_sector_tilted_targets(
            base_weights,
            sector_index,
            ticker_universe,
            base_fund="Combined Maximum Sharpe",
            tilted_fund=fund_name,
            preserve_crypto=True,
        )
    )
    daily, metrics = backtest_tilted_targets(
        equity_calendar_returns,
        weights,
        fund_name=fund_name,
    )
    return FusionResult(
        daily=daily,
        weights=weights,
        metrics=metrics,
        sector_diagnostics=sector_diagnostics,
        rebalance_diagnostics=rebalance_diagnostics,
    )
