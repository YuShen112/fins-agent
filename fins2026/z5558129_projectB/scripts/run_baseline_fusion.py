"""Reproduce the approved Equity Equal Weight baseline sentiment fusion."""
from __future__ import annotations

import importlib
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

fusion = importlib.import_module("src.fusion")

RESULTS_DATA = ROOT / "results" / "data"
RESULTS_TABLES = ROOT / "results" / "tables"


def main() -> None:
    combined = pd.read_csv(
        RESULTS_DATA / "combined_returns_panel.csv", parse_dates=["date"]
    )
    equity_columns = [
        column for column in combined.columns if column != "date" and not column.endswith("-USD")
    ]
    equity_returns = combined[["date", *equity_columns]]
    fund_weights = pd.read_csv(
        RESULTS_DATA / "fund_weights.csv", parse_dates=["rebalance_date"]
    )
    sector_index = pd.read_csv(
        RESULTS_DATA / "sector_sentiment_index.csv",
        parse_dates=["trading_date", "signal_source_date"],
    )
    headline_panel = pd.read_csv(
        RESULTS_DATA / "daily_headline_panel.csv", usecols=["ticker", "sector"]
    )
    ticker_universe = headline_panel.drop_duplicates()

    result = fusion.run_baseline_fusion(
        equity_returns,
        fund_weights,
        sector_index,
        ticker_universe,
    )
    base_returns = pd.read_csv(
        RESULTS_DATA / "fund_returns.csv", parse_dates=["date"]
    ).query("fund == 'Equity Equal Weight'")
    base_metrics = pd.read_csv(RESULTS_TABLES / "performance_metrics.csv").query(
        "fund == 'Equity Equal Weight'"
    )
    comparison_returns = pd.concat(
        [base_returns, result.daily], ignore_index=True
    ).sort_values(["date", "fund"])
    comparison_metrics = pd.concat(
        [base_metrics, result.metrics], ignore_index=True
    )

    returns_path = RESULTS_DATA / "equity_equal_weight_sentiment_returns.csv"
    weights_path = RESULTS_DATA / "equity_equal_weight_sentiment_weights.csv"
    metrics_path = RESULTS_TABLES / "equity_equal_weight_sentiment_metrics.csv"
    sector_path = (
        RESULTS_TABLES / "equity_equal_weight_sentiment_sector_diagnostics.csv"
    )
    rebalance_path = (
        RESULTS_TABLES / "equity_equal_weight_sentiment_rebalance_diagnostics.csv"
    )
    comparison_returns.to_csv(returns_path, index=False)
    result.weights.to_csv(weights_path, index=False)
    comparison_metrics.to_csv(metrics_path, index=False)
    result.sector_diagnostics.to_csv(sector_path, index=False)
    result.rebalance_diagnostics.to_csv(rebalance_path, index=False)

    print(comparison_metrics.to_string(index=False))
    print(f"Saved: {returns_path.relative_to(ROOT)}")
    print(f"Saved: {weights_path.relative_to(ROOT)}")
    print(f"Saved: {metrics_path.relative_to(ROOT)}")
    print(f"Saved: {sector_path.relative_to(ROOT)}")
    print(f"Saved: {rebalance_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
