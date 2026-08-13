"""Run the parallel finVADER robustness and unchanged fusion experiments."""

from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import etl, figures, finvader_robustness, fusion  # noqa: E402

RESULTS_DATA = ROOT / "results" / "data"
RESULTS_TABLES = ROOT / "results" / "tables"
RESULTS_FIGURES = ROOT / "results" / "figures"

HEADLINE_SCORES = RESULTS_DATA / "finvader_headline_scores.csv"
TICKER_DAYS = RESULTS_DATA / "finvader_daily_ticker_sentiment.csv"
SECTOR_INDEX = RESULTS_DATA / "finvader_sector_sentiment_index.csv"
MONTHLY_FIGURE = RESULTS_FIGURES / "finvader_sector_sentiment_monthly.png"
PRIMARY_MONTHLY_FIGURE = RESULTS_FIGURES / "sector_sentiment_monthly_primary.png"
COMPARISON = RESULTS_TABLES / "finvader_vader_comparison.csv"
PRIMARY_FUSION_COMPARISON = RESULTS_TABLES / "sentiment_fusion_comparison.csv"

EQUITY_RETURNS = RESULTS_DATA / "equity_equal_weight_finvader_returns.csv"
EQUITY_WEIGHTS = RESULTS_DATA / "equity_equal_weight_finvader_weights.csv"
EQUITY_METRICS = RESULTS_TABLES / "equity_equal_weight_finvader_metrics.csv"
EQUITY_SECTOR_DIAGNOSTICS = RESULTS_TABLES / "equity_equal_weight_finvader_sector_diagnostics.csv"
EQUITY_REBALANCE_DIAGNOSTICS = (
    RESULTS_TABLES / "equity_equal_weight_finvader_rebalance_diagnostics.csv"
)

COMBINED_RETURNS = RESULTS_DATA / "combined_maximum_sharpe_finvader_returns.csv"
COMBINED_WEIGHTS = RESULTS_DATA / "combined_maximum_sharpe_finvader_weights.csv"
COMBINED_METRICS = RESULTS_TABLES / "combined_maximum_sharpe_finvader_metrics.csv"
COMBINED_SECTOR_DIAGNOSTICS = (
    RESULTS_TABLES / "combined_maximum_sharpe_finvader_sector_diagnostics.csv"
)
COMBINED_REBALANCE_DIAGNOSTICS = (
    RESULTS_TABLES / "combined_maximum_sharpe_finvader_rebalance_diagnostics.csv"
)


def _renamed_result(result: fusion.FusionResult, name: str) -> fusion.FusionResult:
    daily = result.daily.copy()
    weights = result.weights.copy()
    metrics = result.metrics.copy()
    daily["fund"] = name
    weights["fund"] = name
    metrics["fund"] = name
    return fusion.FusionResult(
        daily,
        weights,
        metrics,
        result.sector_diagnostics,
        result.rebalance_diagnostics,
    )


def _save_fusion(
    result: fusion.FusionResult,
    base_returns: pd.DataFrame,
    base_metrics: pd.DataFrame,
    paths: tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path],
) -> pd.DataFrame:
    returns_path, weights_path, metrics_path, sector_path, rebalance_path = paths
    comparison_returns = pd.concat([base_returns, result.daily], ignore_index=True)
    comparison_returns = comparison_returns.sort_values(["date", "fund"])
    comparison_metrics = pd.concat([base_metrics, result.metrics], ignore_index=True)
    comparison_returns.to_csv(returns_path, index=False)
    result.weights.to_csv(weights_path, index=False)
    comparison_metrics.to_csv(metrics_path, index=False)
    result.sector_diagnostics.to_csv(sector_path, index=False)
    result.rebalance_diagnostics.to_csv(rebalance_path, index=False)
    return comparison_metrics


def main() -> None:
    headlines = etl.load_clean_headlines()
    equities = etl.load_clean_equities()
    if len(headlines) != 146_836:
        raise ValueError(f"expected 146,836 deduplicated headlines, got {len(headlines):,}")
    universe = equities[["ticker", "sector"]].drop_duplicates()
    vader_sector = pd.read_csv(
        RESULTS_DATA / "sector_sentiment_index.csv",
        parse_dates=["trading_date", "signal_source_date"],
    )
    robustness = finvader_robustness.run_robustness_check(
        headlines,
        equities["date"],
        universe,
        vader_sector,
        MONTHLY_FIGURE,
    )
    if robustness.headline_scores["headline_id"].duplicated().any():
        raise ValueError("finVADER headline cache contains duplicate IDs")
    excluded = int(robustness.headline_scores["trading_date"].isna().sum())
    if excluded != 6:
        raise ValueError(f"expected six unaligned headlines, got {excluded}")
    if robustness.ticker_days.duplicated(["trading_date", "ticker"]).any():
        raise ValueError("finVADER ticker-day output contains duplicate keys")
    if robustness.sector_index.duplicated(["trading_date", "sector"]).any():
        raise ValueError("finVADER sector output contains duplicate keys")
    coverage_columns = [
        "constituent_ticker_count",
        "covered_ticker_count",
        "covered_ticker_proportion",
        "source_headline_count",
    ]
    pd.testing.assert_frame_equal(
        robustness.sector_index[coverage_columns].reset_index(drop=True),
        vader_sector[coverage_columns].reset_index(drop=True),
        check_dtype=False,
    )

    robustness.headline_scores.to_csv(HEADLINE_SCORES, index=False)
    robustness.ticker_days.to_csv(TICKER_DAYS, index=False)
    robustness.sector_index.to_csv(SECTOR_INDEX, index=False)
    robustness.comparison.to_csv(COMPARISON, index=False)

    combined_panel = pd.read_csv(RESULTS_DATA / "combined_returns_panel.csv", parse_dates=["date"])
    equity_columns = [
        column
        for column in combined_panel.columns
        if column != "date" and not column.endswith("-USD")
    ]
    equity_panel = combined_panel[["date", *equity_columns]]
    fund_weights = pd.read_csv(RESULTS_DATA / "fund_weights.csv", parse_dates=["rebalance_date"])
    fund_returns = pd.read_csv(RESULTS_DATA / "fund_returns.csv", parse_dates=["date"])
    fund_metrics = pd.read_csv(RESULTS_TABLES / "performance_metrics.csv")

    equity_result = _renamed_result(
        fusion.run_baseline_fusion(equity_panel, fund_weights, robustness.sector_index, universe),
        "Equity Equal Weight + finVADER",
    )
    equity_metrics = _save_fusion(
        equity_result,
        fund_returns.query("fund == 'Equity Equal Weight'"),
        fund_metrics.query("fund == 'Equity Equal Weight'"),
        (
            EQUITY_RETURNS,
            EQUITY_WEIGHTS,
            EQUITY_METRICS,
            EQUITY_SECTOR_DIAGNOSTICS,
            EQUITY_REBALANCE_DIAGNOSTICS,
        ),
    )

    combined_result = _renamed_result(
        fusion.run_combined_max_sharpe_fusion(
            combined_panel, fund_weights, robustness.sector_index, universe
        ),
        "Combined Maximum Sharpe + finVADER",
    )
    combined_metrics = _save_fusion(
        combined_result,
        fund_returns.query("fund == 'Combined Maximum Sharpe'"),
        fund_metrics.query("fund == 'Combined Maximum Sharpe'"),
        (
            COMBINED_RETURNS,
            COMBINED_WEIGHTS,
            COMBINED_METRICS,
            COMBINED_SECTOR_DIAGNOSTICS,
            COMBINED_REBALANCE_DIAGNOSTICS,
        ),
    )
    figures.save_primary_finvader_sentiment_figure(robustness.sector_index, PRIMARY_MONTHLY_FIGURE)
    primary_comparison = finvader_robustness.primary_fusion_comparison(
        pd.read_csv(RESULTS_TABLES / "equity_equal_weight_sentiment_metrics.csv"),
        equity_metrics,
        pd.read_csv(RESULTS_TABLES / "combined_maximum_sharpe_sentiment_metrics.csv"),
        combined_metrics,
    )
    primary_comparison.to_csv(PRIMARY_FUSION_COMPARISON, index=False)

    print(f"Headline scores: {len(robustness.headline_scores):,}; excluded={excluded}")
    print(f"Ticker-days: {len(robustness.ticker_days):,}")
    print(f"Sector-days: {len(robustness.sector_index):,}")
    print("\nVADER versus finVADER")
    print(robustness.comparison.to_string(index=False))
    print("\nEquity Equal Weight")
    print(equity_metrics.to_string(index=False))
    print("\nCombined Maximum Sharpe")
    print(combined_metrics.to_string(index=False))
    for path in (
        HEADLINE_SCORES,
        TICKER_DAYS,
        SECTOR_INDEX,
        MONTHLY_FIGURE,
        PRIMARY_MONTHLY_FIGURE,
        COMPARISON,
        PRIMARY_FUSION_COMPARISON,
        EQUITY_RETURNS,
        EQUITY_WEIGHTS,
        EQUITY_METRICS,
        EQUITY_SECTOR_DIAGNOSTICS,
        EQUITY_REBALANCE_DIAGNOSTICS,
        COMBINED_RETURNS,
        COMBINED_WEIGHTS,
        COMBINED_METRICS,
        COMBINED_SECTOR_DIAGNOSTICS,
        COMBINED_REBALANCE_DIAGNOSTICS,
    ):
        print(f"Saved: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
