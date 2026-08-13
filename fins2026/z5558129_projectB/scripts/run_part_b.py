"""Reproduce the Part A data foundation required by Part B.

Run from the project root with ``python scripts/run_part_b.py``.
"""
from __future__ import annotations

import importlib
import json
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

etl = importlib.import_module("src.etl")
features = importlib.import_module("src.features")
figures = importlib.import_module("src.figures")
fusion = importlib.import_module("src.fusion")
portfolios = importlib.import_module("src.portfolios")
sentiment = importlib.import_module("src.sentiment")
suitability = importlib.import_module("src.suitability")

RESULTS_DATA = ROOT / "results" / "data"
COMBINED_RETURNS_PATH = RESULTS_DATA / "combined_returns_panel.csv"
HEADLINE_PANEL_PATH = RESULTS_DATA / "daily_headline_panel.csv"
FUND_RETURNS_PATH = RESULTS_DATA / "fund_returns.csv"
FUND_WEIGHTS_PATH = RESULTS_DATA / "fund_weights.csv"
RESULTS_TABLES = ROOT / "results" / "tables"
RESULTS_FIGURES = ROOT / "results" / "figures"
FUND_METRICS_PATH = RESULTS_TABLES / "performance_metrics.csv"
FUND_DIAGNOSTICS_PATH = RESULTS_TABLES / "fund_diagnostics.csv"
WEIGHT_DISTANCES_PATH = RESULTS_TABLES / "fund_weight_distances.csv"
SECTOR_SENTIMENT_PATH = RESULTS_DATA / "sector_sentiment_index.csv"
FUSION_RETURNS_PATH = RESULTS_DATA / "equity_equal_weight_sentiment_returns.csv"
FUSION_WEIGHTS_PATH = RESULTS_DATA / "equity_equal_weight_sentiment_weights.csv"
FUSION_METRICS_PATH = RESULTS_TABLES / "equity_equal_weight_sentiment_metrics.csv"
FUSION_SECTOR_DIAGNOSTICS_PATH = (
    RESULTS_TABLES / "equity_equal_weight_sentiment_sector_diagnostics.csv"
)
FUSION_REBALANCE_DIAGNOSTICS_PATH = (
    RESULTS_TABLES / "equity_equal_weight_sentiment_rebalance_diagnostics.csv"
)
COMBINED_FUSION_RETURNS_PATH = (
    RESULTS_DATA / "combined_maximum_sharpe_sentiment_returns.csv"
)
COMBINED_FUSION_WEIGHTS_PATH = (
    RESULTS_DATA / "combined_maximum_sharpe_sentiment_weights.csv"
)
COMBINED_FUSION_METRICS_PATH = (
    RESULTS_TABLES / "combined_maximum_sharpe_sentiment_metrics.csv"
)
COMBINED_FUSION_SECTOR_DIAGNOSTICS_PATH = (
    RESULTS_TABLES / "combined_maximum_sharpe_sentiment_sector_diagnostics.csv"
)
COMBINED_FUSION_REBALANCE_DIAGNOSTICS_PATH = (
    RESULTS_TABLES / "combined_maximum_sharpe_sentiment_rebalance_diagnostics.csv"
)
SUITABILITY_CONFIG_PATH = RESULTS_DATA / "suitability_config.json"
SUITABILITY_RECOMMENDATIONS_PATH = RESULTS_DATA / "suitability_recommendations.csv"
FUND_RANKINGS_PATH = RESULTS_TABLES / "fund_rankings.csv"
SUITABILITY_REPLACEMENTS_PATH = RESULTS_TABLES / "suitability_replacements.csv"
SUITABILITY_EXAMPLES_PATH = RESULTS_TABLES / "suitability_examples.csv"

# Part A outputs are external validation benchmarks, not construction inputs.
PART_A_COMBINED_SHAPE = (1_006, 61)
PART_A_HEADLINE_ROWS = 37_977


def main() -> None:
    equities = etl.load_clean_equities()
    crypto = etl.load_clean_crypto()
    headlines = etl.load_clean_headlines()

    equity_returns = features.daily_returns(equities)
    crypto_returns = features.daily_returns(crypto)
    combined = features.combine_returns_on_equity_calendar(
        equity_returns, crypto_returns
    )
    alignment = features.assemble_headline_panel(headlines, equities["date"])

    features.validate_data_foundation(
        combined,
        alignment.panel,
        expected_combined_shape=PART_A_COMBINED_SHAPE,
        expected_headline_rows=PART_A_HEADLINE_ROWS,
    )

    represented_headlines = int(alignment.panel["headline_count"].sum())
    if represented_headlines + len(alignment.excluded) != len(headlines):
        raise ValueError("headline alignment reconciliation failed")

    analyzer = sentiment.get_vader_analyzer(download_if_missing=True)
    scored_ticker_days = sentiment.score_headlines(
        alignment.panel,
        analyzer,
        aligned_headlines=alignment.aligned,
    )
    lagged_ticker_days = sentiment.complete_and_lag_ticker_sentiment(
        scored_ticker_days,
        equities["date"],
        equities[["ticker", "sector"]],
    )
    sector_index = sentiment.sector_sentiment_index(lagged_ticker_days)
    sentiment_figure_path, coverage_figure_path, monthly_sentiment = (
        figures.save_sentiment_figures(sector_index, RESULTS_FIGURES)
    )

    equity_panel = features.returns_to_wide(equity_returns)
    crypto_panel = features.returns_to_wide(crypto_returns)

    equity_backtest = portfolios.oos_backtest(
        equity_panel,
        methods=("equal_weight", "min_variance", "max_sharpe"),
        window=252,
        individual_cap=0.10,
        crypto_cap=None,
        risk_free_rate=0.0,
        transaction_cost=0.0,
        periods_per_year=252,
        method_labels={
            "equal_weight": "Equity Equal Weight",
            "min_variance": "Equity Minimum Variance",
            "max_sharpe": "Equity Maximum Sharpe",
        },
    )
    crypto_backtest = portfolios.oos_backtest(
        crypto_panel,
        methods=("equal_weight", "risk_parity"),
        window=365,
        individual_cap=0.25,
        crypto_cap=None,
        risk_free_rate=0.0,
        transaction_cost=0.0,
        periods_per_year=365,
        method_labels={
            "equal_weight": "Crypto Equal Weight",
            "risk_parity": "Crypto Risk Parity",
        },
    )
    combined_backtest = portfolios.oos_backtest(
        combined,
        methods=("min_variance", "max_sharpe", "risk_parity"),
        window=252,
        individual_cap=0.10,
        crypto_cap=0.25,
        risk_free_rate=0.0,
        transaction_cost=0.0,
        periods_per_year=252,
        method_labels={
            "min_variance": "Combined Minimum Variance",
            "max_sharpe": "Combined Maximum Sharpe",
            "risk_parity": "Combined Risk Parity",
        },
    )
    fusion_result = fusion.run_baseline_fusion(
        equity_panel,
        equity_backtest.weights,
        sector_index,
        equities[["ticker", "sector"]],
    )
    fusion_returns = pd.concat(
        [
            equity_backtest.daily.loc[
                equity_backtest.daily["fund"] == "Equity Equal Weight"
            ],
            fusion_result.daily,
        ],
        ignore_index=True,
    ).sort_values(["date", "fund"]).reset_index(drop=True)
    fusion_metrics = pd.concat(
        [
            equity_backtest.metrics.loc[
                equity_backtest.metrics["fund"] == "Equity Equal Weight"
            ],
            fusion_result.metrics,
        ],
        ignore_index=True,
    )
    combined_fusion_result = fusion.run_combined_max_sharpe_fusion(
        combined,
        combined_backtest.weights,
        sector_index,
        equities[["ticker", "sector"]],
    )
    combined_fusion_returns = pd.concat(
        [
            combined_backtest.daily.loc[
                combined_backtest.daily["fund"] == "Combined Maximum Sharpe"
            ],
            combined_fusion_result.daily,
        ],
        ignore_index=True,
    ).sort_values(["date", "fund"]).reset_index(drop=True)
    combined_fusion_metrics = pd.concat(
        [
            combined_backtest.metrics.loc[
                combined_backtest.metrics["fund"] == "Combined Maximum Sharpe"
            ],
            combined_fusion_result.metrics,
        ],
        ignore_index=True,
    )
    backtests = {
        "Equity": equity_backtest,
        "Crypto": crypto_backtest,
        "Combined": combined_backtest,
    }
    fund_returns = pd.concat(
        [result.daily for result in backtests.values()], ignore_index=True
    ).sort_values(["date", "fund"]).reset_index(drop=True)
    fund_weights = pd.concat(
        [result.weights for result in backtests.values()], ignore_index=True
    ).sort_values(["rebalance_date", "fund", "ticker"]).reset_index(drop=True)
    performance_metrics = pd.concat(
        [result.metrics for result in backtests.values()], ignore_index=True
    ).sort_values("fund").reset_index(drop=True)
    diagnostics = pd.concat(
        [
            result.diagnostics.assign(universe=universe)
            for universe, result in backtests.items()
        ],
        ignore_index=True,
    ).sort_values(["rebalance_date", "fund"]).reset_index(drop=True)
    weight_distances = pd.concat(
        [
            portfolios.method_weight_distances(result.weights).assign(
                universe=universe
            )
            for universe, result in backtests.items()
        ],
        ignore_index=True,
    )
    if not weight_distances["meaningfully_different"].all():
        failed_pairs = weight_distances.loc[
            ~weight_distances["meaningfully_different"], ["fund_a", "fund_b"]
        ]
        raise ValueError(
            "portfolio methods did not produce meaningfully different weights: "
            f"{failed_pairs.to_dict(orient='records')}"
        )
    suitability_grid = suitability.build_recommendation_grid(
        performance_metrics,
        fund_weights,
    )

    RESULTS_DATA.mkdir(parents=True, exist_ok=True)
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    combined.to_csv(COMBINED_RETURNS_PATH, index=False)
    alignment.panel.to_csv(HEADLINE_PANEL_PATH, index=False)
    fund_returns.to_csv(FUND_RETURNS_PATH, index=False)
    fund_weights.to_csv(FUND_WEIGHTS_PATH, index=False)
    performance_metrics.to_csv(FUND_METRICS_PATH, index=False)
    diagnostics.to_csv(FUND_DIAGNOSTICS_PATH, index=False)
    weight_distances.to_csv(WEIGHT_DISTANCES_PATH, index=False)
    sector_index.to_csv(SECTOR_SENTIMENT_PATH, index=False)
    fusion_returns.to_csv(FUSION_RETURNS_PATH, index=False)
    fusion_result.weights.to_csv(FUSION_WEIGHTS_PATH, index=False)
    fusion_metrics.to_csv(FUSION_METRICS_PATH, index=False)
    fusion_result.sector_diagnostics.to_csv(
        FUSION_SECTOR_DIAGNOSTICS_PATH, index=False
    )
    fusion_result.rebalance_diagnostics.to_csv(
        FUSION_REBALANCE_DIAGNOSTICS_PATH, index=False
    )
    combined_fusion_returns.to_csv(COMBINED_FUSION_RETURNS_PATH, index=False)
    combined_fusion_result.weights.to_csv(
        COMBINED_FUSION_WEIGHTS_PATH, index=False
    )
    combined_fusion_metrics.to_csv(COMBINED_FUSION_METRICS_PATH, index=False)
    combined_fusion_result.sector_diagnostics.to_csv(
        COMBINED_FUSION_SECTOR_DIAGNOSTICS_PATH, index=False
    )
    combined_fusion_result.rebalance_diagnostics.to_csv(
        COMBINED_FUSION_REBALANCE_DIAGNOSTICS_PATH, index=False
    )
    SUITABILITY_CONFIG_PATH.write_text(
        json.dumps(suitability.config_as_dict(), indent=2), encoding="utf-8"
    )
    suitability_grid["allocations"].to_csv(
        SUITABILITY_RECOMMENDATIONS_PATH, index=False
    )
    suitability_grid["rankings"].to_csv(FUND_RANKINGS_PATH, index=False)
    suitability_grid["replacements"].to_csv(
        SUITABILITY_REPLACEMENTS_PATH, index=False
    )
    suitability_grid["summary"].to_csv(SUITABILITY_EXAMPLES_PATH, index=False)

    print(f"Combined returns: {combined.shape}; saved to {COMBINED_RETURNS_PATH}")
    print(
        f"Headline panel: {alignment.panel.shape}; "
        f"represents {represented_headlines:,} of {len(headlines):,} cleaned headlines"
    )
    print(f"Excluded after final trading date: {len(alignment.excluded)}")
    if not alignment.excluded.empty:
        print(
            alignment.excluded[["date", "ticker", "sector", "title"]]
            .to_string(index=False)
        )
    print(f"Saved to {HEADLINE_PANEL_PATH}")
    print("\nEight-fund walk-forward backtest")
    for universe, result in backtests.items():
        print(
            f"{universe}: first live {result.daily['date'].min().date()}, "
            f"{result.weights['rebalance_date'].nunique()} rebalances"
        )
    print(performance_metrics.to_string(index=False))
    print("\nPairwise target-weight distances")
    print(weight_distances.to_string(index=False))
    print(f"Saved to {FUND_RETURNS_PATH}")
    print(f"Saved to {FUND_WEIGHTS_PATH}")
    print(f"Saved to {FUND_METRICS_PATH}")
    print(f"Saved to {FUND_DIAGNOSTICS_PATH}")
    print(f"Saved to {WEIGHT_DISTANCES_PATH}")
    print("\nLagged sector sentiment index")
    print(
        f"{len(sector_index)} sector-days from "
        f"{sector_index['trading_date'].min().date()} to "
        f"{sector_index['trading_date'].max().date()}"
    )
    print(f"Saved to {SECTOR_SENTIMENT_PATH}")
    print(f"Saved to {sentiment_figure_path}")
    print(f"Saved to {coverage_figure_path}")
    highlighted = monthly_sentiment.loc[
        monthly_sentiment["extreme_low_coverage"],
        ["month", "sector", "monthly_sentiment", "monthly_coverage"],
    ]
    print("Extreme-sentiment months with unusually low sector coverage")
    print(highlighted.to_string(index=False))
    print("\nBaseline Equity Equal Weight sentiment fusion")
    print(fusion_metrics.to_string(index=False))
    blocked = int(
        fusion_result.sector_diagnostics["coverage_rule_blocked"].sum()
    )
    sector_decisions = len(fusion_result.sector_diagnostics)
    print(
        f"Coverage rule blocked {blocked}/{sector_decisions} "
        "sector-rebalance tilts"
    )
    print(
        "Applied multiplier range: "
        f"{fusion_result.sector_diagnostics['applied_multiplier'].min():.6f} to "
        f"{fusion_result.sector_diagnostics['applied_multiplier'].max():.6f}"
    )
    print(
        "Average absolute asset-weight change: "
        f"{fusion_result.rebalance_diagnostics['mean_absolute_asset_weight_change'].mean():.6f}"
    )
    print(
        "Average half-L1 distance: "
        f"{fusion_result.rebalance_diagnostics['half_l1_weight_distance'].mean():.6f}"
    )
    print(
        "10% cap binding rebalances: "
        f"{int(fusion_result.rebalance_diagnostics['cap_bound'].sum())}"
    )
    print(f"Saved to {FUSION_RETURNS_PATH}")
    print(f"Saved to {FUSION_WEIGHTS_PATH}")
    print(f"Saved to {FUSION_METRICS_PATH}")
    print("\nCombined Maximum Sharpe sentiment fusion")
    print(combined_fusion_metrics.to_string(index=False))
    combined_blocked = int(
        combined_fusion_result.sector_diagnostics[
            "coverage_rule_blocked"
        ].sum()
    )
    print(
        f"Coverage rule blocked {combined_blocked}/"
        f"{len(combined_fusion_result.sector_diagnostics)} sector-rebalance tilts"
    )
    print(
        "Applied multiplier range: "
        f"{combined_fusion_result.sector_diagnostics['applied_multiplier'].min():.6f} "
        "to "
        f"{combined_fusion_result.sector_diagnostics['applied_multiplier'].max():.6f}"
    )
    print(
        "Average absolute equity-weight change: "
        f"{combined_fusion_result.rebalance_diagnostics['mean_absolute_equity_weight_change'].mean():.6f}"
    )
    print(
        "Crypto preservation maximum error: "
        f"{combined_fusion_result.rebalance_diagnostics['crypto_preservation_error'].max():.12f}"
    )
    print(
        "10% redistribution-cap binding rebalances: "
        f"{int(combined_fusion_result.rebalance_diagnostics['cap_bound'].sum())}"
    )
    print(f"Saved to {COMBINED_FUSION_RETURNS_PATH}")
    print(f"Saved to {COMBINED_FUSION_WEIGHTS_PATH}")
    print(f"Saved to {COMBINED_FUSION_METRICS_PATH}")
    print("\nIllustrative suitability recommendation grid")
    print(suitability_grid["summary"].to_string(index=False))
    print(f"Saved to {SUITABILITY_CONFIG_PATH}")
    print(f"Saved to {SUITABILITY_RECOMMENDATIONS_PATH}")
    print(f"Saved to {FUND_RANKINGS_PATH}")
    print(f"Saved to {SUITABILITY_REPLACEMENTS_PATH}")
    print(f"Saved to {SUITABILITY_EXAMPLES_PATH}")


if __name__ == "__main__":
    main()
