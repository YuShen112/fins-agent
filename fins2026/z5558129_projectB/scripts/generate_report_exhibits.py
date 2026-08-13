"""Generate the static portfolio and coverage exhibits used by the report."""

from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import report_figures  # noqa: E402

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


def main() -> None:
    returns = pd.read_csv(RESULTS / "data/fund_returns.csv", parse_dates=["date"])
    metrics = pd.read_csv(RESULTS / "tables/performance_metrics.csv")
    sentiment = pd.read_csv(
        RESULTS / "data/finvader_sector_sentiment_index.csv",
        parse_dates=["trading_date", "signal_source_date"],
    )
    paths = (
        report_figures.save_growth_figure(
            returns,
            FIGURES / "fund_growth_all.png",
            include_crypto_only=True,
        ),
        report_figures.save_growth_figure(
            returns,
            FIGURES / "fund_growth_excluding_crypto.png",
            include_crypto_only=False,
        ),
        report_figures.save_return_volatility_scatter(
            metrics, FIGURES / "fund_return_volatility_scatter.png"
        ),
        report_figures.save_drawdown_comparison(metrics, FIGURES / "fund_maximum_drawdown.png"),
        report_figures.save_coverage_report_figure(
            sentiment, FIGURES / "sector_coverage_monthly.png"
        ),
    )
    for path in paths:
        print(f"Saved: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
