"""Tests for the reproducible A4 report exhibit suite."""

from pathlib import Path

import pandas as pd
from PIL import Image
from src import report_figures

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"


def test_portfolio_and_coverage_exports_have_fixed_report_dimensions(tmp_path) -> None:
    returns = pd.read_csv(RESULTS / "data/fund_returns.csv", parse_dates=["date"])
    metrics = pd.read_csv(RESULTS / "tables/performance_metrics.csv")
    sentiment = pd.read_csv(
        RESULTS / "data/finvader_sector_sentiment_index.csv",
        parse_dates=["trading_date", "signal_source_date"],
    )
    paths = [
        report_figures.save_growth_figure(returns, tmp_path / "all.png", include_crypto_only=True),
        report_figures.save_growth_figure(
            returns, tmp_path / "no_crypto.png", include_crypto_only=False
        ),
        report_figures.save_growth_figure(
            returns,
            tmp_path / "no_crypto_linear.png",
            include_crypto_only=False,
            y_scale="linear",
        ),
        report_figures.save_return_volatility_scatter(metrics, tmp_path / "scatter.png"),
        report_figures.save_drawdown_comparison(metrics, tmp_path / "drawdown.png"),
        report_figures.save_coverage_report_figure(sentiment, tmp_path / "coverage.png"),
    ]
    for path in paths:
        assert path.exists() and path.stat().st_size > 20_000
        with Image.open(path) as image:
            assert image.size == (3000, 1800)


def test_growth_figure_rejects_unknown_display_scale(tmp_path) -> None:
    returns = pd.read_csv(RESULTS / "data/fund_returns.csv", parse_dates=["date"])

    try:
        report_figures.save_growth_figure(
            returns,
            tmp_path / "invalid.png",
            include_crypto_only=False,
            y_scale="unsupported",
        )
    except ValueError as exc:
        assert str(exc) == "y_scale must be 'linear' or 'log'"
    else:
        raise AssertionError("Expected an unsupported display scale to fail")
