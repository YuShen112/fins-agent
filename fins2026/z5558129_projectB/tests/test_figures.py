"""Tests for monthly sentiment aggregation and report figure export."""

from __future__ import annotations

import pandas as pd
from src import figures


def sample_sector_index() -> pd.DataFrame:
    """Create enough monthly history for sector-relative flag calculations."""
    dates = pd.date_range("2020-01-02", "2021-12-31", freq="B")
    rows = []
    for sector, offset in (("Tech", 0.0), ("Energy", 0.02)):
        for date in dates:
            score = 0.05 + offset
            coverage = 0.8
            if date.to_period("M") == pd.Period("2021-06", "M"):
                score = 0.8 + offset
                coverage = 0.2
            rows.append(
                {
                    "trading_date": date,
                    "sector": sector,
                    "sector_sentiment_index": score,
                    "covered_ticker_proportion": coverage,
                    "source_headline_count": 5,
                }
            )
    return pd.DataFrame(rows)


def test_monthly_summary_preserves_sectors_and_flags_low_coverage_extreme() -> None:
    monthly = figures.monthly_sentiment_summary(sample_sector_index())
    assert len(monthly) == 48
    assert monthly["sector"].nunique() == 2
    flagged = monthly.loc[monthly["extreme_low_coverage"]]
    assert set(flagged["month"].dt.strftime("%Y-%m")) == {"2021-06"}
    assert set(flagged["sector"]) == {"Tech", "Energy"}


def test_sentiment_figures_are_exported(tmp_path) -> None:
    sentiment_path, coverage_path, _ = figures.save_sentiment_figures(
        sample_sector_index(), tmp_path
    )
    assert sentiment_path.exists() and sentiment_path.stat().st_size > 10_000
    assert coverage_path.exists() and coverage_path.stat().st_size > 10_000


def test_primary_finvader_figure_is_exported_separately(tmp_path) -> None:
    primary_path = tmp_path / "sector_sentiment_monthly_primary.png"
    figures.save_primary_finvader_sentiment_figure(sample_sector_index(), primary_path)
    assert primary_path.exists() and primary_path.stat().st_size > 10_000
    assert not (tmp_path / "sector_sentiment_monthly.png").exists()
