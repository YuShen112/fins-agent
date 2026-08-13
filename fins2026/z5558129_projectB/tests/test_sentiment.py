"""Tests for headline scoring, no-news handling, lagging, and sector coverage."""
from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
from src import sentiment


class FakeAnalyzer:
    """Deterministic analyzer that makes aggregation behavior explicit."""

    scores: ClassVar[dict[str, float]] = {
        "Good": 0.8,
        "Bad": -0.4,
        "Neutral": 0.0,
        "Line one\nLine two": 0.3,
    }

    def polarity_scores(self, text: str) -> dict[str, float]:
        return {"compound": self.scores[text]}


def test_multiple_headlines_are_scored_separately_then_averaged() -> None:
    panel = pd.DataFrame(
        {
            "trading_date": pd.to_datetime(["2020-01-02"]),
            "ticker": ["AAA"],
            "sector": ["Tech"],
            "headline_count": [2],
            "headline_text": ["Good\nBad"],
        }
    )
    scored = sentiment.score_headlines(panel, FakeAnalyzer())
    assert np.isclose(scored.loc[0, "ticker_sentiment"], 0.2)
    assert scored.loc[0, "has_headline"]


def test_original_titles_resolve_embedded_newline_ambiguity() -> None:
    panel = pd.DataFrame(
        {
            "trading_date": pd.to_datetime(["2020-01-02"]),
            "ticker": ["AAA"],
            "sector": ["Tech"],
            "headline_count": [2],
            "headline_text": ["Line one\nLine two\nGood"],
        }
    )
    aligned = pd.DataFrame(
        {
            "trading_date": pd.to_datetime(["2020-01-02", "2020-01-02"]),
            "ticker": ["AAA", "AAA"],
            "sector": ["Tech", "Tech"],
            "title": ["Line one\nLine two", "Good"],
        }
    )
    scored = sentiment.score_headlines(
        panel, FakeAnalyzer(), aligned_headlines=aligned
    )
    assert np.isclose(scored.loc[0, "ticker_sentiment"], 0.55)


def test_no_news_is_distinct_from_observed_neutral_and_lagged() -> None:
    panel = pd.DataFrame(
        {
            "trading_date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "ticker": ["AAA", "BBB"],
            "sector": ["Tech", "Tech"],
            "headline_count": [1, 1],
            "headline_text": ["Neutral", "Good"],
        }
    )
    scored = sentiment.score_headlines(panel, FakeAnalyzer())
    universe = pd.DataFrame(
        {"ticker": ["AAA", "BBB"], "sector": ["Tech", "Tech"]}
    )
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    ticker_days = sentiment.complete_and_lag_ticker_sentiment(
        scored, dates, universe
    )

    aaa_first = ticker_days.query("ticker == 'AAA'").iloc[0]
    bbb_first = ticker_days.query("ticker == 'BBB'").iloc[0]
    assert aaa_first["ticker_sentiment"] == bbb_first["ticker_sentiment"] == 0.0
    assert bool(aaa_first["has_headline"])
    assert not bool(bbb_first["has_headline"])

    monday = ticker_days.loc[ticker_days["trading_date"] == pd.Timestamp("2020-01-06")]
    assert monday.set_index("ticker").loc["AAA", "lagged_ticker_sentiment"] == 0.0
    assert monday.set_index("ticker").loc["BBB", "lagged_ticker_sentiment"] == 0.8
    assert monday["signal_source_date"].eq(pd.Timestamp("2020-01-03")).all()


def test_sector_index_equal_weights_tickers_and_reports_coverage() -> None:
    lagged = pd.DataFrame(
        {
            "trading_date": pd.to_datetime(["2020-01-03"] * 2),
            "signal_source_date": pd.to_datetime(["2020-01-02"] * 2),
            "ticker": ["AAA", "BBB"],
            "sector": ["Tech", "Tech"],
            "lagged_ticker_sentiment": [0.8, 0.0],
            "lagged_has_headline": [True, False],
            "lagged_headline_count": [4, 0],
        }
    )
    index = sentiment.sector_sentiment_index(lagged)
    assert np.isclose(index.loc[0, "sector_sentiment_index"], 0.4)
    assert index.loc[0, "covered_ticker_count"] == 1
    assert index.loc[0, "covered_ticker_proportion"] == 0.5
    assert index.loc[0, "source_headline_count"] == 4
