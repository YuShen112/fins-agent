"""Focused tests for the parallel finVADER robustness pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src import finvader_robustness


class FakeAnalyzer:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def polarity_scores(self, text: str) -> dict[str, float]:
        return {"compound": self.scores[text]}


def test_headline_scores_are_individual_and_original_titles_are_not_split() -> None:
    headlines = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-02"]),
            "ticker": ["AAA", "AAA"],
            "sector": ["Tech", "Tech"],
            "title": ["Line one\nLine two", "Good"],
        }
    )
    cache, aligned = finvader_robustness.score_deduplicated_headlines(
        headlines,
        pd.to_datetime(["2020-01-02"]),
        vader_analyzer=FakeAnalyzer({"Line one\nLine two": 0.1, "Good": 0.8}),
        finvader_analyzer=FakeAnalyzer({"Line one\nLine two": 0.2, "Good": 0.6}),
    )
    assert len(cache) == len(aligned) == 2
    assert cache["headline_id"].nunique() == 2
    daily = finvader_robustness.aggregate_finvader_ticker_days(aligned)
    assert np.isclose(daily.loc[0, "ticker_sentiment"], 0.4)
    assert daily.loc[0, "headline_count"] == 2


def test_unalignable_headline_is_scored_but_excluded_from_daily_aggregation() -> None:
    headlines = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-03", "2020-01-04"]),
            "ticker": ["AAA", "AAA"],
            "sector": ["Tech", "Tech"],
            "title": ["Inside", "After"],
        }
    )
    analyzer = FakeAnalyzer({"Inside": 0.1, "After": -0.2})
    cache, aligned = finvader_robustness.score_deduplicated_headlines(
        headlines,
        pd.to_datetime(["2020-01-03"]),
        vader_analyzer=analyzer,
        finvader_analyzer=analyzer,
    )
    assert len(cache) == 2
    assert cache["trading_date"].isna().sum() == 1
    assert len(aligned) == 1


def test_primary_fusion_table_keeps_finvader_and_vader_roles() -> None:
    columns = {
        "observations": 753,
        "annualized_return": 0.1,
        "annualized_volatility": 0.2,
        "sharpe_ratio": 0.5,
        "maximum_drawdown": -0.2,
        "final_growth_of_1": 1.3,
    }

    def metrics(base: str, tilt: str) -> pd.DataFrame:
        return pd.DataFrame([{"fund": base, **columns}, {"fund": tilt, **columns}])

    result = finvader_robustness.primary_fusion_comparison(
        metrics("Equity Equal Weight", "Equity Equal Weight + Sentiment"),
        metrics("Equity Equal Weight", "Equity Equal Weight + finVADER"),
        metrics(
            "Combined Maximum Sharpe",
            "Combined Maximum Sharpe + Sentiment",
        ),
        metrics(
            "Combined Maximum Sharpe",
            "Combined Maximum Sharpe + finVADER",
        ),
    )
    assert len(result) == 6
    assert set(result["model_role"]) == {"Base", "Primary", "Benchmark"}
    assert set(result.loc[result["model_role"] == "Primary", "model"]) == {"finVADER tilt"}
