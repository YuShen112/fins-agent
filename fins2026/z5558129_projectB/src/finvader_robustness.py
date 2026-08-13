"""finVADER robustness analysis without changing canonical VADER artifacts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import figures, sentiment


@dataclass(frozen=True)
class FinVaderResult:
    headline_scores: pd.DataFrame
    ticker_days: pd.DataFrame
    sector_index: pd.DataFrame
    monthly: pd.DataFrame
    comparison: pd.DataFrame


class FinVaderAnalyzer:
    """Efficient equivalent of finVADER with both finance lexicons enabled."""

    def __init__(
        self,
        *,
        use_sentibignomics: bool = True,
        use_henry: bool = True,
    ) -> None:
        if not (use_sentibignomics and use_henry):
            raise ValueError("the robustness check requires both finVADER lexicons")
        from finvader.Henry import lexicon2
        from finvader.SentiBignomics import lexicon1

        self.use_sentibignomics = use_sentibignomics
        self.use_henry = use_henry
        self._analyzer = sentiment.get_vader_analyzer(download_if_missing=True)
        sentibignomics = {key: value * 0.1 for key, value in lexicon1().items()}
        combined_lexicon = {**sentibignomics, **lexicon2()}
        self._analyzer.lexicon.update(combined_lexicon)

    def polarity_scores(self, text: str) -> dict[str, float]:
        return self._analyzer.polarity_scores(text)


def validate_against_public_finvader(analyzer: FinVaderAnalyzer) -> None:
    """Confirm the cached analyzer matches both public finVADER switches."""
    from finvader import finvader

    examples = (
        "Profits rose strongly after the company beat expectations.",
        "The firm warned of losses and deteriorating liquidity.",
        "The board announced its quarterly meeting date.",
    )
    for text in examples:
        public_score = finvader(
            text,
            indicator="compound",
            use_sentibignomics=True,
            use_henry=True,
        )
        cached_score = analyzer.polarity_scores(text)["compound"]
        if not np.isclose(public_score, cached_score, atol=1e-12):
            raise ValueError("cached analyzer does not match public finVADER")


def _headline_id(row: pd.Series) -> str:
    key = f"{row['ticker']}\x1f{pd.Timestamp(row['date']).isoformat()}\x1f{row['title']}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def score_deduplicated_headlines(
    headlines: pd.DataFrame,
    trading_dates: pd.Series | pd.Index,
    *,
    vader_analyzer: sentiment.SentimentAnalyzer,
    finvader_analyzer: sentiment.SentimentAnalyzer,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every exact-deduplicated headline and retain alignment separately."""
    required = {"date", "ticker", "sector", "title"}
    if missing := required.difference(headlines.columns):
        raise ValueError(f"headlines are missing columns: {sorted(missing)}")
    clean = headlines.drop_duplicates(["ticker", "date", "title"]).copy()
    clean["date"] = pd.to_datetime(clean["date"])
    if clean["title"].isna().any() or clean["title"].eq("").any():
        raise ValueError("finVADER requires non-empty headline text")

    clean["vader_compound"] = [
        vader_analyzer.polarity_scores(title)["compound"] for title in clean["title"]
    ]
    clean["finvader_compound"] = [
        finvader_analyzer.polarity_scores(title)["compound"] for title in clean["title"]
    ]
    clean["headline_id"] = clean.apply(_headline_id, axis=1)

    calendar = pd.DatetimeIndex(pd.to_datetime(pd.Index(trading_dates))).sort_values()
    positions = calendar.searchsorted(clean["date"], side="left")
    alignable = positions < len(calendar)
    clean["trading_date"] = pd.NaT
    clean.loc[alignable, "trading_date"] = calendar.take(positions[alignable]).to_numpy()
    aligned = clean.loc[alignable].copy()
    cache = clean[
        [
            "headline_id",
            "date",
            "trading_date",
            "ticker",
            "sector",
            "vader_compound",
            "finvader_compound",
        ]
    ].sort_values(["date", "ticker", "headline_id"])
    return cache.reset_index(drop=True), aligned.reset_index(drop=True)


def aggregate_finvader_ticker_days(aligned_scores: pd.DataFrame) -> pd.DataFrame:
    """Average individual finVADER scores within ticker-trading-day groups."""
    return (
        aligned_scores.groupby(["trading_date", "ticker", "sector"], as_index=False)
        .agg(
            headline_count=("finvader_compound", "size"),
            ticker_sentiment=("finvader_compound", "mean"),
        )
        .assign(has_headline=True)
        .sort_values(["trading_date", "ticker"])
        .reset_index(drop=True)
    )


def comparison_table(
    headline_scores: pd.DataFrame,
    vader_sector_index: pd.DataFrame,
    finvader_sector_index: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize headline and sector-month agreement between sentiment models."""
    rows: list[dict[str, object]] = [
        {
            "scope": "Headline",
            "sector": "All",
            "metric": "VADER vs finVADER correlation",
            "value": headline_scores["vader_compound"].corr(headline_scores["finvader_compound"]),
        },
        {
            "scope": "Headline",
            "sector": "All",
            "metric": "VADER exact-zero proportion",
            "value": headline_scores["vader_compound"].eq(0).mean(),
        },
        {
            "scope": "Headline",
            "sector": "All",
            "metric": "finVADER exact-zero proportion",
            "value": headline_scores["finvader_compound"].eq(0).mean(),
        },
        {
            "scope": "Headline",
            "sector": "All",
            "metric": "Mean absolute score difference",
            "value": (headline_scores["vader_compound"] - headline_scores["finvader_compound"])
            .abs()
            .mean(),
        },
    ]
    vader_monthly = figures.monthly_sentiment_summary(vader_sector_index)[
        ["month", "sector", "monthly_sentiment"]
    ].rename(columns={"monthly_sentiment": "vader_monthly_sentiment"})
    finvader_monthly = figures.monthly_sentiment_summary(finvader_sector_index)[
        ["month", "sector", "monthly_sentiment"]
    ].rename(columns={"monthly_sentiment": "finvader_monthly_sentiment"})
    monthly = vader_monthly.merge(
        finvader_monthly,
        on=["month", "sector"],
        how="inner",
        validate="one_to_one",
    )
    rows.extend(
        [
            {
                "scope": "Sector-month",
                "sector": "All",
                "metric": "VADER vs finVADER correlation",
                "value": monthly["vader_monthly_sentiment"].corr(
                    monthly["finvader_monthly_sentiment"]
                ),
            },
            {
                "scope": "Sector-month",
                "sector": "All",
                "metric": "Mean absolute score difference",
                "value": (
                    monthly["vader_monthly_sentiment"] - monthly["finvader_monthly_sentiment"]
                )
                .abs()
                .mean(),
            },
        ]
    )
    for sector, group in monthly.groupby("sector", sort=True):
        rows.append(
            {
                "scope": "Sector-month",
                "sector": sector,
                "metric": "VADER vs finVADER correlation",
                "value": group["vader_monthly_sentiment"].corr(group["finvader_monthly_sentiment"]),
            }
        )
    return pd.DataFrame(rows)


def primary_fusion_comparison(
    equity_vader: pd.DataFrame,
    equity_finvader: pd.DataFrame,
    combined_vader: pd.DataFrame,
    combined_finvader: pd.DataFrame,
) -> pd.DataFrame:
    """Consolidate base, primary finVADER, and VADER benchmark fusion metrics."""
    metric_columns = [
        "observations",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "maximum_drawdown",
        "final_growth_of_1",
    ]
    experiments = (
        (
            "Equity Equal Weight",
            "Equity Equal Weight",
            equity_vader,
            equity_finvader,
        ),
        (
            "Combined Maximum Sharpe",
            "Combined Maximum Sharpe",
            combined_vader,
            combined_finvader,
        ),
    )
    records: list[dict[str, object]] = []
    for experiment, base_fund, vader, finvader in experiments:
        base = finvader.loc[finvader["fund"] == base_fund]
        primary = finvader.loc[finvader["fund"] != base_fund]
        benchmark = vader.loc[vader["fund"] != base_fund]
        if len(base) != 1 or len(primary) != 1 or len(benchmark) != 1:
            raise ValueError(f"{experiment}: expected one row per comparison role")
        for role, model, row in (
            ("Base", "No sentiment tilt", base.iloc[0]),
            ("Primary", "finVADER tilt", primary.iloc[0]),
            ("Benchmark", "VADER tilt", benchmark.iloc[0]),
        ):
            records.append(
                {
                    "experiment": experiment,
                    "model_role": role,
                    "model": model,
                    "fund": row["fund"],
                    **{column: row[column] for column in metric_columns},
                }
            )
    return pd.DataFrame(records)


def run_robustness_check(
    headlines: pd.DataFrame,
    trading_dates: pd.Series | pd.Index,
    ticker_universe: pd.DataFrame,
    vader_sector_index: pd.DataFrame,
    figure_path: str | Path,
) -> FinVaderResult:
    """Run the full parallel finVADER pipeline and comparison."""
    finvader_analyzer = FinVaderAnalyzer(
        use_sentibignomics=True,
        use_henry=True,
    )
    validate_against_public_finvader(finvader_analyzer)
    headline_scores, aligned = score_deduplicated_headlines(
        headlines,
        trading_dates,
        vader_analyzer=sentiment.get_vader_analyzer(download_if_missing=True),
        finvader_analyzer=finvader_analyzer,
    )
    observed = aggregate_finvader_ticker_days(aligned)
    ticker_days = sentiment.complete_and_lag_ticker_sentiment(
        observed, trading_dates, ticker_universe
    )
    sector_index = sentiment.sector_sentiment_index(ticker_days)
    monthly = figures.save_finvader_sentiment_figure(sector_index, figure_path)
    comparison = comparison_table(headline_scores, vader_sector_index, sector_index)
    return FinVaderResult(headline_scores, ticker_days, sector_index, monthly, comparison)
