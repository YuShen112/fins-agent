"""Baseline VADER ticker-day and lagged sector sentiment construction."""
from __future__ import annotations

from typing import Protocol

import pandas as pd


class SentimentAnalyzer(Protocol):
    """Minimal interface supplied by VADER and test doubles."""

    def polarity_scores(self, text: str) -> dict[str, float]: ...


def get_vader_analyzer(*, download_if_missing: bool = False) -> SentimentAnalyzer:
    """Create NLTK's VADER analyzer, optionally downloading its lexicon."""
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer

    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        if not download_if_missing:
            raise RuntimeError(
                "VADER lexicon is unavailable; run nltk.download('vader_lexicon')"
            ) from None
        if not nltk.download("vader_lexicon", quiet=True):
            raise RuntimeError("NLTK could not download the VADER lexicon")
        return SentimentIntensityAnalyzer()


def score_headlines(
    panel: pd.DataFrame,
    analyzer: SentimentAnalyzer,
    aligned_headlines: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Score original headlines and average their scores within each ticker-day.

    ``aligned_headlines`` avoids parsing the panel's display text when an original
    title itself contains a newline. Its group counts are reconciled to ``panel``.
    """
    required = {
        "trading_date", "ticker", "sector", "headline_count", "headline_text"
    }
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"headline panel is missing columns: {sorted(missing)}")
    if panel.duplicated(["trading_date", "ticker"]).any():
        raise ValueError("headline panel contains duplicate trading_date-ticker rows")

    headline_groups: dict[tuple[pd.Timestamp, str], list[str]] = {}
    if aligned_headlines is not None:
        aligned_required = {"trading_date", "ticker", "sector", "title"}
        if missing := aligned_required.difference(aligned_headlines.columns):
            raise ValueError(f"aligned headlines is missing columns: {sorted(missing)}")
        for (date, ticker), group in aligned_headlines.groupby(
            ["trading_date", "ticker"], sort=False
        ):
            headline_groups[(pd.Timestamp(date), ticker)] = group["title"].tolist()

    records: list[dict[str, object]] = []
    for row in panel.itertuples(index=False):
        key = (pd.Timestamp(row.trading_date), row.ticker)
        headlines = (
            headline_groups[key]
            if aligned_headlines is not None
            else str(row.headline_text).splitlines()
        )
        if len(headlines) != int(row.headline_count):
            raise ValueError(
                f"{row.trading_date} {row.ticker}: headline_count is "
                f"{row.headline_count}, but {len(headlines)} original titles were found"
            )
        scores = [analyzer.polarity_scores(title)["compound"] for title in headlines]
        records.append(
            {
                "trading_date": pd.Timestamp(row.trading_date),
                "ticker": row.ticker,
                "sector": row.sector,
                "headline_count": len(headlines),
                "has_headline": True,
                "ticker_sentiment": float(sum(scores) / len(scores)),
            }
        )
    return pd.DataFrame(records).sort_values(
        ["trading_date", "ticker"]
    ).reset_index(drop=True)


def complete_and_lag_ticker_sentiment(
    scored_ticker_days: pd.DataFrame,
    trading_dates: pd.Series | pd.Index,
    ticker_universe: pd.DataFrame,
) -> pd.DataFrame:
    """Add no-news ticker-days as neutral and lag signals one trading day."""
    universe_required = {"ticker", "sector"}
    if missing := universe_required.difference(ticker_universe.columns):
        raise ValueError(f"ticker_universe is missing columns: {sorted(missing)}")
    universe = ticker_universe[["ticker", "sector"]].drop_duplicates()
    if universe["ticker"].duplicated().any():
        raise ValueError("ticker_universe maps a ticker to multiple sectors")

    dates = pd.DatetimeIndex(pd.to_datetime(pd.Index(trading_dates))).drop_duplicates()
    dates = dates.sort_values()
    if dates.empty:
        raise ValueError("trading_dates must not be empty")
    grid = pd.MultiIndex.from_product(
        [dates, universe["ticker"]], names=["trading_date", "ticker"]
    ).to_frame(index=False)
    grid = grid.merge(universe, on="ticker", how="left", validate="many_to_one")
    observed = scored_ticker_days.drop(columns="sector").copy()
    observed["trading_date"] = pd.to_datetime(observed["trading_date"])
    complete = grid.merge(
        observed,
        on=["trading_date", "ticker"],
        how="left",
        validate="one_to_one",
    )
    complete["headline_count"] = complete["headline_count"].fillna(0).astype(int)
    complete["has_headline"] = complete["has_headline"].fillna(False).astype(bool)
    complete["ticker_sentiment"] = complete["ticker_sentiment"].fillna(0.0)
    complete = complete.sort_values(["ticker", "trading_date"]).reset_index(drop=True)
    grouped = complete.groupby("ticker", sort=False)
    complete["signal_source_date"] = grouped["trading_date"].shift(1)
    complete["lagged_ticker_sentiment"] = grouped["ticker_sentiment"].shift(1)
    complete["lagged_has_headline"] = grouped["has_headline"].shift(1)
    complete["lagged_headline_count"] = grouped["headline_count"].shift(1)
    return complete.sort_values(["trading_date", "ticker"]).reset_index(drop=True)


def sector_sentiment_index(lagged_ticker_days: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight lagged ticker signals and report source-day news coverage."""
    required = {
        "trading_date",
        "signal_source_date",
        "ticker",
        "sector",
        "lagged_ticker_sentiment",
        "lagged_has_headline",
        "lagged_headline_count",
    }
    if missing := required.difference(lagged_ticker_days.columns):
        raise ValueError(f"lagged ticker panel is missing columns: {sorted(missing)}")

    index = (
        lagged_ticker_days.groupby(["trading_date", "sector"], as_index=False)
        .agg(
            signal_source_date=("signal_source_date", "first"),
            sector_sentiment_index=("lagged_ticker_sentiment", "mean"),
            constituent_ticker_count=("ticker", "nunique"),
            covered_ticker_count=("lagged_has_headline", "sum"),
            source_headline_count=("lagged_headline_count", "sum"),
        )
        .sort_values(["trading_date", "sector"])
        .reset_index(drop=True)
    )
    index["covered_ticker_proportion"] = (
        index["covered_ticker_count"] / index["constituent_ticker_count"]
    )
    index["covered_ticker_count"] = index["covered_ticker_count"].astype("Int64")
    index["source_headline_count"] = index["source_headline_count"].astype("Int64")
    no_source = index["signal_source_date"].isna()
    index.loc[no_source, "covered_ticker_count"] = pd.NA
    index.loc[no_source, "covered_ticker_proportion"] = float("nan")
    index.loc[no_source, "source_headline_count"] = pd.NA
    return index[
        [
            "trading_date",
            "signal_source_date",
            "sector",
            "sector_sentiment_index",
            "constituent_ticker_count",
            "covered_ticker_count",
            "covered_ticker_proportion",
            "source_headline_count",
        ]
    ]
