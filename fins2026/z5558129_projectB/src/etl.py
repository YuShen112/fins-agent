"""Reusable loading and cleaning for the Part B data foundation."""
from __future__ import annotations

import pandas as pd

from src import data_access

SAMPLE_START = pd.Timestamp("2020-01-01")
SAMPLE_END = pd.Timestamp("2023-12-31")


def normalise_sample_dates(
    frame: pd.DataFrame,
    start: pd.Timestamp = SAMPLE_START,
    end: pd.Timestamp = SAMPLE_END,
) -> pd.DataFrame:
    """Return a copy with timezone-naive dates restricted to the sample."""
    if "date" not in frame.columns:
        raise ValueError("frame is missing required column: date")
    clean = frame.copy()
    clean["date"] = pd.to_datetime(clean["date"], utc=True).dt.tz_localize(None)
    return clean.loc[clean["date"].between(start, end, inclusive="both")].copy()


def clean_prices(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise a price panel and enforce one observation per ticker-date."""
    required = {"ticker", "date", "adjClose"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"price frame is missing required columns: {sorted(missing)}")
    clean = normalise_sample_dates(frame)
    clean = clean.drop_duplicates(["ticker", "date"], keep="first")
    return clean.sort_values(["ticker", "date"]).reset_index(drop=True)


def clean_headlines(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise headlines and remove exact ticker-date-title duplicates."""
    required = {"ticker", "date", "sector", "title"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"headline frame is missing required columns: {sorted(missing)}"
        )
    clean = normalise_sample_dates(frame)
    clean = clean.drop_duplicates(["ticker", "date", "title"], keep="first")
    return clean.sort_values(["ticker", "date", "title"]).reset_index(drop=True)


def load_clean_equities() -> pd.DataFrame:
    """Load and clean equity prices through the provided data-access helper."""
    return clean_prices(data_access.load_equity_prices())


def load_clean_crypto() -> pd.DataFrame:
    """Load and clean crypto prices, including removal of 2024 spillover rows."""
    return clean_prices(data_access.load_crypto_prices())


def load_clean_headlines() -> pd.DataFrame:
    """Load and exactly deduplicate the 2020-2023 headline sample."""
    return clean_headlines(data_access.load_news_headlines())
