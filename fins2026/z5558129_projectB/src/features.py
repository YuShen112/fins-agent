"""Reusable return features and trading-day headline assembly."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class HeadlineAlignmentResult:
    """Headline panel plus source records beyond the available calendar."""

    panel: pd.DataFrame
    aligned: pd.DataFrame
    excluded: pd.DataFrame


def daily_returns(
    prices: pd.DataFrame,
    price_col: str = "adjClose",
) -> pd.DataFrame:
    """Calculate simple daily returns independently within each ticker."""
    required = {"ticker", "date", price_col}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"prices is missing required columns: {sorted(missing)}")

    result = prices.sort_values(["ticker", "date"]).copy()
    result["return"] = result.groupby("ticker", sort=False)[price_col].transform(
        lambda values: values.pct_change(fill_method=None)
    )
    return result


def returns_to_wide(return_panel: pd.DataFrame) -> pd.DataFrame:
    """Convert unique long-form ticker returns to a sorted wide panel."""
    required = {"ticker", "date", "return"}
    missing = required.difference(return_panel.columns)
    if missing:
        raise ValueError(
            f"return_panel is missing required columns: {sorted(missing)}"
        )
    if return_panel.duplicated(["ticker", "date"]).any():
        raise ValueError("return_panel contains duplicate ticker-date rows")
    wide = return_panel.pivot(index="date", columns="ticker", values="return")
    wide = wide.sort_index().sort_index(axis="columns")
    wide.columns.name = None
    return wide.reset_index()


def combine_returns_on_equity_calendar(
    equity_returns: pd.DataFrame,
    crypto_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Join already-computed asset returns on the equity trading calendar."""
    required = {"ticker", "date", "return"}
    for name, frame in (
        ("equity_returns", equity_returns),
        ("crypto_returns", crypto_returns),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
        if frame.duplicated(["ticker", "date"]).any():
            raise ValueError(f"{name} contains duplicate ticker-date rows")

    equities = returns_to_wide(equity_returns).set_index("date")
    crypto = returns_to_wide(crypto_returns).set_index("date")

    overlap = equities.columns.intersection(crypto.columns)
    if not overlap.empty:
        raise ValueError(f"asset tickers overlap: {overlap.tolist()}")
    return equities.join(crypto, how="left").reset_index()


def assemble_headline_panel(
    headlines: pd.DataFrame,
    equity_trading_dates: pd.Series | pd.Index,
) -> HeadlineAlignmentResult:
    """Map titles to the same or next equity trading day and aggregate them.

    Records after the final supplied trading day are returned separately rather
    than silently discarded. Text is preserved for later sentiment modelling.
    """
    required = {"date", "ticker", "sector", "title"}
    missing = required.difference(headlines.columns)
    if missing:
        raise ValueError(f"headlines is missing required columns: {sorted(missing)}")

    calendar = pd.DatetimeIndex(
        pd.to_datetime(pd.Index(equity_trading_dates), utc=True)
    ).tz_localize(None)
    calendar = calendar.drop_duplicates().sort_values()
    if calendar.empty:
        raise ValueError("equity_trading_dates must not be empty")

    source = headlines.copy()
    source["date"] = pd.to_datetime(source["date"], utc=True).dt.tz_localize(None)
    source = source.drop_duplicates(["ticker", "date", "title"], keep="first")
    positions = calendar.searchsorted(source["date"], side="left")
    alignable = positions < len(calendar)

    excluded = source.loc[~alignable].sort_values(
        ["date", "ticker", "title"]
    ).reset_index(drop=True)
    aligned = source.loc[alignable].copy()
    aligned["trading_date"] = calendar.take(positions[alignable]).to_numpy()
    panel = (
        aligned.groupby(["trading_date", "ticker", "sector"], as_index=False)
        .agg(
            headline_count=("title", "size"),
            headline_text=("title", lambda titles: "\n".join(titles)),
        )
        .sort_values(["trading_date", "ticker"])
        .reset_index(drop=True)
    )
    return HeadlineAlignmentResult(
        panel=panel,
        aligned=aligned.sort_values(
            ["trading_date", "ticker", "title"]
        ).reset_index(drop=True),
        excluded=excluded,
    )


def validate_data_foundation(
    combined_returns: pd.DataFrame,
    headline_panel: pd.DataFrame,
    *,
    expected_combined_shape: tuple[int, int] | None = None,
    expected_headline_rows: int | None = None,
) -> None:
    """Validate schemas, uniqueness, and optional external benchmark counts."""
    if "date" not in combined_returns.columns:
        raise ValueError("combined returns panel is missing date")
    if combined_returns["date"].duplicated().any():
        raise ValueError("combined returns panel contains duplicate dates")

    headline_columns = {
        "trading_date", "ticker", "sector", "headline_count", "headline_text"
    }
    missing = headline_columns.difference(headline_panel.columns)
    if missing:
        raise ValueError(f"headline panel is missing columns: {sorted(missing)}")
    duplicate_count = int(
        headline_panel.duplicated(["trading_date", "ticker"]).sum()
    )
    if duplicate_count:
        raise ValueError(
            "headline panel contains "
            f"{duplicate_count} duplicate trading_date-ticker rows"
        )

    if expected_combined_shape and combined_returns.shape != expected_combined_shape:
        raise ValueError(
            "combined returns benchmark mismatch: "
            f"expected {expected_combined_shape}, got {combined_returns.shape}"
        )
    if expected_headline_rows and len(headline_panel) != expected_headline_rows:
        raise ValueError(
            "headline panel benchmark mismatch: "
            f"expected {expected_headline_rows} rows, got {len(headline_panel)}"
        )
