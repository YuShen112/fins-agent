"""Focused tests for the reusable Part A data-foundation logic."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest
from src import etl, features

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def test_returns_use_adjusted_close_and_do_not_cross_tickers() -> None:
    prices = pd.DataFrame(
        {
            "ticker": ["BBB", "AAA", "BBB", "AAA"],
            "date": pd.to_datetime(
                ["2020-01-02", "2020-01-01", "2020-01-01", "2020-01-02"]
            ),
            "adjClose": [40.0, 100.0, 20.0, 110.0],
        }
    )
    result = features.daily_returns(prices)
    wide = result.pivot(index="date", columns="ticker", values="return")
    assert result.groupby("ticker").nth(0)["return"].isna().all()
    assert np.isclose(wide.loc[pd.Timestamp("2020-01-02"), "AAA"], 0.10)
    assert np.isclose(wide.loc[pd.Timestamp("2020-01-02"), "BBB"], 1.00)


def test_cleaning_caps_sample_and_uses_exact_headline_key() -> None:
    headlines = pd.DataFrame(
        {
            "ticker": ["AAA"] * 4,
            "date": pd.to_datetime(
                ["2020-01-02", "2020-01-02", "2020-01-02", "2024-01-01"],
                utc=True,
            ),
            "sector": ["Tech"] * 4,
            "title": ["Same", "Same", "Different", "Outside"],
        }
    )
    clean = etl.clean_headlines(headlines)
    assert clean["title"].tolist() == ["Different", "Same"]
    assert clean["date"].dt.tz is None


def test_headlines_align_forward_and_report_unalignable_records() -> None:
    headlines = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-04", "2020-01-06", "2020-01-07"]),
            "ticker": ["AAA"] * 3,
            "sector": ["Tech"] * 3,
            "title": ["Weekend", "Monday", "After sample"],
        }
    )
    calendar = pd.to_datetime(["2020-01-03", "2020-01-06"])
    result = features.assemble_headline_panel(headlines, calendar)
    assert len(result.panel) == 1
    assert result.panel.loc[0, "trading_date"] == pd.Timestamp("2020-01-06")
    assert result.panel.loc[0, "headline_count"] == 2
    assert result.excluded["title"].tolist() == ["After sample"]
    assert set(result.panel.columns) == {
        "trading_date", "ticker", "sector", "headline_count", "headline_text"
    }


def test_validation_rejects_duplicate_headline_keys_and_benchmark_mismatch() -> None:
    combined = pd.DataFrame({"date": pd.to_datetime(["2020-01-02"]), "AAA": [0.1]})
    panel = pd.DataFrame(
        {
            "trading_date": pd.to_datetime(["2020-01-02", "2020-01-02"]),
            "ticker": ["AAA", "AAA"],
            "sector": ["Tech", "Tech"],
            "headline_count": [1, 1],
            "headline_text": ["One", "Two"],
        }
    )
    with pytest.raises(ValueError, match="duplicate trading_date-ticker"):
        features.validate_data_foundation(combined, panel)

    with pytest.raises(ValueError, match="benchmark mismatch"):
        features.validate_data_foundation(
            combined, panel.iloc[:1], expected_combined_shape=(2, 2)
        )
