"""Pure loading, validation, and questionnaire helpers for the Streamlit app."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class AppArtifacts:
    returns: pd.DataFrame
    weights: pd.DataFrame
    metrics: pd.DataFrame
    sentiment: pd.DataFrame
    recommendations: pd.DataFrame
    rankings: pd.DataFrame
    suitability_config: dict[str, Any]


@dataclass(frozen=True)
class AppProfile:
    score: int
    raw_profile: str
    final_profile: str
    override_reasons: tuple[str, ...]


SCHEMAS = {
    "data/fund_returns.csv": {"date", "fund", "daily_return", "growth_of_1"},
    "data/fund_weights.csv": {"rebalance_date", "fund", "ticker", "asset_class", "target_weight"},
    "tables/performance_metrics.csv": {
        "fund",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "maximum_drawdown",
        "final_growth_of_1",
    },
    "data/finvader_sector_sentiment_index.csv": {
        "trading_date",
        "sector",
        "sector_sentiment_index",
        "covered_ticker_count",
        "covered_ticker_proportion",
        "source_headline_count",
    },
    "data/suitability_recommendations.csv": {
        "profile",
        "crypto_preference",
        "scenario_key",
        "scenario_label",
        "fund",
        "rank",
        "recommended_weight",
        "crypto_contribution",
        "selection_order",
    },
    "tables/fund_rankings.csv": {"profile", "rank", "fund", "ranking_score"},
}


def _read_csv(results_dir: Path, relative_path: str) -> pd.DataFrame:
    path = results_dir / relative_path
    if not path.exists():
        raise FileNotFoundError(relative_path)
    frame = pd.read_csv(path)
    missing = SCHEMAS[relative_path].difference(frame.columns)
    if missing:
        raise ValueError(f"{relative_path} is missing columns {sorted(missing)}")
    return frame


def load_artifacts(results_dir: Path) -> AppArtifacts:
    """Load only precomputed app artifacts and validate their public schemas."""
    returns = _read_csv(results_dir, "data/fund_returns.csv")
    weights = _read_csv(results_dir, "data/fund_weights.csv")
    metrics = _read_csv(results_dir, "tables/performance_metrics.csv")
    sentiment = _read_csv(results_dir, "data/finvader_sector_sentiment_index.csv")
    recommendations = _read_csv(results_dir, "data/suitability_recommendations.csv")
    rankings = _read_csv(results_dir, "tables/fund_rankings.csv")
    config_path = results_dir / "data/suitability_config.json"
    if not config_path.exists():
        raise FileNotFoundError("data/suitability_config.json")
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)

    returns["date"] = pd.to_datetime(returns["date"])
    weights["rebalance_date"] = pd.to_datetime(weights["rebalance_date"])
    sentiment["trading_date"] = pd.to_datetime(sentiment["trading_date"])
    if returns.duplicated(["date", "fund"]).any():
        raise ValueError("fund_returns.csv contains duplicate date/fund rows")
    if metrics["fund"].duplicated().any():
        raise ValueError("performance_metrics.csv contains duplicate funds")
    return AppArtifacts(returns, weights, metrics, sentiment, recommendations, rankings, config)


def score_questionnaire(answers: dict[str, str], config: dict[str, Any]) -> AppProfile:
    """Score answers from config and apply capacity overrides; no fund logic runs."""
    questions = config["questions"]
    if set(answers) != set(questions):
        raise ValueError("All five questionnaire questions must be answered")
    score = sum(questions[key]["options"][value]["score"] for key, value in answers.items())
    raw_profile = next(
        profile
        for profile, bounds in config["profile_cutoffs"].items()
        if bounds[0] <= score <= bounds[1]
    )
    short = answers["horizon"] == config["short_horizon_key"]
    low_loss = answers["loss_tolerance"] == config["low_loss_key"]
    long = answers["horizon"] == config["long_horizon_key"]
    high_loss = answers["loss_tolerance"] == config["high_loss_key"]
    final_profile = raw_profile
    reasons: list[str] = []
    if short and low_loss:
        final_profile = "Conservative"
        if raw_profile != final_profile:
            reasons.append("A short horizon and low loss tolerance require Conservative.")
    elif (short or low_loss) and raw_profile == "Growth":
        final_profile = "Balanced"
        reasons.append("A short horizon or low loss tolerance caps the profile at Balanced.")
    if final_profile == "Growth" and not (long and high_loss):
        final_profile = "Balanced"
        reasons.append("Growth requires both a long horizon and high loss tolerance.")
    return AppProfile(score, raw_profile, final_profile, tuple(reasons))


def recommendation_rows(
    recommendations: pd.DataFrame,
    profile: str,
    preference: str,
    scenario_key: str | None = None,
) -> pd.DataFrame:
    """Look up, but never construct, a precomputed recommendation allocation."""
    selected = recommendations.loc[
        (recommendations["profile"] == profile)
        & (recommendations["crypto_preference"] == preference)
    ]
    if scenario_key is not None:
        selected = selected.loc[selected["scenario_key"] == scenario_key]
    if selected.empty:
        raise ValueError(f"No precomputed recommendation for {profile}/{preference}")
    return selected.sort_values("selection_order").copy()
