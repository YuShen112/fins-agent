"""Tests for questionnaire overrides, ranking, and illustrative allocations."""
from __future__ import annotations

import numpy as np
import pandas as pd
from src import suitability


def answers(**updates: str) -> dict[str, str]:
    """Return high-scoring answers with selected overrides."""
    result = {
        "objective": "growth",
        "horizon": "long",
        "loss_tolerance": "high",
        "decline_behavior": "buy",
        "stability_growth": "growth",
    }
    result.update(updates)
    return result


def test_questionnaire_override_rules() -> None:
    growth = suitability.score_questionnaire(answers())
    assert growth.raw_profile == growth.final_profile == "Growth"

    both = suitability.score_questionnaire(
        answers(horizon="short", loss_tolerance="low")
    )
    assert both.final_profile == "Conservative"

    one = suitability.score_questionnaire(answers(horizon="short"))
    assert one.raw_profile == "Growth"
    assert one.final_profile == "Balanced"

    capacity = suitability.score_questionnaire(answers(horizon="medium"))
    assert capacity.raw_profile == "Growth"
    assert capacity.final_profile == "Balanced"


def metric_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fund": ["A", "B", "C", "D"],
            "annualized_return": [0.05, 0.10, 0.15, 0.20],
            "sharpe_ratio": [0.5, 0.7, 0.9, 1.1],
            "annualized_volatility": [0.10, 0.15, 0.20, 0.25],
            "maximum_drawdown": [-0.10, -0.15, -0.20, -0.25],
        }
    )


def test_profile_rankings_use_configured_metric_weights() -> None:
    conservative = suitability.rank_funds(metric_fixture(), "Conservative")
    growth = suitability.rank_funds(metric_fixture(), "Growth")
    assert conservative.iloc[0]["fund"] == "A"
    assert growth.iloc[0]["fund"] == "D"
    assert conservative["ranking_score"].between(0, 1).all()


def recommendation_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = pd.DataFrame(
        {
            "profile": ["Balanced"] * 6,
            "rank": range(1, 7),
            "fund": ["Crypto A", "Crypto B", "Combined", "Equity A", "Equity B", "Equity C"],
            "ranking_score": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        }
    )
    exposure = pd.DataFrame(
        {
            "fund": ranked["fund"],
            "crypto_exposure": [1.0, 1.0, 0.25, 0.0, 0.0, 0.0],
        }
    )
    return ranked, exposure


def test_recommendation_replaces_funds_then_minimally_adjusts_template() -> None:
    ranked, exposure = recommendation_fixture()
    result = suitability.recommend_funds(
        ranked, exposure, "Balanced", "limited"
    )
    assert not result.replacements.empty
    assert result.allocation["recommended_weight"].sum() == 1.0
    assert result.allocation["recommended_weight"].le(0.50 + 1e-9).all()
    assert result.lookthrough_crypto_exposure <= 0.10 + 1e-8
    assert result.template_adjusted


def test_crypto_preferences_and_warning() -> None:
    ranked, exposure = recommendation_fixture()
    none = suitability.recommend_funds(ranked, exposure, "Balanced", "none")
    assert np.isclose(none.lookthrough_crypto_exposure, 0.0)
    assert none.allocation["crypto_exposure"].eq(0).all()

    higher = suitability.recommend_funds(ranked, exposure, "Balanced", "higher")
    assert higher.lookthrough_crypto_exposure <= 0.25 + 1e-8
    assert higher.warning == suitability.DEFAULT_CONFIG.crypto_warning


def test_profile_specific_scenarios_and_balanced_alternative_metadata() -> None:
    scenarios = suitability.DEFAULT_CONFIG.crypto_scenarios
    assert scenarios["Conservative"]["limited"].maximum_exposure == 0.05
    assert scenarios["Conservative"]["higher"].maximum_exposure == 0.10
    assert scenarios["Balanced"]["higher"].maximum_exposure == 0.15
    assert scenarios["Growth"]["limited"].maximum_exposure == 0.15
    assert scenarios["Growth"]["higher"].maximum_exposure == 0.25
    alternative = scenarios["Balanced"]["higher_25_alternative"]
    assert alternative.maximum_exposure == 0.25
    assert not alternative.is_main_recommendation
    assert alternative.is_higher_risk_alternative
    assert alternative.requires_explicit_confirmation


def test_grid_builds_ten_scenarios_with_strong_balanced_warning() -> None:
    funds = [
        "Equity A",
        "Equity B",
        "Equity C",
        "Combined A",
        "Combined B",
        "Combined C",
        "Crypto A",
        "Crypto B",
    ]
    metrics = pd.DataFrame(
        {
            "fund": funds,
            "annualized_return": np.linspace(0.05, 0.40, 8),
            "sharpe_ratio": np.linspace(0.5, 0.9, 8),
            "annualized_volatility": [0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.80, 0.82],
            "maximum_drawdown": [-0.12, -0.14, -0.16, -0.20, -0.22, -0.25, -0.80, -0.82],
        }
    )
    exposures = [0.0, 0.0, 0.0, 0.03, 0.09, 0.16, 1.0, 1.0]
    weight_rows = []
    for fund, exposure in zip(funds, exposures):
        date = pd.Timestamp("2023-12-01")
        if exposure > 0:
            weight_rows.append(
                {
                    "fund": fund,
                    "rebalance_date": date,
                    "asset_class": "Crypto",
                    "target_weight": exposure,
                }
            )
        if exposure < 1:
            weight_rows.append(
                {
                    "fund": fund,
                    "rebalance_date": date,
                    "asset_class": "Equity",
                    "target_weight": 1 - exposure,
                }
            )
    grid = suitability.build_recommendation_grid(
        metrics, pd.DataFrame(weight_rows)
    )
    summary = grid["summary"]
    assert len(summary) == 10
    alternative = summary.loc[
        (summary["profile"] == "Balanced")
        & (summary["scenario_key"] == "higher_25_alternative")
    ].iloc[0]
    assert alternative["crypto_limit"] == 0.25
    assert not bool(alternative["is_main_recommendation"])
    assert bool(alternative["requires_explicit_confirmation"])
    assert "80.0%-82.0%" in alternative["warning"]
    assert "Explicit user confirmation" in alternative["warning"]

    for row in summary.itertuples(index=False):
        assert row.lookthrough_crypto_exposure <= row.crypto_limit + 1e-8
