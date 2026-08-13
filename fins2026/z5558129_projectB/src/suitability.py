"""Configurable educational investor-profile and fund recommendation engine."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class OptionConfig:
    """One questionnaire response option."""

    label: str
    score: int


@dataclass(frozen=True)
class QuestionConfig:
    """One scored questionnaire item."""

    prompt: str
    options: dict[str, OptionConfig]


@dataclass(frozen=True)
class CryptoScenarioConfig:
    """One profile-specific cryptocurrency preference scenario."""

    preference: str
    label: str
    maximum_exposure: float
    is_main_recommendation: bool = True
    is_higher_risk_alternative: bool = False
    requires_explicit_confirmation: bool = False
    warning_level: str = "none"


@dataclass(frozen=True)
class SuitabilityConfig:
    """All configurable questionnaire, ranking, and allocation rules."""

    questions: dict[str, QuestionConfig]
    profile_cutoffs: dict[str, tuple[int, int]]
    ranking_weights: dict[str, dict[str, float]]
    allocation_templates: dict[str, dict[int, tuple[float, ...]]]
    crypto_scenarios: dict[str, dict[str, CryptoScenarioConfig]]
    crypto_warning: str
    maximum_fund_weight: float = 0.50
    short_horizon_key: str = "short"
    long_horizon_key: str = "long"
    low_loss_key: str = "low"
    high_loss_key: str = "high"


DEFAULT_CONFIG = SuitabilityConfig(
    questions={
        "objective": QuestionConfig(
            "What is your primary investment objective?",
            {
                "preservation": OptionConfig("Preserve capital and earn income", 1),
                "balanced": OptionConfig("Balance growth and stability", 2),
                "growth": OptionConfig("Prioritise long-term growth", 3),
            },
        ),
        "horizon": QuestionConfig(
            "How long do you expect to remain invested?",
            {
                "short": OptionConfig("Under 3 years", 1),
                "medium": OptionConfig("3-7 years", 2),
                "long": OptionConfig("Over 7 years", 3),
            },
        ),
        "loss_tolerance": QuestionConfig(
            "What temporary portfolio loss could you tolerate?",
            {
                "low": OptionConfig("Under 10%", 1),
                "medium": OptionConfig("10-25%", 2),
                "high": OptionConfig("Over 25%", 3),
            },
        ),
        "decline_behavior": QuestionConfig(
            "What would you most likely do during a market decline?",
            {
                "sell": OptionConfig("Sell to avoid further losses", 1),
                "hold": OptionConfig("Hold the investment", 2),
                "buy": OptionConfig("Invest more at lower prices", 3),
            },
        ),
        "stability_growth": QuestionConfig(
            "Which trade-off do you prefer?",
            {
                "stability": OptionConfig("More stability", 1),
                "balanced": OptionConfig("A balance of both", 2),
                "growth": OptionConfig("More potential growth", 3),
            },
        ),
    },
    profile_cutoffs={
        "Conservative": (5, 8),
        "Balanced": (9, 11),
        "Growth": (12, 15),
    },
    ranking_weights={
        "Conservative": {
            "annualized_return": 0.10,
            "sharpe_ratio": 0.20,
            "annualized_volatility": 0.35,
            "maximum_drawdown": 0.35,
        },
        "Balanced": {
            "annualized_return": 0.25,
            "sharpe_ratio": 0.25,
            "annualized_volatility": 0.25,
            "maximum_drawdown": 0.25,
        },
        "Growth": {
            "annualized_return": 0.45,
            "sharpe_ratio": 0.30,
            "annualized_volatility": 0.15,
            "maximum_drawdown": 0.10,
        },
    },
    allocation_templates={
        "Conservative": {3: (0.50, 0.30, 0.20)},
        "Balanced": {
            3: (0.40, 0.35, 0.25),
            4: (0.35, 0.30, 0.20, 0.15),
        },
        "Growth": {
            3: (0.50, 0.30, 0.20),
            4: (0.40, 0.30, 0.20, 0.10),
        },
    },
    crypto_scenarios={
        "Conservative": {
            "none": CryptoScenarioConfig("none", "None (0% crypto)", 0.0),
            "limited": CryptoScenarioConfig("limited", "Limited", 0.05),
            "higher": CryptoScenarioConfig(
                "higher", "Higher", 0.10, warning_level="standard"
            ),
        },
        "Balanced": {
            "none": CryptoScenarioConfig("none", "None (0% crypto)", 0.0),
            "limited": CryptoScenarioConfig("limited", "Limited", 0.10),
            "higher": CryptoScenarioConfig(
                "higher", "Higher (default)", 0.15, warning_level="standard"
            ),
            "higher_25_alternative": CryptoScenarioConfig(
                preference="higher",
                label="Optional 25% crypto (higher-risk alternative)",
                maximum_exposure=0.25,
                is_main_recommendation=False,
                is_higher_risk_alternative=True,
                requires_explicit_confirmation=True,
                warning_level="strong",
            ),
        },
        "Growth": {
            "none": CryptoScenarioConfig("none", "None (0% crypto)", 0.0),
            "limited": CryptoScenarioConfig("limited", "Limited", 0.15),
            "higher": CryptoScenarioConfig(
                "higher", "Higher", 0.25, warning_level="standard"
            ),
        },
    },
    crypto_warning=(
        "Higher cryptocurrency exposure can involve exceptional volatility and "
        "deep historical drawdowns. This educational allocation is not personal "
        "financial advice."
    ),
)


@dataclass(frozen=True)
class ProfileResult:
    """Questionnaire score plus transparent override results."""

    score: int
    raw_profile: str
    final_profile: str
    override_reasons: tuple[str, ...]


@dataclass(frozen=True)
class RecommendationResult:
    """Ranked shortlist and illustrative constrained allocation."""

    profile: str
    crypto_preference: str
    scenario_key: str
    scenario_label: str
    allocation: pd.DataFrame
    replacements: pd.DataFrame
    template_adjusted: bool
    adjustment_l2: float
    lookthrough_crypto_exposure: float
    warning: str | None
    is_main_recommendation: bool
    is_higher_risk_alternative: bool
    requires_explicit_confirmation: bool


def config_as_dict(config: SuitabilityConfig = DEFAULT_CONFIG) -> dict[str, Any]:
    """Return JSON-serializable configuration for later UI use."""
    return asdict(config)


def _profile_for_score(score: int, config: SuitabilityConfig) -> str:
    for profile, (lower, upper) in config.profile_cutoffs.items():
        if lower <= score <= upper:
            return profile
    raise ValueError(f"questionnaire score {score} is outside configured cutoffs")


def score_questionnaire(
    answers: dict[str, str],
    config: SuitabilityConfig = DEFAULT_CONFIG,
) -> ProfileResult:
    """Score five responses and apply capacity-based profile caps."""
    if set(answers) != set(config.questions):
        missing = set(config.questions).difference(answers)
        extra = set(answers).difference(config.questions)
        raise ValueError(
            f"questionnaire answers mismatch; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    score = 0
    for question, answer in answers.items():
        options = config.questions[question].options
        if answer not in options:
            raise ValueError(f"unsupported answer {answer!r} for {question}")
        score += options[answer].score

    raw_profile = _profile_for_score(score, config)
    final_profile = raw_profile
    reasons: list[str] = []
    short_horizon = answers["horizon"] == config.short_horizon_key
    low_loss = answers["loss_tolerance"] == config.low_loss_key
    long_horizon = answers["horizon"] == config.long_horizon_key
    high_loss = answers["loss_tolerance"] == config.high_loss_key

    if short_horizon and low_loss:
        final_profile = "Conservative"
        if raw_profile != final_profile:
            reasons.append("short horizon and low loss tolerance require Conservative")
    elif short_horizon or low_loss:
        if raw_profile == "Growth":
            final_profile = "Balanced"
            reasons.append(
                "short horizon or low loss tolerance caps the profile at Balanced"
            )
    if final_profile == "Growth" and not (long_horizon and high_loss):
        final_profile = "Balanced"
        reasons.append(
            "Growth requires both a long horizon and high loss tolerance"
        )
    return ProfileResult(score, raw_profile, final_profile, tuple(reasons))


def rank_funds(
    metrics: pd.DataFrame,
    profile: str,
    config: SuitabilityConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Rank all funds using profile-weighted out-of-sample percentile scores."""
    if profile not in config.ranking_weights:
        raise ValueError(f"unsupported profile: {profile}")
    weights = config.ranking_weights[profile]
    required = {"fund", *weights}
    if missing := required.difference(metrics.columns):
        raise ValueError(f"metrics is missing columns: {sorted(missing)}")
    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError(f"ranking weights for {profile} do not sum to 1")

    ranked = metrics.copy()
    benefit_direction = {
        "annualized_return": True,
        "sharpe_ratio": True,
        "annualized_volatility": False,
        "maximum_drawdown": True,
    }
    component_columns = []
    for metric, weight in weights.items():
        column = f"{metric}_score"
        ranked[column] = ranked[metric].rank(
            method="average",
            pct=True,
            ascending=benefit_direction[metric],
        )
        ranked[column] *= weight
        component_columns.append(column)
    ranked["ranking_score"] = ranked[component_columns].sum(axis=1)
    ranked = ranked.sort_values(
        ["ranking_score", "fund"], ascending=[False, True]
    ).reset_index(drop=True)
    ranked.insert(0, "profile", profile)
    ranked.insert(1, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def latest_fund_crypto_exposure(fund_weights: pd.DataFrame) -> pd.DataFrame:
    """Calculate each fund's look-through crypto share at its latest rebalance."""
    required = {"fund", "rebalance_date", "asset_class", "target_weight"}
    if missing := required.difference(fund_weights.columns):
        raise ValueError(f"fund_weights is missing columns: {sorted(missing)}")
    weights = fund_weights.copy()
    weights["rebalance_date"] = pd.to_datetime(weights["rebalance_date"])
    latest_dates = weights.groupby("fund")["rebalance_date"].transform("max")
    latest = weights.loc[weights["rebalance_date"] == latest_dates]
    exposure = (
        latest.assign(
            crypto_contribution=np.where(
                latest["asset_class"].eq("Crypto"), latest["target_weight"], 0.0
            )
        )
        .groupby("fund", as_index=False)
        .agg(
            latest_rebalance_date=("rebalance_date", "first"),
            crypto_exposure=("crypto_contribution", "sum"),
        )
    )
    return exposure


def _minimum_crypto_exposure(
    exposures: np.ndarray, maximum_fund_weight: float
) -> float:
    """Return the smallest attainable exposure under the per-fund cap."""
    remaining = 1.0
    minimum = 0.0
    for exposure in np.sort(exposures):
        weight = min(maximum_fund_weight, remaining)
        minimum += weight * exposure
        remaining -= weight
        if remaining <= 1e-12:
            return float(minimum)
    return float("inf")


def _constraint_aware_shortlist(
    eligible: pd.DataFrame,
    count: int,
    crypto_limit: float,
    maximum_fund_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select top funds, replacing the lowest-ranked only when infeasible."""
    selected = eligible.head(count).copy()
    replacements: list[dict[str, object]] = []
    while (
        _minimum_crypto_exposure(
            selected["crypto_exposure"].to_numpy(), maximum_fund_weight
        )
        > crypto_limit + 1e-10
    ):
        outside = eligible.loc[~eligible["fund"].isin(selected["fund"])]
        replacement_made = False
        for selected_index in selected.sort_values("rank", ascending=False).index:
            outgoing = selected.loc[selected_index]
            for _, incoming in outside.sort_values("rank").iterrows():
                if incoming["crypto_exposure"] >= outgoing["crypto_exposure"]:
                    continue
                trial = selected.drop(index=selected_index)
                trial = pd.concat([trial, incoming.to_frame().T], ignore_index=True)
                old_minimum = _minimum_crypto_exposure(
                    selected["crypto_exposure"].to_numpy(), maximum_fund_weight
                )
                new_minimum = _minimum_crypto_exposure(
                    trial["crypto_exposure"].to_numpy(), maximum_fund_weight
                )
                if new_minimum >= old_minimum - 1e-12:
                    continue
                replacements.append(
                    {
                        "removed_fund": outgoing["fund"],
                        "removed_rank": int(outgoing["rank"]),
                        "added_fund": incoming["fund"],
                        "added_rank": int(incoming["rank"]),
                        "reason": (
                            f"reduce minimum crypto exposure from {old_minimum:.2%} "
                            f"toward the {crypto_limit:.0%} limit"
                        ),
                    }
                )
                selected = trial
                replacement_made = True
                break
            if replacement_made:
                break
        if not replacement_made:
            raise ValueError("no feasible shortlist satisfies the crypto constraint")
    return selected.sort_values("rank").reset_index(drop=True), pd.DataFrame(replacements)


def _minimum_adjustment_allocation(
    template: np.ndarray,
    crypto_exposure: np.ndarray,
    crypto_limit: float,
    maximum_fund_weight: float,
) -> tuple[np.ndarray, bool, float]:
    """Project template weights onto the crypto-constrained feasible set."""
    template_exposure = float(template @ crypto_exposure)
    if template_exposure <= crypto_limit + 1e-10:
        return template.copy(), False, 0.0
    feasible_start = np.zeros(len(template))
    remaining = 1.0
    for index in np.argsort(crypto_exposure):
        assigned = min(maximum_fund_weight, remaining)
        feasible_start[index] = assigned
        remaining -= assigned
        if remaining <= 1e-12:
            break
    if float(feasible_start @ crypto_exposure) > crypto_limit + 1e-10:
        raise ValueError("selected funds cannot satisfy the crypto limit")
    constraints = [
        {"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        {
            "type": "ineq",
            "fun": lambda weights: crypto_limit - weights @ crypto_exposure,
        },
    ]
    result = minimize(
        lambda weights: float(np.square(weights - template).sum()),
        feasible_start,
        method="SLSQP",
        bounds=[(0.0, maximum_fund_weight)] * len(template),
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1_000, "disp": False},
    )
    if not result.success:
        raise RuntimeError(
            f"allocation adjustment failed with status {result.status}: "
            f"{result.message}"
        )
    if float(result.x @ crypto_exposure) > crypto_limit + 1e-8:
        raise RuntimeError("allocation solver returned an infeasible crypto exposure")
    adjustment_l2 = float(np.sqrt(np.square(result.x - template).sum()))
    return result.x, True, adjustment_l2


def recommend_funds(
    rankings: pd.DataFrame,
    fund_crypto_exposure: pd.DataFrame,
    profile: str,
    crypto_preference: str,
    config: SuitabilityConfig = DEFAULT_CONFIG,
    *,
    scenario_key: str | None = None,
) -> RecommendationResult:
    """Select ranked funds and construct a transparent illustrative allocation."""
    if profile not in config.allocation_templates:
        raise ValueError(f"unsupported profile: {profile}")
    profile_scenarios = config.crypto_scenarios[profile]
    if scenario_key is None:
        matches = [
            (key, scenario)
            for key, scenario in profile_scenarios.items()
            if scenario.preference == crypto_preference
            and scenario.is_main_recommendation
        ]
        if len(matches) != 1:
            raise ValueError(
                f"no unique main {crypto_preference} scenario for {profile}"
            )
        scenario_key, scenario = matches[0]
    else:
        if scenario_key not in profile_scenarios:
            raise ValueError(f"unsupported scenario {scenario_key!r} for {profile}")
        scenario = profile_scenarios[scenario_key]
        if scenario.preference != crypto_preference:
            raise ValueError("scenario preference does not match crypto_preference")
    ranked = rankings.loc[rankings["profile"] == profile].merge(
        fund_crypto_exposure[["fund", "crypto_exposure"]],
        on="fund",
        how="left",
        validate="one_to_one",
    )
    if ranked["crypto_exposure"].isna().any():
        raise ValueError("one or more ranked funds lack crypto exposure data")
    limit = scenario.maximum_exposure
    if crypto_preference == "none":
        eligible = ranked.loc[ranked["crypto_exposure"].le(1e-12)].copy()
    else:
        eligible = ranked.copy()

    profile_templates = config.allocation_templates[profile]
    preferred_count = max(profile_templates)
    selected_count = min(preferred_count, len(eligible))
    if selected_count not in profile_templates:
        raise ValueError(
            f"no {selected_count}-fund template is configured for {profile}"
        )
    template = np.array(profile_templates[selected_count], dtype=float)
    selected, replacements = _constraint_aware_shortlist(
        eligible,
        selected_count,
        limit,
        config.maximum_fund_weight,
    )
    selected = selected.sort_values("rank").reset_index(drop=True)
    adjusted, was_adjusted, adjustment_l2 = _minimum_adjustment_allocation(
        template,
        selected["crypto_exposure"].to_numpy(dtype=float),
        limit,
        config.maximum_fund_weight,
    )
    allocation = selected[
        ["fund", "rank", "ranking_score", "crypto_exposure"]
    ].copy()
    allocation["template_weight"] = template
    allocation["recommended_weight"] = adjusted
    allocation["weight_adjustment"] = adjusted - template
    allocation["crypto_contribution"] = (
        allocation["recommended_weight"] * allocation["crypto_exposure"]
    )
    lookthrough = float(allocation["crypto_contribution"].sum())
    warning = None
    if scenario.warning_level == "standard":
        warning = config.crypto_warning
    elif scenario.warning_level == "strong":
        crypto_metrics = ranked.loc[ranked["fund"].str.startswith("Crypto ")]
        if crypto_metrics.empty:
            raise ValueError("strong crypto warning requires crypto-fund metrics")
        volatility_low = crypto_metrics["annualized_volatility"].min()
        volatility_high = crypto_metrics["annualized_volatility"].max()
        drawdown_low = crypto_metrics["maximum_drawdown"].abs().min()
        drawdown_high = crypto_metrics["maximum_drawdown"].abs().max()
        warning = (
            "Higher-risk alternative: the crypto-only funds recorded annualized "
            f"volatility of {volatility_low:.1%}-{volatility_high:.1%} and maximum "
            f"drawdowns of {drawdown_low:.1%}-{drawdown_high:.1%}. Explicit user "
            "confirmation is required before displaying this optional scenario. "
            "This educational allocation is not personal financial advice."
        )
    return RecommendationResult(
        profile=profile,
        crypto_preference=crypto_preference,
        scenario_key=scenario_key,
        scenario_label=scenario.label,
        allocation=allocation,
        replacements=replacements,
        template_adjusted=was_adjusted,
        adjustment_l2=adjustment_l2,
        lookthrough_crypto_exposure=lookthrough,
        warning=warning,
        is_main_recommendation=scenario.is_main_recommendation,
        is_higher_risk_alternative=scenario.is_higher_risk_alternative,
        requires_explicit_confirmation=scenario.requires_explicit_confirmation,
    )


def build_recommendation_grid(
    metrics: pd.DataFrame,
    fund_weights: pd.DataFrame,
    config: SuitabilityConfig = DEFAULT_CONFIG,
) -> dict[str, pd.DataFrame]:
    """Build rankings and examples for every configured profile-preference pair."""
    exposure = latest_fund_crypto_exposure(fund_weights)
    rankings = pd.concat(
        [rank_funds(metrics, profile, config) for profile in config.ranking_weights],
        ignore_index=True,
    )
    allocation_frames = []
    replacement_frames = []
    summaries = []
    for profile in config.ranking_weights:
        for scenario_key, scenario in config.crypto_scenarios[profile].items():
            preference = scenario.preference
            result = recommend_funds(
                rankings,
                exposure,
                profile,
                preference,
                config,
                scenario_key=scenario_key,
            )
            allocation = result.allocation.copy()
            allocation.insert(0, "scenario_label", result.scenario_label)
            allocation.insert(0, "scenario_key", result.scenario_key)
            allocation.insert(0, "crypto_preference", preference)
            allocation.insert(0, "profile", profile)
            allocation.insert(4, "selection_order", range(1, len(allocation) + 1))
            allocation_frames.append(allocation)
            if not result.replacements.empty:
                replacements = result.replacements.copy()
                replacements.insert(0, "scenario_label", result.scenario_label)
                replacements.insert(0, "scenario_key", result.scenario_key)
                replacements.insert(0, "crypto_preference", preference)
                replacements.insert(0, "profile", profile)
                replacement_frames.append(replacements)
            summaries.append(
                {
                    "profile": profile,
                    "crypto_preference": preference,
                    "scenario_key": result.scenario_key,
                    "scenario_label": result.scenario_label,
                    "is_main_recommendation": result.is_main_recommendation,
                    "is_higher_risk_alternative": result.is_higher_risk_alternative,
                    "requires_explicit_confirmation": (
                        result.requires_explicit_confirmation
                    ),
                    "selected_fund_count": len(allocation),
                    "selected_funds": " | ".join(allocation["fund"]),
                    "lookthrough_crypto_exposure": result.lookthrough_crypto_exposure,
                    "crypto_limit": scenario.maximum_exposure,
                    "template_adjusted": result.template_adjusted,
                    "adjustment_l2": result.adjustment_l2,
                    "replacement_count": len(result.replacements),
                    "warning": result.warning or "",
                }
            )
    replacement_columns = [
        "profile",
        "crypto_preference",
        "scenario_key",
        "scenario_label",
        "removed_fund",
        "removed_rank",
        "added_fund",
        "added_rank",
        "reason",
    ]
    return {
        "exposure": exposure,
        "rankings": rankings,
        "allocations": pd.concat(allocation_frames, ignore_index=True),
        "replacements": (
            pd.concat(replacement_frames, ignore_index=True)
            if replacement_frames
            else pd.DataFrame(columns=replacement_columns)
        ),
        "summary": pd.DataFrame(summaries),
    }
