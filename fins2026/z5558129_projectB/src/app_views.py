"""Streamlit view renderers for EasyFunds."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.app_allocation import (
    allocation_total_is_valid,
    compare_allocations,
    fee_adjusted_growth,
    lookthrough_crypto,
)
from src.app_charts import (
    allocation_growth_chart,
    drawdown_chart,
    growth_chart,
    holdings_chart,
    monthly_coverage_chart,
    monthly_sentiment_chart,
)
from src.app_data import AppArtifacts, recommendation_rows, score_questionnaire
from src.app_fund_analysis import build_fact_sheet_analysis


def _heading(eyebrow: str, title: str, copy: str) -> None:
    st.markdown(f"<div class='eyebrow'>{eyebrow}</div>", unsafe_allow_html=True)
    st.title(title)
    st.caption(copy)


def _metric_strip(row: pd.Series) -> None:
    cols = st.columns(5)
    values = [
        ("Annual return", f"{row['annualized_return']:.1%}"),
        ("Volatility", f"{row['annualized_volatility']:.1%}"),
        ("Sharpe ratio", f"{row['sharpe_ratio']:.2f}"),
        ("Maximum drawdown", f"{row['maximum_drawdown']:.1%}"),
        ("Final value of $1", f"${row['final_growth_of_1']:.2f}"),
    ]
    for col, (label, value) in zip(cols, values, strict=True):
        col.metric(label, value)


def render_comparison(a: AppArtifacts) -> None:
    _heading(
        "Portfolio library",
        "Compare the eight funds",
        "One out-of-sample lens across risk, return and capital growth.",
    )
    default = ["Equity Equal Weight", "Combined Maximum Sharpe", "Crypto Risk Parity"]
    funds = st.multiselect("Funds on chart", sorted(a.metrics["fund"]), default=default)
    period = st.radio("Chart period", ["1M", "3M", "6M", "1Y", "ALL"], index=4, horizontal=True)
    if funds:
        st.plotly_chart(growth_chart(a.returns, funds, period), width="stretch")
    else:
        st.info("Select at least one fund to draw the comparison.")
    st.caption(
        "Performance metrics below use the full out-of-sample backtest and do not "
        "change with the chart period."
    )
    table = a.metrics.copy().sort_values("sharpe_ratio", ascending=False)
    table = table.rename(
        columns={
            "fund": "Fund",
            "annualized_return": "Annual return",
            "annualized_volatility": "Volatility",
            "sharpe_ratio": "Sharpe ratio",
            "maximum_drawdown": "Maximum drawdown",
            "final_growth_of_1": "Final value of $1",
            "observations": "Observations",
        }
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Annual return": st.column_config.NumberColumn(format="percent"),
            "Volatility": st.column_config.NumberColumn(format="percent"),
            "Maximum drawdown": st.column_config.NumberColumn(format="percent"),
            "Final value of $1": st.column_config.NumberColumn(format="$%.2f"),
            "Sharpe ratio": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.caption("Risk-free rate and transaction costs are zero in these historical backtests.")


def render_fact_sheet(a: AppArtifacts) -> None:
    _heading(
        "Fund detail",
        "A closer look",
        "Performance and the latest precomputed target portfolio in one fact sheet.",
    )
    fund = st.selectbox("Choose a fund", sorted(a.metrics["fund"]))
    metric = a.metrics.loc[a.metrics["fund"] == fund].iloc[0]
    st.caption(
        "Performance cards use the full out-of-sample backtest and do not change "
        "with the chart period."
    )
    _metric_strip(metric)
    analysis = build_fact_sheet_analysis(a.returns, a.weights, a.metrics, fund)

    downside_block, recovery_block = st.columns(2)
    with downside_block:
        st.subheader("Downside risk")
        downside_cols = st.columns(2)
        downside_cols[0].metric("Downside deviation", f"{analysis.downside_deviation:.1%}")
        downside_cols[1].metric("Sortino ratio", f"{analysis.sortino_ratio:.2f}")
        st.caption(
            f"Annualised using {analysis.annualization} days. Sortino = "
            "(annualised return - 0% MAR) / annualised downside deviation."
        )
    with recovery_block:
        st.subheader("Recovery")
        recovery_cols = st.columns(2)
        recovery_cols[0].metric(
            "Longest drawdown",
            f"{analysis.recovery.longest_drawdown_days:,} days",
        )
        if analysis.recovery.maximum_drawdown_recovered:
            recovery_value = f"{analysis.recovery.maximum_drawdown_recovery_days:,} days"
            recovery_help = "From the maximum-drawdown trough back to its prior peak."
        else:
            recovery_value = "Not recovered"
            recovery_help = (
                f"Still below the prior peak after "
                f"{analysis.recovery.maximum_drawdown_elapsed_days:,} calendar days "
                "at sample end."
            )
        recovery_cols[1].metric("Maximum-drawdown recovery", recovery_value)
        st.caption(f"Durations are calendar days. {recovery_help}")

    st.subheader("Diversification and relative position")
    diversification_cols = st.columns([1, 1, 2])
    diversification_cols[0].metric("Effective holdings", f"{analysis.effective_holdings:.1f}")
    diversification_cols[1].metric("Top-5 concentration", f"{analysis.top_five_concentration:.1%}")
    with diversification_cols[2]:
        st.dataframe(
            analysis.ranks,
            hide_index=True,
            width="stretch",
            height=178,
            column_config={"Rank among 8 funds": st.column_config.NumberColumn(format="#%d")},
        )
    st.caption(
        f"Diversification uses target weights from {analysis.holdings_as_of:%d %b %Y}. "
        "Effective holdings expresses the portfolio as an equivalent number of "
        "evenly weighted positions; a higher number means broader diversification. "
        "Top-5 concentration is the combined weight of the five largest positions. "
        "Rank #1 is best: higher return and Sharpe, lower volatility, and a smaller "
        "drawdown."
    )
    period = st.radio("Chart period", ["1M", "3M", "6M", "1Y", "ALL"], index=4, horizontal=True)
    left, right = st.columns([1.5, 1])
    with left:
        st.subheader("Capital growth")
        st.plotly_chart(growth_chart(a.returns, [fund], period), width="stretch")
        st.subheader("Drawdown from prior peak")
        st.plotly_chart(drawdown_chart(a.returns, fund, period), width="stretch")
    latest_date = a.weights.loc[a.weights["fund"] == fund, "rebalance_date"].max()
    latest = a.weights.loc[
        (a.weights["fund"] == fund) & (a.weights["rebalance_date"] == latest_date)
    ].sort_values("target_weight", ascending=False)
    with right:
        st.subheader("Latest target holdings")
        st.caption(f"Rebalance date: {latest_date:%d %b %Y}")
        crypto = latest.loc[latest["asset_class"] == "Crypto", "target_weight"].sum()
        st.metric("Look-through crypto", f"{crypto:.1%}")
        st.caption("Top 10 holdings by target weight")
        st.plotly_chart(holdings_chart(latest.head(10)), width="stretch")
        with st.expander(f"View all {len(latest)} holdings"):
            holdings_table = latest[["ticker", "asset_class", "target_weight"]].rename(
                columns={
                    "ticker": "Ticker",
                    "asset_class": "Asset class",
                    "target_weight": "Target weight",
                }
            )
            st.dataframe(
                holdings_table,
                hide_index=True,
                width="stretch",
                column_config={"Target weight": st.column_config.NumberColumn(format="percent")},
            )


def render_sentiment(a: AppArtifacts) -> None:
    _heading(
        "Market context",
        "Sector pulse",
        "Lagged news sentiment alongside the breadth of ticker coverage.",
    )
    sectors = sorted(a.sentiment["sector"].dropna().unique())
    sector = st.selectbox("Sector", ["All sectors", *sectors])
    period = st.radio("Chart period", ["6M", "1Y", "2Y", "ALL"], index=3, horizontal=True)
    selected_sentiment = (
        a.sentiment if sector == "All sectors" else a.sentiment.loc[a.sentiment["sector"] == sector]
    )
    st.subheader("Monthly sentiment")
    st.plotly_chart(monthly_sentiment_chart(selected_sentiment, period), width="stretch")
    st.info(
        "finVADER is the primary sentiment model, adding the SentiBignomics and "
        "Henry finance lexicons; plain VADER is retained as the benchmark. "
        "Daily scores are shown as monthly averages."
    )
    st.subheader("Monthly news coverage")
    st.plotly_chart(monthly_coverage_chart(selected_sentiment, period), width="stretch")
    st.caption(
        "Coverage is the proportion of each sector's five constituent tickers "
        "with at least one headline before lagging."
    )
    latest = a.sentiment["trading_date"].max()
    st.info(
        "Sentiment uses information available by the prior equity trading day. "
        f"Latest signal date: {latest:%d %b %Y}."
    )


def _show_allocation(rows: pd.DataFrame, title: str) -> None:
    st.subheader(title)
    for row in rows.itertuples():
        st.markdown(
            f"<div class='allocation'><strong>{row.fund}</strong><br>"
            f"{row.recommended_weight:.0%} allocation · fund rank #{int(row.rank)}"
            "</div>",
            unsafe_allow_html=True,
        )
    crypto = rows["crypto_contribution"].sum()
    st.metric("Portfolio crypto exposure", f"{crypto:.1%}")
    st.caption(f"Precomputed scenario: {rows['scenario_label'].iloc[0]}")


def _render_allocation_adjuster(
    a: AppArtifacts,
    rows: pd.DataFrame,
    profile: str,
    scenario_key: str,
    widget_prefix: str,
) -> None:
    st.subheader("Adjust the fund mix")
    st.caption(
        "Adjust fund allocations only. The underlying fund holdings and strategy "
        "returns remain unchanged."
    )
    annual_fee_percent = st.number_input(
        "Illustrative annual management fee (%)",
        min_value=0.0,
        max_value=2.0,
        value=0.5,
        step=0.05,
        format="%.2f",
        key=f"{widget_prefix}_management_fee",
    )
    st.caption("Illustrative fee only; not an actual product charge.")
    original = rows.set_index("fund")["recommended_weight"].astype(float)
    adjusted_values = {}
    columns = st.columns(2)
    for index, (fund, weight) in enumerate(original.items()):
        adjusted_values[fund] = (
            columns[index % 2].number_input(
                f"{fund} (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(weight * 100),
                step=0.5,
                format="%.2f",
                key=f"{widget_prefix}_{fund}",
            )
            / 100
        )
    adjusted = pd.Series(adjusted_values, dtype=float)
    total = adjusted.sum()
    st.metric("Adjusted allocation total", f"{total:.2%}")
    if not allocation_total_is_valid(adjusted):
        st.warning("Adjusted fund weights must sum to 100.00% before results are shown.")
        return

    comparison = compare_allocations(a.returns, original, adjusted)
    fund_crypto = rows.set_index("fund")["crypto_exposure"].astype(float)
    original_crypto = lookthrough_crypto(original, fund_crypto)
    adjusted_crypto = lookthrough_crypto(adjusted, fund_crypto)
    scenario = a.suitability_config["crypto_scenarios"][profile][scenario_key]
    crypto_limit = float(scenario["maximum_exposure"])
    if adjusted_crypto > crypto_limit + 1e-8:
        st.warning(
            f"Adjusted crypto exposure is {adjusted_crypto:.1%}, above this "
            f"scenario's {crypto_limit:.1%} limit by "
            f"{adjusted_crypto - crypto_limit:.1%}."
        )
    else:
        st.caption(
            f"Adjusted crypto exposure remains within the {crypto_limit:.1%} scenario limit."
        )
    if comparison.annualization == 365:
        st.info(
            "Metrics use 365-day annualisation because this allocation includes a crypto-only fund."
        )
    st.caption(
        f"Both allocations use the same calendar and common start date: "
        f"{comparison.start_date:%d %b %Y}."
    )
    metrics = comparison.metrics.copy()
    metrics["Look-through crypto"] = [original_crypto, adjusted_crypto]
    st.dataframe(
        metrics,
        hide_index=True,
        width="stretch",
        column_config={
            "Annual return": st.column_config.NumberColumn(format="percent"),
            "Volatility": st.column_config.NumberColumn(format="percent"),
            "Sharpe ratio": st.column_config.NumberColumn(format="%.2f"),
            "Maximum drawdown": st.column_config.NumberColumn(format="percent"),
            "Final value of $1": st.column_config.NumberColumn(format="$%.2f"),
            "Look-through crypto": st.column_config.NumberColumn(format="percent"),
        },
    )
    st.plotly_chart(allocation_growth_chart(comparison.growth), width="stretch")
    fee_growth = fee_adjusted_growth(
        comparison.returns["User-adjusted"],
        annual_fee_percent / 100.0,
        comparison.annualization,
    )
    st.subheader("Illustrative fee impact")
    fee_cols = st.columns(2)
    fee_cols[0].metric(
        "Gross historical growth of $1",
        f"${fee_growth['Gross historical growth of $1'].iloc[-1]:.2f}",
    )
    fee_cols[1].metric(
        "Net-of-fee historical growth of $1",
        f"${fee_growth['Net-of-fee historical growth of $1'].iloc[-1]:.2f}",
    )
    st.plotly_chart(allocation_growth_chart(fee_growth), width="stretch")
    st.caption(
        "Fees reduce the value of an investment over time, so the net-of-fee path "
        "shows the historical portfolio growth after the illustrative annual charge."
    )


def render_suitability(a: AppArtifacts) -> None:
    _heading(
        "Investor profile",
        "Find your fit",
        "Five questions connect your risk capacity and preference to a "
        "precomputed illustrative allocation.",
    )
    config = a.suitability_config
    with st.form("suitability_form"):
        answers: dict[str, str] = {}
        for key, question in config["questions"].items():
            option_keys = list(question["options"])
            labels = {item: question["options"][item]["label"] for item in option_keys}
            selected_label = st.radio(
                question["prompt"], [labels[item] for item in option_keys], horizontal=True
            )
            answers[key] = next(item for item in option_keys if labels[item] == selected_label)
        preference_labels = {"none": "None", "limited": "Limited", "higher": "Higher"}
        selected_preference = st.radio(
            "How much cryptocurrency exposure would you consider?",
            list(preference_labels.values()),
            horizontal=True,
        )
        preference = next(
            key for key, label in preference_labels.items() if label == selected_preference
        )
        submitted = st.form_submit_button("See my illustrative allocation", type="primary")

    if submitted:
        st.session_state["profile_result"] = score_questionnaire(answers, config)
        st.session_state["crypto_preference"] = preference
        st.session_state["confirm_balanced_25"] = False
    result = st.session_state.get("profile_result")
    if result is None:
        st.info(
            "Complete the questionnaire to reveal a profile and matching "
            "precomputed recommendation."
        )
        return
    preference = st.session_state["crypto_preference"]
    profile_cols = st.columns(3)
    profile_cols[0].metric("Raw score", f"{result.score}/15")
    profile_cols[1].metric("Initial profile", result.raw_profile)
    profile_cols[2].metric("Final profile", result.final_profile)
    for reason in result.override_reasons:
        st.info(f"Profile safeguard: {reason}")
    if not result.override_reasons:
        st.caption("No profile override was required.")
    main = recommendation_rows(a.recommendations, result.final_profile, preference)
    main = main.loc[main["scenario_key"] != "higher_25_alternative"]
    _show_allocation(main, "Main recommendation")
    _render_allocation_adjuster(
        a,
        main,
        result.final_profile,
        str(main["scenario_key"].iloc[0]),
        f"main_{result.final_profile}_{preference}",
    )

    ranks = a.rankings.loc[a.rankings["profile"] == result.final_profile].sort_values("rank")
    with st.expander("How the funds ranked"):
        st.dataframe(
            ranks[["rank", "fund", "ranking_score"]].rename(
                columns={"rank": "Rank", "fund": "Fund", "ranking_score": "Profile score"}
            ),
            hide_index=True,
            width="stretch",
        )

    if result.final_profile == "Balanced" and preference == "higher":
        st.warning(
            "Higher-risk alternative: the crypto-only funds in the historical sample had "
            "annualised volatility near 80% and drawdowns around 80%. A 25% crypto limit can "
            "materially increase portfolio losses. This is not the main recommendation."
        )
        confirmed = st.checkbox(
            "I understand the additional historical volatility and drawdown risk, "
            "and want to view the optional 25% scenario.",
            key="confirm_balanced_25",
        )
        if confirmed:
            alternative = recommendation_rows(
                a.recommendations, "Balanced", "higher", "higher_25_alternative"
            )
            _show_allocation(alternative, "Optional 25% crypto scenario")
            _render_allocation_adjuster(
                a,
                alternative,
                "Balanced",
                "higher_25_alternative",
                "balanced_higher_25",
            )

    st.caption(
        "Illustrative educational output only. The app looks up existing rankings "
        "and allocations; it does not optimise a portfolio."
    )
