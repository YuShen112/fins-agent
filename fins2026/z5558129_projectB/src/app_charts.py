"""Plotly chart builders for precomputed EasyFunds artifacts."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

NAVY = "#14213D"
GOLD = "#C49A4A"
TEAL = "#3D7C78"
MUTED = "#8A94A6"

PERIOD_OFFSETS = {
    "1M": pd.DateOffset(months=1),
    "3M": pd.DateOffset(months=3),
    "6M": pd.DateOffset(months=6),
    "1Y": pd.DateOffset(years=1),
    "2Y": pd.DateOffset(years=2),
}


def visible_period(data: pd.DataFrame, date_column: str, period: str) -> pd.DataFrame:
    """Return a display-only trailing window ending at the latest date."""
    if period == "ALL" or data.empty:
        return data.copy()
    if period not in PERIOD_OFFSETS:
        raise ValueError(f"Unsupported chart period: {period}")
    latest = data[date_column].max()
    return data.loc[data[date_column] >= latest - PERIOD_OFFSETS[period]].copy()


def _finish(fig: go.Figure, y_title: str) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=430,
        margin=dict(l=8, r=8, t=24, b=8),
        font=dict(family="Inter, system-ui, sans-serif", color=NAVY),
        legend_title_text="",
        hovermode="x unified",
        xaxis_title=None,
        yaxis_title=y_title,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#E9EDF2")
    return fig


def growth_chart(returns: pd.DataFrame, funds: list[str], period: str = "ALL") -> go.Figure:
    data = returns.loc[returns["fund"].isin(funds)].sort_values("date")
    data = visible_period(data, "date", period)
    fig = px.line(data, x="date", y="growth_of_1", color="fund")
    return _finish(fig, "Growth of $1")


def drawdown_chart(returns: pd.DataFrame, fund: str, period: str = "ALL") -> go.Figure:
    data = returns.loc[returns["fund"] == fund].sort_values("date").copy()
    data["drawdown"] = data["growth_of_1"] / data["growth_of_1"].cummax() - 1
    data = visible_period(data, "date", period)
    fig = px.area(data, x="date", y="drawdown", color_discrete_sequence=[GOLD])
    fig.update_yaxes(tickformat=".0%")
    return _finish(fig, "Drawdown")


def holdings_chart(weights: pd.DataFrame) -> go.Figure:
    data = weights.sort_values("target_weight", ascending=True)
    fig = px.bar(
        data,
        x="target_weight",
        y="ticker",
        orientation="h",
        color="asset_class",
        color_discrete_map={"Equity": TEAL, "Crypto": GOLD},
    )
    fig.update_xaxes(tickformat=".0%")
    return _finish(fig, "Target weight")


def allocation_growth_chart(growth: pd.DataFrame) -> go.Figure:
    """Compare original and adjusted allocation growth on their shared calendar."""
    data = (
        growth.rename_axis("date")
        .reset_index()
        .melt(id_vars="date", var_name="Allocation", value_name="growth_of_1")
    )
    fig = px.line(data, x="date", y="growth_of_1", color="Allocation")
    return _finish(fig, "Growth of $1")


def monthly_sentiment_chart(sentiment: pd.DataFrame, period: str = "ALL") -> go.Figure:
    data = sentiment.dropna(subset=["sector_sentiment_index"]).copy()
    data["month"] = data["trading_date"].dt.to_period("M").dt.to_timestamp()
    monthly = data.groupby(["month", "sector"], as_index=False)["sector_sentiment_index"].mean()
    monthly = visible_period(monthly, "month", period)
    fig = px.line(monthly, x="month", y="sector_sentiment_index", color="sector")
    fig.add_hline(y=0, line_color=MUTED, line_width=1)
    return _finish(fig, "Average finVADER sentiment")


def monthly_coverage_chart(sentiment: pd.DataFrame, period: str = "ALL") -> go.Figure:
    data = sentiment.dropna(subset=["covered_ticker_proportion"]).copy()
    data["month"] = data["trading_date"].dt.to_period("M").dt.to_timestamp()
    monthly = data.groupby(["month", "sector"], as_index=False)["covered_ticker_proportion"].mean()
    monthly = visible_period(monthly, "month", period)
    fig = px.line(monthly, x="month", y="covered_ticker_proportion", color="sector")
    fig.add_hline(y=0.60, line_dash="dot", line_color=GOLD, annotation_text="60% tilt threshold")
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    return _finish(fig, "Ticker coverage")
