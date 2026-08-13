"""A4-ready static portfolio and coverage exhibits from existing artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.figures import SECTOR_COLORS, monthly_sentiment_summary

FIGURE_SIZE = (10.0, 6.0)
FIGURE_DPI = 300

EQUITY_SECTORS = {
    "Tech": ("NVDA", "AMD", "INTC", "QCOM", "ADBE"),
    "Financials": ("GS", "MS", "WFC", "V", "USB"),
    "Energy": ("XOM", "CVX", "COP", "SLB", "OXY"),
    "Consumer": ("DIS", "WMT", "NKE", "SBUX", "KO"),
    "Industrials": ("GE", "BA", "CAT", "UPS", "MMM"),
    "Healthcare": ("MRK", "ABBV", "AMGN", "GILD", "ABT"),
    "Comm": ("T", "CMCSA", "TMUS", "EA", "TTWO"),
    "Materials": ("SHW", "NEM", "DOW", "NUE", "DD"),
    "Utilities": ("NEE", "DUK", "SO", "D", "AEP"),
    "RealEstate": ("AMT", "O", "PLD", "CCI", "PSA"),
}

WEIGHT_CATEGORY_COLORS = {
    **SECTOR_COLORS,
    "Crypto": "#B57614",
}

WEIGHT_CATEGORY_ORDER = [
    "Tech",
    "Financials",
    "Energy",
    "Consumer",
    "Industrials",
    "Healthcare",
    "Comm",
    "Materials",
    "Utilities",
    "RealEstate",
    "Crypto",
]

FUND_COLORS = {
    "Equity Equal Weight": "#4E79A7",
    "Equity Minimum Variance": "#76B7B2",
    "Equity Maximum Sharpe": "#59A14F",
    "Crypto Equal Weight": "#E15759",
    "Crypto Risk Parity": "#F28E2B",
    "Combined Minimum Variance": "#B07AA1",
    "Combined Maximum Sharpe": "#9C755F",
    "Combined Risk Parity": "#EDC948",
}

SHORT_LABELS = {
    "Equity Equal Weight": "Equity equal weight",
    "Equity Minimum Variance": "Equity minimum variance",
    "Equity Maximum Sharpe": "Equity maximum Sharpe",
    "Crypto Equal Weight": "Crypto equal weight",
    "Crypto Risk Parity": "Crypto risk parity",
    "Combined Minimum Variance": "Combined minimum variance",
    "Combined Maximum Sharpe": "Combined maximum Sharpe",
    "Combined Risk Parity": "Combined risk parity",
}


def _validate_funds(frame: pd.DataFrame, required_columns: set[str]) -> None:
    if missing := required_columns.difference(frame.columns):
        raise ValueError(f"fund artifact is missing columns: {sorted(missing)}")
    missing_funds = set(FUND_COLORS).difference(frame["fund"].unique())
    if missing_funds:
        raise ValueError(f"fund artifact is missing funds: {sorted(missing_funds)}")


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="#444444", labelsize=8.5)


def _save(fig: plt.Figure, output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=FIGURE_DPI, facecolor="white")
    plt.close(fig)
    return destination


def save_growth_figure(
    fund_returns: pd.DataFrame,
    output_path: str | Path,
    *,
    include_crypto_only: bool,
    y_scale: str = "log",
) -> Path:
    """Plot precomputed fund growth paths using the requested display scale."""
    if y_scale not in {"linear", "log"}:
        raise ValueError("y_scale must be 'linear' or 'log'")
    _validate_funds(fund_returns, {"date", "fund", "growth_of_1"})
    data = fund_returns.copy()
    data["date"] = pd.to_datetime(data["date"])
    if not include_crypto_only:
        data = data.loc[~data["fund"].str.startswith("Crypto ")]
    funds = [fund for fund in FUND_COLORS if fund in set(data["fund"])]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    for fund in funds:
        group = data.loc[data["fund"] == fund].sort_values("date")
        ax.plot(
            group["date"],
            group["growth_of_1"],
            color=FUND_COLORS[fund],
            linewidth=1.55,
            alpha=0.92,
            label=SHORT_LABELS[fund],
        )
    title = (
        "Growth of $1 Across All Funds"
        if include_crypto_only
        else "Growth of $1 Excluding Crypto-Only Funds"
    )
    ax.set_title(title, loc="left", fontsize=13, fontweight="semibold", pad=10)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio value" if y_scale == "linear" else "Portfolio value (log scale)")
    ax.set_yscale(y_scale)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:.2f}"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _style_axis(ax)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3 if len(funds) <= 6 else 4,
        frameon=False,
        fontsize=7.8,
    )
    fig.text(
        0.01,
        0.022,
        "Note: precomputed walk-forward out-of-sample growth paths; crypto-only "
        "funds use the daily calendar,\nwhile equity and combined funds use the "
        "equity calendar. Sample: Jan 2021 to Dec 2023. Source: project backtests.",
        fontsize=7.4,
        color="#555555",
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.28)
    return _save(fig, output_path)


def combined_risk_parity_weight_summary(
    fund_weights: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | bool]]:
    """Aggregate precomputed Combined Risk Parity weights into report categories."""
    required = {"rebalance_date", "fund", "ticker", "asset_class", "target_weight"}
    if missing := required.difference(fund_weights.columns):
        raise ValueError(f"fund weights are missing columns: {sorted(missing)}")
    data = fund_weights.loc[fund_weights["fund"] == "Combined Risk Parity"].copy()
    if data.empty:
        raise ValueError("Combined Risk Parity weights are unavailable")
    data["rebalance_date"] = pd.to_datetime(data["rebalance_date"])
    sector_by_ticker = {
        ticker: sector for sector, tickers in EQUITY_SECTORS.items() for ticker in tickers
    }
    equity = data["asset_class"].eq("Equity")
    data["category"] = "Crypto"
    data.loc[equity, "category"] = data.loc[equity, "ticker"].map(sector_by_ticker)
    if missing_tickers := sorted(data.loc[equity & data["category"].isna(), "ticker"].unique()):
        raise ValueError(f"equity tickers are missing sector mappings: {missing_tickers}")

    totals = data.groupby("rebalance_date")["target_weight"].sum()
    sums_to_one = bool(np.isclose(totals.to_numpy(), 1.0, atol=1e-8).all())
    if not sums_to_one:
        failed = totals.loc[~np.isclose(totals, 1.0, atol=1e-8)]
        raise ValueError(f"weights do not sum to one: {failed.to_dict()}")
    crypto = data.loc[data["category"] == "Crypto"].groupby("rebalance_date")["target_weight"].sum()
    active = data.loc[data["target_weight"] > 1e-8].groupby("rebalance_date")["ticker"].nunique()
    summary = {
        "average_crypto_weight": float(crypto.mean()),
        "maximum_crypto_weight": float(crypto.max()),
        "average_active_holdings": float(active.mean()),
        "weights_sum_to_one": sums_to_one,
    }
    aggregated = (
        data.pivot_table(
            index="rebalance_date",
            columns="category",
            values="target_weight",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(columns=WEIGHT_CATEGORY_ORDER, fill_value=0.0)
        .sort_index()
    )
    return aggregated, summary


def save_combined_risk_parity_weights_figure(
    fund_weights: pd.DataFrame,
    output_path: str | Path,
) -> tuple[Path, dict[str, float | bool]]:
    """Export sector-level Combined Risk Parity target weights over time."""
    aggregated, summary = combined_risk_parity_weight_summary(fund_weights)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.stackplot(
        aggregated.index,
        *[aggregated[column] for column in aggregated.columns],
        labels=aggregated.columns,
        colors=[WEIGHT_CATEGORY_COLORS[column] for column in aggregated.columns],
        alpha=0.92,
        linewidth=0.35,
        edgecolor="white",
    )
    ax.set_title(
        "Combined Risk Parity - Portfolio Weights Over Time",
        loc="left",
        fontsize=13,
        fontweight="semibold",
        pad=10,
    )
    ax.set_xlabel("Rebalance date")
    ax.set_ylabel("Portfolio weight (%)")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _style_axis(ax)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=len(WEIGHT_CATEGORY_ORDER),
        frameon=False,
        fontsize=6.8,
        columnspacing=0.85,
        handlelength=1.5,
    )
    fig.text(
        0.01,
        0.052,
        "Crypto exposure remained moderate over the sample: average 6.96%, maximum 11.83%.",
        fontsize=7.6,
        color="#333333",
    )
    fig.text(
        0.01,
        0.016,
        "Note: equity target weights are aggregated by sector; all crypto holdings "
        "are shown as one category.\nSample: Jan 2021 to Dec 2023. Source: "
        "precomputed monthly project weights.",
        fontsize=7.4,
        color="#555555",
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.27)
    return _save(fig, output_path), summary


def save_return_volatility_scatter(
    metrics: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot annualized return against volatility with collision-aware labels."""
    _validate_funds(metrics, {"fund", "annualized_return", "annualized_volatility"})
    offsets = {
        "Equity Equal Weight": (-104, -4),
        "Equity Minimum Variance": (9, -16),
        "Equity Maximum Sharpe": (-108, -14),
        "Crypto Equal Weight": (-104, -15),
        "Crypto Risk Parity": (-98, 10),
        "Combined Minimum Variance": (9, 11),
        "Combined Maximum Sharpe": (9, -13),
        "Combined Risk Parity": (9, 10),
    }
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    for row in metrics.itertuples(index=False):
        color = FUND_COLORS[row.fund]
        ax.scatter(
            row.annualized_volatility,
            row.annualized_return,
            s=58,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.annotate(
            SHORT_LABELS[row.fund],
            (row.annualized_volatility, row.annualized_return),
            xytext=offsets[row.fund],
            textcoords="offset points",
            fontsize=7.7,
            color="#333333",
            arrowprops={"arrowstyle": "-", "color": "#888888", "lw": 0.6},
            va="center",
        )
    ax.set_title(
        "Annual Return Versus Volatility",
        loc="left",
        fontsize=13,
        fontweight="semibold",
        pad=10,
    )
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized return")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlim(0.08, 0.88)
    ax.set_ylim(0.025, 0.475)
    _style_axis(ax)
    fig.text(
        0.01,
        0.025,
        "Note: annualization uses 252 days for equity and combined funds and 365 "
        "days for crypto-only funds.\nSample: Jan 2021 to Dec 2023. Source: "
        "project backtests.",
        fontsize=7.4,
        color="#555555",
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.18)
    return _save(fig, output_path)


def save_drawdown_comparison(
    metrics: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Compare maximum peak-to-trough losses across all eight funds."""
    _validate_funds(metrics, {"fund", "maximum_drawdown"})
    data = metrics[["fund", "maximum_drawdown"]].copy()
    data["drawdown_magnitude"] = data["maximum_drawdown"].abs()
    data = data.sort_values("drawdown_magnitude", ascending=True)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    bars = ax.barh(
        [SHORT_LABELS[fund] for fund in data["fund"]],
        data["drawdown_magnitude"],
        color=[FUND_COLORS[fund] for fund in data["fund"]],
        height=0.64,
    )
    ax.bar_label(
        bars,
        labels=[f"{value:.1%}" for value in data["drawdown_magnitude"]],
        padding=4,
        fontsize=8,
        color="#333333",
    )
    ax.set_title(
        "Maximum Drawdown by Fund",
        loc="left",
        fontsize=13,
        fontweight="semibold",
        pad=10,
    )
    ax.set_xlabel("Maximum drawdown (loss from prior peak)")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlim(0, data["drawdown_magnitude"].max() * 1.13)
    _style_axis(ax)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.grid(axis="y", visible=False)
    fig.text(
        0.01,
        0.025,
        "Note: maximum drawdown is the largest peak-to-trough decline in each "
        "precomputed out-of-sample growth path. Sample: Jan 2021 to Dec 2023. "
        "Source: project backtests.",
        fontsize=7.4,
        color="#555555",
    )
    fig.subplots_adjust(left=0.25, right=0.97, top=0.90, bottom=0.17)
    return _save(fig, output_path)


def save_coverage_report_figure(
    sector_index: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Export unclipped monthly coverage with the fusion eligibility threshold."""
    monthly = monthly_sentiment_summary(sector_index)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    for sector in sorted(monthly["sector"].unique()):
        group = monthly.loc[monthly["sector"] == sector]
        color = SECTOR_COLORS[sector]
        ax.plot(
            group["month"],
            group["monthly_coverage"],
            color=color,
            linewidth=1.35,
            alpha=0.88,
            label=sector,
        )
        highlighted = group.loc[group["extreme_low_coverage"]]
        ax.scatter(
            highlighted["month"],
            highlighted["monthly_coverage"],
            s=46,
            facecolor=color,
            edgecolor="#111111",
            linewidth=1.0,
            zorder=5,
        )
    ax.axhline(0.60, color="#666666", linewidth=0.9, linestyle="--")
    ax.text(
        0.99,
        0.605,
        "60% tilt threshold",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
    )
    ax.set_title(
        "Monthly Sector News Coverage",
        loc="left",
        fontsize=13,
        fontweight="semibold",
        pad=10,
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Constituent tickers with at least one headline")
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _style_axis(ax)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="#111111",
            markersize=6,
        )
    )
    labels.append("Extreme sentiment with low coverage")
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=4,
        frameon=False,
        fontsize=7.8,
    )
    fig.text(
        0.01,
        0.022,
        "Note: monthly mean share of each sector's five tickers with news before "
        "the one-day lag; dashed line marks the 60% fusion threshold.\n"
        "Sample: Jan 2020 to Dec 2023. Source: course project headlines.",
        fontsize=7.4,
        color="#555555",
    )
    fig.subplots_adjust(left=0.10, right=0.985, top=0.90, bottom=0.30)
    return _save(fig, output_path)
