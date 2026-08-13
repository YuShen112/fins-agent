"""Self-contained report figures for the Part B project artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from matplotlib.lines import Line2D

SECTOR_COLORS = {
    "Comm": "#4E79A7",
    "Consumer": "#F28E2B",
    "Energy": "#E15759",
    "Financials": "#76B7B2",
    "Healthcare": "#59A14F",
    "Industrials": "#EDC948",
    "Materials": "#B07AA1",
    "RealEstate": "#FF9DA7",
    "Tech": "#9C755F",
    "Utilities": "#BAB0AC",
}


def monthly_sentiment_summary(sector_index: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the daily index and flag sector-relative low-coverage extremes."""
    required = {
        "trading_date",
        "sector",
        "sector_sentiment_index",
        "covered_ticker_proportion",
        "source_headline_count",
    }
    if missing := required.difference(sector_index.columns):
        raise ValueError(f"sector_index is missing columns: {sorted(missing)}")

    data = sector_index.copy()
    data["trading_date"] = pd.to_datetime(data["trading_date"])
    data = data.dropna(subset=["sector_sentiment_index", "covered_ticker_proportion"])
    data["month"] = data["trading_date"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        data.groupby(["month", "sector"], as_index=False)
        .agg(
            monthly_sentiment=("sector_sentiment_index", "mean"),
            monthly_coverage=("covered_ticker_proportion", "mean"),
            source_headlines=("source_headline_count", "sum"),
            trading_days=("trading_date", "nunique"),
        )
        .sort_values(["month", "sector"])
        .reset_index(drop=True)
    )
    sector_group = monthly.groupby("sector", sort=False)
    means = sector_group["monthly_sentiment"].transform("mean")
    standard_deviations = sector_group["monthly_sentiment"].transform("std")
    monthly["sentiment_z_score"] = (monthly["monthly_sentiment"] - means) / standard_deviations
    monthly["sector_low_coverage_threshold"] = sector_group["monthly_coverage"].transform(
        lambda values: values.quantile(0.25)
    )
    monthly["extreme_low_coverage"] = monthly["sentiment_z_score"].abs().ge(2.0) & monthly[
        "monthly_coverage"
    ].le(monthly["sector_low_coverage_threshold"])
    return monthly


def _style_axis(ax: plt.Axes) -> None:
    """Apply a restrained report style to one axis."""
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="#444444", labelsize=8.5)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def _plot_monthly_lines(
    monthly: pd.DataFrame,
    *,
    value_column: str,
    ylabel: str,
    title: str,
    output_path: str | Path,
    percentage_axis: bool = False,
    source_model: str = "VADER",
    include_sample: bool = False,
) -> Path:
    """Plot ten monthly sector series and mark flagged observations."""
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    for sector in sorted(monthly["sector"].unique()):
        group = monthly.loc[monthly["sector"] == sector]
        color = SECTOR_COLORS[sector]
        ax.plot(
            group["month"],
            group[value_column],
            color=color,
            linewidth=1.35,
            alpha=0.88,
            label=sector,
        )
        highlighted = group.loc[group["extreme_low_coverage"]]
        ax.scatter(
            highlighted["month"],
            highlighted[value_column],
            s=48,
            facecolor=color,
            edgecolor="#111111",
            linewidth=1.1,
            zorder=5,
        )

    if value_column == "monthly_sentiment":
        ax.axhline(0, color="#666666", linewidth=0.8, linestyle="--", zorder=1)
    if percentage_axis:
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.set_title(title, loc="left", fontsize=13, fontweight="semibold", pad=10)
    ax.set_xlabel("Month")
    ax.set_ylabel(ylabel)
    _style_axis(ax)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#FFFFFF",
            markeredgecolor="#111111",
            markersize=6,
            label="Extreme sentiment with low coverage",
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
        fontsize=8,
    )
    sample_note = ""
    if include_sample:
        sample_note = f" Sample: {monthly['month'].min():%b %Y} to {monthly['month'].max():%b %Y}."
    fig.text(
        0.01,
        0.012,
        "Note: monthly averages of the one-equity-trading-day-lagged index. "
        "Outlined points have |sector z-score| >= 2 and coverage at or below "
        f"that sector's 25th percentile.{sample_note} "
        f"Source: course project headlines; {source_model}.",
        fontsize=7.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.105, 1, 1))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return destination


def save_sentiment_figures(
    sector_index: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[Path, Path, pd.DataFrame]:
    """Save monthly sentiment and coverage figures plus their monthly data."""
    monthly = monthly_sentiment_summary(sector_index)
    output = Path(output_dir)
    sentiment_path = _plot_monthly_lines(
        monthly,
        value_column="monthly_sentiment",
        ylabel="Mean VADER compound score",
        title="Monthly Sector Sentiment",
        output_path=output / "sector_sentiment_monthly.png",
    )
    coverage_path = _plot_monthly_lines(
        monthly,
        value_column="monthly_coverage",
        ylabel="Tickers with at least one headline",
        title="Monthly Sector News Coverage",
        output_path=output / "sector_coverage_monthly.png",
        percentage_axis=True,
    )
    return sentiment_path, coverage_path, monthly


def save_finvader_sentiment_figure(
    sector_index: pd.DataFrame,
    output_path: str | Path,
) -> pd.DataFrame:
    """Save the parallel finVADER monthly sector sentiment robustness figure."""
    monthly = monthly_sentiment_summary(sector_index)
    _plot_monthly_lines(
        monthly,
        value_column="monthly_sentiment",
        ylabel="Mean finVADER compound score",
        title="Monthly Sector Sentiment — finVADER",
        output_path=output_path,
        source_model="finVADER",
    )
    return monthly


def save_primary_finvader_sentiment_figure(
    sector_index: pd.DataFrame,
    output_path: str | Path,
) -> pd.DataFrame:
    """Save the report-primary finVADER figure with explicit sample context."""
    monthly = monthly_sentiment_summary(sector_index)
    _plot_monthly_lines(
        monthly,
        value_column="monthly_sentiment",
        ylabel="Mean finVADER compound score",
        title="Monthly Sector Sentiment — finVADER",
        output_path=output_path,
        source_model="finVADER",
        include_sample=True,
    )
    return monthly
