"""Export the report-ready Combined Risk Parity weights-over-time figure."""

from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.report_figures import save_combined_risk_parity_weights_figure  # noqa: E402


def main() -> None:
    weights = pd.read_csv(ROOT / "results/data/fund_weights.csv")
    output, summary = save_combined_risk_parity_weights_figure(
        weights,
        ROOT / "results/figures/combined_risk_parity_weights_over_time_report.png",
    )
    print(f"Saved: {output.relative_to(ROOT)}")
    print(f"Average crypto weight: {summary['average_crypto_weight']:.2%}")
    print(f"Maximum crypto weight: {summary['maximum_crypto_weight']:.2%}")
    print(f"Average active holdings: {summary['average_active_holdings']:.1f}")
    print(
        "Weights sum to 100% at every rebalance date: "
        f"{'Yes' if summary['weights_sum_to_one'] else 'No'}"
    )


if __name__ == "__main__":
    main()
