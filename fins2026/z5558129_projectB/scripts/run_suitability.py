"""Build rankings and nine illustrative suitability recommendations."""
from __future__ import annotations

import importlib
import json
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

suitability = importlib.import_module("src.suitability")

RESULTS_DATA = ROOT / "results" / "data"
RESULTS_TABLES = ROOT / "results" / "tables"


def main() -> None:
    metrics = pd.read_csv(RESULTS_TABLES / "performance_metrics.csv")
    fund_weights = pd.read_csv(
        RESULTS_DATA / "fund_weights.csv", parse_dates=["rebalance_date"]
    )
    grid = suitability.build_recommendation_grid(metrics, fund_weights)
    rankings = grid["rankings"]
    allocations = grid["allocations"]
    replacements = grid["replacements"]
    summary = grid["summary"]

    config_path = RESULTS_DATA / "suitability_config.json"
    ranking_path = RESULTS_TABLES / "fund_rankings.csv"
    allocation_path = RESULTS_DATA / "suitability_recommendations.csv"
    replacement_path = RESULTS_TABLES / "suitability_replacements.csv"
    summary_path = RESULTS_TABLES / "suitability_examples.csv"
    config_path.write_text(
        json.dumps(suitability.config_as_dict(), indent=2), encoding="utf-8"
    )
    rankings.to_csv(ranking_path, index=False)
    allocations.to_csv(allocation_path, index=False)
    replacements.to_csv(replacement_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("Illustrative suitability recommendations")
    print(summary.to_string(index=False))
    print("\nAllocations")
    print(
        allocations[
            [
                "profile",
                "crypto_preference",
                "scenario_label",
                "selection_order",
                "fund",
                "template_weight",
                "recommended_weight",
                "crypto_contribution",
            ]
        ].to_string(index=False)
    )
    print("\nReplacements")
    print(replacements.to_string(index=False) if len(replacements) else "None")
    for path in (
        config_path,
        ranking_path,
        allocation_path,
        replacement_path,
        summary_path,
    ):
        print(f"Saved: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
