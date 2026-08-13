"""Local app tests: artifact contracts, scoring, lookups, and rendered views."""

from pathlib import Path

import pandas as pd
from src.app_charts import visible_period
from src.app_data import load_artifacts, recommendation_rows, score_questionnaire
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = load_artifacts(PROJECT_ROOT / "results")


def test_artifact_contracts_and_precomputed_recommendations():
    assert ARTIFACTS.metrics["fund"].nunique() == 8
    assert ARTIFACTS.returns["fund"].nunique() == 8
    main = recommendation_rows(ARTIFACTS.recommendations, "Balanced", "higher")
    assert set(main["scenario_key"]) == {"higher", "higher_25_alternative"}
    alternative = recommendation_rows(
        ARTIFACTS.recommendations, "Balanced", "higher", "higher_25_alternative"
    )
    assert abs(alternative["recommended_weight"].sum() - 1) < 1e-12
    assert abs(alternative["crypto_contribution"].sum() - 0.25) < 1e-8
    promoted = pd.read_csv(
        PROJECT_ROOT / "results/data/finvader_sector_sentiment_index.csv",
        parse_dates=["trading_date"],
    )
    pd.testing.assert_frame_equal(
        ARTIFACTS.sentiment.reset_index(drop=True),
        promoted.reset_index(drop=True),
        check_dtype=False,
    )


def test_questionnaire_scoring_and_overrides_from_artifact_config():
    config = ARTIFACTS.suitability_config
    growth = {key: list(question["options"])[-1] for key, question in config["questions"].items()}
    assert score_questionnaire(growth, config).final_profile == "Growth"
    growth["horizon"] = "short"
    assert score_questionnaire(growth, config).final_profile == "Balanced"
    growth["loss_tolerance"] = "low"
    assert score_questionnaire(growth, config).final_profile == "Conservative"


def _app() -> AppTest:
    return AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py"), default_timeout=20).run()


def test_each_app_view_renders_without_exception():
    app = _app()
    assert not app.exception
    assert "Compare the eight funds" in [title.value for title in app.title]
    for index, expected in [(1, "A closer look"), (2, "Sector pulse"), (3, "Find your fit")]:
        app.sidebar.radio[0].set_value(app.sidebar.radio[0].options[index]).run()
        assert not app.exception
        assert expected in [title.value for title in app.title]


def test_visible_period_filters_a_copy_without_changing_full_data():
    original = ARTIFACTS.returns.copy()
    visible = visible_period(original, "date", "1M")
    assert visible["date"].max() == original["date"].max()
    assert visible["date"].min() >= original["date"].max() - pd.DateOffset(months=1)
    pd.testing.assert_frame_equal(original, ARTIFACTS.returns)


def test_chart_period_controls_offer_required_ranges():
    app = _app()
    assert app.radio[0].options == ["1M", "3M", "6M", "1Y", "ALL"]
    app.sidebar.radio[0].set_value("Sector pulse").run()
    assert app.radio[0].options == ["6M", "1Y", "2Y", "ALL"]


def test_fact_sheet_limits_chart_and_exposes_full_holdings():
    app = _app()
    app.sidebar.radio[0].set_value("Fund fact sheet").run()
    assert "Top 10 holdings by target weight" in [item.value for item in app.caption]
    assert len(app.expander) == 1
    subheaders = [item.value for item in app.subheader]
    assert "Downside risk" in subheaders
    assert "Recovery" in subheaders
    assert "Diversification and relative position" in subheaders
    metric_labels = [metric.label for metric in app.metric]
    assert "Downside deviation" in metric_labels
    assert "Sortino ratio" in metric_labels
    assert "Longest drawdown" in metric_labels
    assert "Maximum-drawdown recovery" in metric_labels
    assert "Effective holdings" in metric_labels
    assert "Top-5 concentration" in metric_labels
    metric_values = {metric.label: metric.value for metric in app.metric}
    assert metric_values["Longest drawdown"] == "461 days"
    assert metric_values["Maximum-drawdown recovery"] == "391 days"
    captions = [item.value for item in app.caption]
    assert any("Durations are calendar days." in item for item in captions)
    assert any("equivalent number of evenly weighted positions" in item for item in captions)
    assert not any("sum of squared weights" in item for item in captions)


def test_sentiment_sector_selector_filters_without_exception():
    app = _app()
    app.sidebar.radio[0].set_value("Sector pulse").run()
    assert app.selectbox[0].label == "Sector"
    app.selectbox[0].set_value(app.selectbox[0].options[1]).run()
    assert not app.exception
    model_notes = [item.value for item in app.info if "finVADER" in item.value]
    assert len(model_notes) == 1
    assert "SentiBignomics and Henry" in model_notes[0]
    assert "VADER is retained as the benchmark" in model_notes[0]


def test_balanced_alternative_requires_confirmation():
    app = _app()
    app.sidebar.radio[0].set_value("Find your fit").run()
    # Default answers total 5. Choose middle answers to produce Balanced.
    for radio in app.radio[:5]:
        radio.set_value(radio.options[1])
    app.radio[5].set_value("Higher")
    app.button[0].click().run()
    metric_labels = [metric.label for metric in app.metric]
    assert metric_labels[:3] == ["Raw score", "Initial profile", "Final profile"]
    assert "Main recommendation" in [item.value for item in app.subheader]
    assert "Optional 25% crypto scenario" not in [item.value for item in app.subheader]
    assert len(app.checkbox) == 1
    app.checkbox[0].check().run()
    assert "Optional 25% crypto scenario" in [item.value for item in app.subheader]


def test_allocation_adjuster_shows_illustrative_fee_control_and_outputs():
    app = _app()
    app.sidebar.radio[0].set_value("Find your fit").run()
    for radio in app.radio[:5]:
        radio.set_value(radio.options[1])
    app.radio[5].set_value("Limited")
    app.button[0].click().run()
    fee_inputs = [
        item for item in app.number_input if item.label == "Illustrative annual management fee (%)"
    ]
    assert fee_inputs and fee_inputs[0].value == 0.5
    captions = [item.value for item in app.caption]
    assert "Illustrative fee only; not an actual product charge." in captions
    assert any("Fees reduce the value of an investment over time" in item for item in captions)
    metric_labels = [metric.label for metric in app.metric]
    assert "Gross historical growth of $1" in metric_labels
    assert "Net-of-fee historical growth of $1" in metric_labels
