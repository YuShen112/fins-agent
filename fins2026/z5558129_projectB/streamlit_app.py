"""EasyFunds: a local, artifact-driven investment dashboard."""

from __future__ import annotations

import pathlib
import sys

import streamlit as st

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.app_data import load_artifacts  # noqa: E402
from src.app_style import apply_style  # noqa: E402
from src.app_views import (  # noqa: E402
    render_comparison,
    render_fact_sheet,
    render_sentiment,
    render_suitability,
)

st.set_page_config(
    page_title="EasyFunds",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_style()


@st.cache_data(show_spinner="Loading portfolio research…")
def _load_artifacts():
    return load_artifacts(PROJECT_ROOT / "results")


try:
    artifacts = _load_artifacts()
except (FileNotFoundError, ValueError) as exc:
    st.error(f"The precomputed research artifacts could not be loaded: {exc}")
    st.stop()

with st.sidebar:
    st.markdown("<div class='brand'>◆ EASYFUNDS</div>", unsafe_allow_html=True)
    st.caption("Systematic portfolios, made inspectable.")
    view = st.radio(
        "Explore",
        ["Compare funds", "Fund fact sheet", "Sector pulse", "Find your fit"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption(
        "Educational analysis based on historical backtests. Not personal "
        "financial advice. Past performance is not a guarantee of future results."
    )

if view == "Compare funds":
    render_comparison(artifacts)
elif view == "Fund fact sheet":
    render_fact_sheet(artifacts)
elif view == "Sector pulse":
    render_sentiment(artifacts)
else:
    render_suitability(artifacts)
