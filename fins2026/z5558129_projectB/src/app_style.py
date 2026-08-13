"""Central visual system for EasyFunds."""

import streamlit as st


def apply_style() -> None:
    st.markdown(
        """
        <style>
        :root { --ink:#14213D; --muted:#667085; --gold:#C49A4A; --paper:#F7F8FA; }
        .stApp { background: var(--paper); }
        .block-container { max-width: 1240px; padding-top: 2.4rem; padding-bottom: 4rem; }
        h1, h2, h3 { color: var(--ink); letter-spacing: -0.025em; }
        h1 { font-size: 2.45rem !important; }
        .brand { color: var(--ink); font-size: 1.28rem; font-weight: 900;
          letter-spacing: .13em; margin: .35rem 0; }
        [data-testid="stMetric"] { background:white; border:1px solid #E5E9EF; padding:1rem;
          border-radius:12px; box-shadow:0 4px 18px rgba(20,33,61,.035); }
        [data-testid="stMetricLabel"] { color:var(--muted); }
        .eyebrow { color:var(--gold); font-weight:700; letter-spacing:.12em;
          text-transform:uppercase; font-size:.76rem; }
        .allocation { background:white; border:1px solid #E5E9EF; padding:1rem 1.15rem;
          border-radius:12px; margin:.45rem 0; }
        .allocation strong { color:var(--ink); font-size:1.05rem; }
        div[data-testid="stAlert"] { border-radius:12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
