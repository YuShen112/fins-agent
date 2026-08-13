# AGENTS.md

## Project
FINS3645 FinTech Project 2026 Part B — EasyFunds.

## Product goal
EasyFunds is designed for novice investors. The app should explain risk and return clearly rather than simply promoting the fund with the highest historical return.

## Portfolio rules
- Long-only portfolios.
- Monthly rebalancing.
- Equity and combined funds use a 252-day estimation window.
- Crypto-only funds use a 365-day estimation window.
- Equity-only individual asset cap: 10%.
- Crypto-only individual asset cap: 25%.
- Combined individual asset cap: 10%.
- Combined total crypto exposure cap: 25%.
- Risk-free rate = 0.
- Transaction costs = 0.

## Sentiment rules
- finVADER is the primary sentiment model.
- VADER is retained as a benchmark.
- Sentiment is lagged by one equity trading day.
- Sector sentiment can affect portfolio weights only when coverage is at least 60%.
- Sentiment is a supporting market-monitoring signal, not the main recommendation driver.

## App rules
- Streamlit should read precomputed portfolio and sentiment outputs.
- Do not recompute portfolio optimisation or sentiment scoring inside the app.
- Suitability recommendations must remain explainable.
- Users may adjust recommended allocations.
- Show warnings when adjusted crypto exposure exceeds the suitability limit rather than silently changing the allocation.

## Code quality
- Keep functions modular and testable.
- Do not modify canonical outputs without checking downstream dependencies.
- Run relevant tests after material changes.
- Preserve existing working behaviour unless a change is explicitly requested.
