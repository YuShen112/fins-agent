# EasyFunds Part B Agent Instructions

## Project purpose

Build EasyFunds, a prototype investment app for users with limited investment experience who are interested in systematic investing. The app should simplify the comparison of systematically managed equity, cryptocurrency, and combined funds while still allowing users to consider clearly labelled higher-risk investment options.

EasyFunds should emphasise transparency, risk explanation, and user choice rather than restricting inexperienced users to low-risk products only.

## Core product decisions

- App name: EasyFunds.
- Target users: investors with limited investment experience who are interested in systematic investing and want clear explanations of risk, performance, and portfolio construction.
- The app may offer higher-risk products, including cryptocurrency funds, provided that risks and historical drawdowns are clearly communicated.
- Build the following eight funds:
  1. Equity Equal Weight
  2. Equity Minimum Variance
  3. Equity Maximum Sharpe
  4. Crypto Equal Weight
  5. Crypto Risk Parity
  6. Combined Minimum Variance
  7. Combined Maximum Sharpe
  8. Combined Risk Parity
- All portfolios are initially long-only.
- Initial maximum individual asset weight: 10%. This may be revised later if concentration, diversification, solver diagnostics, or out-of-sample evidence justify a change.
- Maximum total cryptocurrency exposure in combined funds: 25%.
- Initial estimation window: 252 equity trading days.
- Rebalance frequency: monthly.
- Initial risk-free rate: 0.
- Initial transaction costs: 0.
- Primary innovation: a transparent, data-driven investor suitability allocation engine combining:
  - a five-question investor risk-profile questionnaire;
  - a separate cryptocurrency-preference constraint;
  - out-of-sample fund performance and risk metrics;
  - an illustrative fund allocation that users may subsequently adjust.

## Critical modelling rules

- Never use future data when estimating portfolio weights or generating investment signals.
- Portfolio weights for each holding period must be formed only using information available before that holding period begins.
- Use a walk-forward out-of-sample backtest.
- The first live backtest date must occur after the initial estimation window.
- Sentiment used for investment decisions must be lagged by at least one equity trading day.
- Compute equity and cryptocurrency returns separately before aligning calendars.
- Compute cryptocurrency returns before mapping them onto the equity trading calendar.
- Use adjusted closing prices to calculate returns.
- Do not remove genuine extreme returns merely because they are large.
- Combined-fund cryptocurrency exposure must not exceed 25%.
- Initial individual asset weights must not exceed 10%.
- Confirm that optimisation methods produce meaningfully different weights.
- Test all important assumptions, including:
  - weight sums;
  - long-only constraints;
  - individual asset caps;
  - total cryptocurrency caps;
  - date ordering;
  - missing values;
  - annualisation factors;
  - rebalance dates;
  - absence of look-ahead bias.
- Clearly report optimisation failures, fallback methods, or solver warnings rather than silently replacing results.
- Keep financial calculations separate from presentation and Streamlit code.

## Part A data foundation

- Reuse the Part A logic and outputs where appropriate.
- Load source data through `src/data_access.py`.
- Do not commit raw source data.
- The expected Part A derived structures are:
  - a combined return panel containing one date column, 50 equity return columns, and 10 cryptocurrency return columns;
  - a daily headline panel containing `trading_date`, `ticker`, `sector`, `headline_count`, and `headline_text`.
- Treat the existing Part A outputs as validation benchmarks, but ensure the Part B project remains reproducible from the provided hosted data.

## Sentiment rules

- Apply VADER or another documented sentiment model to the original headline text.
- Do not remove punctuation, casing, negation, intensifiers, or other text features required by the selected sentiment model unless the effect is tested and justified.
- Construct ticker-day sentiment before producing the sector sentiment index.
- Equal-weight tickers when constructing sector-level sentiment.
- Explicitly document how ticker-days with no headlines are treated.
- Any sentiment-based portfolio signal must apply only to equities.
- Compare the base portfolio with the sentiment-augmented portfolio.
- A negative sentiment-fusion result is acceptable and must be interpreted honestly.

## Investor suitability feature

- Treat the suitability output as educational and illustrative, not personalised financial advice.
- Use five questionnaire items covering:
  - investment objective;
  - time horizon;
  - tolerable temporary loss;
  - likely behaviour during a market decline;
  - preference between stability and potential growth.
- Classify users initially as Conservative, Balanced, or Growth.
- Use a separate cryptocurrency-preference control rather than including cryptocurrency preference directly in the profile score.
- Cryptocurrency preference options should support:
  - no cryptocurrency exposure;
  - limited cryptocurrency exposure;
  - higher cryptocurrency exposure.
- A user selecting higher cryptocurrency exposure must receive a clear warning about historical volatility and drawdown risk.
- Base fund recommendations on out-of-sample performance and risk metrics rather than fixed marketing labels alone.
- Allow users to modify the illustrative allocation after receiving the model suggestion.
- Show how the modified allocation changes historical return, volatility, drawdown, and cryptocurrency exposure.
- Clearly explain why the recommended funds were selected.

## App and deployment rules

- The deployed Streamlit app must read precomputed artifacts from `results/`.
- Do not run portfolio backtests, optimisation, VADER, or other heavy modelling inside `streamlit_app.py`.
- Do not commit raw data, API keys, credentials, or secrets.
- Keep the app capable of running on Streamlit Community Cloud.
- The app must support the investor journey:
  - compare funds;
  - open each fund fact sheet;
  - review growth of $1, drawdown, Sharpe ratio, risk, and current holdings;
  - complete the suitability questionnaire;
  - receive and adjust an illustrative allocation;
  - view equity-sector sentiment analytics.
- Use a restrained, minimal, premium visual direction with neutral and low-saturation colours.
- Keep visual settings centralised so colours and styling can be changed later without rewriting the full app.
- Prioritise readability and clear risk communication over decoration.

## Project structure

- `src/`: reusable data, portfolio, sentiment, fusion, suitability, and metrics functions.
- `scripts/`: runnable scripts that reproduce project outputs.
- `results/data/`: precomputed CSV files read by the app.
- `results/tables/`: report-ready tables.
- `results/figures/`: report-ready figures.
- `report/`: report source and final PDF.
- `ai/`: curated prompt logs, AI outputs, corrections, decisions, and reflections.
- `streamlit_app.py`: lightweight app entrypoint only.
- Do not place the full modelling pipeline inside one large file.

## Working process

- Read `PROJECT_BRIEF.md`, `README.md`, and the files under `context/` before making substantial changes.
- Work only inside this Part B project folder.
- Make small, testable changes.
- Before editing, inspect the relevant existing files and explain the intended change.
- After editing, summarise:
  - files created;
  - files changed;
  - tests run;
  - assumptions made;
  - remaining issues.
- Do not overwrite working code without explaining why.
- Add validation checks for every important financial or data assumption.
- Prefer clear, reusable functions with descriptive names and docstrings.
- Avoid unnecessary complexity and duplicated code.
- Keep the project runnable end to end through `scripts/run_part_b.py`.
- Record important prompts, mistakes, corrections, model decisions, and reasons in `ai/`.
- Do not generate final economic interpretation or report conclusions without first examining the actual outputs.
- Flag uncertainty and ask for a decision when a modelling choice materially affects the product or results.
