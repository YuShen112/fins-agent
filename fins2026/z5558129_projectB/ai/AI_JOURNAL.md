# AI Journal - EasyFunds Part B

This journal records important uses of the coding assistant. The prompt wording and
project decisions below come from the project conversation and saved outputs. The
entries distinguish the assistant's work from decisions and corrections made by the
student.

---

# Prompt log - Portfolio concentration limits

## What I wanted
I wanted long-only funds with limits that controlled concentration but still allowed
the optimisation methods to produce different portfolios.

## Prompt(s)
The earlier project instruction set an initial 10% maximum individual asset weight
for the portfolios. After reviewing the crypto-only case, I corrected this with:

"The crypto-only cap must be higher because 10 assets with a 10% cap would force
every asset to exactly 10%, making Equal Weight and Risk Parity identical."

I then specified the final constraints: "Equity-only: 10% individual cap; Combined:
10% individual cap and 25% total crypto cap; Crypto-only: 25% individual cap."

## What the assistant produced
The assistant implemented and tested the three constraint sets across the eight
funds. It also reported optimiser failures explicitly rather than silently replacing
a method.

## What was wrong or risky
The earlier 10% individual cap was suitable for the 50-stock equity universe and the
combined funds, but applying it to ten crypto assets would leave only one feasible
allocation: 10% in every asset. This would make the crypto methods mechanically
identical.

## What I changed and why
I kept the 10% individual cap for equity and combined funds and the 25% total crypto
cap for combined funds. I raised the crypto-only individual cap to 25% so Equal
Weight and Risk Parity could produce meaningfully different weights.

---

# Prompt log - Trading calendars, estimation windows and annualisation

## What I wanted
I wanted the eight-fund backtest to respect the different trading frequencies of
equities and cryptocurrencies.

## Prompt(s)
"Use the equity calendar and 252-day annualisation for equity and combined funds,
and the native crypto calendar with 365-day annualisation for crypto-only funds."

"The 365-observation window for crypto-only funds is approved."

## What the assistant produced
The assistant expanded the engine so equity and combined funds used equity trading
dates, a 252-observation estimation window and 252-day annualisation. Crypto-only
funds used daily crypto dates, a 365-observation window and 365-day annualisation.

## What was wrong or risky
Using one common calendar or 252-day annualisation for every fund would understate
the number of crypto trading days. Using a 252-observation crypto window would also
represent a shorter period of history than the equity window.

## What I changed and why
I explicitly approved a 365-observation window for crypto-only funds. This made the
estimation horizon approximately one year for both asset classes while preserving
their native calendars.

---

# Prompt log - No-look-ahead sentiment controls

## What I wanted
I wanted a daily sector sentiment signal that could be used in a historical test
without using information that was unavailable at the decision date.

## Prompt(s)
"Score the daily headline panel with VADER, create ticker-day sentiment, lag it by
one equity trading day, build the sector sentiment index, and save
sector_sentiment_index.csv."

"Use the lagged sector signal only when at least 3 of the 5 sector tickers have
headlines that day. If coverage is below 60%, do not tilt that sector."

## What the assistant produced
The assistant aggregated multiple headlines to ticker-day scores, retained a
complete equity-date grid, separated no-news observations from neutral scores,
equal-weighted tickers by sector, and shifted the signal by one equity trading day.
The fusion code applied a multiplier of one when coverage was below 60%.

## What was wrong or risky
Same-day headlines can arrive after a portfolio decision. Treating them as available
for that day's trade would create look-ahead bias. Thin news coverage could also make
a sector score depend on only one or two firms.

## What I changed and why
I required a one-equity-trading-day lag and a minimum of three covered tickers out of
five. These rules use only prior information and stop low-coverage sector signals
from changing weights.

---

# Prompt log - VADER and finVADER robustness test

## What I wanted
I wanted to test a finance-specific sentiment model without deleting or changing the
existing VADER benchmark.

## Prompt(s)
"Please add a finVADER robustness check without replacing the current VADER pipeline
yet."

"Enable both finance lexicons: use_sentibignomics=True, use_henry=True."

"The finVADER robustness results support using finVADER as the primary sentiment
model."

## What the assistant produced
The assistant scored the same deduplicated headlines with finVADER, reproduced the
same aggregation, coverage and lag pipeline, and saved separate outputs. The
comparison table reported a headline correlation of 0.604, exact-zero proportions
of 48.85% for VADER and 16.86% for finVADER, and a sector-month correlation of
0.692.

## What was wrong or risky
Fewer zero scores and finance-specific lexicons do not prove that finVADER is more
accurate because the project has no manually labelled sentiment truth set. Replacing
VADER outputs would also remove the benchmark needed for robustness comparison.

## What I changed and why
I promoted finVADER only as the primary reporting and app model because it provides
broader finance-domain coverage. I kept every VADER artifact unchanged and retained
VADER as the benchmark. I did not claim that finVADER had verified predictive
accuracy.

---

# Prompt log - Sentiment tilt experiment and negative result

## What I wanted
I wanted a simple test of whether the lagged sector signal improved an existing fund
without changing the sentiment method between experiments.

## Prompt(s)
"Use the same raw sector-sentiment tilt as the first fusion: multiplier = 1 + lagged
sector sentiment, 60% coverage threshold, no centring, scaling, or new threshold."

"Apply it to Combined Maximum Sharpe instead of Equity Equal Weight. Only reallocate
the equity holdings; keep the base fund's crypto weights unchanged."

## What the assistant produced
The assistant applied the same transparent tilt to Equity Equal Weight and Combined
Maximum Sharpe, enforced long-only weights and the individual cap, and preserved the
combined fund's crypto weights. It repeated both tests with finVADER so only the
sentiment input changed.

## What was wrong or risky
Both primary finVADER tilts reduced annualised return and Sharpe. Equity Equal Weight
return fell from 12.64% to 12.09% and Sharpe from 0.819 to 0.790. Combined Maximum
Sharpe return fell from 14.73% to 14.48% and Sharpe from 0.784 to 0.773. It would be
misleading to describe the signal as improving investment performance.

## What I changed and why
I kept and reported the negative result. I treated sector sentiment as supporting
market information in EasyFunds rather than a proven source of portfolio
outperformance. I did not tune the formula after seeing the result.

---

# Prompt log - Suitability profile overrides

## What I wanted
I wanted a five-question score that also respected an investor's capacity to take
risk.

## Prompt(s)
"If either horizon under 3 years or loss tolerance under 10% applies, cap the profile
at Balanced; if both apply, classify as Conservative; Growth requires both a long
horizon and high loss tolerance."

## What the assistant produced
The assistant first proposed score-based Conservative, Balanced and Growth cutoffs.
It then implemented the revised override rules in configurable suitability logic and
showed the raw score, initial profile, final profile and override reason in the app.

## What was wrong or risky
A total score alone could classify someone as Growth even when a short horizon or
low loss tolerance showed limited capacity to accept losses.

## What I changed and why
I added downside-only overrides. Either limiting condition caps the result at
Balanced, both conditions produce Conservative, and Growth requires both a long
horizon and high loss tolerance. The overrides can reduce risk but cannot increase
it.

---

# Prompt log - Profile-specific crypto exposure rules

## What I wanted
I wanted crypto preference to constrain recommendations differently for
Conservative, Balanced and Growth investors.

## Prompt(s)
"Please revise the crypto limits in the suitability engine."

"Conservative: Limited max 5%, Higher max 10%, no 25% alternative. Balanced:
Limited max 10%, Higher default max 15%, and a separate optional 25% higher-risk
alternative. Growth: Limited max 15%, Higher max 25%."

## What the assistant produced
The assistant had initially implemented preference-wide limits of 0%, 10% and 25%.
It then kept the ranking and allocation logic unchanged while implementing the
profile-specific limits and precomputing the Balanced 25% alternative.

## What was wrong or risky
A common 25% Higher limit ignored differences in risk profile. In particular, a 25%
crypto allocation was too aggressive as the main Balanced recommendation given the
crypto funds' high historical volatility and drawdowns.

## What I changed and why
I introduced lower limits for Conservative and Balanced profiles. The Balanced 25%
scenario became a separately labelled higher-risk alternative with a strong warning
and explicit confirmation in Streamlit. The main Balanced Higher recommendation is
capped at 15%.

---

# Prompt log - Fund Fact Sheet improvements

## What I wanted
I wanted the fact sheet to explain downside risk, recovery, diversification and a
fund's position relative to the other seven funds.

## Prompt(s)
"Please enhance the Fund Fact Sheet with three finance-focused analysis blocks."

"Define the Sortino ratio explicitly as (annualised return - MAR) / annualised
downside deviation, with MAR = 0%. Keep recovery durations in calendar days and
label them clearly as calendar days in the UI."

"Change recovery cards from truncated text like '461 calenda...' to simple values
such as '461 days' and '391 days', with 'calendar days' explained in the small note
below."

## What the assistant produced
The assistant added downside deviation, Sortino ratio, longest drawdown, recovery
from maximum drawdown, effective holdings, top-five concentration, and four ranks
among the eight funds. The calculations use only precomputed fund returns, weights
and performance metrics.

## What was wrong or risky
The first plan needed a precise Sortino numerator and explicit calendar-day units.
The first UI wording was also too long for metric cards, causing values such as
"461 calendar days" to be truncated. The effective-holdings formula was too
technical for the main investor-facing note.

## What I changed and why
I specified MAR = 0% and required calendar-day recovery measures. I later shortened
the cards to "days", moved the calendar-day explanation below them, and replaced the
formula with a plain-language explanation. The calculations and ranking logic were
not changed.

---

# Prompt log - Portfolio weights over time figure

## What I wanted
I wanted to add a portfolio-weights-over-time exhibit to show how sector and crypto exposure changed across monthly rebalances.

## Prompt(s)
"Create a report-ready portfolio weights over time figure for Combined Risk Parity using the existing precomputed fund_weights.csv. Aggregate equity holdings by sector and show crypto as one separate category."

Later, I asked the assistant to make the legend order match the stacked-area order, keep Crypto as the top layer, and add the average and maximum crypto exposure to the figure.

## What the assistant produced
The assistant created a stacked area chart using the 36 precomputed monthly rebalance weights. It reported average crypto exposure of 6.96%, maximum crypto exposure of 11.83%, and confirmed that weights summed to 100% at every rebalance.

## What was wrong or risky
The first version was analytically correct, but the legend order did not match the stack order and the financial interpretation of crypto exposure was not immediately visible.

## What I changed and why
I asked for the legend to follow the stack order, kept Crypto as the top layer, and added the average and maximum crypto exposure directly to the figure. This made the chart easier to read and more useful for financial interpretation without changing the underlying data or portfolio calculations.

---

# Prompt log - Illustrative management fee

## What I wanted
I wanted the allocation section to show how an annual management fee could affect the historical value of a user-adjusted portfolio.

## Prompt(s)
"Add an illustrative annual management fee to the adjusted allocation and show gross and net-of-fee historical growth."

## What the assistant produced
The assistant added an adjustable annual management fee to the allocation section and displayed both gross and net-of-fee historical growth of $1. It also added a chart comparing the two growth paths.

## What was wrong or risky
The fee could be mistaken for an actual EasyFunds product charge. The fee calculation also needed to remain consistent with the existing 252-day and 365-day return calendars.

## What I changed and why
I required the fee to be clearly labelled as illustrative only and kept all existing gross portfolio metrics unchanged. I also required tests to confirm that a 0% fee produces identical gross and net results and that a positive fee reduces the final portfolio value consistently under both calendar conventions.
