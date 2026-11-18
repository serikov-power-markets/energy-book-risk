# Risk of a TTF trading book

Daily marks, P&L attribution, limits and VaR and ES backtests for an illustrative gas book through the
2021-2023 crisis, with Black-76, Kirk's spread approximation and a swing by least-squares Monte Carlo.
A research study on the public TTF front-month series, not a production system or a trading strategy.

## What is real and what is assumed

Only the front-month price is observed (Yahoo TTF=F, October 2017 on). There is no open archive of forward
curves or option prices, so the rest is constructed:
- **Curve:** near maturities follow the front month and far ones move less, with the mean reversion from
  [eu-gas-storage-valuation](https://github.com/serikov-power-markets/eu-gas-storage-valuation), a seasonal shape
  and the mean price of the past year as the long-run level.
- **Volatility:** EWMA of front-month returns (lambda 0.97), reduced with maturity, floor 20%. It stands in
  for implied volatility.
- **Book:** a fixed recipe reset each month: a 200 GWh calendar spread in futures, a long call, a short put
  and a calendar spread option. It exists to carry delta, gamma, vega and curve risk, so its total profit
  means nothing.
- **Power is not in the daily book,** because no open history of power forwards exists. Kirk's formula for
  a spark-spread shape is only checked against Monte Carlo.

## Results

Book from October 2018 to October 2026 (2,016 days), risk forecasts from October 2019 (1,766 days).

### VaR and ES backtest, one day at 99%, on hypothetical P&L

| Model | Exceptions (expected 17.7) | Kupiec p | Christoffersen p | Worst 250 days | Mean VaR, kEUR |
|---|---|---|---|---|---|
| Historical simulation, full revaluation | 29 | 0.013 | 0.091 | 10, red | 841 |
| Filtered historical simulation | 31 | 0.004 | 0.292 | 10, red | 763 |
| Delta-gamma on historical returns | 24 | 0.151 | 0.335 | 8, yellow | 878 |
| Delta-normal with EWMA volatility | 13 | 0.242 | 0.661 | 4, green | 835 |

Historical simulation fails coverage, and in the crisis its 9 exceptions cluster (Christoffersen p = 0.012).
A window of 250 days reacts late when volatility jumps. Filtered simulation was expected to fix this and did
not: it does better in the crisis (6 exceptions) but worse afterward (17 against 9 expected), apparently because
it scales past returns down when volatility falls. Delta-normal passes all tests here, though the sample holds a book with little
gamma, and on exception days the realized loss was 1.31 times its expected shortfall, as with the other models
(1.25 to 1.29), so all of them understate the tail beyond VaR.

### P&L attribution

Delta explains most of the P&L (mean absolute 249 kEUR per day), then time decay and roll along the curve
(30), gamma (12), vega (7) and the level of the curve (6). Unexplained is 4 kEUR, 0.06% of the variance. The
largest gap was on 29 August 2022: a loss of 7.7 MEUR, with 236 kEUR unexplained after a 20% fall.

The Basel-style test (Spearman above 0.99, KS 0.005) is green, but this proves little. Marks and sensitivities come
from one model, so the test only measures the quality of the Taylor expansion. A real desk would fail on factors
the model leaves out, such as basis and independent curve moves.

An earlier version showed a large unexplained P&L on the first day of each month. The cause was in the curve,
not the market: fixed delivery months jumped when the front contract changed. Time now runs continuously.

### Limits

Limits come from a rule fixed before the test: VaR limit twice the mean VaR of the first tested year
(226 kEUR), stop loss three times that, delta limit 150 GWh of front-month equivalent.

| | Value |
|---|---|
| Days over the VaR limit | 1,383 of 1,766 |
| Days over the delta limit | 408 |
| Months with a stop-loss breach | 27 |

A limit calibrated in a calm year is breached on 78% of later days, so it stops carrying information. Sizing
each monthly book to stay inside the VaR limit cuts the worst day from 7.7 to 0.68 MEUR and the worst month
from 10.0 to 0.93 MEUR. Daily volatility falls by a factor of 5.6 and the total result by 2.7. Positions change
only at month start, so the limit still fails within a month.

### Pricing checks

- **Kirk against Monte Carlo:** four cases with 400,000 antithetic paths: within 1.5 standard errors, and Kirk
  is lower in all four, by 0.05% to 0.8%, so the gap is small but one-sided.
- **Swing by LSMC,** one month of daily rights at the money, volatility 80%, mean reversion 2.9: with all 30
  rights 52.11 (standard error 0.35) against the exact strip of calls 52.30; one right 2.61 against the best
  single call 2.44. The value of a right falls from 2.61 to 1.74 as rights increase.

| Rights | 1 | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|---|
| Value, EUR/MWh of daily volume | 2.61 | 12.55 | 23.78 | 33.61 | 41.91 | 48.31 | 52.11 |

## Limits of the study

Illustrative curve, volatility and positions. One market factor, plus volatility and the slow level. The
correlation between the legs of a spread is assumed. Zero rates, no bid-ask spread, no liquidity, margin or
credit risk. The swing is not part of the daily book. Never a substitute for an ETRM or a desk's risk system.

## Run

`pip install -e ".[test]"`, `pytest` (offline, synthetic data), `python scripts/run_study.py` (downloads TTF=F
to `data/`, which is not committed; under half an hour on one core).
