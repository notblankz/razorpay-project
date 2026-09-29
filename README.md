# Financial Portfolio Rebalancing Advisor

A backend service that models a portfolio of simulated stocks and bonds,
detects when it has drifted from a target allocation, and produces a concrete,
tax- and cost-aware plan of buy/sell trades to bring it back — with an
explanation for every trade.

Built in **Python + Flask**. All market data is **simulated** (no external
APIs). The "AI" layer is a transparent **statistical signal engine**, not a
trained model — see [Why statistics, not ML](#why-statistics-not-ml).

---

## Table of contents

- [What it does](#what-it-does)
- [Requirement coverage](#requirement-coverage)
- [Key design decisions](#key-design-decisions)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Running the server](#running-the-server)
- [API reference](#api-reference)
- [The signals, explained](#the-signals-explained)
- [The rebalancing strategy](#the-rebalancing-strategy)
- [Simulated tax & transaction costs](#simulated-tax--transaction-costs)
- [Testing](#testing)
- [Assumptions & limitations](#assumptions--limitations)

---

## What it does

1. **Simulates** ~1 year of daily prices for a mixed universe of 6 assets
   (4 stocks, 2 bonds) using Geometric Brownian Motion.
2. **Values** a portfolio and computes its allocation by ticker and by asset
   class (stock vs. bond).
3. **Computes signals** per asset: moving averages, momentum, volatility,
   Sharpe ratio, and a trend label.
4. **Detects drift** from a target allocation using the **5/25 rule**.
5. **Plans trades** to correct drift, choosing what to sell in a way that
   minimizes simulated capital-gains tax, and skipping trades too small to be
   worth their transaction cost.

---

## Requirement coverage

| Problem requirement | Where it lives |
|---|---|
| Model assets with simulated price data | `scripts/generate_prices.py`, `data/prices.csv` |
| Define target asset allocations | `target` input on `/drift` and `/rebalance-plan` (default 60/40) |
| Calculate current allocation | `app/portfolio.py` |
| Predict market trends & recommend rebalancing | `app/signals.py` + `app/rebalance.py` |
| Automated detection of significant drift | `app/rebalance.py` (`detect_bands`, 5/25 rule) |
| Optimal buy/sell w/ transaction cost & tax | `app/rebalance.py` (`plan_trades`) |
| Accurate value / % / cost calculations | `app/portfolio.py`, `app/rebalance.py` |
| Robust error handling for data | `app/data_loader.py` (`DataError`) |
| Tests for rebalancing, prediction, financial calcs | `tests/` (109 tests) |

> **Personalized investment advice from risk tolerance** is the one remaining
> "AI-assisted" item and is intended as a follow-up (`app/advice.py`): a small
> rule engine mapping a risk profile — Conservative / Moderate / Aggressive —
> to a default target allocation and advice text. It is not yet implemented.

---

## Key design decisions

### Why statistics, not ML

The problem statement frames several features as "AI." In practice none of them
benefit from a trained model, and a statistical engine is the stronger choice:

- **No learnable signal.** Over simulated Geometric Brownian Motion data there
  is nothing to learn beyond drift and volatility; an ML model would overfit
  noise. A moving-average / momentum signal is more defensible.
- **Explainability.** Financial advice must be justifiable. Every trade this
  system emits carries a plain-language reason ("stock overweight by 16.8%;
  trimming (downtrend, long-term gain)"). A black-box model cannot do that.
- **Determinism = testability.** "Prediction accuracy" is trivial to test with
  a deterministic signal (feed a known rising series, assert `uptrend`) and
  meaningless for a model trained ad hoc.

### Why Python only (not Go + Python)

An earlier plan split a Go API from a Python analytics service. For a
time-boxed project the cross-process integration cost outweighs the benefit, so
everything lives in one Flask app. The logic modules are pure Python with no
Flask imports, so the boundary between "web layer" and "domain logic" is still
clean.

### Why stateless

The API keeps **no server-side portfolio** between requests. Each call is
independent: holdings come either from the committed sample
(`data/holdings.csv`) or from the request body. This keeps the demo fully
reproducible and free of file-mutation / concurrency concerns. Persisting a
per-user portfolio would require storage and auth, which are out of scope.

---

## Architecture

```
                        HTTP / JSON (Postman, curl)
                                   │
                    ┌──────────────▼──────────────┐
                    │        app/routes.py         │   thin Flask layer:
                    │  (parse → call logic → JSON) │   no calculations
                    └──────────────┬──────────────┘
                                   │
        ┌──────────────┬───────────┴───────────┬──────────────┐
        ▼              ▼                       ▼              ▼
  data_loader.py   portfolio.py            signals.py     rebalance.py
  load + validate  value / alloc /         SMA/EMA/mom/   5/25 bands +
  the CSV files    drift (pure math)       vol/Sharpe/    tax-aware trade
                                           trend          planner
        │
        ▼
  data/*.csv  (simulated prices, asset metadata, sample holdings)
```

**Layering rule:** `routes.py` only wires HTTP to logic. Every other module is
pure Python (`portfolio`, `signals`, `rebalance` do no I/O; `data_loader` is the
sole gateway to disk). This is what makes the money math unit-testable to exact
numbers without a running server.

---

## Project structure

```
.
├── app/
│   ├── __init__.py        # Flask app factory (create_app)
│   ├── routes.py          # HTTP endpoints (thin)
│   ├── data_loader.py     # load + validate CSVs; raises DataError
│   ├── portfolio.py       # value, allocation, drift (pure math)
│   ├── signals.py         # SMA/EMA/momentum/volatility/Sharpe/trend
│   └── rebalance.py       # 5/25 bands + cost/tax-aware trade planner
├── scripts/
│   └── generate_prices.py # GBM price simulator -> data/*.csv
├── data/
│   ├── prices.csv         # simulated daily prices (committed)
│   ├── assets.csv         # ticker, name, asset_class
│   └── holdings.csv       # sample portfolio (committed fixture)
├── tests/                 # pytest suite (109 tests)
├── conftest.py            # puts repo root on sys.path for tests
├── requirements.txt
└── README.md
```

---

## Setup

Requires **Python 3.11+** (developed on 3.13).

```bash
python -m pip install -r requirements.txt
```

Dependencies: **Flask** (API), **pandas** (CSV loading/validation),
**pytest** (tests).

### Regenerate the simulated data (optional)

The data files are committed, but you can regenerate them reproducibly:

```bash
python scripts/generate_prices.py --days 252 --seed 42
```

Flags: `--days`, `--seed`, `--start` (ISO date), `--out-dir`.

---

## Running the server

```bash
python -m flask --app app run --debug
```

Serves at `http://127.0.0.1:5000`. `--debug` enables auto-reload. Stop with
Ctrl+C. (This is Flask's development server — appropriate for this project, not
for production.)

---

## API reference

Base URL: `http://127.0.0.1:5000`.
Valid tickers: **VOO, QQQ, AAPL, MSFT** (stock) · **BND, TLT** (bond).

| Method | Route | Body | Purpose |
|---|---|---|---|
| GET | `/health` | — | Liveness check |
| GET | `/portfolio` | — | Summarize the sample portfolio |
| POST | `/portfolio` | `holdings` (required) | Summarize a supplied portfolio |
| GET | `/signals` | — | Per-asset trend & risk signals |
| POST | `/drift` | `holdings`, `target` (both optional) | Drift vs. target + band breaches |
| GET | `/rebalance-plan` | — | Full plan for the sample vs. 60/40 |
| POST | `/rebalance-plan` | `holdings`, `target` (both optional) | Full plan for supplied inputs |

On `POST /drift` and `POST /rebalance-plan`, omitting `holdings` uses the sample
portfolio and omitting `target` uses a 60/40 stock/bond default — so both are
callable with an empty body `{}`.

**Errors:** bad caller input (unknown ticker, negative shares, malformed body)
→ `400`; a broken/missing committed data file → `500`. Both return
`{"error": "..."}`.

### Examples

`GET /health`
```bash
curl http://127.0.0.1:5000/health
# {"status":"ok"}
```

`GET /portfolio` — value, per-position detail, allocation by ticker and class.

`POST /portfolio`
```json
{"holdings":[{"ticker":"VOO","shares":50},{"ticker":"BND","shares":100}]}
```

`GET /signals` — for each ticker: `latest_price`, `sma_short`, `sma_long`,
`momentum`, `annualized_volatility`, `sharpe_ratio`, `trend`.

`POST /drift` (empty body uses sample + 60/40)
```json
{}
```
Returns `current_allocation`, `target_allocation`, `drift`, `band_breaches`,
`needs_rebalance`.

`POST /rebalance-plan` (custom, exercising the tax logic)
```json
{
  "holdings": [
    {"ticker":"VOO","shares":80,"cost_basis":450,"long_term":true},
    {"ticker":"AAPL","shares":40,"cost_basis":170,"long_term":false},
    {"ticker":"BND","shares":20,"cost_basis":71,"long_term":true}
  ],
  "target": {"stock":0.6,"bond":0.4}
}
```

Example plan output (abridged):
```json
{
  "portfolio_value": 66687.4,
  "current_allocation": {"stock": 0.768, "bond": 0.232},
  "target_allocation": {"stock": 0.6, "bond": 0.4},
  "drift": {"stock": 0.168, "bond": -0.168},
  "band_breaches": {"stock": true, "bond": true},
  "needs_rebalance": true,
  "estimated_transaction_cost": 22.4,
  "estimated_tax": 40.09,
  "trades": [
    {"ticker":"AAPL","action":"SELL","shares":40.0,"notional":5850.4,
     "est_tax":0.0,"reason":"stock overweight by 16.8%; trimming (uptrend, loss lot, no tax)"},
    {"ticker":"QQQ","action":"SELL","shares":13.38,"notional":5350.06,
     "est_tax":40.09,"reason":"stock overweight by 16.8%; trimming (downtrend, long-term gain)"},
    {"ticker":"BND","action":"BUY","shares":108.57,"notional":7569.51,"est_tax":0.0},
    {"ticker":"TLT","action":"BUY","shares":36.19,"notional":3630.95,"est_tax":0.0}
  ]
}
```

Note the tax-aware sequencing: the **loss lot (AAPL) is sold first** despite
being in an uptrend, then the long-term gain (QQQ); the untouched positions
avoid realizing further gains.

---

## The signals, explained

All operate on one asset's price series (oldest → newest).

- **SMA (Simple Moving Average)** — average of the last *N* prices. Smooths
  daily noise to reveal the price level; lags by design.
- **EMA (Exponential Moving Average)** — like SMA but weights recent prices
  more (`α = 2/(N+1)`), so it reacts faster.
- **Momentum** — percent change over the last *N* days; direction and speed.
- **Volatility (annualized)** — standard deviation of daily returns × √252; the
  core risk measure.
- **Sharpe ratio (annualized)** — return per unit of risk; higher is better.
  Returns 0.0 when volatility is zero (rather than dividing by zero).
- **trend_label** — combines an SMA crossover with momentum:
  short SMA meaningfully above long SMA **and** positive momentum → `uptrend`;
  the mirror → `downtrend`; otherwise `flat`. Used only to sequence sells.

---

## The rebalancing strategy

**Threshold-band rebalancing (the "5/25 rule").** Rather than rebalancing on a
fixed calendar, the system acts only when an asset class drifts outside a
tolerance band: **5 absolute percentage points, or 25% of its target weight,
whichever is smaller.** This trades only when it matters, controlling risk with
fewer transactions.

*Why this over calendar rebalancing:* calendar rebalancing trades on a schedule
regardless of need (wasting cost and tax when nothing has drifted) and ignores
drift between dates. Band rebalancing is the approach favored by Vanguard's
research and the Daryanani (2008) "opportunistic rebalancing" study for giving
the best risk control per trade.

The **trend signal** is used to *sequence* sells: when an overweight class must
be trimmed and multiple assets qualify, weaker trends are sold first.

---

## Simulated tax & transaction costs

All simulated, configurable via `RebalanceConfig` in `app/rebalance.py`:

- **Transaction cost** — a fraction of trade notional (default 10 bps).
- **Capital-gains tax** — computed only on realized gains when selling.
  `long_term` lots use a lower rate (default 15%); short-term a higher one
  (default 35%). Loss lots realize no gain, so no tax.
- **Sell ordering** minimizes tax: loss lots first, then long-term, then
  short-term; trend breaks ties.
- **Minimum trade value** — trades below a threshold (default $50) are skipped,
  since the transaction cost isn't worth the tiny drift correction.

`cost_basis` and `long_term` are optional per-holding fields (columns in
`holdings.csv`, or fields in a POST body). Without them, tax is treated as zero.

---

## Testing

```bash
python -m pytest -q
```

109 tests covering:

| File | Focus |
|---|---|
| `test_generate_prices.py` | GBM simulator: length, reproducibility, weekend skipping, CSV output |
| `test_data_loader.py` | Load + every validation error (missing file, NaN, non-numeric, bad class, duplicates) |
| `test_portfolio.py` | Exact-number value / allocation / drift math + edge cases |
| `test_signals.py` | SMA/EMA/momentum/volatility/Sharpe exact values; trend labels (the "prediction accuracy" tests) |
| `test_rebalance.py` | 5/25 bands, buy/sell notionals, tax on gains, no-tax loss lots, sell ordering, min-trade skip |
| `test_health.py`, `test_portfolio_endpoints.py`, `test_signals_endpoint.py`, `test_rebalance_endpoint.py` | HTTP wiring, response shapes, error codes |

Tests never touch the committed data files: file-based tests write throwaway
CSVs to a pytest `tmp_path`.

---

## Assumptions & limitations

- **Simulated data.** Prices are GBM, not real markets — there is no genuine
  signal to predict, by design.
- **Fractional shares.** Trades are computed as fractional shares for clean
  math; a real broker might round to whole shares.
- **Stateless.** No portfolio is persisted between requests (see
  [Key design decisions](#key-design-decisions)).
- **Two asset classes.** Only `stock` and `bond` are modeled.
- **Simplified tax.** A flat long/short-term rate per lot, not a full tax-lot
  accounting engine.
- **Not investment advice.** This is a technical demonstration, not a financial
  product.

### Possible follow-ups

- `app/advice.py`: risk-profile-driven target allocations and advice text.
- Per-ticker (not just per-class) targets.
- An `execute` step that applies a plan (would require the stateful model).
```
