"""Simulated market-data generator.

Produces reproducible daily price history for a small mix of stock and bond
assets using Geometric Brownian Motion (GBM), the standard textbook model for
asset prices:

    S_{t+1} = S_t * exp( (mu - 0.5 * sigma^2) * dt  +  sigma * sqrt(dt) * Z )

where mu is the annual drift, sigma the annual volatility, dt one trading day
(1/252 of a year) and Z a standard-normal shock. Bonds get low drift and low
volatility; stocks get higher both.

Outputs two files under data/:
    prices.csv  - wide format: date + one close-price column per ticker
    assets.csv  - metadata: ticker, name, asset_class (stock/bond)

Usage:
    python scripts/generate_prices.py                 # defaults
    python scripts/generate_prices.py --days 252 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class AssetSpec:
    """Immutable description of one simulated asset.

    Holds the parameters the GBM model needs, plus display/classification
    metadata. Frozen so a spec can't be mutated mid-run, keeping a given
    seed fully reproducible.

    Attributes:
        ticker: Short symbol used as the CSV column header (e.g. "VOO").
        name: Human-readable asset name.
        asset_class: Either "stock" or "bond"; used later to group holdings
            into asset-class allocations.
        start_price: Price on day 0, before any GBM steps.
        mu: Annual expected return (drift), e.g. 0.10 for +10%/yr.
        sigma: Annual volatility (std. dev. of returns), e.g. 0.16 for 16%.
    """

    ticker: str
    name: str
    asset_class: str  # "stock" or "bond"
    start_price: float
    mu: float          # annual drift
    sigma: float       # annual volatility


# A small, deliberately-mixed universe: 4 equities, 2 fixed income.
ASSETS: list[AssetSpec] = [
    AssetSpec("VOO",  "Vanguard S&P 500 ETF",       "stock", 400.0, 0.10, 0.16),
    AssetSpec("QQQ",  "Invesco QQQ (Nasdaq 100)",   "stock", 380.0, 0.13, 0.22),
    AssetSpec("AAPL", "Apple Inc.",                 "stock", 180.0, 0.12, 0.28),
    AssetSpec("MSFT", "Microsoft Corp.",            "stock", 370.0, 0.11, 0.25),
    AssetSpec("BND",  "Vanguard Total Bond Market", "bond",   72.0, 0.03, 0.05),
    AssetSpec("TLT",  "20+ Year Treasury Bond ETF", "bond",   95.0, 0.02, 0.12),
]

TRADING_DAYS_PER_YEAR = 252


def simulate_prices(spec: AssetSpec, days: int, rng: random.Random) -> list[float]:
    """Generate a daily closing-price series for one asset using GBM.

    Each step applies the discrete Geometric Brownian Motion update
    S_next = S * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z), where Z is a
    standard-normal shock drawn from `rng`. The first element is the spec's
    start_price (no shock applied), so exactly `days` prices are returned.

    Args:
        spec: The asset's GBM parameters (start_price, mu, sigma).
        days: Number of prices to produce; must be >= 1 (the caller's
            argparse guard enforces >= 2 for a meaningful series).
        rng: A seeded random.Random instance. Passing one shared, seeded
            generator across all assets is what makes a run reproducible.

    Returns:
        A list of `days` prices, each rounded to 2 decimal places (the first
        being start_price unrounded-but-typically-whole).
    """
    dt = 1.0 / TRADING_DAYS_PER_YEAR
    drift = (spec.mu - 0.5 * spec.sigma ** 2) * dt
    shock = spec.sigma * math.sqrt(dt)

    prices = [spec.start_price]
    for _ in range(days - 1):
        z = rng.gauss(0.0, 1.0)
        next_price = prices[-1] * math.exp(drift + shock * z)
        prices.append(round(next_price, 2))
    return prices


def business_days(n: int, start: date) -> list[date]:
    """Return the first `n` weekday dates on or after `start`.

    Weekends (Saturday and Sunday) are skipped so the series loosely mimics a
    real trading calendar. Public holidays are not modeled. If `start` itself
    falls on a weekday it is included as the first date.

    Args:
        n: How many weekday dates to return; must be >= 1.
        start: The earliest date to consider (inclusive).

    Returns:
        A list of `n` dates in ascending order, each a Mon-Fri weekday.
    """
    out: list[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:  # Mon-Fri
            out.append(d)
        d += timedelta(days=1)
    return out


def write_prices(path: str, dates: list[date], series: dict[str, list[float]]) -> None:
    """Write the price history to CSV in wide format.

    Produces a header row of `date` followed by one column per ticker (in the
    fixed order of `ASSETS`), then one row per date with that day's price for
    each ticker. Row i uses dates[i] and series[ticker][i], so all series and
    `dates` are expected to be the same length.

    Args:
        path: Destination CSV path; overwritten if it already exists.
        dates: Ordered list of dates, one per row.
        series: Mapping of ticker -> price list, keyed by every ticker in
            `ASSETS`.

    Returns:
        None. Writes the file as a side effect.
    """
    tickers = [a.ticker for a in ASSETS]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date"] + tickers)
        for i, d in enumerate(dates):
            writer.writerow([d.isoformat()] + [series[t][i] for t in tickers])


def write_assets(path: str) -> None:
    """Write the asset metadata table to CSV.

    Emits a header row (`ticker`, `name`, `asset_class`) and one row per asset
    in `ASSETS`. This is the lookup the downstream logic uses to group holdings
    into stock vs. bond allocations.

    Args:
        path: Destination CSV path; overwritten if it already exists.

    Returns:
        None. Writes the file as a side effect.
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker", "name", "asset_class"])
        for a in ASSETS:
            writer.writerow([a.ticker, a.name, a.asset_class])


def main() -> None:
    """CLI entry point: parse arguments and write the data files.

    Reads --days, --seed, --start and --out-dir from the command line, builds
    a seeded RNG and a weekday calendar, simulates each asset in the fixed
    order of `ASSETS` (so the seed is fully reproducible), then writes
    prices.csv and assets.csv into the output directory (defaulting to
    <repo>/data, which is created if missing). Prints a short summary.

    Exits with an argparse error if --days < 2.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description="Generate simulated price data.")
    parser.add_argument("--days", type=int, default=TRADING_DAYS_PER_YEAR,
                        help="number of trading days to simulate (default: 252)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducibility (default: 42)")
    parser.add_argument("--start", type=str, default="2025-01-01",
                        help="first date, ISO format (default: 2025-01-01)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="output directory (default: <repo>/data)")
    args = parser.parse_args()

    if args.days < 2:
        parser.error("--days must be at least 2")

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )
    os.makedirs(out_dir, exist_ok=True)

    rng = random.Random(args.seed)
    start = date.fromisoformat(args.start)
    dates = business_days(args.days, start)

    # Draw each asset in a fixed order so a given seed is fully reproducible.
    series = {a.ticker: simulate_prices(a, args.days, rng) for a in ASSETS}

    prices_path = os.path.join(out_dir, "prices.csv")
    assets_path = os.path.join(out_dir, "assets.csv")
    write_prices(prices_path, dates, series)
    write_assets(assets_path)

    print(f"Wrote {args.days} rows x {len(ASSETS)} assets -> {prices_path}")
    print(f"Wrote asset metadata -> {assets_path}")


if __name__ == "__main__":
    main()
