"""Tests for the simulated market-data generator (scripts/generate_prices.py)."""

from __future__ import annotations

import csv
import random
from datetime import date

import pytest

from scripts import generate_prices as gp


# --------------------------------------------------------------------------- #
# simulate_prices
# --------------------------------------------------------------------------- #

def test_simulate_prices_length_and_start():
    spec = gp.AssetSpec("TST", "Test", "stock", 100.0, 0.10, 0.20)
    prices = gp.simulate_prices(spec, days=10, rng=random.Random(1))
    assert len(prices) == 10
    assert prices[0] == 100.0  # first element is the start price, unshocked


def test_simulate_prices_is_reproducible_with_same_seed():
    spec = gp.AssetSpec("TST", "Test", "stock", 100.0, 0.10, 0.20)
    a = gp.simulate_prices(spec, days=50, rng=random.Random(42))
    b = gp.simulate_prices(spec, days=50, rng=random.Random(42))
    assert a == b


def test_simulate_prices_differs_with_different_seed():
    spec = gp.AssetSpec("TST", "Test", "stock", 100.0, 0.10, 0.20)
    a = gp.simulate_prices(spec, days=50, rng=random.Random(1))
    b = gp.simulate_prices(spec, days=50, rng=random.Random(2))
    assert a != b


def test_simulate_prices_are_positive():
    spec = gp.AssetSpec("TST", "Test", "stock", 100.0, 0.10, 0.30)
    prices = gp.simulate_prices(spec, days=200, rng=random.Random(7))
    assert all(p > 0 for p in prices)  # GBM can never go non-positive


def test_zero_volatility_grows_deterministically():
    # With sigma=0 the random shock vanishes, so every step multiplies by a
    # fixed factor exp(mu*dt) and the series is strictly increasing for mu>0.
    spec = gp.AssetSpec("TST", "Test", "bond", 100.0, 0.10, 0.0)
    prices = gp.simulate_prices(spec, days=20, rng=random.Random(0))
    assert all(b >= a for a, b in zip(prices, prices[1:]))
    assert prices[-1] > prices[0]


# --------------------------------------------------------------------------- #
# business_days
# --------------------------------------------------------------------------- #

def test_business_days_count_and_order():
    days = gp.business_days(10, date(2025, 1, 1))
    assert len(days) == 10
    assert days == sorted(days)  # ascending


def test_business_days_skips_weekends():
    # 2025-01-01 is a Wednesday. The 3rd/4th are Fri/Sat/Sun boundary:
    # Wed 1, Thu 2, Fri 3, (skip Sat 4, Sun 5), Mon 6 ...
    days = gp.business_days(4, date(2025, 1, 1))
    assert days == [date(2025, 1, 1), date(2025, 1, 2),
                    date(2025, 1, 3), date(2025, 1, 6)]
    assert all(d.weekday() < 5 for d in days)


def test_business_days_starting_on_weekend_advances():
    # 2025-01-04 is a Saturday -> first returned date should be Monday the 6th.
    days = gp.business_days(1, date(2025, 1, 4))
    assert days == [date(2025, 1, 6)]


# --------------------------------------------------------------------------- #
# write_prices / write_assets
# --------------------------------------------------------------------------- #

def test_write_prices_roundtrip(tmp_path):
    dates = [date(2025, 1, 1), date(2025, 1, 2)]
    series = {a.ticker: [1.0 + i, 2.0 + i] for i, a in enumerate(gp.ASSETS)}
    out = tmp_path / "prices.csv"

    gp.write_prices(str(out), dates, series)

    with open(out, newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    assert header == ["date"] + [a.ticker for a in gp.ASSETS]
    assert len(rows) == 1 + len(dates)  # header + one row per date
    assert rows[1][0] == "2025-01-01"


def test_write_assets_has_all_assets(tmp_path):
    out = tmp_path / "assets.csv"
    gp.write_assets(str(out))

    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(gp.ASSETS)
    assert {r["ticker"] for r in rows} == {a.ticker for a in gp.ASSETS}
    assert {r["asset_class"] for r in rows} <= {"stock", "bond"}


# --------------------------------------------------------------------------- #
# main (CLI)
# --------------------------------------------------------------------------- #

def test_main_writes_both_files(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", [
        "generate_prices.py",
        "--days", "20",
        "--seed", "5",
        "--out-dir", str(tmp_path),
    ])
    gp.main()

    prices = tmp_path / "prices.csv"
    assets = tmp_path / "assets.csv"
    assert prices.exists() and assets.exists()

    with open(prices, newline="") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 1 + 20  # header + 20 days


def test_main_rejects_too_few_days(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", [
        "generate_prices.py", "--days", "1", "--out-dir", str(tmp_path),
    ])
    with pytest.raises(SystemExit):  # argparse parser.error() exits
        gp.main()
