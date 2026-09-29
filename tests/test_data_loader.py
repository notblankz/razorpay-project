"""Tests for data loading + validation (app/data_loader.py).

Two kinds of tests:
  * happy-path against the real committed data files, and
  * validation tests that write deliberately-broken CSVs to a tmp dir and
    assert DataError is raised with the right trigger.
"""

from __future__ import annotations

import pytest

from app import data_loader as dl


# --------------------------------------------------------------------------- #
# Happy path against the real data/ files
# --------------------------------------------------------------------------- #

def test_load_price_history_real_file():
    df = dl.load_price_history()
    assert "date" in df.columns
    assert len(df) > 0
    # sorted ascending by date
    assert df["date"].is_monotonic_increasing


def test_load_latest_prices_real_file():
    prices = dl.load_latest_prices()
    assert isinstance(prices, dict)
    assert all(isinstance(v, float) and v > 0 for v in prices.values())


def test_load_asset_classes_real_file():
    classes = dl.load_asset_classes()
    assert set(classes.values()) <= dl.VALID_ASSET_CLASSES


def test_load_holdings_real_file():
    holdings = dl.load_holdings()
    assert len(holdings) > 0
    # ticker + shares are required; cost_basis/long_term are optional extras.
    assert all({"ticker", "shares"} <= set(h) for h in holdings)
    assert all(h["shares"] > 0 for h in holdings)


# --------------------------------------------------------------------------- #
# Helpers for writing broken files
# --------------------------------------------------------------------------- #

def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


# --------------------------------------------------------------------------- #
# _read_csv / generic file problems
# --------------------------------------------------------------------------- #

def test_missing_file_raises():
    with pytest.raises(dl.DataError):
        dl.load_price_history("does/not/exist.csv")


def test_empty_file_raises(tmp_path):
    path = _write(tmp_path, "prices.csv", "")
    with pytest.raises(dl.DataError):
        dl.load_price_history(path)


# --------------------------------------------------------------------------- #
# load_price_history validation
# --------------------------------------------------------------------------- #

def test_prices_missing_date_column(tmp_path):
    path = _write(tmp_path, "prices.csv", "VOO,BND\n1,2\n")
    with pytest.raises(dl.DataError):
        dl.load_price_history(path)


def test_prices_no_ticker_columns(tmp_path):
    path = _write(tmp_path, "prices.csv", "date\n2025-01-01\n")
    with pytest.raises(dl.DataError):
        dl.load_price_history(path)


def test_prices_with_nan(tmp_path):
    path = _write(tmp_path, "prices.csv", "date,VOO\n2025-01-01,\n")
    with pytest.raises(dl.DataError):
        dl.load_price_history(path)


def test_prices_non_positive(tmp_path):
    path = _write(tmp_path, "prices.csv", "date,VOO\n2025-01-01,-5\n")
    with pytest.raises(dl.DataError):
        dl.load_price_history(path)


def test_prices_non_numeric(tmp_path):
    path = _write(tmp_path, "prices.csv", "date,VOO\n2025-01-01,abc\n")
    with pytest.raises(dl.DataError):
        dl.load_price_history(path)


def test_prices_valid_file_loads_and_sorts(tmp_path):
    # Rows out of order -> should come back sorted ascending.
    path = _write(
        tmp_path, "prices.csv",
        "date,VOO\n2025-01-03,102\n2025-01-01,100\n2025-01-02,101\n",
    )
    df = dl.load_price_history(path)
    assert list(df["VOO"]) == [100, 101, 102]


def test_latest_prices_takes_last_row(tmp_path):
    path = _write(
        tmp_path, "prices.csv",
        "date,VOO,BND\n2025-01-01,100,50\n2025-01-02,110,55\n",
    )
    assert dl.load_latest_prices(path) == {"VOO": 110.0, "BND": 55.0}


# --------------------------------------------------------------------------- #
# load_asset_classes validation
# --------------------------------------------------------------------------- #

def test_assets_missing_column(tmp_path):
    path = _write(tmp_path, "assets.csv", "ticker\nVOO\n")
    with pytest.raises(dl.DataError):
        dl.load_asset_classes(path)


def test_assets_unknown_class(tmp_path):
    path = _write(tmp_path, "assets.csv", "ticker,asset_class\nDOGE,crypto\n")
    with pytest.raises(dl.DataError):
        dl.load_asset_classes(path)


def test_assets_valid(tmp_path):
    path = _write(
        tmp_path, "assets.csv",
        "ticker,name,asset_class\nVOO,S&P,stock\nBND,Bonds,bond\n",
    )
    assert dl.load_asset_classes(path) == {"VOO": "stock", "BND": "bond"}


# --------------------------------------------------------------------------- #
# load_holdings validation
# --------------------------------------------------------------------------- #

def test_holdings_missing_column(tmp_path):
    path = _write(tmp_path, "holdings.csv", "ticker\nVOO\n")
    with pytest.raises(dl.DataError):
        dl.load_holdings(path)


def test_holdings_non_positive_shares(tmp_path):
    path = _write(tmp_path, "holdings.csv", "ticker,shares\nVOO,0\n")
    with pytest.raises(dl.DataError):
        dl.load_holdings(path)


def test_holdings_non_numeric_shares(tmp_path):
    path = _write(tmp_path, "holdings.csv", "ticker,shares\nVOO,many\n")
    with pytest.raises(dl.DataError):
        dl.load_holdings(path)


def test_holdings_duplicate_ticker(tmp_path):
    path = _write(tmp_path, "holdings.csv", "ticker,shares\nVOO,10\nVOO,5\n")
    with pytest.raises(dl.DataError):
        dl.load_holdings(path)


def test_holdings_valid(tmp_path):
    path = _write(tmp_path, "holdings.csv", "ticker,shares\nVOO,10\nBND,20\n")
    assert dl.load_holdings(path) == [
        {"ticker": "VOO", "shares": 10.0},
        {"ticker": "BND", "shares": 20.0},
    ]


def test_holdings_with_optional_columns(tmp_path):
    path = _write(
        tmp_path, "holdings.csv",
        "ticker,shares,cost_basis,long_term\nVOO,10,450,true\nBND,20,71,false\n",
    )
    assert dl.load_holdings(path) == [
        {"ticker": "VOO", "shares": 10.0, "cost_basis": 450.0, "long_term": True},
        {"ticker": "BND", "shares": 20.0, "cost_basis": 71.0, "long_term": False},
    ]


def test_holdings_non_positive_cost_basis(tmp_path):
    path = _write(tmp_path, "holdings.csv",
                  "ticker,shares,cost_basis\nVOO,10,0\n")
    with pytest.raises(dl.DataError):
        dl.load_holdings(path)
