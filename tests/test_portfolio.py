"""Tests for the pure portfolio math (app/portfolio.py).

Uses a tiny hand-built fixture so every expected number can be verified by
hand — no randomness, no file I/O.
"""

from __future__ import annotations

import pytest

from app import portfolio as pf

# Fixture: 3 assets, round prices so values are easy to check by hand.
PRICES = {"AAA": 100.0, "BBB": 50.0, "CCC": 10.0}
CLASSES = {"AAA": "stock", "BBB": "stock", "CCC": "bond"}
HOLDINGS = [
    {"ticker": "AAA", "shares": 10},   # 10 * 100 = 1000
    {"ticker": "BBB", "shares": 20},   # 20 *  50 = 1000
    {"ticker": "CCC", "shares": 50},   # 50 *  10 =  500
]
# total = 2500 ; stock = 2000 (0.8) ; bond = 500 (0.2)


# --------------------------------------------------------------------------- #
# position_values / portfolio_value
# --------------------------------------------------------------------------- #

def test_position_values_exact():
    assert pf.position_values(HOLDINGS, PRICES) == {
        "AAA": 1000.0, "BBB": 1000.0, "CCC": 500.0,
    }


def test_portfolio_value_exact():
    assert pf.portfolio_value(HOLDINGS, PRICES) == 2500.0


def test_duplicate_tickers_are_summed():
    holdings = [{"ticker": "AAA", "shares": 3}, {"ticker": "AAA", "shares": 2}]
    assert pf.position_values(holdings, PRICES) == {"AAA": 500.0}


def test_empty_portfolio_value_is_zero():
    assert pf.portfolio_value([], PRICES) == 0.0


def test_unknown_ticker_raises():
    with pytest.raises(pf.PortfolioError):
        pf.position_values([{"ticker": "ZZZ", "shares": 1}], PRICES)


def test_negative_shares_raise():
    with pytest.raises(pf.PortfolioError):
        pf.position_values([{"ticker": "AAA", "shares": -5}], PRICES)


# --------------------------------------------------------------------------- #
# current_allocation
# --------------------------------------------------------------------------- #

def test_current_allocation_weights():
    alloc = pf.current_allocation(HOLDINGS, PRICES)
    assert alloc == {"AAA": 0.4, "BBB": 0.4, "CCC": 0.2}


def test_current_allocation_sums_to_one():
    alloc = pf.current_allocation(HOLDINGS, PRICES)
    assert pytest.approx(sum(alloc.values())) == 1.0


def test_current_allocation_empty_portfolio():
    assert pf.current_allocation([], PRICES) == {}


# --------------------------------------------------------------------------- #
# allocation_by_class
# --------------------------------------------------------------------------- #

def test_allocation_by_class_groups_correctly():
    by_class = pf.allocation_by_class(HOLDINGS, PRICES, CLASSES)
    assert by_class == {"stock": 0.8, "bond": 0.2}


def test_allocation_by_class_missing_class_raises():
    incomplete = {"AAA": "stock"}  # BBB, CCC unmapped
    with pytest.raises(pf.PortfolioError):
        pf.allocation_by_class(HOLDINGS, PRICES, incomplete)


def test_allocation_by_class_empty_portfolio():
    assert pf.allocation_by_class([], PRICES, CLASSES) == {}


# --------------------------------------------------------------------------- #
# compute_drift
# --------------------------------------------------------------------------- #

def test_compute_drift_signed():
    current = {"stock": 0.8, "bond": 0.2}
    target = {"stock": 0.6, "bond": 0.4}
    drift = pf.compute_drift(current, target)
    assert drift == {"stock": pytest.approx(0.2), "bond": pytest.approx(-0.2)}


def test_compute_drift_missing_key_treated_as_zero():
    # 'cash' exists only in target -> current side is 0.
    current = {"stock": 1.0}
    target = {"stock": 0.9, "cash": 0.1}
    drift = pf.compute_drift(current, target)
    assert drift["stock"] == pytest.approx(0.1)
    assert drift["cash"] == pytest.approx(-0.1)
