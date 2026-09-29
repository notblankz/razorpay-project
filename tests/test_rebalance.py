"""Tests for the rebalancing logic (app/rebalance.py).

Hand-built fixtures with round numbers so trades and taxes can be checked by
hand. Prices are chosen so 1 share == a clean dollar amount.
"""

from __future__ import annotations

import pytest

from app import rebalance as rb

# 2-asset universe, $1 and $1 per "unit" made easy: STK=$100, BND=$100.
PRICES = {"STK": 100.0, "BND": 100.0}
CLASSES = {"STK": "stock", "BND": "bond"}
TARGET = {"stock": 0.6, "bond": 0.4}


# --------------------------------------------------------------------------- #
# band_tolerance / detect_bands / needs_rebalance
# --------------------------------------------------------------------------- #

def test_band_tolerance_uses_smaller_of_two():
    # target 0.6 -> abs 0.05 vs rel 0.25*0.6=0.15 -> min is 0.05
    assert rb.band_tolerance(0.6) == 0.05
    # target 0.10 -> abs 0.05 vs rel 0.25*0.10=0.025 -> min is 0.025
    assert rb.band_tolerance(0.10) == pytest.approx(0.025)


def test_needs_rebalance_false_when_within_band():
    current = {"stock": 0.62, "bond": 0.38}   # 2% drift < 5% band
    assert rb.needs_rebalance(current, TARGET) is False


def test_needs_rebalance_true_when_outside_band():
    current = {"stock": 0.75, "bond": 0.25}   # 15% drift > 5% band
    assert rb.needs_rebalance(current, TARGET) is True


def test_detect_bands_flags_the_right_class():
    current = {"stock": 0.75, "bond": 0.25}
    breaches = rb.detect_bands(current, TARGET)
    assert breaches == {"stock": True, "bond": True}


# --------------------------------------------------------------------------- #
# plan_trades: no-op when balanced
# --------------------------------------------------------------------------- #

def test_plan_no_trades_when_within_band():
    # 60/40 exactly -> no drift -> no trades.
    holdings = [{"ticker": "STK", "shares": 60}, {"ticker": "BND", "shares": 40}]
    plan = rb.plan_trades(holdings, PRICES, CLASSES, TARGET)
    assert plan["needs_rebalance"] is False
    assert plan["trades"] == []


def test_plan_empty_portfolio_no_trades():
    # Holding nothing is technically 100% off a 60/40 target, so it *reads* as
    # needing rebalance -- but there's $0 to trade, so no trades are produced.
    plan = rb.plan_trades([], PRICES, CLASSES, TARGET)
    assert plan["portfolio_value"] == 0.0
    assert plan["trades"] == []


# --------------------------------------------------------------------------- #
# plan_trades: generates correct buy/sell
# --------------------------------------------------------------------------- #

def test_plan_sells_overweight_and_buys_underweight():
    # 80 STK + 20 BND = $10,000 ; 80/20 vs target 60/40.
    # Need stock -> $6000 (sell $2000), bond -> $4000 (buy $2000).
    holdings = [{"ticker": "STK", "shares": 80}, {"ticker": "BND", "shares": 20}]
    plan = rb.plan_trades(holdings, PRICES, CLASSES, TARGET)

    assert plan["needs_rebalance"] is True
    by_action = {t["action"] for t in plan["trades"]}
    assert by_action == {"SELL", "BUY"}

    sell = next(t for t in plan["trades"] if t["action"] == "SELL")
    buy = next(t for t in plan["trades"] if t["action"] == "BUY")
    assert sell["ticker"] == "STK"
    assert sell["notional"] == pytest.approx(2000.0)
    assert buy["ticker"] == "BND"
    assert buy["notional"] == pytest.approx(2000.0)


def test_transaction_cost_applied():
    holdings = [{"ticker": "STK", "shares": 80}, {"ticker": "BND", "shares": 20}]
    plan = rb.plan_trades(holdings, PRICES, CLASSES, TARGET)
    sell = next(t for t in plan["trades"] if t["action"] == "SELL")
    # 10 bps of $2000 = $2.00
    assert sell["est_transaction_cost"] == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# Tax logic
# --------------------------------------------------------------------------- #

def test_tax_on_long_term_gain():
    # Sell $2000 of STK bought at $50 (now $100) -> $1000 gain on the sold
    # shares (20 shares * $50 gain). Long-term rate 15% -> $150 tax.
    holdings = [
        {"ticker": "STK", "shares": 80, "cost_basis": 50.0, "long_term": True},
        {"ticker": "BND", "shares": 20, "cost_basis": 100.0, "long_term": True},
    ]
    plan = rb.plan_trades(holdings, PRICES, CLASSES, TARGET)
    sell = next(t for t in plan["trades"] if t["action"] == "SELL")
    assert sell["est_tax"] == pytest.approx(150.0)


def test_no_tax_on_loss_lot():
    # cost_basis 120 > price 100 -> loss -> no tax.
    holdings = [
        {"ticker": "STK", "shares": 80, "cost_basis": 120.0, "long_term": True},
        {"ticker": "BND", "shares": 20, "cost_basis": 100.0, "long_term": True},
    ]
    plan = rb.plan_trades(holdings, PRICES, CLASSES, TARGET)
    sell = next(t for t in plan["trades"] if t["action"] == "SELL")
    assert sell["est_tax"] == 0.0


def test_sell_prefers_loss_lot_first():
    # Two stock tickers, both overweight class. STKA is a loss lot, STKB a gain.
    # Only need to sell a little; the loss lot should be chosen first.
    prices = {"STKA": 100.0, "STKB": 100.0, "BND": 100.0}
    classes = {"STKA": "stock", "STKB": "stock", "BND": "bond"}
    holdings = [
        {"ticker": "STKA", "shares": 40, "cost_basis": 130.0, "long_term": True},  # loss
        {"ticker": "STKB", "shares": 40, "cost_basis": 50.0, "long_term": True},   # gain
        {"ticker": "BND", "shares": 20, "cost_basis": 100.0, "long_term": True},
    ]
    plan = rb.plan_trades(holdings, prices, classes, TARGET)
    sells = [t for t in plan["trades"] if t["action"] == "SELL"]
    # The first (and, for a $2000 need, only) sell should be the loss lot STKA.
    assert sells[0]["ticker"] == "STKA"
    assert sells[0]["est_tax"] == 0.0


# --------------------------------------------------------------------------- #
# min trade value
# --------------------------------------------------------------------------- #

def test_tiny_trades_are_skipped():
    # Barely off target: a very high min_trade_value suppresses the trades.
    holdings = [{"ticker": "STK", "shares": 61}, {"ticker": "BND", "shares": 39}]
    cfg = rb.RebalanceConfig(min_trade_value=1_000_000)
    plan = rb.plan_trades(holdings, PRICES, CLASSES, TARGET, cfg=cfg)
    # It still *needs* rebalance (band uses drift), but no trade clears the bar.
    assert plan["trades"] == []


# --------------------------------------------------------------------------- #
# plan structure
# --------------------------------------------------------------------------- #

def test_plan_has_expected_keys():
    holdings = [{"ticker": "STK", "shares": 80}, {"ticker": "BND", "shares": 20}]
    plan = rb.plan_trades(holdings, PRICES, CLASSES, TARGET)
    assert set(plan) == {
        "portfolio_value", "current_allocation", "target_allocation",
        "drift", "band_breaches", "needs_rebalance", "trades",
        "estimated_transaction_cost", "estimated_tax",
    }
