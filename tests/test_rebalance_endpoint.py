"""Tests for the /rebalance-plan and updated /drift endpoints (app/routes.py)."""

from __future__ import annotations

import pytest

from app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


# --------------------------------------------------------------------------- #
# GET /rebalance-plan  (sample portfolio, default 60/40 target)
# --------------------------------------------------------------------------- #

def test_get_rebalance_plan_shape(client):
    resp = client.get("/rebalance-plan")
    assert resp.status_code == 200
    plan = resp.get_json()
    assert set(plan) == {
        "portfolio_value", "current_allocation", "target_allocation",
        "drift", "band_breaches", "needs_rebalance", "trades",
        "estimated_transaction_cost", "estimated_tax",
    }


def test_get_rebalance_plan_sample_is_drifted(client):
    # The committed sample is deliberately stock-heavy -> should need rebalance.
    plan = client.get("/rebalance-plan").get_json()
    assert plan["needs_rebalance"] is True
    assert len(plan["trades"]) > 0
    for t in plan["trades"]:
        assert t["action"] in {"BUY", "SELL"}
        assert set(t) >= {"ticker", "action", "shares", "notional",
                          "est_transaction_cost", "est_tax", "reason"}


# --------------------------------------------------------------------------- #
# POST /rebalance-plan  (custom holdings + target)
# --------------------------------------------------------------------------- #

def test_post_rebalance_plan_balanced_no_trades(client):
    # Choose holdings already near 60/40 so no trades are produced.
    # Using two bond tickers wouldn't hit 60/40; use a stock + bond mix.
    body = {
        "holdings": [
            {"ticker": "VOO", "shares": 60},
            {"ticker": "BND", "shares": 40},
        ],
        "target": {"stock": 1.0},  # all-stock target vs a mixed book -> breach
    }
    resp = client.post("/rebalance-plan", json=body)
    assert resp.status_code == 200
    plan = resp.get_json()
    assert plan["needs_rebalance"] is True


def test_post_rebalance_plan_defaults_target(client):
    body = {"holdings": [{"ticker": "VOO", "shares": 100}]}  # 100% stock
    plan = client.post("/rebalance-plan", json=body).get_json()
    assert plan["target_allocation"] == {"stock": 0.6, "bond": 0.4}
    assert plan["needs_rebalance"] is True


def test_post_rebalance_plan_bad_holdings_400(client):
    resp = client.post("/rebalance-plan", json={"holdings": "nope"})
    assert resp.status_code == 400


def test_post_rebalance_plan_empty_body_uses_sample(client):
    # No holdings, no target -> sample portfolio vs default 60/40.
    resp = client.post("/rebalance-plan", json={})
    assert resp.status_code == 200
    plan = resp.get_json()
    assert plan["target_allocation"] == {"stock": 0.6, "bond": 0.4}
    assert plan["needs_rebalance"] is True  # sample is deliberately drifted


# --------------------------------------------------------------------------- #
# /drift now uses the 5/25 band logic
# --------------------------------------------------------------------------- #

def test_drift_includes_band_breaches(client):
    body = {
        "holdings": [{"ticker": "VOO", "shares": 10}, {"ticker": "BND", "shares": 10}],
        "target": {"stock": 0.6, "bond": 0.4},
    }
    data = client.post("/drift", json=body).get_json()
    assert "band_breaches" in data
    assert isinstance(data["needs_rebalance"], bool)
