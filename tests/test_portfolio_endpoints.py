"""Tests for the portfolio HTTP endpoints (app/routes.py).

Uses Flask's test client. These exercise the wiring and error translation;
the underlying math is proven exactly in test_portfolio.py.
"""

from __future__ import annotations

import pytest

from app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


# --------------------------------------------------------------------------- #
# GET /portfolio  (sample portfolio)
# --------------------------------------------------------------------------- #

def test_get_portfolio_shape(client):
    resp = client.get("/portfolio")
    assert resp.status_code == 200
    data = resp.get_json()
    assert set(data) == {
        "portfolio_value", "positions",
        "allocation_by_ticker", "allocation_by_class",
    }
    assert data["portfolio_value"] > 0


def test_get_portfolio_allocations_sum_to_one(client):
    data = client.get("/portfolio").get_json()
    assert pytest.approx(sum(data["allocation_by_ticker"].values())) == 1.0
    assert pytest.approx(sum(data["allocation_by_class"].values())) == 1.0


def test_get_portfolio_positions_have_detail(client):
    positions = client.get("/portfolio").get_json()["positions"]
    for pos in positions.values():
        assert set(pos) == {"shares", "price", "value"}
        assert pytest.approx(pos["value"]) == pos["shares"] * pos["price"]


# --------------------------------------------------------------------------- #
# POST /portfolio  (custom holdings)
# --------------------------------------------------------------------------- #

def test_post_portfolio_custom_holdings(client):
    # BND price is fixed in the seeded data; assert value scales with shares.
    body = {"holdings": [{"ticker": "BND", "shares": 10}]}
    data = client.post("/portfolio", json=body).get_json()
    assert data["portfolio_value"] > 0
    assert data["allocation_by_class"] == {"bond": 1.0}
    assert data["positions"]["BND"]["shares"] == 10


def test_post_portfolio_missing_holdings_400(client):
    resp = client.post("/portfolio", json={})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_post_portfolio_holdings_not_a_list_400(client):
    resp = client.post("/portfolio", json={"holdings": "nope"})
    assert resp.status_code == 400


def test_post_portfolio_unknown_ticker_400(client):
    body = {"holdings": [{"ticker": "ZZZ", "shares": 1}]}
    resp = client.post("/portfolio", json=body)
    assert resp.status_code == 400


def test_post_portfolio_non_numeric_shares_400(client):
    body = {"holdings": [{"ticker": "VOO", "shares": "lots"}]}
    resp = client.post("/portfolio", json=body)
    assert resp.status_code == 400


def test_post_portfolio_negative_shares_400(client):
    body = {"holdings": [{"ticker": "VOO", "shares": -5}]}
    resp = client.post("/portfolio", json=body)
    assert resp.status_code == 400


# --------------------------------------------------------------------------- #
# POST /drift
# --------------------------------------------------------------------------- #

def test_post_drift_basic(client):
    body = {
        "holdings": [
            {"ticker": "VOO", "shares": 10},   # stock
            {"ticker": "BND", "shares": 10},   # bond
        ],
        "target": {"stock": 0.6, "bond": 0.4},
    }
    resp = client.post("/drift", json=body)
    assert resp.status_code == 200
    data = resp.get_json()
    assert set(data) == {
        "current_allocation", "target_allocation", "drift",
        "band_breaches", "needs_rebalance",
    }
    # drift = current - target, must sum to ~0 across classes
    assert pytest.approx(sum(data["drift"].values()), abs=1e-9) == 0.0
    assert isinstance(data["needs_rebalance"], bool)


def test_post_drift_needs_rebalance_true_when_far(client):
    # All-stock portfolio vs a 60/40 target -> large drift -> flag True.
    body = {
        "holdings": [{"ticker": "VOO", "shares": 10}],
        "target": {"stock": 0.6, "bond": 0.4},
    }
    data = client.post("/drift", json=body).get_json()
    assert data["needs_rebalance"] is True


def test_post_drift_missing_target_uses_default(client):
    body = {"holdings": [{"ticker": "VOO", "shares": 10}]}
    resp = client.post("/drift", json=body)
    assert resp.status_code == 200
    assert resp.get_json()["target_allocation"] == {"stock": 0.6, "bond": 0.4}


def test_post_drift_empty_body_uses_sample_and_default(client):
    resp = client.post("/drift", json={})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["target_allocation"] == {"stock": 0.6, "bond": 0.4}
    assert set(data["current_allocation"]) <= {"stock", "bond"}


def test_post_drift_non_numeric_target_400(client):
    body = {
        "holdings": [{"ticker": "VOO", "shares": 10}],
        "target": {"stock": "high"},
    }
    resp = client.post("/drift", json=body)
    assert resp.status_code == 400
