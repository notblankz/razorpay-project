"""Tests for the GET /signals endpoint (app/routes.py).

Runs against the committed simulated data (data/prices.csv), which has enough
history (252 days) for the trend windows.
"""

from __future__ import annotations

import pytest

from app import create_app
from app.data_loader import load_asset_classes


@pytest.fixture
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def test_signals_status_and_covers_all_tickers(client):
    resp = client.get("/signals")
    assert resp.status_code == 200
    data = resp.get_json()
    assert set(data) == set(load_asset_classes())      # one entry per known ticker


def test_signals_each_asset_has_all_fields(client):
    data = client.get("/signals").get_json()
    for asset in data.values():
        assert set(asset) == {
            "latest_price", "sma_short", "sma_long", "momentum",
            "annualized_volatility", "sharpe_ratio", "trend",
        }


def test_signals_trend_labels_are_valid(client):
    data = client.get("/signals").get_json()
    for asset in data.values():
        assert asset["trend"] in {"uptrend", "downtrend", "flat"}


def test_signals_metrics_are_sensible(client):
    data = client.get("/signals").get_json()
    for asset in data.values():
        assert asset["latest_price"] > 0
        assert asset["annualized_volatility"] >= 0
