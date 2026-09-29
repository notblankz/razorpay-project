"""Tests for the Flask app factory and the /health endpoint."""

from __future__ import annotations

import pytest

from app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}


def test_health_is_json(client):
    resp = client.get("/health")
    assert resp.is_json


def test_unknown_route_404s(client):
    resp = client.get("/does-not-exist")
    assert resp.status_code == 404
