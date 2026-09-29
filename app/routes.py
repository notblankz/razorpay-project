"""HTTP endpoints (thin wrappers around the logic modules).

Routes here stay small: parse the request, call a logic function, return JSON.
No calculations belong in this file. Two error types are translated to clean
HTTP responses:
    DataError      -> 500 (a committed data file is broken; server's fault)
    PortfolioError -> 400 (caller sent bad holdings/target; client's fault)
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from app import portfolio as pf
from app import signals as sg
from app.data_loader import (
    DataError,
    load_asset_classes,
    load_holdings,
    load_latest_prices,
    load_price_series,
)

bp = Blueprint("api", __name__)


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #

@bp.errorhandler(pf.PortfolioError)
def _handle_portfolio_error(exc: pf.PortfolioError):
    """Bad caller input (unknown ticker, negative shares) -> 400."""
    return jsonify({"error": str(exc)}), 400


@bp.errorhandler(DataError)
def _handle_data_error(exc: DataError):
    """Broken/missing committed data file -> 500."""
    return jsonify({"error": str(exc)}), 500


# --------------------------------------------------------------------------- #
# Request parsing helpers
# --------------------------------------------------------------------------- #

def _parse_holdings(body: dict) -> list[dict]:
    """Validate and normalize the 'holdings' field of a request body.

    Args:
        body: The parsed JSON request body.

    Returns:
        A list of {"ticker": str, "shares": float} dicts.

    Raises:
        pf.PortfolioError: If 'holdings' is absent, not a list, or any entry is
            missing 'ticker'/'shares' or has a non-numeric share count.
    """
    holdings = body.get("holdings")
    if not isinstance(holdings, list):
        raise pf.PortfolioError("'holdings' must be a list")

    parsed: list[dict] = []
    for i, h in enumerate(holdings):
        if not isinstance(h, dict) or "ticker" not in h or "shares" not in h:
            raise pf.PortfolioError(
                f"holding #{i} must have 'ticker' and 'shares'"
            )
        try:
            shares = float(h["shares"])
        except (TypeError, ValueError):
            raise pf.PortfolioError(
                f"holding #{i} has non-numeric shares: {h['shares']!r}"
            )
        parsed.append({"ticker": str(h["ticker"]), "shares": shares})
    return parsed


def _summarize(holdings: list[dict]) -> dict:
    """Build the standard portfolio summary payload for a set of holdings.

    Args:
        holdings: List of {"ticker", "shares"} dicts.

    Returns:
        A dict with total value, per-position detail, and allocations by
        ticker and by asset class.

    Raises:
        pf.PortfolioError: For unknown tickers / negative shares / unmapped class.
        DataError: If the underlying data files fail to load.
    """
    prices = load_latest_prices()
    classes = load_asset_classes()

    values = pf.position_values(holdings, prices)
    total = pf.portfolio_value(holdings, prices)
    positions = {
        h["ticker"]: {
            "shares": h["shares"],
            "price": prices[h["ticker"]],
            "value": values[h["ticker"]],
        }
        for h in holdings
    }
    return {
        "portfolio_value": total,
        "positions": positions,
        "allocation_by_ticker": pf.current_allocation(holdings, prices),
        "allocation_by_class": pf.allocation_by_class(holdings, prices, classes),
    }


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@bp.get("/health")
def health():
    """Liveness check."""
    return jsonify({"status": "ok"})


@bp.get("/portfolio")
def get_portfolio():
    """Summarize the committed sample portfolio (no request body needed)."""
    holdings = load_holdings()
    return jsonify(_summarize(holdings))


@bp.post("/portfolio")
def post_portfolio():
    """Summarize a caller-supplied portfolio.

    Body: {"holdings": [{"ticker": "VOO", "shares": 50}, ...]}
    """
    body = request.get_json(silent=True) or {}
    holdings = _parse_holdings(body)
    return jsonify(_summarize(holdings))


@bp.get("/signals")
def get_signals():
    """Per-asset trend and risk signals over the full price history.

    Returns, for every ticker: latest price, short/long SMA, momentum,
    annualized volatility, Sharpe ratio, and a trend label.
    """
    series = load_price_series()
    return jsonify({
        ticker: sg.asset_signals(prices)
        for ticker, prices in series.items()
    })


@bp.post("/drift")
def post_drift():
    """Compute allocation drift for a portfolio against a target.

    Body: {
        "holdings": [{"ticker": "VOO", "shares": 50}, ...],
        "target": {"stock": 0.6, "bond": 0.4}
    }
    Returns current class allocation, the target, signed drift, and a naive
    needs_rebalance flag (any |drift| > 0.05). The real 5/25 band logic will
    live in rebalance.py.
    """
    body = request.get_json(silent=True) or {}
    holdings = _parse_holdings(body)

    target = body.get("target")
    if not isinstance(target, dict) or not target:
        raise pf.PortfolioError("'target' must be a non-empty object of weights")
    try:
        target = {str(k): float(v) for k, v in target.items()}
    except (TypeError, ValueError):
        raise pf.PortfolioError("'target' weights must be numbers")

    prices = load_latest_prices()
    classes = load_asset_classes()
    current = pf.allocation_by_class(holdings, prices, classes)
    drift = pf.compute_drift(current, target)

    return jsonify({
        "current_allocation": current,
        "target_allocation": target,
        "drift": drift,
        "needs_rebalance": any(abs(d) > 0.05 for d in drift.values()),
    })
