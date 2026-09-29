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
from app import rebalance as rb
from app import signals as sg
from app.data_loader import (
    DataError,
    load_asset_classes,
    load_holdings,
    load_latest_prices,
    load_price_series,
)

DEFAULT_TARGET = {"stock": 0.6, "bond": 0.4}

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

        entry: dict = {"ticker": str(h["ticker"]), "shares": shares}
        # Optional fields used by the rebalancer's simulated tax logic.
        if "cost_basis" in h and h["cost_basis"] is not None:
            try:
                entry["cost_basis"] = float(h["cost_basis"])
            except (TypeError, ValueError):
                raise pf.PortfolioError(
                    f"holding #{i} has non-numeric cost_basis: {h['cost_basis']!r}"
                )
        if "long_term" in h:
            entry["long_term"] = bool(h["long_term"])
        parsed.append(entry)
    return parsed


def _holdings_from_body(body: dict) -> list[dict]:
    """Return holdings from the request body, or the sample portfolio default.

    If the body omits 'holdings' entirely, the committed sample portfolio
    (data/holdings.csv) is used — so these endpoints are callable with an empty
    body. If 'holdings' is present but malformed, validation still raises 400.

    Args:
        body: The parsed JSON request body.

    Returns:
        A validated list of holding dicts.
    """
    if "holdings" not in body:
        return load_holdings()
    return _parse_holdings(body)


def _parse_target(body: dict) -> dict[str, float]:
    """Validate and normalize the 'target' allocation field of a request body.

    Args:
        body: The parsed JSON request body.

    Returns:
        A {class: weight} dict of floats.

    Raises:
        pf.PortfolioError: If 'target' is missing/empty or has non-numeric weights.
    """
    target = body.get("target")
    if not isinstance(target, dict) or not target:
        raise pf.PortfolioError("'target' must be a non-empty object of weights")
    try:
        return {str(k): float(v) for k, v in target.items()}
    except (TypeError, ValueError):
        raise pf.PortfolioError("'target' weights must be numbers")


def _current_trends() -> dict[str, str]:
    """Compute the current trend label for every ticker from the price history.

    Returns:
        Mapping of ticker -> "uptrend"/"flat"/"downtrend". A ticker whose
        series is too short for the trend windows falls back to "flat".
    """
    trends: dict[str, str] = {}
    for ticker, series in load_price_series().items():
        try:
            trends[ticker] = sg.trend_label(series)
        except ValueError:
            trends[ticker] = "flat"
    return trends


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
    Both fields are optional: 'holdings' defaults to the sample portfolio and
    'target' defaults to 60/40 stock/bond, so this is callable with an empty
    body. Returns current class allocation, the target, signed drift, per-class
    band breaches and a needs_rebalance flag, using the 5/25 rule.
    """
    body = request.get_json(silent=True) or {}
    holdings = _holdings_from_body(body)
    target = _parse_target(body) if body.get("target") else DEFAULT_TARGET

    prices = load_latest_prices()
    classes = load_asset_classes()
    current = pf.allocation_by_class(holdings, prices, classes)
    drift = pf.compute_drift(current, target)

    return jsonify({
        "current_allocation": current,
        "target_allocation": target,
        "drift": drift,
        "band_breaches": rb.detect_bands(current, target),
        "needs_rebalance": rb.needs_rebalance(current, target),
    })


@bp.get("/rebalance-plan")
def get_rebalance_plan():
    """Full rebalancing plan for the sample portfolio vs the default target.

    Uses the committed holdings, a 60/40 stock/bond default target, and current
    trend signals. Returns band breaches, per-ticker trades, and simulated
    transaction cost and tax totals.
    """
    holdings = load_holdings()
    prices = load_latest_prices()
    classes = load_asset_classes()
    plan = rb.plan_trades(holdings, prices, classes, DEFAULT_TARGET,
                          trends=_current_trends())
    return jsonify(plan)


@bp.post("/rebalance-plan")
def post_rebalance_plan():
    """Full rebalancing plan for a caller-supplied portfolio and target.

    Body: {
        "holdings": [{"ticker": "VOO", "shares": 50,
                      "cost_basis": 450, "long_term": true}, ...],
        "target": {"stock": 0.6, "bond": 0.4}   # optional; defaults to 60/40
    }
    Both fields are optional: 'holdings' defaults to the sample portfolio and
    'target' defaults to 60/40. cost_basis / long_term are optional per holding
    and drive the simulated tax.
    """
    body = request.get_json(silent=True) or {}
    holdings = _holdings_from_body(body)
    target = _parse_target(body) if body.get("target") else DEFAULT_TARGET

    prices = load_latest_prices()
    classes = load_asset_classes()
    plan = rb.plan_trades(holdings, prices, classes, target,
                          trends=_current_trends())
    return jsonify(plan)
