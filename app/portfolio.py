"""Portfolio valuation and allocation math.

Pure functions only: every function takes plain Python data (lists/dicts) and
returns plain Python data. There is no file I/O and no Flask here, which keeps
this correctness-critical money math trivial to unit-test with exact numbers.

Shared shapes used throughout:
    holdings       - list of {"ticker": str, "shares": float}
    prices         - {ticker: float} latest price per ticker
    asset_classes  - {ticker: "stock" | "bond"}
    allocation     - {key: weight} where weights are fractions summing to 1.0
"""

from __future__ import annotations

from typing import Iterable, Mapping


class PortfolioError(Exception):
    """Raised when portfolio inputs are inconsistent.

    Examples: a holding references a ticker with no known price, or a share
    count is negative. Kept distinct from data-loading errors so the API layer
    can map it to a 400 (bad request) rather than a 500.
    """


def position_values(
    holdings: Iterable[Mapping],
    prices: Mapping[str, float],
) -> dict[str, float]:
    """Compute the market value of each position.

    Market value of a position is ``shares * latest_price``. Multiple entries
    for the same ticker are summed, so the result has one entry per ticker.

    Args:
        holdings: Iterable of {"ticker", "shares"} mappings.
        prices: Mapping of ticker to latest price.

    Returns:
        Mapping of ticker to its total market value.

    Raises:
        PortfolioError: If a holding's ticker is absent from ``prices`` or its
            share count is negative.
    """
    values: dict[str, float] = {}
    for h in holdings:
        ticker = h["ticker"]
        shares = h["shares"]
        if shares < 0:
            raise PortfolioError(f"negative share count for {ticker}: {shares}")
        if ticker not in prices:
            raise PortfolioError(f"no price available for ticker '{ticker}'")
        values[ticker] = values.get(ticker, 0.0) + shares * prices[ticker]
    return values


def portfolio_value(
    holdings: Iterable[Mapping],
    prices: Mapping[str, float],
) -> float:
    """Compute the total market value of the whole portfolio.

    Args:
        holdings: Iterable of {"ticker", "shares"} mappings.
        prices: Mapping of ticker to latest price.

    Returns:
        The summed market value of all positions (0.0 for an empty portfolio).

    Raises:
        PortfolioError: Propagated from :func:`position_values` for unknown
            tickers or negative share counts.
    """
    return sum(position_values(holdings, prices).values())


def current_allocation(
    holdings: Iterable[Mapping],
    prices: Mapping[str, float],
) -> dict[str, float]:
    """Compute each ticker's share of total portfolio value.

    Args:
        holdings: Iterable of {"ticker", "shares"} mappings.
        prices: Mapping of ticker to latest price.

    Returns:
        Mapping of ticker to weight (fraction of total value). Weights sum to
        1.0. Returns an empty dict for an empty or zero-value portfolio, so
        callers never divide by zero.

    Raises:
        PortfolioError: Propagated from :func:`position_values`.
    """
    values = position_values(holdings, prices)
    total = sum(values.values())
    if total <= 0:
        return {}
    return {ticker: value / total for ticker, value in values.items()}


def allocation_by_class(
    holdings: Iterable[Mapping],
    prices: Mapping[str, float],
    asset_classes: Mapping[str, str],
) -> dict[str, float]:
    """Aggregate portfolio weights by asset class (e.g. stock vs. bond).

    Groups each ticker's market value into its asset class, then normalizes to
    fractions of the total.

    Args:
        holdings: Iterable of {"ticker", "shares"} mappings.
        prices: Mapping of ticker to latest price.
        asset_classes: Mapping of ticker to its asset class.

    Returns:
        Mapping of asset class to weight (fractions summing to 1.0). Empty dict
        for an empty or zero-value portfolio.

    Raises:
        PortfolioError: From :func:`position_values`, or if a held ticker has
            no entry in ``asset_classes``.
    """
    values = position_values(holdings, prices)
    total = sum(values.values())
    if total <= 0:
        return {}

    by_class: dict[str, float] = {}
    for ticker, value in values.items():
        if ticker not in asset_classes:
            raise PortfolioError(f"no asset class known for ticker '{ticker}'")
        cls = asset_classes[ticker]
        by_class[cls] = by_class.get(cls, 0.0) + value
    return {cls: value / total for cls, value in by_class.items()}


def compute_drift(
    current: Mapping[str, float],
    target: Mapping[str, float],
) -> dict[str, float]:
    """Compute signed allocation drift: current minus target.

    Works over the union of keys in ``current`` and ``target``; a key missing
    from either side is treated as weight 0. A positive drift means the
    portfolio is *overweight* that key relative to target; negative means
    underweight.

    Args:
        current: Current allocation weights (e.g. from :func:`allocation_by_class`).
        target: Desired allocation weights.

    Returns:
        Mapping of key to signed drift (current - target).
    """
    keys = set(current) | set(target)
    return {k: current.get(k, 0.0) - target.get(k, 0.0) for k in keys}
