"""Rebalancing decisions: drift bands and a cost/tax-aware trade planner.

This is where the pieces combine: current holdings + latest prices + asset
classes + a target allocation (and, optionally, trend signals) become a
concrete list of buy/sell actions.

Two proven ideas drive it:

1. Threshold-band rebalancing (the "5/25 rule"). Rather than rebalancing on a
   fixed calendar, we act only when an asset class drifts outside a tolerance
   band: 5 absolute percentage points, OR 25% of its target weight, whichever
   is *smaller*. This trades only when it matters, controlling risk with fewer
   transactions (Vanguard / Daryanani "opportunistic rebalancing").

2. Cost- and tax-aware execution. Once we know *what* to rebalance, we decide
   *how* cheaply. When selling an overweight class we prefer, in order, lots
   with losses (no capital-gains tax), then long-term lots (lower simulated
   rate), then short-term lots; ties are broken by trend (sell downtrend
   assets first). Trades smaller than a minimum value are skipped, since the
   transaction cost isn't worth the tiny drift correction.

All functions are pure. Tax and transaction costs are simulated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from app.portfolio import allocation_by_class, compute_drift, position_values

# Rank used to break sell-ordering ties: sell weaker trends first.
_TREND_RANK = {"downtrend": 0, "flat": 1, "uptrend": 2}


@dataclass(frozen=True)
class RebalanceConfig:
    """Tunable knobs for band detection and simulated costs.

    Attributes:
        abs_band: Absolute drift tolerance in weight (0.05 = 5 percentage pts).
        rel_band: Relative drift tolerance as a fraction of target weight
            (0.25 = 25%). The effective band is min(abs_band, rel_band*target).
        transaction_cost_rate: Fee per trade as a fraction of notional
            (0.001 = 10 bps).
        short_term_tax_rate: Simulated capital-gains rate on short-term lots.
        long_term_tax_rate: Simulated capital-gains rate on long-term lots.
        min_trade_value: Trades below this dollar value are skipped.
    """

    abs_band: float = 0.05
    rel_band: float = 0.25
    transaction_cost_rate: float = 0.001
    short_term_tax_rate: float = 0.35
    long_term_tax_rate: float = 0.15
    min_trade_value: float = 50.0


DEFAULT_CONFIG = RebalanceConfig()


def band_tolerance(target_weight: float, cfg: RebalanceConfig = DEFAULT_CONFIG) -> float:
    """Return the 5/25-rule tolerance band for a given target weight.

    Args:
        target_weight: The class's target weight (fraction).
        cfg: Rebalance configuration.

    Returns:
        The smaller of the absolute band and the relative band, i.e. how far
        the current weight may drift before a rebalance is triggered.
    """
    return min(cfg.abs_band, cfg.rel_band * target_weight)


def detect_bands(
    current: Mapping[str, float],
    target: Mapping[str, float],
    cfg: RebalanceConfig = DEFAULT_CONFIG,
) -> dict[str, bool]:
    """Flag which classes have drifted outside their tolerance band.

    Args:
        current: Current allocation weights by class.
        target: Target allocation weights by class.
        cfg: Rebalance configuration.

    Returns:
        Mapping of class -> True if it breaches its band, over the union of
        keys in `current` and `target`.
    """
    drift = compute_drift(current, target)
    return {
        cls: abs(d) >= band_tolerance(target.get(cls, 0.0), cfg)
        for cls, d in drift.items()
    }


def needs_rebalance(
    current: Mapping[str, float],
    target: Mapping[str, float],
    cfg: RebalanceConfig = DEFAULT_CONFIG,
) -> bool:
    """Return True if any class breaches its tolerance band.

    Args:
        current: Current allocation weights by class.
        target: Target allocation weights by class.
        cfg: Rebalance configuration.

    Returns:
        Whether a rebalance is warranted under the 5/25 rule.
    """
    return any(detect_bands(current, target, cfg).values())


def _tax_rate(holding: Mapping, cfg: RebalanceConfig) -> float:
    """Pick the simulated tax rate for a holding (long- vs short-term)."""
    return cfg.long_term_tax_rate if holding.get("long_term", True) else cfg.short_term_tax_rate


def _gain_per_share(holding: Mapping, price: float) -> float:
    """Unrealized gain per share; <= 0 (a loss) when no gain or no basis known."""
    basis = holding.get("cost_basis")
    if basis is None:
        return 0.0
    return price - basis


def _sell_priority(holding: Mapping, price: float, trend: str, cfg: RebalanceConfig):
    """Sort key for choosing what to sell first (lower sorts earlier).

    Prefers, in order: loss/no-tax lots, then cheaper tax per dollar, then
    weaker trends. Selling loss lots and long-term lots first minimizes the
    simulated tax bill.
    """
    gain = _gain_per_share(holding, price)
    taxable_gain = max(0.0, gain)
    tax_per_dollar = (taxable_gain / price) * _tax_rate(holding, cfg) if price else 0.0
    return (tax_per_dollar, _TREND_RANK.get(trend, 1))


def plan_trades(
    holdings: Sequence[Mapping],
    prices: Mapping[str, float],
    asset_classes: Mapping[str, str],
    target: Mapping[str, float],
    trends: Mapping[str, str] | None = None,
    cfg: RebalanceConfig = DEFAULT_CONFIG,
) -> dict:
    """Produce a full rebalancing plan for a portfolio against a target.

    Detects band breaches (5/25 rule); if any class is out of band, computes
    the dollar move needed per class and turns it into concrete per-ticker
    buy/sell trades, each annotated with simulated transaction cost, simulated
    tax, and a human-readable reason. If nothing breaches, no trades are
    returned.

    Sell selection within an overweight class is tax-aware (loss lots first,
    then long-term, then short-term; ties broken by weaker trend). Buys within
    an underweight class are spread across that class's tickers in proportion
    to what's already held (equally if none is held). Trades below
    ``cfg.min_trade_value`` are skipped.

    Args:
        holdings: Holdings, each with ``ticker`` and ``shares`` and optionally
            ``cost_basis`` and ``long_term``.
        prices: Latest price per ticker.
        asset_classes: Ticker -> asset class; defines the universe of tickers
            available to buy within each class.
        target: Target allocation weights by class (should sum to ~1.0).
        trends: Optional ticker -> trend label ("uptrend"/"flat"/"downtrend"),
            used only to sequence sells.
        cfg: Rebalance configuration.

    Returns:
        A dict with: portfolio_value, current_allocation, target_allocation,
        drift, band_breaches, needs_rebalance, trades (list), and the summed
        estimated_transaction_cost / estimated_tax.

    Raises:
        PortfolioError: Propagated from the portfolio math (unknown ticker,
            negative shares, or a held ticker with no class).
    """
    trends = trends or {}
    values = position_values(holdings, prices)
    total = sum(values.values())

    current = allocation_by_class(holdings, prices, asset_classes)
    drift = compute_drift(current, target)
    breaches = detect_bands(current, target, cfg)

    plan: dict = {
        "portfolio_value": total,
        "current_allocation": current,
        "target_allocation": dict(target),
        "drift": drift,
        "band_breaches": breaches,
        "needs_rebalance": any(breaches.values()),
        "trades": [],
        "estimated_transaction_cost": 0.0,
        "estimated_tax": 0.0,
    }

    if total <= 0 or not plan["needs_rebalance"]:
        return plan

    holding_by_ticker = {h["ticker"]: h for h in holdings}
    class_current_value = {
        cls: sum(values.get(t, 0.0) for t, c in asset_classes.items() if c == cls)
        for cls in set(asset_classes.values())
    }

    trades: list[dict] = []
    for cls in set(target) | set(current):
        target_value = target.get(cls, 0.0) * total
        delta = target_value - class_current_value.get(cls, 0.0)

        if delta < 0:  # overweight -> sell
            trades += _plan_sells(
                cls, -delta, holding_by_ticker, prices, asset_classes,
                trends, drift.get(cls, 0.0), cfg,
            )
        elif delta > 0:  # underweight -> buy
            trades += _plan_buys(
                cls, delta, values, prices, asset_classes,
                drift.get(cls, 0.0), cfg,
            )

    plan["trades"] = trades
    plan["estimated_transaction_cost"] = round(
        sum(t["est_transaction_cost"] for t in trades), 2)
    plan["estimated_tax"] = round(sum(t["est_tax"] for t in trades), 2)
    return plan


def _plan_sells(cls, dollars_needed, holding_by_ticker, prices, asset_classes,
                trends, class_drift, cfg) -> list[dict]:
    """Raise `dollars_needed` from an overweight class, tax-aware.

    Sells from the class's held tickers in tax-then-trend priority order until
    the needed amount is raised, skipping any individual sale below the minimum
    trade value.
    """
    candidates = [
        holding_by_ticker[t]
        for t in holding_by_ticker
        if asset_classes.get(t) == cls and holding_by_ticker[t]["shares"] > 0
    ]
    candidates.sort(key=lambda h: _sell_priority(h, prices[h["ticker"]],
                                                  trends.get(h["ticker"], "flat"), cfg))

    trades: list[dict] = []
    remaining = dollars_needed
    for h in candidates:
        if remaining < cfg.min_trade_value:
            break
        ticker = h["ticker"]
        price = prices[ticker]
        available_value = h["shares"] * price
        sell_value = min(remaining, available_value)
        if sell_value < cfg.min_trade_value:
            continue

        shares = sell_value / price
        tx_cost = sell_value * cfg.transaction_cost_rate
        taxable_gain = max(0.0, _gain_per_share(h, price)) * shares
        tax = taxable_gain * _tax_rate(h, cfg)

        trades.append({
            "ticker": ticker,
            "action": "SELL",
            "shares": round(shares, 4),
            "price": price,
            "notional": round(sell_value, 2),
            "est_transaction_cost": round(tx_cost, 2),
            "est_tax": round(tax, 2),
            "reason": _sell_reason(cls, class_drift, h, price,
                                   trends.get(ticker, "flat")),
        })
        remaining -= sell_value
    return trades


def _plan_buys(cls, dollars_to_deploy, values, prices, asset_classes,
               class_drift, cfg) -> list[dict]:
    """Deploy `dollars_to_deploy` into an underweight class.

    Spreads the purchase across the class's tickers in proportion to what's
    already held (equal split if the class holds nothing), skipping any buy
    below the minimum trade value. Buys incur transaction cost but no tax.
    """
    class_tickers = [t for t, c in asset_classes.items() if c == cls]
    if not class_tickers:
        return []

    held_value = {t: values.get(t, 0.0) for t in class_tickers}
    total_held = sum(held_value.values())
    if total_held > 0:
        weights = {t: held_value[t] / total_held for t in class_tickers}
    else:
        weights = {t: 1.0 / len(class_tickers) for t in class_tickers}

    trades: list[dict] = []
    for ticker in class_tickers:
        buy_value = dollars_to_deploy * weights[ticker]
        if buy_value < cfg.min_trade_value:
            continue
        price = prices[ticker]
        shares = buy_value / price
        tx_cost = buy_value * cfg.transaction_cost_rate
        trades.append({
            "ticker": ticker,
            "action": "BUY",
            "shares": round(shares, 4),
            "price": price,
            "notional": round(buy_value, 2),
            "est_transaction_cost": round(tx_cost, 2),
            "est_tax": 0.0,
            "reason": (f"{cls} underweight by {abs(class_drift) * 100:.1f}%; "
                       f"buying to reach target"),
        })
    return trades


def _sell_reason(cls, class_drift, holding, price, trend) -> str:
    """Compose a short, explainable reason string for a sell trade."""
    gain = _gain_per_share(holding, price)
    if holding.get("cost_basis") is None:
        tax_note = "tax n/a (no cost basis)"
    elif gain <= 0:
        tax_note = "loss lot, no tax"
    else:
        term = "long-term" if holding.get("long_term", True) else "short-term"
        tax_note = f"{term} gain"
    return (f"{cls} overweight by {abs(class_drift) * 100:.1f}%; "
            f"trimming ({trend}, {tax_note})")
