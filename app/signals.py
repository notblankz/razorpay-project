"""Statistical trend and risk signals (the project's "AI" layer).

Despite the "AI" framing in the problem statement, none of this is a trained
model. Over simulated Geometric Brownian Motion data there is no learnable
signal beyond drift and momentum, so a transparent, deterministic stats engine
is both more defensible and fully testable. Every function here is pure: it
takes a plain list of prices (oldest-first) and returns a number or label.

Signals computed:
    sma / ema   - smoothed price level (trend baseline)
    momentum    - recent % change (trend confirmation)
    volatility  - annualized std. dev. of daily returns (risk)
    sharpe      - annualized return per unit of risk
    trend_label - "uptrend" / "flat" / "downtrend" from an SMA crossover
"""

from __future__ import annotations

import math
from typing import Sequence

TRADING_DAYS_PER_YEAR = 252

# Trend-label windows and threshold. Short SMA reacts fast, long SMA is the
# slow baseline; a crossover beyond GAP_THRESHOLD (0.5%) confirmed by momentum
# sets the trend, otherwise it's flat.
SHORT_WINDOW = 20
LONG_WINDOW = 50
MOMENTUM_WINDOW = 10
GAP_THRESHOLD = 0.005


def _daily_returns(series: Sequence[float]) -> list[float]:
    """Compute simple day-over-day returns from a price series.

    Return on day t is ``price[t] / price[t-1] - 1``. A series of N prices
    yields N-1 returns.

    Args:
        series: Ordered price list (oldest-first), length >= 2.

    Returns:
        List of N-1 simple returns.

    Raises:
        ValueError: If fewer than two prices are provided.
    """
    if len(series) < 2:
        raise ValueError("need at least two prices to compute returns")
    return [series[i] / series[i - 1] - 1.0 for i in range(1, len(series))]


def _std(values: Sequence[float]) -> float:
    """Sample standard deviation (ddof=1) of a sequence.

    Args:
        values: The numbers to measure. Length >= 2.

    Returns:
        The sample standard deviation, or 0.0 if fewer than two values.
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return math.sqrt(var)


def sma(series: Sequence[float], window: int) -> float:
    """Simple Moving Average of the most recent `window` prices.

    The average of the last `window` elements — a smoothed, lagging view of the
    price level. Used as the trend baseline.

    Args:
        series: Ordered price list (oldest-first).
        window: Number of trailing prices to average; 1 <= window <= len(series).

    Returns:
        The mean of the last `window` prices.

    Raises:
        ValueError: If `window` is not in the range [1, len(series)].
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if window > len(series):
        raise ValueError(f"window {window} exceeds series length {len(series)}")
    return sum(series[-window:]) / window


def ema(series: Sequence[float], window: int) -> float:
    """Exponential Moving Average (current value) of a price series.

    Like the SMA but weights recent prices more heavily, using smoothing factor
    ``alpha = 2 / (window + 1)``. Seeded with the first price and rolled forward
    across the whole series; the final value is returned.

    Args:
        series: Ordered price list (oldest-first), non-empty.
        window: EMA period; must be positive.

    Returns:
        The current (most recent) EMA value.

    Raises:
        ValueError: If the series is empty or `window` is not positive.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if not series:
        raise ValueError("series is empty")
    alpha = 2.0 / (window + 1)
    value = series[0]
    for price in series[1:]:
        value = alpha * price + (1 - alpha) * value
    return value


def momentum(series: Sequence[float], window: int) -> float:
    """Percent price change over the last `window` periods.

    Computed as ``price[-1] / price[-1-window] - 1``. Positive means the price
    is higher than `window` steps ago (upward momentum), negative the reverse.

    Args:
        series: Ordered price list (oldest-first).
        window: Look-back length; requires len(series) > window.

    Returns:
        The fractional change over the window (e.g. 0.1 == +10%).

    Raises:
        ValueError: If `window` is not positive or the series is too short.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if len(series) <= window:
        raise ValueError(
            f"series length {len(series)} too short for momentum window {window}"
        )
    return series[-1] / series[-1 - window] - 1.0


def volatility(series: Sequence[float]) -> float:
    """Annualized volatility: std. dev. of daily returns, scaled by sqrt(252).

    A standard risk measure — how much the asset's daily returns fluctuate,
    expressed on an annual basis. A perfectly steady series (constant daily
    return) has volatility 0.

    Args:
        series: Ordered price list (oldest-first), length >= 2.

    Returns:
        Annualized volatility as a fraction (e.g. 0.16 == 16%).

    Raises:
        ValueError: If fewer than two prices are provided.
    """
    returns = _daily_returns(series)
    return _std(returns) * math.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe(series: Sequence[float], risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio: excess return per unit of risk.

    ``(mean_daily_return - daily_risk_free) / std_daily_return`` scaled by
    sqrt(252). Higher is better (more return per unit of volatility). When daily
    returns have zero standard deviation the ratio is undefined, and 0.0 is
    returned rather than raising or returning infinity.

    Args:
        series: Ordered price list (oldest-first), length >= 2.
        risk_free_rate: Annual risk-free rate as a fraction (default 0.0).

    Returns:
        The annualized Sharpe ratio (0.0 when volatility is zero).

    Raises:
        ValueError: If fewer than two prices are provided.
    """
    returns = _daily_returns(series)
    std = _std(returns)
    if std == 0:
        return 0.0
    mean = sum(returns) / len(returns)
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    return (mean - daily_rf) / std * math.sqrt(TRADING_DAYS_PER_YEAR)


def trend_label(
    series: Sequence[float],
    short: int = SHORT_WINDOW,
    long: int = LONG_WINDOW,
    mom_window: int = MOMENTUM_WINDOW,
) -> str:
    """Classify the current trend as 'uptrend', 'downtrend', or 'flat'.

    Combines the classic SMA crossover with a momentum confirmation:
      * short SMA sufficiently above long SMA AND positive momentum -> uptrend
      * short SMA sufficiently below long SMA AND negative momentum -> downtrend
      * otherwise (mixed or too small a gap) -> flat

    "Sufficiently" means the relative gap ``(sma_short - sma_long) / sma_long``
    exceeds :data:`GAP_THRESHOLD`, which keeps small wobbles labeled flat.

    Args:
        series: Ordered price list (oldest-first).
        short: Short SMA window (default 20).
        long: Long SMA window (default 50).
        mom_window: Momentum look-back (default 10).

    Returns:
        One of "uptrend", "downtrend", "flat".

    Raises:
        ValueError: If the series is shorter than the long window.
    """
    if len(series) < long:
        raise ValueError(
            f"series length {len(series)} shorter than long window {long}"
        )
    sma_short = sma(series, short)
    sma_long = sma(series, long)
    mom = momentum(series, mom_window)
    gap = (sma_short - sma_long) / sma_long

    if gap > GAP_THRESHOLD and mom > 0:
        return "uptrend"
    if gap < -GAP_THRESHOLD and mom < 0:
        return "downtrend"
    return "flat"


def asset_signals(series: Sequence[float]) -> dict:
    """Compute the full signal bundle for one asset's price series.

    Convenience aggregator used by the /signals endpoint: runs every signal
    function and returns them in one dict.

    Args:
        series: Ordered price list (oldest-first), long enough for the trend
            windows (>= :data:`LONG_WINDOW`).

    Returns:
        A dict with latest_price, sma_short, sma_long, momentum,
        annualized_volatility, sharpe_ratio and trend.

    Raises:
        ValueError: If the series is too short for the trend windows.
    """
    return {
        "latest_price": series[-1],
        "sma_short": sma(series, SHORT_WINDOW),
        "sma_long": sma(series, LONG_WINDOW),
        "momentum": momentum(series, MOMENTUM_WINDOW),
        "annualized_volatility": volatility(series),
        "sharpe_ratio": sharpe(series),
        "trend": trend_label(series),
    }
