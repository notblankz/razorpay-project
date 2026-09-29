"""Tests for the statistical signals (app/signals.py).

Pure-function tests on hand-built series. The trend tests double as the
"prediction accuracy" tests the problem statement asks for: feed a known
up/down/flat series and assert the label.
"""

from __future__ import annotations

import math

import pytest

from app import signals as sg


# --------------------------------------------------------------------------- #
# sma
# --------------------------------------------------------------------------- #

def test_sma_exact():
    assert sg.sma([1, 2, 3, 4, 5], 3) == 4.0        # mean(3,4,5)
    assert sg.sma([2, 4, 6], 3) == 4.0
    assert sg.sma([10, 20], 1) == 20.0              # last value only


def test_sma_invalid_window():
    with pytest.raises(ValueError):
        sg.sma([1, 2, 3], 0)
    with pytest.raises(ValueError):
        sg.sma([1, 2, 3], 4)                         # window > length


# --------------------------------------------------------------------------- #
# ema
# --------------------------------------------------------------------------- #

def test_ema_of_constant_series_is_constant():
    assert sg.ema([5, 5, 5, 5], 2) == 5.0


def test_ema_weights_recent_more_than_sma():
    # Rising series: EMA should sit above the equal-weighted SMA.
    series = [1, 2, 3, 4, 5]
    assert sg.ema(series, 3) > sg.sma(series, 5)


def test_ema_invalid():
    with pytest.raises(ValueError):
        sg.ema([], 3)
    with pytest.raises(ValueError):
        sg.ema([1, 2], 0)


# --------------------------------------------------------------------------- #
# momentum
# --------------------------------------------------------------------------- #

def test_momentum_positive_and_negative():
    assert sg.momentum([100, 110], 1) == pytest.approx(0.10)
    assert sg.momentum([100, 90], 1) == pytest.approx(-0.10)


def test_momentum_over_multiple_periods():
    assert sg.momentum([100, 999, 120], 2) == pytest.approx(0.20)  # 120/100 - 1


def test_momentum_too_short():
    with pytest.raises(ValueError):
        sg.momentum([100], 1)


# --------------------------------------------------------------------------- #
# volatility
# --------------------------------------------------------------------------- #

def test_volatility_constant_returns_is_zero():
    # Each step +10%: returns are all identical -> zero dispersion.
    series = [100, 110, 121, 133.1]
    assert sg.volatility(series) == pytest.approx(0.0, abs=1e-12)


def test_volatility_positive_when_returns_vary():
    series = [100, 110, 100, 110, 100]
    assert sg.volatility(series) > 0


def test_volatility_too_short():
    with pytest.raises(ValueError):
        sg.volatility([100])


# --------------------------------------------------------------------------- #
# sharpe
# --------------------------------------------------------------------------- #

def test_sharpe_zero_when_no_volatility():
    # Constant returns -> std 0 -> defined as 0.0 (not inf).
    assert sg.sharpe([100, 110, 121]) == 0.0


def test_sharpe_positive_for_upward_noisy_series():
    up = [100, 102, 101, 104, 103, 107, 106, 110]
    assert sg.sharpe(up) > 0


def test_sharpe_negative_for_downward_series():
    down = [110, 108, 109, 105, 106, 102, 103, 99]
    assert sg.sharpe(down) < 0


# --------------------------------------------------------------------------- #
# trend_label  (the "prediction accuracy" tests)
# --------------------------------------------------------------------------- #

def test_trend_uptrend_on_rising_series():
    rising = [100 + i for i in range(60)]
    assert sg.trend_label(rising) == "uptrend"


def test_trend_downtrend_on_falling_series():
    falling = [200 - i for i in range(60)]
    assert sg.trend_label(falling) == "downtrend"


def test_trend_flat_on_constant_series():
    flat = [100.0] * 60
    assert sg.trend_label(flat) == "flat"


def test_trend_too_short_raises():
    with pytest.raises(ValueError):
        sg.trend_label([100, 101, 102])             # shorter than long window


# --------------------------------------------------------------------------- #
# asset_signals bundle
# --------------------------------------------------------------------------- #

def test_asset_signals_has_all_fields():
    series = [100 + i * 0.5 for i in range(60)]
    out = sg.asset_signals(series)
    assert set(out) == {
        "latest_price", "sma_short", "sma_long", "momentum",
        "annualized_volatility", "sharpe_ratio", "trend",
    }
    assert out["latest_price"] == series[-1]
    assert out["trend"] in {"uptrend", "downtrend", "flat"}
