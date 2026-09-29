"""Load and validate the simulated data files.

This module is the single gateway between the CSV files on disk and the rest
of the app. Everything downstream (portfolio math, signals, rebalancing)
receives clean, validated Python data from here, never a raw file handle.

It reads three files, all produced by ``scripts/generate_prices.py`` (except
holdings.csv, a hand-authored sample portfolio):

    data/prices.csv    - wide price history: date + one column per ticker
    data/assets.csv    - metadata: ticker, name, asset_class (stock/bond)
    data/holdings.csv  - sample portfolio: ticker, shares

Any structural problem (missing file, missing column, empty file, NaN,
non-numeric or non-positive price, unknown asset class, bad share count)
raises :class:`DataError` with a message that names the file and the problem,
so callers can surface a clean error instead of a pandas stack trace.
"""

from __future__ import annotations

import os

import pandas as pd

# Repo root is two levels up from this file (app/ -> repo root).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_REPO_ROOT, "data")

PRICES_PATH = os.path.join(DATA_DIR, "prices.csv")
ASSETS_PATH = os.path.join(DATA_DIR, "assets.csv")
HOLDINGS_PATH = os.path.join(DATA_DIR, "holdings.csv")

VALID_ASSET_CLASSES = {"stock", "bond"}


class DataError(Exception):
    """Raised when a data file is missing, malformed, or fails validation.

    Carries a human-readable message naming the offending file and the
    specific problem so the API layer can return a clean 4xx/5xx response
    instead of leaking a pandas/OS traceback.
    """


def _read_csv(path: str) -> pd.DataFrame:
    """Read a CSV into a DataFrame, converting I/O problems to DataError.

    Wraps :func:`pandas.read_csv` so that a missing file or an empty/unparseable
    file becomes a :class:`DataError` (with the path named) rather than a bare
    ``FileNotFoundError`` or pandas exception.

    Args:
        path: Absolute or relative path to the CSV file.

    Returns:
        The parsed DataFrame (not yet schema-validated).

    Raises:
        DataError: If the file does not exist, is empty, or cannot be parsed.
    """
    if not os.path.exists(path):
        raise DataError(f"Data file not found: {path}")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise DataError(f"Data file is empty: {path}") from exc
    except pd.errors.ParserError as exc:
        raise DataError(f"Could not parse CSV: {path} ({exc})") from exc
    if df.empty:
        raise DataError(f"Data file has no rows: {path}")
    return df


def load_price_history(path: str | None = None) -> pd.DataFrame:
    """Load and validate the full price history.

    Validates that the file has a ``date`` column plus at least one ticker
    column, contains no missing values, and that every price is numeric and
    strictly positive (prices from Geometric Brownian Motion are always > 0,
    so a non-positive value signals corrupt data). The ``date`` column is
    parsed to datetimes and the frame is sorted ascending by date.

    Args:
        path: Path to the prices CSV. Defaults to ``data/prices.csv``.

    Returns:
        A DataFrame indexed 0..N-1 with a datetime ``date`` column followed by
        one float column per ticker, sorted oldest-first.

    Raises:
        DataError: If the file is missing/empty, lacks a ``date`` column or any
            ticker column, contains NaN, or has non-numeric/non-positive prices.
    """
    path = path or PRICES_PATH
    df = _read_csv(path)

    if "date" not in df.columns:
        raise DataError(f"prices file missing required 'date' column: {path}")

    ticker_cols = [c for c in df.columns if c != "date"]
    if not ticker_cols:
        raise DataError(f"prices file has no ticker columns: {path}")

    if df.isnull().any().any():
        raise DataError(f"prices file contains missing values: {path}")

    # Every ticker column must be numeric.
    for col in ticker_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DataError(
                f"prices file has non-numeric values in column '{col}': {path}"
            )

    if (df[ticker_cols] <= 0).any().any():
        raise DataError(f"prices file contains non-positive prices: {path}")

    try:
        df["date"] = pd.to_datetime(df["date"])
    except (ValueError, TypeError) as exc:
        raise DataError(f"prices file has unparseable dates: {path} ({exc})") from exc

    return df.sort_values("date").reset_index(drop=True)


def latest_prices(history: pd.DataFrame) -> dict[str, float]:
    """Extract the most recent price for each ticker from a price history.

    Takes the last (newest) row of a validated price-history frame and returns
    it as a plain ``{ticker: price}`` dict — the shape the portfolio math
    expects. The ``date`` column is dropped.

    Args:
        history: A DataFrame as returned by :func:`load_price_history`.

    Returns:
        Mapping of ticker to its latest closing price as a float.

    Raises:
        DataError: If the history is empty.
    """
    if history.empty:
        raise DataError("price history is empty; cannot take latest prices")
    last_row = history.iloc[-1]
    return {
        col: float(last_row[col])
        for col in history.columns
        if col != "date"
    }


def load_latest_prices(path: str | None = None) -> dict[str, float]:
    """Convenience: load the price history and return only the latest prices.

    Equivalent to ``latest_prices(load_price_history(path))``; useful for the
    endpoints, which only need current prices to value a portfolio.

    Args:
        path: Path to the prices CSV. Defaults to ``data/prices.csv``.

    Returns:
        Mapping of ticker to its latest closing price.

    Raises:
        DataError: For any validation failure in the underlying load.
    """
    return latest_prices(load_price_history(path))


def price_series(history: pd.DataFrame) -> dict[str, list[float]]:
    """Convert a price-history frame into per-ticker price lists.

    Turns the wide DataFrame into ``{ticker: [p0, p1, ...]}`` (oldest-first),
    dropping the ``date`` column. This is the plain-list shape the signals math
    expects, keeping numpy/pandas out of ``signals.py``.

    Args:
        history: A DataFrame as returned by :func:`load_price_history`.

    Returns:
        Mapping of ticker to its ordered list of prices as floats.
    """
    return {
        col: [float(x) for x in history[col]]
        for col in history.columns
        if col != "date"
    }


def load_price_series(path: str | None = None) -> dict[str, list[float]]:
    """Convenience: load the price history and return per-ticker price lists.

    Equivalent to ``price_series(load_price_history(path))``.

    Args:
        path: Path to the prices CSV. Defaults to ``data/prices.csv``.

    Returns:
        Mapping of ticker to its ordered list of prices.

    Raises:
        DataError: For any validation failure in the underlying load.
    """
    return price_series(load_price_history(path))


def load_asset_classes(path: str | None = None) -> dict[str, str]:
    """Load the ticker -> asset-class mapping from the assets metadata file.

    Validates the presence of ``ticker`` and ``asset_class`` columns and that
    every class is one of :data:`VALID_ASSET_CLASSES` ("stock"/"bond"). This is
    the lookup used to group holdings into asset-class allocations.

    Args:
        path: Path to the assets CSV. Defaults to ``data/assets.csv``.

    Returns:
        Mapping of ticker to its asset class.

    Raises:
        DataError: If required columns are missing or an unknown class appears.
    """
    path = path or ASSETS_PATH
    df = _read_csv(path)

    required = {"ticker", "asset_class"}
    missing = required - set(df.columns)
    if missing:
        raise DataError(
            f"assets file missing required column(s) {sorted(missing)}: {path}"
        )

    classes = dict(zip(df["ticker"].astype(str), df["asset_class"].astype(str)))
    unknown = set(classes.values()) - VALID_ASSET_CLASSES
    if unknown:
        raise DataError(
            f"assets file has unknown asset_class value(s) {sorted(unknown)}: {path}"
        )
    return classes


def load_holdings(path: str | None = None) -> list[dict]:
    """Load and validate the sample portfolio holdings.

    Validates the presence of ``ticker`` and ``shares`` columns, that share
    counts are numeric and strictly positive, and that no ticker is duplicated.
    Returns the holdings in the shape the portfolio math expects.

    Args:
        path: Path to the holdings CSV. Defaults to ``data/holdings.csv``.

    Returns:
        A list of ``{"ticker": str, "shares": float}`` dicts.

    Raises:
        DataError: If required columns are missing, shares are non-numeric or
            non-positive, or a ticker appears more than once.
    """
    path = path or HOLDINGS_PATH
    df = _read_csv(path)

    required = {"ticker", "shares"}
    missing = required - set(df.columns)
    if missing:
        raise DataError(
            f"holdings file missing required column(s) {sorted(missing)}: {path}"
        )

    shares = pd.to_numeric(df["shares"], errors="coerce")
    if shares.isnull().any():
        raise DataError(f"holdings file has non-numeric share counts: {path}")
    if (shares <= 0).any():
        raise DataError(f"holdings file has non-positive share counts: {path}")

    tickers = df["ticker"].astype(str)
    if tickers.duplicated().any():
        dupes = sorted(tickers[tickers.duplicated()].unique())
        raise DataError(f"holdings file has duplicate ticker(s) {dupes}: {path}")

    return [
        {"ticker": t, "shares": float(s)}
        for t, s in zip(tickers, shares)
    ]
