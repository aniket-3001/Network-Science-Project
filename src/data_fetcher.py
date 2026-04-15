"""
Data Fetcher — Download and cache S&P 500 price data
======================================================

Uses yfinance to download daily adjusted close prices for all tickers
in the S&P 500 CSV.  Results are cached locally so subsequent runs
skip the download.

References
----------
- yfinance documentation: https://pypi.org/project/yfinance/
"""

import os

import numpy as np
import pandas as pd
import yfinance as yf


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def load_metadata(csv_path: str) -> pd.DataFrame:
    """Load and clean the S&P 500 metadata CSV.

    Returns a DataFrame with columns: symbol, name, sector, sub_industry.
    """
    df = pd.read_csv(csv_path)
    df = df.rename(columns={
        "Symbol":            "symbol",
        "Security":          "name",
        "GICS Sector":       "sector",
        "GICS Sub-Industry": "sub_industry",
    })
    df = df[["symbol", "name", "sector", "sub_industry"]].dropna()
    df = df.drop_duplicates(subset="symbol")
    # Clean tickers — some have dots (BRK.B → BRK-B for yfinance)
    df["yf_ticker"] = df["symbol"].str.replace(".", "-", regex=False)
    df = df.reset_index(drop=True)
    return df


def download_prices(tickers: list,
                    period: str = "2y",
                    cache_path: str | None = None) -> pd.DataFrame:
    """Download adjusted close prices for *tickers* using yfinance.

    Parameters
    ----------
    tickers : list of str
        Yahoo Finance compatible ticker symbols.
    period : str
        Look-back period, e.g. "1y", "2y".  Default = "2y".
    cache_path : str or None
        If given and the file exists, load from cache instead of downloading.
        If given and the file does NOT exist, save after downloading.

    Returns
    -------
    prices : pd.DataFrame
        Columns = tickers, Index = DatetimeIndex.
        Only tickers with at least 80 % non-NaN rows are kept.
    """
    # ── Try cache ──
    if cache_path and os.path.exists(cache_path):
        print(f"  ↻ Loading cached prices from {cache_path}")
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return prices

    # ── Download ──
    print(f"  ⬇ Downloading {len(tickers)} tickers ({period} history) …")
    raw = yf.download(
        tickers,
        period=period,
        auto_adjust=True,
        progress=True,
        threads=True,
    )

    # yfinance returns MultiIndex columns for many tickers: (Price, Ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw  # single ticker edge case

    # ── Quality filter: keep tickers with ≥ 80 % data ──
    threshold = 0.80 * len(prices)
    good = prices.columns[prices.count() >= threshold]
    prices = prices[good]
    dropped = len(tickers) - len(good)
    if dropped:
        print(f"  ⚠ Dropped {dropped} tickers with < 80 % price coverage")
    print(f"  ✓ Price matrix: {prices.shape[0]} days × {prices.shape[1]} tickers")

    # ── Cache ──
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        prices.to_csv(cache_path)
        print(f"  ✓ Cached prices → {cache_path}")

    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log-returns: r_t = ln(P_t / P_{t-1}).

    Drops the first row (NaN) and any remaining NaN columns.
    """
    log_ret = np.log(prices / prices.shift(1)).iloc[1:]
    log_ret = log_ret.dropna(axis=1, how="all")
    return log_ret


def compute_correlation_matrix(log_returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix of log-returns."""
    return log_returns.corr()
