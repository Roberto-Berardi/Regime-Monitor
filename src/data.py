"""
Data layer for the Cross-Asset Regime Monitor.

Handles: download from Yahoo Finance + FRED, merge, cache with parquet,
validate, and fall back gracefully on failure.

Built one function at a time, tested at each step.
"""
from pathlib import Path
import sys

import pandas as pd
import numpy as np
import yfinance as yf

# --- Import our project config -----------------------------------------------
# src/data.py needs to import from config.py at the project root.
# The line below adds the project root to Python's search path.
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# =============================================================================
# 1. YAHOO FINANCE DOWNLOAD (one ticker at a time)
# =============================================================================

def fetch_one_yahoo(ticker: str, start: str = None) -> pd.Series:
    """
    Download adjusted-close daily prices for ONE Yahoo Finance ticker.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol (e.g., "^GSPC" for S&P 500).
    start : str, optional
        Start date "YYYY-MM-DD". Defaults to config.START_DATE.

    Returns
    -------
    pd.Series
        Daily adjusted close prices, indexed by date.
        Empty Series if download fails.
    """
    if start is None:
        start = config.START_DATE

    print(f"[fetch_one_yahoo] downloading {ticker} from {start}...")

    try:
        df = yf.download(
            ticker,
            start=start,
            progress=False,       # no progress bar (cleaner for scripts)
            auto_adjust=True,     # use adjusted close by default
        )
    except Exception as e:
        print(f"[fetch_one_yahoo] ERROR downloading {ticker}: {e}")
        return pd.Series(dtype=float)

    if df.empty:
        print(f"[fetch_one_yahoo] WARNING: empty result for {ticker}")
        return pd.Series(dtype=float)

    # yfinance returns a DataFrame; we want the "Close" column as a named Series
    prices = df["Close"].squeeze()   # squeeze in case of a 1-col multiindex
    prices.name = ticker

    print(f"[fetch_one_yahoo] {ticker}: {len(prices)} rows, "
          f"{prices.index.min().date()} to {prices.index.max().date()}, "
          f"{prices.isna().sum()} NaN")

    return prices

# =============================================================================
# 2. YAHOO FINANCE - ALL ASSETS TOGETHER
# =============================================================================

def fetch_all_yahoo(start: str = None) -> pd.DataFrame:
    """
    Download all Yahoo Finance assets defined in config.ASSETS.

    Loops through the config.ASSETS dict, downloads each ticker one at a time
    (safer than yf.download's multi-ticker mode, which handles failures poorly),
    and combines them into one DataFrame.

    Parameters
    ----------
    start : str, optional
        Start date "YYYY-MM-DD". Defaults to config.START_DATE.

    Returns
    -------
    pd.DataFrame
        Wide-format daily prices: rows = dates, columns = asset names
        (using config.ASSETS keys, e.g., "SP500", "Gold", not tickers).
    """
    if start is None:
        start = config.START_DATE

    print(f"\n[fetch_all_yahoo] fetching {len(config.ASSETS)} assets\n")

    series_list = []
    failed = []

    for name, ticker in config.ASSETS.items():
        s = fetch_one_yahoo(ticker, start=start)
        if s.empty:
            failed.append(name)
            continue
        s.name = name   # rename from ticker (e.g., "^GSPC") to friendly ("SP500")
        series_list.append(s)

    if not series_list:
        print("[fetch_all_yahoo] ERROR: nothing downloaded")
        return pd.DataFrame()

    # Combine into a single wide DataFrame; outer join keeps all dates
    df = pd.concat(series_list, axis=1)

    print(f"\n[fetch_all_yahoo] SUMMARY")
    print(f"  shape:    {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  dates:    {df.index.min().date()} to {df.index.max().date()}")
    print(f"  columns:  {list(df.columns)}")
    print(f"  NaN per column:")
    for col in df.columns:
        n = df[col].isna().sum()
        print(f"    {col:15s} {n:5d} NaN")
    if failed:
        print(f"  FAILED:   {failed}")

    return df