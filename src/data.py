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

# =============================================================================
# 3. FRED DOWNLOAD (macro/rates series)
# =============================================================================

# Import the extras we need for FRED. Kept here for grouping;
# in a bigger project they'd all live at the top of the file.
import os
from functools import lru_cache
from dotenv import load_dotenv
from fredapi import Fred

# Load environment variables from .env into the process (once, at import).
# In production (Streamlit Cloud), .env doesn't exist and the key comes from
# Streamlit's secrets manager instead — we handle that in Phase 10.
load_dotenv()


@lru_cache(maxsize=1)
def _get_fred_client() -> Fred:
    """
    Create (or reuse) a FRED API client.

    lru_cache means this runs ONCE per Python session. Every subsequent
    call returns the same client — no re-authentication overhead.
    """
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY not found. Check that .env exists in the project "
            "root and contains 'FRED_API_KEY=your_key_here'."
        )
    return Fred(api_key=api_key)


def fetch_one_fred(series_id: str, start: str = None) -> pd.Series:
    """
    Download ONE FRED series (e.g., "DGS10" for 10-year Treasury yield).

    Parameters
    ----------
    series_id : str
        FRED series code (see https://fred.stlouisfed.org).
    start : str, optional
        Start date "YYYY-MM-DD". Defaults to config.START_DATE.

    Returns
    -------
    pd.Series
        Daily series indexed by date; name = series_id.
        Empty Series if download fails.
    """
    if start is None:
        start = config.START_DATE

    print(f"[fetch_one_fred] downloading {series_id} from {start}...")

    try:
        client = _get_fred_client()
        s = client.get_series(series_id, observation_start=start)
    except Exception as e:
        print(f"[fetch_one_fred] ERROR downloading {series_id}: {e}")
        return pd.Series(dtype=float)

    if s.empty:
        print(f"[fetch_one_fred] WARNING: empty result for {series_id}")
        return pd.Series(dtype=float)

    s.name = series_id
    print(f"[fetch_one_fred] {series_id}: {len(s)} rows, "
          f"{s.index.min().date()} to {s.index.max().date()}, "
          f"{s.isna().sum()} NaN")

    return s


def fetch_all_fred(start: str = None) -> pd.DataFrame:
    """
    Download all FRED series defined in config.FRED_SERIES.
    """
    if start is None:
        start = config.START_DATE

    print(f"\n[fetch_all_fred] fetching {len(config.FRED_SERIES)} series\n")

    series_list = []
    failed = []

    for name, sid in config.FRED_SERIES.items():
        s = fetch_one_fred(sid, start=start)
        if s.empty:
            failed.append(name)
            continue
        s.name = name   # rename from FRED code to friendly key
        series_list.append(s)

    if not series_list:
        print("[fetch_all_fred] ERROR: nothing downloaded")
        return pd.DataFrame()

    df = pd.concat(series_list, axis=1)

    print(f"\n[fetch_all_fred] SUMMARY")
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

# =============================================================================
# 4. MERGE YAHOO + FRED INTO ONE PANEL
# =============================================================================

def fetch_all(start: str = None) -> pd.DataFrame:
    """
    Download all Yahoo and FRED data and merge into one wide DataFrame.

    - Yahoo columns: raw adjusted-close prices.
    - FRED columns: yields (%) and credit spreads (%).

    Returns
    -------
    pd.DataFrame
        Rows = business days (union of Yahoo + FRED indices).
        Columns = 7 Yahoo assets + 6 FRED series = 13 total.
        NaN patterns are expected: different exchanges have different holidays,
        Yahoo doesn't publish yields, FRED doesn't publish equity prices.
    """
    print("\n" + "="*70)
    print("[fetch_all] downloading full data panel")
    print("="*70)

    yahoo_df = fetch_all_yahoo(start=start)
    fred_df  = fetch_all_fred(start=start)

    if yahoo_df.empty and fred_df.empty:
        print("[fetch_all] ERROR: both Yahoo and FRED failed")
        return pd.DataFrame()

    if yahoo_df.empty:
        print("[fetch_all] WARNING: Yahoo empty, using FRED only")
        return fred_df
    if fred_df.empty:
        print("[fetch_all] WARNING: FRED empty, using Yahoo only")
        return yahoo_df

    # Outer join preserves every date in either source; NaN pattern is expected.
    df = yahoo_df.join(fred_df, how="outer")

    print(f"\n[fetch_all] MERGED PANEL")
    print(f"  shape:    {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  dates:    {df.index.min().date()} to {df.index.max().date()}")
    print(f"  columns:  {list(df.columns)}")

    return df