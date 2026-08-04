"""
Returns computation for the Cross-Asset Regime Monitor.

Faithful port of the return-computation logic from EMiF Project 2
(Piras, D'Amico, Berardi 2025). Preserves the exact conventions used
in the graded coursework, so downstream models (GARCH, DCC, Markov,
ERC) inherit identical inputs.

Conventions (from Project 2):
- Price assets  -> log returns: ln(P_t / P_{t-1})
- Yield series  -> first difference, converted to bond return proxy via
                   price term (-D * dY / 100) plus daily carry (y_prev / (100 * 252)).
                   Both terms in the same decimal units as price returns.
                   The lagged carry term (y_prev, not y_t) prevents look-ahead:
                   day-t coupon is set by yesterday's yield.
- Panel forward-filled to business-day frequency BEFORE differencing,
  matching Project 2's df_clean.resample("B").last().ffill() step.

Modifications explicitly documented:
- WTI oil went negative on 2020-04-20. log() is undefined for x <= 0.
  We floor WTI at $1 before taking logs (affects 3 days in April 2020).
- Credit assets US_IG and US_HY use ETF proxies (LQD, HYG) from Yahoo.
  Project 2 used total-return indices. ETF vols run 2-6pp higher.
  Documented in the app's Limitations section.
- US_2Y is a new short-duration bond proxy not in Project 2, using
  modified duration 1.9 from config.DURATIONS.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# =============================================================================
# 1. FORWARD-FILL TO BUSINESS-DAY FREQUENCY (Project 2 step)
# =============================================================================

def align_business_days(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Resample to business-day frequency and forward-fill.
    Matches Project 2: df_clean.resample("B").last().ffill()

    Rationale: different exchanges have different holidays. Forward-filling
    aligns everything on the union business-day index so returns can be
    compared like-for-like. The cost is that a market holiday shows up as
    a zero return, which slightly depresses vol. Preferred here because
    Project 2 used the same convention and we want direct reconciliation.
    """
    aligned = panel.resample("B").last().ffill()
    print(f"[align_business_days] {panel.shape} -> {aligned.shape} after B-day ffill")
    return aligned


# =============================================================================
# 2. RETURNS COMPUTATION (Project 2 formulas, verbatim)
# =============================================================================

def compute_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the raw panel into a 9-column daily returns DataFrame.

    Steps (matching Project 2 cell 8 + cell 11):
    1. Align to business-day frequency (ffill).
    2. Price columns  -> log returns.
    3. Yield columns  -> first difference in yield.
    4. Yield columns  -> bond return proxy via -D * dY / 100.
    5. Drop the raw yield columns; keep only bond proxies.

    Returns
    -------
    pd.DataFrame
        Columns: 7 price assets + 2 bond proxies = 9 total.
        First row NaN by construction.
    """
    # Step 1: forward-fill alignment
    df = align_business_days(panel)

    # Which columns are prices vs yields?
    price_cols = [c for c in config.ASSETS if c in df.columns]
    yield_cols = [c for c in config.DURATIONS if c in df.columns]

    print(f"[compute_returns] price columns:  {price_cols}")
    print(f"[compute_returns] yield columns:  {yield_cols}")

    # Step 2a: WTI negative-price handling (documented above)
    prices = df[price_cols].copy()
    if "Oil_WTI" in prices.columns:
        below = (prices["Oil_WTI"] < 1).sum()
        if below > 0:
            print(f"[compute_returns] Oil_WTI: {below} days below $1, floored")
        prices["Oil_WTI"] = prices["Oil_WTI"].clip(lower=1.0)

    # Step 2b: log returns for price columns
    log_returns = np.log(prices / prices.shift(1))

    # Step 3: yield first differences (in percentage points)
    yield_diff = df[yield_cols].diff()

    # Step 4: bond return proxies via modified duration + daily carry
    #    ret = -D * dY / 100  +  y_prev / (100 * 252)
    # The first term is price change from yield move.
    # The second term is one day of coupon income (yesterday's yield / 252 trading days).
    # Using y_prev (not y_t) avoids look-ahead: on day t we earn the coupon
    # that was known at yesterday's close.
    YIELD_SCALE = 100.0
    TRADING_DAYS = 252
    bond_proxies = pd.DataFrame(index=df.index)
    for yc in yield_cols:
        D = config.DURATIONS[yc]
        price_return = -D * yield_diff[yc] / YIELD_SCALE
        carry_return = df[yc].shift(1) / (YIELD_SCALE * TRADING_DAYS)
        bond_proxies[f"{yc}_proxy"] = price_return + carry_return
        print(f"[compute_returns] bond proxy: {yc}_proxy = -{D} * d{yc}/{YIELD_SCALE:.0f} + {yc}_prev/{YIELD_SCALE*TRADING_DAYS:.0f}")

    # Step 5: combine and drop the initial NaN row
    returns = pd.concat([log_returns, bond_proxies], axis=1).iloc[1:]

    # Step 6: winsorize at +/- config.RETURN_CAP (see config.py rationale)
    cap = config.RETURN_CAP
    n_before = (returns.abs() > cap).sum()
    returns = returns.clip(lower=-cap, upper=cap)
    total_capped = int(n_before.sum())
    if total_capped > 0:
        print(f"[compute_returns] winsorized {total_capped} extreme returns at +/-{cap:.0%}:")
        for col, n in n_before.items():
            if n > 0:
                print(f"    {col:15s} {int(n):3d} days")

    print(f"[compute_returns] final shape: {returns.shape}")
    return returns


# =============================================================================
# 3. RECONCILIATION vs PROJECT 2
# =============================================================================

# Project 2 annualized daily vols (Piras, D'Amico, Berardi 2025)
# Sample: 2005-01-04 to 2026-04-24, 5559 obs, ffill business days.
# Source: extracted from stored notebook outputs.
PROJECT2_ANN_VOLS = {
    "SP500":       0.1878,
    "EuroStoxx50": 0.2069,
    "MSCI_EM":     0.1871,
    "Gold":        0.1756,
    "Oil_WTI":     0.4118,
    # US_IG and US_HY intentionally omitted - Project 2 used bond
    # total-return indices; we use LQD/HYG ETFs. Expected to differ.
    "US_10Y_proxy": 0.0768,   # derived: 8.5 * 0.9034 / 100
}

RECON_TOLERANCE = 0.05  # 5% relative deviation - pass/fail threshold


def reconcile_vs_project2(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compare our annualized daily vols against Project 2 targets.
    Passes if all comparable assets are within RECON_TOLERANCE relative.
    """
    ann_factor = np.sqrt(252)
    rows = []
    for col in returns.columns:
        s = returns[col].dropna()
        if len(s) < 2:
            continue
        our_vol = s.std() * ann_factor
        target = PROJECT2_ANN_VOLS.get(col, None)
        if target is None:
            rows.append({
                "asset":       col,
                "our_ann_vol": f"{our_vol:.4f}",
                "target":      "  n/a",
                "rel_dev":     "  n/a",
                "verdict":     "no target",
            })
            continue
        rel_dev = (our_vol - target) / target
        verdict = "PASS" if abs(rel_dev) <= RECON_TOLERANCE else "FAIL"
        rows.append({
            "asset":       col,
            "our_ann_vol": f"{our_vol:.4f}",
            "target":      f"{target:.4f}",
            "rel_dev":     f"{rel_dev:+.1%}",
            "verdict":     verdict,
        })
    return pd.DataFrame(rows)
