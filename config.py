"""
Central configuration for the Cross-Asset Regime Monitor.

Every assumption lives here. If a PM asks "why?", the answer is one line in
this file. Do not hardcode constants elsewhere.
"""

# ---------------------------------------------------------------------------
# 1. ASSET UNIVERSE
# ---------------------------------------------------------------------------
# 9 assets from EMiF Project 2. Yahoo Finance tickers.
# Bonds are represented via yields (from FRED) later converted to return
# proxies via modified duration in src/returns.py.
ASSETS = {
    "SP500":       "^GSPC",     # S&P 500 index
    "EuroStoxx50": "^STOXX50E", # Euro Stoxx 50
    "MSCI_EM":     "EEM",       # iShares MSCI EM ETF (proxy)
    "Gold":        "GC=F",      # Gold futures
    "Oil_WTI":     "CL=F",      # WTI crude oil futures
    "US_IG":       "LQD",       # iShares IG corporate bond ETF (proxy)
    "US_HY":       "HYG",       # iShares HY corporate bond ETF (proxy)
}

# Bonds via yields from FRED (converted to returns in src/returns.py).
# Two categories of FRED series.
# ------------------------------
# CURRENT: recent values for dashboard KPIs. ICE BofA OAS series are the
# institutional standard, but FRED restricted them to a rolling 3-year
# window in April 2026 (licensing change with ICE Data Indices).
# HISTORY: Fed-computed Moody's spreads, freely available since 1986/1953,
# used for long-history context charts and any regime work that needs
# pre-2023 credit data.
FRED_SERIES = {
    # Rates
    "US_2Y":       "DGS2",       # 2-Year Treasury constant maturity, since 1976
    "US_10Y":      "DGS10",      # 10-Year Treasury constant maturity, since 1962

    # Credit spreads - CURRENT (ICE BofA, rolling 3-year window since Apr 2026)
    "HY_SPREAD":   "BAMLH0A0HYM2",   # ICE BofA US HY OAS, daily
    "IG_SPREAD":   "BAMLC0A0CM",     # ICE BofA US IG OAS, daily

    # Credit spreads - HISTORY (Moody's, Fed-computed, long history)
    "BAA_SPREAD":  "BAA10Y",     # Moody's Baa - 10Y Treasury, daily since 1986
    "AAA_SPREAD":  "AAA10Y",     # Moody's Aaa - 10Y Treasury, daily since 1986

    # Risk-free rate for excess-Sharpe computation
    "RF_RATE":     "DGS3MO",     # 3-Month Treasury Constant Maturity

    # ALFRED point-in-time data
    "PAYEMS":       "PAYEMS",      # Nonfarm Payrolls (revised heavily; classic macro signal)
    "INDPRO":       "INDPRO",      # Industrial Production (also heavily revised)
    
}

# Modified durations for yield-to-return conversion (approximate, in years).
# ret ~= -duration * change_in_yield
DURATIONS = {
    "US_2Y":  1.9,
    "US_10Y": 8.5,
}

# ---------------------------------------------------------------------------
# 2. SAMPLE PERIOD
# ---------------------------------------------------------------------------
START_DATE = "2005-01-01"

# ---------------------------------------------------------------------------
# 3. GARCH MODEL
# ---------------------------------------------------------------------------
GARCH_SPEC = {
    "vol":  "GARCH",   # GARCH(1,1)
    "p":    1,
    "q":    1,
    "dist": "t",       # Student-t innovations (fat tails)
}
EWMA_LAMBDA = 0.94     # RiskMetrics-standard fallback if GARCH fails

# ---------------------------------------------------------------------------
# 4. DCC PARAMETERS
# ---------------------------------------------------------------------------
# Fixed per course convention (Engle 2002 finds these values typical).
# Optionally re-estimated by QMLE in Phase 4 (P8) as a robustness check.
DCC_A = 0.05
DCC_B = 0.93

# ---------------------------------------------------------------------------
# 4b. RETURNS PROCESSING - OUTLIER CAP (winsorization)
# ---------------------------------------------------------------------------
# Daily returns are winsorized at +/- RETURN_CAP before entering any model.
# Rationale: extreme single-day events (notably WTI on 2020-04-20 during
# COVID demand collapse when oil futures went to -$37/bbl) can dominate
# GARCH/DCC estimation and inflate long-run vol. Standard practice in
# production risk modeling (Chan et al. 1992; Hyndman & Athanasopoulos).
# The sign and direction of every event is preserved; only the magnitude
# is capped. The threshold is chosen so that in a Student-t(nu=6) with
# annualised vol of 40%, a +/- 25% single-day return is above the 99.9th
# percentile - i.e. only true tail events are affected.
RETURN_CAP = 0.25   # cap daily log returns at +/- 25%

# ---------------------------------------------------------------------------
# 5. MOMENTUM SIGNAL
# ---------------------------------------------------------------------------
# 12-1 month time-series momentum (Moskowitz-Ooi-Pedersen 2012).
MOM_LOOKBACK_DAYS = 252    # ~12 months of trading days
MOM_SKIP_DAYS     = 21     # skip most recent month to avoid short-term reversal
MA_WINDOW_DAYS    = 200    # confirmation filter: 200-day MA

# ---------------------------------------------------------------------------
# 6. TILT AND REGIME GATE
# ---------------------------------------------------------------------------
TILT_CAP_PP        = 4     # max +/- percentage points around ERC weight
REGIME_THRESHOLD   = 0.70  # P(high-corr regime) above which tilt is halved
REGIME_HALVING     = 0.5   # scaling factor applied when gate is triggered

# ---------------------------------------------------------------------------
# 7. BACKTEST
# ---------------------------------------------------------------------------
TX_COST_BPS        = 5     # one-way transaction cost in basis points
TX_COST_BPS_STRESS = 10    # stress test scenario
REBAL_FREQ         = "W-FRI"  # weekly on Fridays (pandas offset alias)

# ---------------------------------------------------------------------------
# 8. PATHS
# ---------------------------------------------------------------------------
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent
DATA_DIR     = PROJECT_ROOT / "data"
CACHE_FILE   = DATA_DIR / "prices_cache.parquet"
