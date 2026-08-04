"""
Strategy B — Enhanced Trend + Vol-Scaled Leverage.

Pre-committed spec (see config.py):
- Base: SP500 60%, US_10Y_proxy 40% (classic 60/40)
- Trend filter: SP500 weight -> 0 when combined momentum signal = -1
  (bond stays at 40% baseline; bonds don't get trend-filtered)
- Vol-scaled leverage: clip(vol_target / realized_vol, 0.5, 1.25)
- Regime gate: leverage cap halved to 1.0x when filtered P(high-corr) > 0.70
- Financing cost: 50bp/yr on borrowed portion above 1.0x

The design philosophy:
    Strategy A (Tilted ERC) is a drawdown-controlled multi-asset core.
    Strategy B is a higher-octane sleeve targeting SPX-like returns
    with modest active risk overlay. Different client, different pitch.

The regime gate matters more here than in Strategy A: when correlations
are elevated, trend signals lose reliability AND leveraged losses hurt more.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# =============================================================================
# 1. PORTFOLIO REALIZED VOL (63-day rolling)
# =============================================================================

def portfolio_realized_vol(daily_returns: pd.DataFrame,
                           base_weights: dict,
                           lookback: int = None) -> pd.Series:
    """
    Compute rolling realized vol of the un-levered base portfolio.
    Uses the FIXED base weights, not the trend-filtered ones - the vol
    target is a property of the strategy, not of the current position.

    Rolling window: config.STRAT_B_VOL_LOOKBACK_DAYS.
    Annualized: sqrt(252).
    """
    if lookback is None:
        lookback = config.STRAT_B_VOL_LOOKBACK_DAYS

    # Portfolio daily return series using base weights only
    port_daily = pd.Series(0.0, index=daily_returns.index)
    for asset, w in base_weights.items():
        if asset in daily_returns.columns:
            port_daily = port_daily + w * daily_returns[asset]

    # 63-day rolling std, annualized
    realized_vol = port_daily.rolling(window=lookback, min_periods=lookback).std() * np.sqrt(252)
    realized_vol.name = "portfolio_realized_vol"
    return realized_vol


# =============================================================================
# 2. LEVERAGE FROM VOL-TARGETING + REGIME GATE
# =============================================================================

def leverage_series(realized_vol: pd.Series,
                    regime_prob: pd.Series) -> pd.Series:
    """
    Compute the leverage multiplier at each weekly Friday:
    1. Raw leverage: clip(vol_target / realized_vol, LEV_MIN, LEV_MAX)
    2. Regime gate: when P(high-corr) > threshold, cap at LEV_CAP_HIGH_REGIME

    Both realized_vol and regime_prob are weekly series aligned to the
    strategy's rebalance grid.
    """
    vol_target = config.STRAT_B_VOL_TARGET
    lev_min    = config.STRAT_B_LEV_MIN
    lev_max    = config.STRAT_B_LEV_MAX
    lev_gated  = config.STRAT_B_LEV_CAP_HIGH_REGIME
    threshold  = config.REGIME_THRESHOLD

    # Raw vol-scaled leverage
    raw_lev = (vol_target / realized_vol).clip(lev_min, lev_max)

    # Regime gate: on high-corr weeks, cap at lev_gated
    gated_lev = raw_lev.copy()
    high_regime_mask = regime_prob > threshold
    # min ensures gate lowers but never raises
    gated_lev[high_regime_mask] = np.minimum(raw_lev[high_regime_mask], lev_gated)

    gated_lev.name = "leverage"
    return gated_lev


# =============================================================================
# 3. BUILD STRATEGY B WEIGHTS (WEEKLY)
# =============================================================================

def build_strategy_b_weights(
    daily_returns: pd.DataFrame,
    signals: pd.DataFrame,
    regime_prob: pd.Series,
    asset_columns: list,
    verbose: bool = True,
) -> dict:
    """
    Produce the weekly weights, leverage, and diagnostics for Strategy B.

    Parameters
    ----------
    daily_returns : pd.DataFrame
        Daily returns for all 9 assets. Used for realized-vol calc.
    signals : pd.DataFrame
        Daily momentum signals from build_signal_panel.
    regime_prob : pd.Series
        Weekly filtered P(high-corr) from Phase 5.
    asset_columns : list
        Full asset universe order (for output DataFrame consistency).

    Returns
    -------
    dict with:
        weights          : pd.DataFrame of weekly weights per asset
                           (each column = weight in that asset)
        leverage         : pd.Series of applied leverage
        cash_weight      : pd.Series = 1 - sum(weights)
                           Positive = uninvested cash. Negative = borrowing.
        signal_spx       : pd.Series of the SPX trend signal actually used
        realized_vol     : pd.Series of the 63-day realized vol
        passed_tests     : bool
        test_details     : dict
    """
    base = config.STRAT_B_BASE_WEIGHTS

    # Realized vol on the base 60/40 portfolio
    r_vol_daily = portfolio_realized_vol(daily_returns, base_weights=base)

    # Resample to Fri weekly for all decision-making
    signals_wk = signals.resample("W-FRI").last()
    r_vol_wk   = r_vol_daily.resample("W-FRI").last()
    regime_wk  = regime_prob.resample("W-FRI").last()

    # Common index
    valid_idx = (
        signals_wk.dropna(how="any").index
        .intersection(r_vol_wk.dropna().index)
        .intersection(regime_wk.dropna().index)
    )

    if verbose:
        print(f"[strategy_b] {len(valid_idx)} weekly decisions")
        print(f"[strategy_b] sample: {valid_idx[0].date()} to {valid_idx[-1].date()}")

    # Leverage series
    lev = leverage_series(r_vol_wk.loc[valid_idx], regime_wk.loc[valid_idx])

    # Weight rows
    W_rows = []
    signal_spx_series = []
    for t in valid_idx:
        # Base allocation
        w = {asset: 0.0 for asset in asset_columns}
        # SP500: apply trend filter
        sig_spx = signals_wk.loc[t, "SP500"] if "SP500" in signals_wk.columns else 0.0
        signal_spx_series.append(sig_spx)
        if sig_spx >= 0:  # >=0 means neutral or positive; only actively-negative is filtered out
            w["SP500"] = base["SP500"]
        # else: SP500 weight stays 0 (trend cut)
        # Bond: always at base
        w["US_10Y_proxy"] = base["US_10Y_proxy"]

        # Apply leverage multiplier
        L_t = lev.loc[t]
        w = {k: v * L_t for k, v in w.items()}

        W_rows.append(w)

    W = pd.DataFrame(W_rows, index=valid_idx, columns=asset_columns).fillna(0.0)
    cash = 1.0 - W.sum(axis=1)

    signal_spx = pd.Series(signal_spx_series, index=valid_idx, name="signal_spx")

    # Unit tests
    test_details = {}

    all_nonneg = (W >= 0).all().all()
    test_details["weights_non_negative"] = {"passed": bool(all_nonneg)}

    max_lev = W.sum(axis=1).max()
    test_details["max_leverage_within_cap"] = {
        "max_sum_weights": float(max_lev),
        "passed": max_lev <= config.STRAT_B_LEV_MAX + 1e-9,
    }

    min_lev = W.sum(axis=1).min()
    # If SPX is filtered out, effective sum = 0.4 * L; if L = 0.5, that's 0.2
    # Still non-negative, but lower than 1.
    test_details["min_leverage_positive"] = {
        "min_sum_weights": float(min_lev),
        "passed": min_lev >= 0.0,
    }

    n_gated = ((regime_wk.loc[valid_idx] > config.REGIME_THRESHOLD)).sum()
    test_details["regime_gate_active"] = {
        "n_gated_weeks": int(n_gated),
        "pct_gated": float(n_gated / len(valid_idx) * 100),
    }

    passed_tests = all(v.get("passed", True) for v in test_details.values())

    if verbose:
        print(f"[strategy_b] UNIT TESTS:")
        for name, d in test_details.items():
            print(f"    {name}: {d}")

    return {
        "weights":       W,
        "leverage":      lev,
        "cash_weight":   cash,
        "signal_spx":    signal_spx,
        "realized_vol":  r_vol_wk.loc[valid_idx],
        "passed_tests":  passed_tests,
        "test_details":  test_details,
    }

# =============================================================================
# 4. STRATEGY B BACKTEST RUNNER (handles cash + financing)
# =============================================================================

def run_strategy_b(
    weights: pd.DataFrame,
    cash: pd.Series,
    weekly_returns: pd.DataFrame,
    rf: pd.Series,
    cost_bps: float = None,
    financing_bps: float = None,
    label: str = "StrategyB",
    verbose: bool = True,
) -> dict:
    """
    Backtest a levered/cash portfolio.

    Portfolio return  = sum(w_i * r_i) + cash_contribution
      where cash_contribution:
        - if cash > 0 (uninvested):  cash * rf_weekly   (earn RF)
        - if cash < 0 (borrowing):   cash * (rf_weekly + financing_spread_weekly)
                                     (pay RF + 50bp/yr spread on borrowed amount)

    Weight lag = 1 week (same convention as run_strategy). Turnover cost
    applies to asset weight changes only; cash rebalancing is costless.

    Returns dict matching run_strategy's output shape so it plugs into
    compare_strategies unchanged.
    """
    if cost_bps is None:
        cost_bps = config.TX_COST_BPS
    if financing_bps is None:
        financing_bps = config.STRAT_B_FINANCING_BPS

    # --- Lag weights + cash by one week ---
    w_lagged    = weights.shift(1).dropna(how="all")
    cash_lagged = cash.shift(1).dropna()

    # --- Align on common index and columns ---
    common_cols  = [c for c in w_lagged.columns if c in weekly_returns.columns]
    common_dates = (w_lagged.index
                    .intersection(weekly_returns.index)
                    .intersection(cash_lagged.index)
                    .intersection(rf.index))

    W  = w_lagged.loc[common_dates, common_cols]
    R  = weekly_returns.loc[common_dates, common_cols]
    C  = cash_lagged.loc[common_dates]
    RF = rf.loc[common_dates]

    # --- Asset gross return ---
    asset_gross = (W * R).sum(axis=1)

    # --- Cash contribution ---
    # Uninvested cash (cash > 0) earns RF; borrowed cash (cash < 0) pays RF + spread.
    # Financing spread in weekly simple terms: (spread_bps/10000) / 52.
    fin_weekly = (financing_bps / 10000.0) / 52.0
    cash_pos = C.clip(lower=0) * RF                        # uninvested at RF
    cash_neg = C.clip(upper=0) * (RF + fin_weekly)         # borrowed at RF + spread
    cash_contrib = cash_pos + cash_neg

    gross = asset_gross + cash_contrib

    # --- Turnover on asset weights only ---
    delta_w = W.diff()
    delta_w.iloc[0] = W.iloc[0]
    turnover = 0.5 * delta_w.abs().sum(axis=1)
    cost = turnover * (cost_bps / 10000.0)

    net = gross - cost

    if verbose:
        n_years = (common_dates[-1] - common_dates[0]).days / 365.25
        ann_ret = net.mean() * 52
        ann_vol = net.std() * np.sqrt(52)
        sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
        ann_turn = turnover.mean() * 52
        cost_drag = cost.mean() * 52 * 10000
        pct_borrowing = (C < 0).mean() * 100

        print(f"[run_strategy_b] {label}")
        print(f"    period:            {common_dates[0].date()} to {common_dates[-1].date()} ({n_years:.1f}y)")
        print(f"    cost model:        {cost_bps} bps one-way + {financing_bps} bps/yr financing")
        print(f"    ann return net:    {ann_ret:+.2%}")
        print(f"    ann vol:           {ann_vol:.2%}")
        print(f"    raw Sharpe:        {sharpe:.2f}")
        print(f"    ann turnover:      {ann_turn:.0%}")
        print(f"    cost drag:         {cost_drag:.0f} bps/yr")
        print(f"    % weeks borrowing: {pct_borrowing:.1f}%")

    return {
        "gross":    gross,
        "net":      net,
        "turnover": turnover,
        "cost":     cost,
        "weights":  W,
        "label":    label,
    }