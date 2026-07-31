"""
Two-state Markov regime switching model on weekly stock-bond DCC correlation.

CRITICAL for production: the app must use FILTERED probabilities only
(P(regime_t | data up to t)) - NOT smoothed probabilities
(P(regime_t | full sample)). Smoothed uses future data, which would leak
into any historical backtest. This module provides BOTH: full-sample fit
(Block 1, for reconciliation) and recursive-filtered fit (Block 2, for
production).

Interpretation of the 2 states:
- LOW-corr regime  : DCC clearly negative -> diversification working
- HIGH-corr regime : DCC near-zero / positive -> diversification broken
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# =============================================================================
# 1. WEEKLY RESAMPLING
# =============================================================================

def to_weekly(dcc_series: pd.Series) -> pd.Series:
    """
    Resample daily DCC to weekly Friday closes (Project 2 convention).
    Weekly frequency is standard for regime models on correlation data:
    less noisy than daily, still responsive enough to catch turning points.
    """
    weekly = dcc_series.resample("W-FRI").last().dropna()
    weekly.name = dcc_series.name
    return weekly


# =============================================================================
# 2. FULL-SAMPLE FIT (Project 2 style; SMOOTHED probs - not for production)
# =============================================================================

def fit_markov_full(weekly_dcc: pd.Series,
                    search_reps: int = 20,
                    maxiter: int = 200) -> dict:
    """
    Fit 2-state Markov switching model on weekly DCC.
    Multiple random starts (default 20) to avoid local optima.

    Returns dict with:
      model, result       : statsmodels objects
      means, sigmas       : per-regime constant and variance
      trans_mat           : 2x2 transition matrix
      high_idx, low_idx   : which regime index is "high correlation"
      smoothed_high       : P(high-corr regime | full sample), aligned to series
      pct_high, pct_low   : share of weeks in each state (smoothed >= 0.5)
      converged           : bool
    """
    print(f"[fit_markov_full] fitting on {len(weekly_dcc)} weekly observations")
    print(f"[fit_markov_full] range: {weekly_dcc.index[0].date()} to {weekly_dcc.index[-1].date()}")

    model = MarkovRegression(
        weekly_dcc,
        k_regimes=2,
        trend="c",
        switching_variance=True,
    )
    result = model.fit(
        search_reps=search_reps,
        search_iter=20,
        disp=False,
        maxiter=maxiter,
    )

    means  = np.array([result.params[f"const[{k}]"]  for k in range(2)])
    sigmas = np.array([result.params[f"sigma2[{k}]"] for k in range(2)])

    # By convention: the HIGH-corr regime is the one with the LARGER mean
    # (closer to zero or positive - diversification broken)
    high_idx = int(np.argmax(means))
    low_idx  = 1 - high_idx

    # Transition matrix
    p00 = result.params["p[0->0]"]
    p10 = result.params["p[1->0]"]
    p01 = 1 - p00
    p11 = 1 - p10
    trans_mat = np.array([[p00, p01], [p10, p11]])

    # Smoothed probability of being in the high-corr state at each date
    smoothed = result.smoothed_marginal_probabilities
    smoothed_high = smoothed.iloc[:, high_idx]
    smoothed_high.name = "P_high_smoothed"

    regime_ind = (smoothed_high >= 0.5).astype(int)
    pct_high = regime_ind.mean() * 100
    pct_low  = 100 - pct_high

    # Convergence check
    probs_ok = np.isfinite(smoothed.values).all() and np.allclose(smoothed.sum(axis=1), 1.0, atol=0.01)

    print(f"[fit_markov_full] converged: {probs_ok}")
    print(f"[fit_markov_full] regime {high_idx} (HIGH-corr): mean={means[high_idx]:+.4f}, sigma2={sigmas[high_idx]:.4f}")
    print(f"[fit_markov_full] regime {low_idx}  (LOW-corr):  mean={means[low_idx]:+.4f}, sigma2={sigmas[low_idx]:.4f}")
    print(f"[fit_markov_full] transition: p[0->0]={p00:.4f}, p[1->0]={p10:.4f}")
    print(f"[fit_markov_full] sample time in HIGH-corr: {pct_high:.1f}%   in LOW-corr: {pct_low:.1f}%")
    print(f"[fit_markov_full] log-likelihood: {result.llf:.2f}  AIC: {result.aic:.2f}")

    return {
        "model":         model,
        "result":        result,
        "means":         means,
        "sigmas":        sigmas,
        "trans_mat":     trans_mat,
        "high_idx":      high_idx,
        "low_idx":       low_idx,
        "smoothed_high": smoothed_high,
        "pct_high":      pct_high,
        "pct_low":       pct_low,
        "converged":     probs_ok,
    }