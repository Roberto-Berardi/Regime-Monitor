"""
GARCH(1,1) with Student-t innovations for the Cross-Asset Regime Monitor.

Faithful in spirit to Project 2's hand-rolled GARCH but uses the production
`arch` library (Kevin Sheppard) for reliability under unattended weekly
re-fits on Streamlit Cloud.

Design decisions:
- GARCH(1,1) is fixed as the model order (industry-standard baseline,
  no data-mined order selection).
- Student-t innovations (Project 2 documented excess kurtosis 5-15 across
  assets; Gaussian innovations would misprice tails).
- Input returns multiplied by 100 (percent scale) before fitting, matching
  Project 2's convention exactly - makes reconciliation direct.
- Convergence hardening (added in Block 2): multi-start retry + EWMA(0.94)
  fallback if all attempts fail.
"""
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from arch import arch_model

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Silence a chatty warning the arch library prints on some Mac setups.
warnings.filterwarnings("ignore", category=FutureWarning, module="arch")


# =============================================================================
# 1. SINGLE-ASSET GARCH FIT
# =============================================================================

def fit_garch(returns: pd.Series, name: str = "asset") -> dict:
    """
    Fit GARCH(1,1) with Student-t innovations on a single return series.

    Parameters
    ----------
    returns : pd.Series
        Daily returns for one asset. NaN values dropped internally.
    name : str
        Asset name for diagnostic printing.

    Returns
    -------
    dict with:
      name        : asset name
      mu, omega   : mean and unconditional-variance-scale parameters (percent scale)
      alpha, beta : ARCH and GARCH coefficients
      persistence : alpha + beta (should be <1 for stationarity)
      nu          : Student-t degrees of freedom
      sigma       : pd.Series of conditional daily vol (DECIMAL units, aligned to input index)
      std_resid   : pd.Series of standardised residuals (aligned to input index)
      loglik      : optimised log-likelihood
      converged   : True/False
      method      : "GARCH-t"  (Block 2 will add "EWMA-fallback" option)
    """
    r = returns.dropna()
    if len(r) < 100:
        raise ValueError(f"[{name}] too few observations ({len(r)}) for GARCH fit")

    # Scale to percent, matching Project 2's convention for numerical stability
    r_pct = r * 100.0

    # Fit GARCH(1,1) with Student-t
    am = arch_model(r_pct, mean="constant", vol="GARCH", p=1, q=1, dist="t", rescale=False)
    res = am.fit(disp="off", show_warning=False)

    # Extract parameters (arch library naming)
    params = res.params
    mu    = float(params["mu"])
    omega = float(params["omega"])
    alpha = float(params["alpha[1]"])
    beta  = float(params["beta[1]"])
    nu    = float(params["nu"])

    # Conditional vol is on percent scale; convert back to decimal
    sigma_pct     = res.conditional_volatility   # daily vol in percent
    sigma_decimal = sigma_pct / 100.0
    sigma_decimal.name = f"{name}_sigma"

    # Standardised residuals (scale-invariant, use arch's directly)
    z = res.std_resid
    z.name = f"{name}_z"

    result = {
        "name":        name,
        "mu":          mu,
        "omega":       omega,
        "alpha":       alpha,
        "beta":        beta,
        "persistence": alpha + beta,
        "nu":          nu,
        "sigma":       sigma_decimal,
        "std_resid":   z,
        "loglik":      float(res.loglikelihood),
        "converged":   bool(res.convergence_flag == 0),
        "method":      "GARCH-t",
    }

    # Diagnostic printout
    print(f"[fit_garch] {name}:")
    print(f"    method={result['method']}  converged={result['converged']}  n={len(r)}")
    print(f"    mu={mu:+.4f}  omega={omega:.4f}  alpha={alpha:.4f}  beta={beta:.4f}")
    print(f"    persistence={result['persistence']:.4f}  nu={nu:.2f}  loglik={result['loglik']:.1f}")

    return result

# =============================================================================
# 2. HARDENING - MULTI-START RETRY + EWMA FALLBACK
# =============================================================================

def _ewma_sigma(returns: pd.Series, lam: float = None) -> pd.Series:
    """
    EWMA (RiskMetrics) conditional vol as a fallback when GARCH fails.
    Recursion: sigma2_t = (1-lam) * r_{t-1}^2 + lam * sigma2_{t-1}
    """
    if lam is None:
        lam = config.EWMA_LAMBDA

    r = returns.dropna()
    r2 = r ** 2
    var = r2.ewm(alpha=(1 - lam), adjust=False).mean()
    sigma = np.sqrt(var)
    sigma.name = returns.name
    return sigma


def fit_garch_hardened(returns: pd.Series, name: str = "asset",
                      n_retries: int = 3) -> dict:
    """
    Robust wrapper around fit_garch:
      1. Try the standard fit.
      2. If it fails OR persistence >= 0.999 (near-IGARCH is fine, but
         >= 1.0 is degenerate), retry from perturbed starting values.
      3. If all retries fail, fall back to EWMA(lambda=0.94) and flag it.

    The app's methodology footnote can display which assets used the
    fallback - full transparency without breaking the pipeline.
    """
    r = returns.dropna()
    if len(r) < 100:
        print(f"[fit_garch_hardened] {name}: too few observations, EWMA fallback")
        sigma = _ewma_sigma(returns)
        z = (returns / sigma).dropna()
        return {
            "name":        name,
            "mu":          float(r.mean() * 100),
            "omega":       np.nan,
            "alpha":       np.nan,
            "beta":        np.nan,
            "persistence": np.nan,
            "nu":          np.nan,
            "sigma":       sigma,
            "std_resid":   z,
            "loglik":      np.nan,
            "converged":   False,
            "method":      "EWMA-fallback",
        }

    # Attempt 1: standard fit
    try:
        result = fit_garch(returns, name=name)
        if result["converged"] and result["persistence"] < 0.9999:
            return result
        print(f"[fit_garch_hardened] {name}: primary fit poor "
              f"(converged={result['converged']}, persistence={result['persistence']:.4f}), retrying")
    except Exception as e:
        print(f"[fit_garch_hardened] {name}: primary fit raised '{e}', retrying")

    # Retries with perturbed starting values via arch's internal 'starting_values'.
    # We pass in different mu/vol starts to nudge the optimizer to a new basin.
    r_pct = r * 100.0
    for attempt in range(1, n_retries + 1):
        try:
            # Perturb: use asset's own rolling stats to generate alternative starts
            rng = np.random.default_rng(seed=attempt * 17)
            init_mu    = float(r_pct.mean()) + rng.normal(scale=0.05)
            init_omega = float(r_pct.var()) * rng.uniform(0.02, 0.15)
            init_alpha = rng.uniform(0.03, 0.20)
            init_beta  = rng.uniform(0.70, 0.94)
            starting = [init_mu, init_omega, init_alpha, init_beta, 8.0]

            am = arch_model(r_pct, mean="constant", vol="GARCH", p=1, q=1,
                            dist="t", rescale=False)
            res = am.fit(disp="off", show_warning=False, starting_values=starting)

            p = res.params
            alpha = float(p["alpha[1]"])
            beta  = float(p["beta[1]"])
            if res.convergence_flag == 0 and (alpha + beta) < 0.9999:
                print(f"[fit_garch_hardened] {name}: retry {attempt} converged")
                sigma = res.conditional_volatility / 100.0
                sigma.name = f"{name}_sigma"
                z = res.std_resid
                z.name = f"{name}_z"
                return {
                    "name":        name,
                    "mu":          float(p["mu"]),
                    "omega":       float(p["omega"]),
                    "alpha":       alpha,
                    "beta":        beta,
                    "persistence": alpha + beta,
                    "nu":          float(p["nu"]),
                    "sigma":       sigma,
                    "std_resid":   z,
                    "loglik":      float(res.loglikelihood),
                    "converged":   True,
                    "method":      f"GARCH-t (retry {attempt})",
                }
        except Exception as e:
            print(f"[fit_garch_hardened] {name}: retry {attempt} raised '{e}'")

    # All GARCH attempts failed; fall back to EWMA
    print(f"[fit_garch_hardened] {name}: all GARCH attempts failed, EWMA fallback")
    sigma = _ewma_sigma(returns)
    z = (returns / sigma).dropna()
    return {
        "name":        name,
        "mu":          float(r.mean() * 100),
        "omega":       np.nan,
        "alpha":       np.nan,
        "beta":        np.nan,
        "persistence": np.nan,
        "nu":          np.nan,
        "sigma":       sigma,
        "std_resid":   z,
        "loglik":      np.nan,
        "converged":   False,
        "method":      "EWMA-fallback",
    }