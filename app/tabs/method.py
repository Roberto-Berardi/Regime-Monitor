"""Method & Limitations - the tab that earns trust."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

NAVY, GREY, AMBER, RED, INK = "#0B3040", "#8A9199", "#B45309", "#9B1C1C", "#1A1A1A"


def render(D: dict, M: dict):
    # ------------------------------------------------------- look-ahead -----
    st.markdown("### Look-ahead testing")
    st.caption(
        "Every backtest that peeks at future data looks excellent. Three "
        "safeguards are built into this one, and the third is an adversarial "
        "test designed to break it."
    )

    a, b = st.columns([3, 2])

    with a:
        st.markdown(f"""
<div style='font-size:0.85rem;line-height:1.65;color:{INK}'>
<b>Recursive filtered regime probabilities.</b> The Markov model is estimated
on data through 2015 only, then re-estimated each quarter using data available
at that date. Probabilities are <i>filtered</i>, never smoothed. A smoothed
probability at any historical date is computed using the whole sample - on
6 February 2009 the smoothed model says the high-correlation regime has
probability 0.02, the filtered model says 0.98. Opposite conclusions, same
date. Using the smoothed series would have quietly inserted hindsight into
every position in the backtest.
<br><br>
<b>Weight lag.</b> Weights decided at Friday's close are applied to the
following week's returns. No position benefits from information available
the day it was set.
<br><br>
<b>Point-in-time macro data.</b> Macro signals use first-release values via
ALFRED, not the revised series that FRED serves today. Testing the same
payrolls momentum rule both ways gives Sharpe 0.69 on first-release data
against 0.72 on revised - the 0.03 gap is revision alpha that would not have
existed live.
</div>""", unsafe_allow_html=True)

    with b:
        la = M.get("lookahead", {}).get("Tilted", {})
        if la:
            st.markdown(f"""
<div style='border:1px solid #E3E6E8;padding:1rem;font-size:0.85rem;
            line-height:1.6;color:{INK}'>
<b>Adversarial lag test</b><br>
<span style='color:{GREY}'>Delay every signal by one extra week and re-run. A
backtest that was peeking collapses; one that is clean degrades mildly.</span>
<br><br>
Base Sharpe &nbsp;&nbsp;<b>{la.get('base_sharpe', float('nan')):.3f}</b><br>
+1 week lag &nbsp;<b>{la.get('lagged_sharpe', float('nan')):.3f}</b><br>
Degradation &nbsp;<b>{la.get('degradation', float('nan')):+.1%}</b>
<br><br>
<span style='color:{NAVY};font-weight:600'>{la.get('verdict', 'n/a')}</span>
<span style='color:{GREY}'> - threshold is 30%. Some decay is expected and
healthy: it confirms the signal has genuine timing content.</span>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # -------------------------------------------------- crisis episodes -----
    st.markdown("### Behaviour in every crisis of the sample")
    st.caption(
        "Drawdown ratios cannot be bootstrapped meaningfully - maximum drawdown "
        "is a path-dependent extreme, and resampling blocks destroys the "
        "multi-year episodes that produce it. Seven independent stress periods "
        "is the better evidence."
    )

    ce = D.get("crisis_episodes")
    if ce is None or ce.empty:
        st.info("Crisis episode table not available - re-run precompute.py")
    else:
        cols_keep = [c for c in ce.columns
                     if c.startswith(("Tilted_", "EW_")) and c.endswith(("_ret", "_dd"))]
        t = ce[cols_keep].copy()
        t.columns = [c.replace("Tilted_ret", "ERC+tilt return")
                      .replace("Tilted_dd", "ERC+tilt max DD")
                      .replace("EW_ret", "Buy & hold return")
                      .replace("EW_dd", "Buy & hold max DD") for c in t.columns]
        order = ["ERC+tilt return", "Buy & hold return",
                 "ERC+tilt max DD", "Buy & hold max DD"]
        t = t[[c for c in order if c in t.columns]]

        st.dataframe(
            t.style.format("{:.1%}")
             .map(lambda v: f"color:{RED}" if isinstance(v, float) and v < -0.10 else "",
                  subset=[c for c in t.columns if "DD" in c]),
            use_container_width=True,
        )

        dd_t = t.get("ERC+tilt max DD")
        dd_e = t.get("Buy & hold max DD")
        if dd_t is not None and dd_e is not None:
            wins = int((dd_t > dd_e).sum())
            st.caption(
                f"Smaller drawdown in {wins} of {len(t)} episodes. Independent "
                "crises rather than resampled blocks - this is the evidence the "
                "drawdown claim rests on."
            )

    st.markdown("---")

    # ------------------------------------------------- statistical honesty --
    st.markdown("### What the statistics do and do not support")

    bci = D.get("bootstrap_ci")
    c1, c2 = st.columns([2, 3])

    with c1:
        if bci is not None and not bci.empty:
            rows = [r for r in ["Tilted", "EW"] if r in bci.index]
            names = {"Tilted": "ERC + tilt", "EW": "Buy & hold"}
            fig = go.Figure()
            for i, r in enumerate(rows):
                fig.add_trace(go.Scatter(
                    x=[bci.loc[r, "ci_low"], bci.loc[r, "ci_high"]],
                    y=[names[r], names[r]], mode="lines",
                    line=dict(color=GREY, width=2), showlegend=False,
                    hoverinfo="skip"))
                fig.add_trace(go.Scatter(
                    x=[bci.loc[r, "mean"]], y=[names[r]], mode="markers",
                    marker=dict(color=NAVY, size=10), showlegend=False,
                    hovertemplate=f"{names[r]}<br>mean %{{x:.2f}}<extra></extra>"))
            fig.add_vline(x=0, line_dash="dot", line_color=GREY, line_width=1)
            fig.update_layout(
                template="simple_white", height=200,
                margin=dict(l=0, r=10, t=20, b=30),
                xaxis=dict(title="excess Sharpe, 95% CI"), yaxis=dict(title=None),
                font=dict(size=11))
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown(f"""
<div style='font-size:0.85rem;line-height:1.65;color:{INK}'>
Excess-Sharpe confidence intervals come from a block bootstrap - 5,000
resamples of 52-week blocks, which preserves the volatility clustering that an
IID bootstrap would destroy. The intervals overlap almost entirely, and the
pairwise difference has a p-value above 0.3.
<br><br>
<b>So no Sharpe-superiority claim is made.</b> Nineteen years of weekly data
cannot distinguish two strategies whose true Sharpes differ by less than
roughly 0.4. Any project reporting a 0.05 Sharpe edge as a finding, without an
interval around it, is reporting noise.
<br><br>
Drawdown is different. A worst loss of 14% against 32% is not a marginal
statistical gap - it is a structural consequence of holding nine risk-balanced
assets rather than nine equally-weighted ones, and it repeats across
independent crises.
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ---------------------------------------------------------- limitations -
    st.markdown("### Limitations")

    l1, l2 = st.columns(2)

    with l1:
        st.markdown(f"""
<div style='font-size:0.82rem;line-height:1.6;color:{INK}'>
<b>Instruments.</b> Rates exposure uses modified-duration return proxies built
from constant-maturity yields, not investable bonds. The proxy captures price
change and carry but ignores roll-down, and holds duration fixed at 1.9 and 8.5
across a yield range of 0-5% where true modified duration varies by more than a
year. SHY and IEF are the implementable equivalents; the proxies are retained
because they reconcile against the coursework this project extends.
<br><br>
<b>Credit and EM.</b> LQD, HYG and EEM are ETFs, not the indices they track.
Their realised volatility runs 2-6pp above index equivalents because of
intraday pricing. Euro Stoxx exposure via FEZ is USD-denominated and unhedged,
so it carries EUR/USD risk a hedged mandate would not.
<br><br>
<b>Winsorization.</b> Daily returns are capped at &plusmn;25% for both
estimation and P&amp;L. This is defensible for liquid ETFs - none of the nine
genuinely moves 25% in a day - but it does mean the 2020 crude collapse enters
the record smaller than the raw log-return arithmetic implies.
</div>""", unsafe_allow_html=True)

    with l2:
        methods = M.get("garch_methods", {})
        ewma = [a for a, m in methods.items() if "EWMA" in str(m)]
        st.markdown(f"""
<div style='font-size:0.82rem;line-height:1.6;color:{INK}'>
<b>Model fitting.</b> {', '.join(ewma) if ewma else 'One asset'} hit the IGARCH
boundary and falls back to EWMA(0.94). Because scalar DCC evolves each pair
from its own standardised residuals, this does not touch the S&amp;P/10Y
element that drives the regime model - only the covariance the optimiser sees.
<br><br>
<b>Specification.</b> GARCH(1,1)-t, DCC at Engle's (0.05, 0.93), the 12-1M and
200-day lookbacks and the &plusmn;4pp cap were all fixed before any
performance was measured, and taken from published work rather than tuned.
The residual concern is that the specification was chosen in 2026 with
knowledge of which methods are well regarded - true of every backtest, and
mitigated only by using standard rather than bespoke choices.
<br><br>
<b>Discarded work.</b> A second strategy - a cross-asset trend book with a
regime-gated exposure overlay - was built, ablated layer by layer, and left out
of this dashboard. Its vol-targeting layer proved actively harmful and was
removed after testing. A variant deploying the full book rather than holding
cash was also tested and rejected: identical return, three times the drawdown.
Both are documented in the repository rather than quietly dropped.
<br><br>
<b>Scale.</b> Costs assume an institutional book in liquid ETFs. There is no
market-impact model, so the results do not describe a size at which a
&plusmn;4pp tilt in EEM or HYG would move prices.
</div>""", unsafe_allow_html=True)

    st.markdown("")
    st.caption(
        "Model positioning, not investment advice. Built on EMiF Project 2 "
        "(Piras, D'Amico, Berardi 2025) and extended: production data layer with "
        "cached fallback, bond carry, recursive filtered regime estimation, "
        "point-in-time macro data, cost and look-ahead testing."
    )
