"""The Strategy - what ERC + tilt is, and the evidence it works."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

NAVY, GREY, AMBER, RED, INK = "#0B3040", "#8A9199", "#B45309", "#9B1C1C", "#1A1A1A"


def render(D: dict, M: dict):
    equity = D.get("equity_curves")
    dd = D.get("drawdowns")
    comp = D.get("comparison")

    # ---------------------------------------------------------- explanation --
    st.markdown("### How positions are set")
    st.caption(
        "Three layers. Each answers a different question and each was specified "
        "before any performance was measured."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
<div style='font-size:0.85rem;line-height:1.6;color:{INK}'>
<b>1 · Strategic anchor — Equal Risk Contribution</b><br>
<span style='color:{GREY}'>Question: how much of each asset, absent any view?</span><br><br>
Weights are solved monthly so every asset contributes an equal share of
portfolio variance, using the GARCH-filtered DCC covariance. Low-volatility
assets take large capital weights and high-volatility assets take small ones,
which is why rates dominate the book. Long-only, weights sum to 100%.
<br><br>
<span style='color:{GREY}'>Maillard, Roncalli &amp; Teiletche (2010)</span>
</div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
<div style='font-size:0.85rem;line-height:1.6;color:{INK}'>
<b>2 · Tactical tilt — time-series momentum</b><br>
<span style='color:{GREY}'>Question: which assets are trending, and by how much
should that move the weights?</span><br><br>
Each week an asset is tilted &plusmn;4pp around its anchor weight if its
12-1 month return AND its 200-day moving average agree on direction.
Disagreement produces no tilt. Requiring agreement means fewer trades and
fewer whipsaws in choppy markets.
<br><br>
<span style='color:{GREY}'>Moskowitz, Ooi &amp; Pedersen (2012); Faber (2007)</span>
</div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
<div style='font-size:0.85rem;line-height:1.6;color:{INK}'>
<b>3 · Regime gate — Markov switching</b><br>
<span style='color:{GREY}'>Question: is this an environment where trend signals
can be trusted?</span><br><br>
A two-state model on the weekly S&amp;P/10Y correlation. When the filtered
probability of the high-correlation state exceeds 0.70, the tilt cap halves to
&plusmn;2pp. Correlations rising together is when diversification fails and
trend signals get noisiest, so the model takes less active risk there.
<br><br>
<span style='color:{GREY}'>Hamilton (1989); Engle (2002)</span>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ----------------------------------------------------------- evidence ----
    st.markdown("### Does it work?")
    st.caption(
        "Against equal-weight buy-and-hold of the same nine assets - the "
        "honest 'what if I did nothing' comparison. Weekly rebalancing, one-week "
        "weight lag, 5bp one-way transaction costs, excess of the 3M T-bill."
    )

    if equity is None or "Tilted" not in equity.columns:
        st.info("Backtest artifacts not available - re-run precompute.py")
        return

    show = [c for c in ["Tilted", "EW"] if c in equity.columns]
    names = {"Tilted": "ERC + tilt", "EW": "Equal-weight buy & hold"}
    colours = {"Tilted": NAVY, "EW": GREY}

    fig = go.Figure()
    for c in show:
        fig.add_trace(go.Scatter(
            x=equity.index, y=equity[c], name=names[c],
            line=dict(color=colours[c], width=1.4),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}x<extra></extra>",
        ))
    fig.update_layout(
        template="simple_white", height=340,
        margin=dict(l=0, r=0, t=30, b=0),
        yaxis=dict(title="growth of 1.00", type="log",
                   tickvals=[1, 1.5, 2, 3, 4, 5], ticktext=["1.0x", "1.5x", "2.0x", "3.0x", "4.0x", "5.0x"]),
        xaxis=dict(title=None),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
        font=dict(size=11), hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    if dd is not None:
        dfig = go.Figure()
        for c in show:
            dfig.add_trace(go.Scatter(
                x=dd.index, y=dd[c], name=names[c], fill="tozeroy",
                line=dict(color=colours[c], width=0.8),
                fillcolor=("rgba(11,48,64,0.20)" if c == "Tilted"
                           else "rgba(138,145,153,0.25)"),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1%}<extra></extra>",
            ))
        dfig.update_layout(
            template="simple_white", height=210,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(title="drawdown", tickformat=".0%"),
            xaxis=dict(title=None), showlegend=False,
            font=dict(size=11), hovermode="x unified",
        )
        st.plotly_chart(dfig, use_container_width=True)
        st.caption("Log scale above; drawdown from running peak below.")

    # ------------------------------------------------------------- stats -----
    if comp is not None and not comp.empty:
        rows = [r for r in ["Tilted", "EW"] if r in comp.index]
        t = comp.loc[rows, ["ann_return", "ann_vol", "sharpe_excess",
                            "max_dd", "calmar", "ann_turnover"]].copy()
        t.index = [names[r] for r in rows]
        t.columns = ["Ann. return", "Ann. vol", "Excess Sharpe",
                     "Max drawdown", "Calmar", "Ann. turnover"]

        st.markdown("")
        st.dataframe(
            t.style.format({
                "Ann. return": "{:.2%}", "Ann. vol": "{:.2%}",
                "Excess Sharpe": "{:.2f}", "Max drawdown": "{:.1%}",
                "Calmar": "{:.2f}", "Ann. turnover": "{:.0%}",
            }),
            use_container_width=True,
        )

        cs = D.get("cost_sensitivity")
        if cs is not None and not cs.empty:
            st.markdown("")
            st.markdown("#### What transaction costs do to the comparison")
            cst = cs.copy()
            cst.index = [f"{int(i)}bp one-way" for i in cst.index]
            show = cst[["drag_bps", "tilted_sharpe", "ew_sharpe", "edge", "tilted_maxdd"]]
            show.columns = ["Annual drag (bp)", "ERC + tilt Sharpe",
                            "Buy & hold Sharpe", "Edge", "ERC + tilt max DD"]
            st.dataframe(
                show.style
                    .format({"Annual drag (bp)": "{:.0f}", "ERC + tilt Sharpe": "{:.2f}",
                             "Buy & hold Sharpe": "{:.2f}", "Edge": "{:+.2f}",
                             "ERC + tilt max DD": "{:.1%}"})
                    .map(lambda v: f"color:{RED}" if isinstance(v, float) and v < 0 else "",
                         subset=["Edge"]),
                use_container_width=True,
            )
            st.caption(
                "The strategy turns over about twice its value a year against "
                "roughly nothing for buy-and-hold, so it is the side that pays. "
                "5bp one-way is realistic for an institutional book trading these "
                "nine ETFs, which are among the most liquid instruments listed. "
                "For a retail account paying spread plus commission, 10-20bp is "
                "the honest assumption - and at that level the Sharpe edge is "
                "gone. Note the drawdown column: it barely moves. That is the "
                "reason the claim on this page is about drawdown and not Sharpe."
            )

        tl, ew = comp.loc["Tilted"], comp.loc["EW"]
        st.markdown(f"""
<div style='font-size:0.85rem;line-height:1.65;color:{INK}'>
<b>The honest read.</b> Equal-weight buy-and-hold compounds faster
({ew['ann_return']:.1%} a year against {tl['ann_return']:.1%}) and ends with more
money. What the strategy delivers is a different risk path: a worst drawdown of
{abs(tl['max_dd']):.0%} against {abs(ew['max_dd']):.0%}, roughly a third of the
tail loss, and a Calmar ratio of {tl['calmar']:.2f} against {ew['calmar']:.2f}.
<br><br>
On excess Sharpe the two are {tl['sharpe_excess']:.2f} and {ew['sharpe_excess']:.2f}
&mdash; a gap that a block bootstrap cannot distinguish from zero over this
sample. No Sharpe-superiority claim is made here. The claim is about drawdown,
where nineteen years is long enough to see a structural difference rather than
noise, and it comes at a cost of {tl['ann_turnover']:.0%} annual turnover against
almost none.
<br><br>
This suits a mandate where the drawdown constraint binds harder than the return
target. Where it does not, buy-and-hold is the better answer and this dashboard
says so.
</div>""", unsafe_allow_html=True)
