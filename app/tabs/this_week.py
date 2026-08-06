"""This Week — the weekly market read and current positioning."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

NAVY, GREY, AMBER, INK = "#0B3040", "#8A9199", "#B45309", "#1A1A1A"

LABELS = {
    "SP500": "S&P 500", "EuroStoxx50": "Euro Stoxx 50", "MSCI_EM": "MSCI EM",
    "Gold": "Gold", "Oil_WTI": "WTI Crude", "US_IG": "US IG Credit",
    "US_HY": "US HY Credit", "US_2Y_proxy": "US 2Y", "US_10Y_proxy": "US 10Y",
}


def _signal_prices(panel: pd.DataFrame, returns_daily: pd.DataFrame,
                   assets: list) -> pd.DataFrame:
    """
    Reproduce EXACTLY the price basis src.momentum.build_signal_panel_full uses:
      - price assets  -> raw forward-filled prices from the panel
      - bond proxies  -> cumulative-return synthetic prices

    This matters. The returns series is winsorized at +/-25%, so a cumulative
    return series diverges sharply from raw prices for assets with extreme
    days (WTI, April 2020). Computing the displayed momentum on a different
    basis than the signal makes the table contradict itself.
    """
    cols = {}
    for a in assets:
        if a.endswith("_proxy"):
            cols[a] = (1.0 + returns_daily[a].fillna(0.0)).cumprod()
        elif a in panel.columns:
            cols[a] = panel[a].ffill()
    # panel and returns_daily sit on different indices (raw daily vs business-day),
    # so the union introduces gaps; ffill before any rolling window is applied.
    return pd.DataFrame(cols).reindex(columns=assets).ffill()


def render(D: dict, M: dict):
    narr = M.get("narrative", {})
    sig_w = D["signals_weekly"]
    tilt_w = D["tilted_weights_weekly"]
    erc_w = D["erc_weights_weekly"]
    rets = D["returns_daily"]
    moves = D.get("weekly_moves")

    assets = [c for c in tilt_w.columns if c in LABELS]
    last = sig_w.index[-1]
    prev = sig_w.index[-2] if len(sig_w) > 1 else last

    # ---------------------------------------------------------------- brief --
    st.markdown("### The week in brief")
    period = narr.get("period_label", "week")
    st.caption(
        f"Week ending {narr.get('week_end', last.date())} - {period}. "
        "Generated from model output, not written by hand. No causal claims: "
        "releases and price moves are reported side by side, not linked."
    )

    c1, c2 = st.columns(2)
    with c1:
        for key in ["regime", "moves", "signals"]:
            if narr.get(key):
                st.markdown(
                    f"<p style='font-size:0.88rem;line-height:1.6;color:{INK}'>"
                    f"{narr[key]}</p>", unsafe_allow_html=True)
    with c2:
        for key in ["positioning", "macro", "sentiment"]:
            if narr.get(key):
                st.markdown(
                    f"<p style='font-size:0.88rem;line-height:1.6;color:{INK}'>"
                    f"{narr[key]}</p>", unsafe_allow_html=True)

    st.markdown("---")

    # ------------------------------------------------------------ risk strip -
    rn = M.get("risk_now", {})
    if rn:
        st.markdown("### Risk being run right now")
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Ex-ante vol", f"{rn['vol_ann']*100:.1f}%")
        r2.metric("Duration", f"{rn['duration_years']:.1f} yrs")
        r3.metric("Equity beta", f"{rn['equity_beta']:.2f}")
        r4.metric("1w 95% ES", f"{rn['es95_weekly']*100:.2f}%")
        r5.metric("Largest position", f"{rn['largest_weight']*100:.0f}%")
        st.caption(
            "Forward-looking, from the current GARCH/DCC covariance - not "
            "realised history. Duration counts the rate proxies only, so it "
            "understates true rate sensitivity: the credit ETFs carry spread "
            "duration this figure ignores. Expected shortfall assumes normal "
            "innovations; the fitted residuals are Student-t with 4-10 degrees "
            "of freedom, so treat it as a floor rather than a worst case."
        )

    # --------------------------------------------------------- signal panel --
    prices = _signal_prices(D["panel"], rets, assets)
    mom_12_1 = (prices.shift(21) / prices.shift(252) - 1.0).iloc[-1]
    ret_1m = (prices.iloc[-1] / prices.shift(21).iloc[-1] - 1.0)
    ret_3m = (prices.iloc[-1] / prices.shift(63).iloc[-1] - 1.0)
    ma200 = prices.rolling(200, min_periods=200).mean()
    dist_ma = (prices.iloc[-1] / ma200.iloc[-1] - 1.0)

    move_map = {}
    if moves is not None and "ret" in moves.columns:
        move_map = moves["ret"].to_dict()

    rows = []
    for a in assets:
        s_now = float(sig_w.loc[last, a])
        s_prev = float(sig_w.loc[prev, a])
        rows.append({
            "Asset":     LABELS[a],
            "Signal":    "UP" if s_now > 0 else ("DOWN" if s_now < 0 else "-"),
            "Flip":      "*" if s_now != s_prev else "",
            "Week":      move_map.get(a, np.nan),
            "1M":        ret_1m.get(a, np.nan),
            "3M":        ret_3m.get(a, np.nan),
            "12-1M":     mom_12_1.get(a, np.nan),
            "vs 200DMA": dist_ma.get(a, np.nan),
            "Weight":    tilt_w.loc[last, a],
            "vs ERC":    tilt_w.loc[last, a] - erc_w.loc[last, a],
        })
    tbl = pd.DataFrame(rows).set_index("Asset")

    st.markdown("### Signal panel")
    st.caption(
        "UP = 12-1M momentum and 200DMA both positive; DOWN = both negative; "
        "- = disagreement, no position. Asterisk marks a signal that changed "
        "this week. 1M and 3M are context only, not decision inputs - the "
        "signal spec was fixed before any performance was measured."
    )

    st.dataframe(
        tbl.style
           .format({"Week": "{:+.1%}", "1M": "{:+.1%}", "3M": "{:+.1%}",
                    "12-1M": "{:+.1%}", "vs 200DMA": "{:+.1%}",
                    "Weight": "{:.1%}", "vs ERC": "{:+.1%}"})
           .map(lambda v: f"color:{AMBER};font-weight:700" if v == "*" else "",
                subset=["Flip"])
           .map(lambda v: (f"color:{NAVY};font-weight:600" if v == "UP" else
                           (f"color:{AMBER};font-weight:600" if v == "DOWN" else
                            f"color:{GREY}")),
                subset=["Signal"]),
        width="stretch",
        height=(len(tbl) + 1) * 35 + 3,
    )

    st.markdown("---")

    # ------------------------------------------------------------ positions --
    left, right = st.columns([3, 2])

    with left:
        st.markdown("### Positioning vs strategic anchor")
        order = tilt_w.loc[last, assets].sort_values().index.tolist()
        fig = go.Figure()
        fig.add_bar(y=[LABELS[a] for a in order],
                    x=[erc_w.loc[last, a] for a in order],
                    name="ERC anchor", orientation="h", marker_color=GREY)
        fig.add_bar(y=[LABELS[a] for a in order],
                    x=[tilt_w.loc[last, a] for a in order],
                    name="After tilt", orientation="h", marker_color=NAVY)
        fig.update_layout(
            barmode="group", template="simple_white", height=340,
            margin=dict(l=0, r=10, t=10, b=0),
            xaxis=dict(tickformat=".0%", title=None),
            yaxis=dict(title=None),
            legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
            font=dict(size=11),
        )
        st.plotly_chart(fig, width="stretch")
        gated = M["active_cap_latest"] < 4
        st.caption(
            f"Equal Risk Contribution anchor, rebalanced monthly, then tilted "
            f"weekly by +/-{M['active_cap_latest']:.0f}pp per asset on the trend "
            f"signal{' - halved because the regime gate is active' if gated else ''}. "
            "Long-only; weights sum to 100%."
        )

    with right:
        st.markdown("### Capital by asset class")
        w = tilt_w.loc[last, assets]
        groups = {
            "Equities":    ["SP500", "EuroStoxx50", "MSCI_EM"],
            "Rates":       ["US_2Y_proxy", "US_10Y_proxy"],
            "Credit":      ["US_IG", "US_HY"],
            "Commodities": ["Gold", "Oil_WTI"],
        }
        gsum = pd.DataFrame(
            {"Weight": [sum(w.get(a, 0) for a in mem) for mem in groups.values()]},
            index=list(groups.keys()),
        )
        gfig = go.Figure(go.Bar(
            y=gsum.index.tolist(), x=gsum["Weight"].tolist(),
            orientation="h", marker_color=NAVY))
        gfig.update_layout(
            template="simple_white", height=340,
            margin=dict(l=0, r=10, t=30, b=0),
            xaxis=dict(tickformat=".0%", title=None), yaxis=dict(title=None),
            font=dict(size=11), showlegend=False,
        )
        st.plotly_chart(gfig, width="stretch")
        st.caption(
            "Capital weights. The ERC anchor equalises RISK contribution across "
            "the nine assets, which is why low-volatility rates carry the largest "
            "capital weight - each asset contributes ~1/9th of portfolio variance "
            "regardless of how much capital it holds."
        )
