"""Market Conditions - where each asset sits relative to its own history."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

NAVY, GREY, AMBER, RED, INK = "#0B3040", "#8A9199", "#B45309", "#9B1C1C", "#1A1A1A"

LABELS = {
    "SP500": "S&P 500", "EuroStoxx50": "Euro Stoxx 50", "MSCI_EM": "MSCI EM",
    "Gold": "Gold", "Oil_WTI": "WTI Crude", "US_IG": "US IG Credit",
    "US_HY": "US HY Credit", "US_2Y_proxy": "US 2Y", "US_10Y_proxy": "US 10Y",
}


def render(D: dict, M: dict):
    vol = D.get("vol_percentiles")
    corr = D.get("corr_matrix_latest")
    regime = D.get("regime")
    dnsi = D.get("dnsi")

    # ------------------------------------------------- conditional vol ------
    st.markdown("### Volatility vs own history")
    st.caption(
        "GARCH(1,1)-t conditional volatility, annualised, with each asset ranked "
        "against its own distribution since 2007. Comparing an asset to itself is "
        "the only meaningful comparison - a 3% move in US 2Y and a 60% move in "
        "oil are both normal for their respective series."
    )

    if vol is None or vol.empty:
        st.info("Volatility percentiles not available - re-run precompute.py")
    else:
        v = vol.sort_values("pctile", ascending=True)
        labels = [LABELS.get(a, a) for a in v.index]
        # Colour is reserved for genuine outliers. Most assets sit in a normal
        # range for themselves; painting five of nine amber destroys the signal.
        colours = [RED if p >= 90 else (GREY if p <= 15 else NAVY)
                   for p in v["pctile"]]

        fig = go.Figure(go.Bar(
            y=labels, x=v["pctile"], orientation="h",
            marker_color=colours,
            text=[f"{r.vol_ann:.0%} ann" for r in v.itertuples()],
            textposition="outside",
            hovertemplate="%{y}<br>%{x:.0f}th percentile<extra></extra>",
        ))
        fig.add_vline(x=50, line_dash="dot", line_color=GREY, line_width=1)
        fig.update_layout(
            template="simple_white", height=360,
            margin=dict(l=0, r=60, t=10, b=30),
            xaxis=dict(title="percentile of own history since 2007",
                       range=[0, 108], ticksuffix=""),
            yaxis=dict(title=None), font=dict(size=11), showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        hot = v[v["pctile"] >= 85]
        cold = v[v["pctile"] <= 15]
        bits = []
        if len(hot):
            bits.append("Elevated: " + ", ".join(
                f"{LABELS.get(a, a)} ({r.pctile:.0f}th)" for a, r in hot.iterrows()))
        if len(cold):
            bits.append("Subdued: " + ", ".join(
                f"{LABELS.get(a, a)} ({r.pctile:.0f}th)" for a, r in cold.iterrows()))
        if bits:
            st.markdown(
                f"<p style='font-size:0.85rem;color:{INK}'>" + " · ".join(bits) + "</p>",
                unsafe_allow_html=True)

        methods = M.get("garch_methods", {})
        ewma = [LABELS.get(a, a) for a, m in methods.items() if "EWMA" in str(m)]
        if ewma:
            st.caption(
                f"Note: {', '.join(ewma)} uses an EWMA(0.94) fallback rather than "
                "GARCH - the GARCH fit hit the IGARCH boundary. EWMA responds "
                "faster to quiet periods, so read its percentile with that in mind."
            )

    st.markdown("---")

    # ------------------------------------- correlation + regime side by side -
    left, right = st.columns([1, 1])

    with left:
        st.markdown("### Current correlation structure")
        if corr is None or corr.empty:
            st.info("Correlation matrix not available.")
        else:
            order = [a for a in LABELS if a in corr.columns]
            c = corr.loc[order, order]
            fig = go.Figure(go.Heatmap(
                z=c.values,
                x=[LABELS[a] for a in order],
                y=[LABELS[a] for a in order],
                colorscale=[[0.0, "#B45309"], [0.5, "#FFFFFF"], [1.0, NAVY]],
                zmid=0, zmin=-1, zmax=1,
                hovertemplate="%{y} / %{x}<br>rho = %{z:.2f}<extra></extra>",
                colorbar=dict(thickness=10, len=0.8, tickvals=[-1, 0, 1]),
            ))
            fig.update_layout(
                template="simple_white", height=420,
                margin=dict(l=0, r=0, t=10, b=0),
                font=dict(size=10),
                xaxis=dict(tickangle=-45), yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "DCC(1,1) conditional correlations as of the latest observation. "
                "Navy = positive, amber = negative. These feed the ERC optimiser; "
                "the S&P/10Y cell drives the regime model. Note that WTI is "
                "currently the only asset negatively correlated with the rest of "
                "the book - in a panel where equities, credit and rates are all "
                "co-moving, crude is carrying the diversification."
            )

    with right:
        st.markdown("### Regime tape")
        if regime is None or regime.empty:
            st.info("Regime series not available.")
        else:
            r = regime.dropna(subset=["p_high_filtered"])

            # Regime as BACKGROUND shading, DCC correlation as the foreground
            # line. Filtered probabilities whipsaw week to week - genuine model
            # behaviour, but unreadable as a foreground series. Shading contiguous
            # blocks where P > threshold shows regime persistence at a glance.
            # Shade on the 13-week average, not the raw weekly series. The
            # weekly filtered probability flips constantly, producing hundreds of
            # one-week bands that communicate nothing. The smoothed version shows
            # regime PERSISTENCE, which is what the reader needs to see.
            p_smooth = r["p_high_filtered"].rolling(13, min_periods=4).mean()
            gated = (p_smooth > 0.70).astype(int)
            blocks, start_i = [], None
            for ts, on in gated.items():
                if on and start_i is None:
                    start_i = ts
                elif not on and start_i is not None:
                    blocks.append((start_i, ts))
                    start_i = None
            if start_i is not None:
                blocks.append((start_i, gated.index[-1]))

            fig = go.Figure()
            for b0, b1 in blocks:
                fig.add_vrect(x0=b0, x1=b1, fillcolor=AMBER, opacity=0.10,
                              layer="below", line_width=0)
            fig.add_trace(go.Scatter(
                x=r.index, y=r["dcc_weekly"], name="S&P / 10Y DCC correlation",
                line=dict(color=NAVY, width=1.2),
                hovertemplate="%{x|%Y-%m-%d}<br>rho = %{y:.2f}<extra></extra>",
            ))
            fig.add_hline(y=0, line_dash="dot", line_color=GREY, line_width=1)
            fig.add_vline(x=pd.Timestamp("2015-12-31"), line_dash="dash",
                          line_color=GREY, line_width=1)
            fig.add_annotation(
                x=pd.Timestamp("2015-12-31"), y=0.72, yref="y",
                text="burn-in ends", showarrow=False,
                font=dict(size=9, color=GREY), xanchor="left")
            fig.update_layout(
                template="simple_white", height=420,
                margin=dict(l=0, r=0, t=40, b=0),
                yaxis=dict(title="S&P / 10Y correlation", range=[-0.9, 0.9]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                            font=dict(size=10)),
                font=dict(size=11), hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Amber shading marks weeks the model assigns to the "
                "high-correlation regime (P > 0.70), when the tactical tilt is "
                "halved. Probabilities are FILTERED and recursively estimated - "
                "parameters at each date use only data available up to that date; "
                "smoothed probabilities would leak hindsight. The dotted line is a "
                "13-week average: the weekly series flips frequently, which is why "
                "the gate uses a threshold rather than reacting to every move."
            )

    st.markdown("---")

    # -------------------------------------------------------- news sentiment -
    st.markdown("### Economic news sentiment")
    if dnsi is None or dnsi.empty:
        st.info("Sentiment series not available.")
    else:
        s = dnsi.squeeze() if dnsi.shape[1] == 1 else dnsi.iloc[:, 0]
        s = s.dropna()
        roll_lo = s.rolling(252 * 5, min_periods=252).quantile(0.10)
        roll_hi = s.rolling(252 * 5, min_periods=252).quantile(0.90)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s.index, y=roll_hi, line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=s.index, y=roll_lo, line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(138,145,153,0.18)",
                                 name="10th-90th pctile, trailing 5y",
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=s.index, y=s, line=dict(color=NAVY, width=1),
                                 name="Daily News Sentiment Index",
                                 hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3f}<extra></extra>"))
        fig.add_hline(y=0, line_dash="dot", line_color=GREY, line_width=1)
        fig.update_layout(
            template="simple_white", height=300,
            margin=dict(l=0, r=0, t=30, b=0),
            yaxis=dict(title=None), xaxis=dict(title=None),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                        font=dict(size=10)),
            font=dict(size=11),
        )
        st.plotly_chart(fig, use_container_width=True)
        d = M.get("dnsi", {})
        st.caption(
            f"San Francisco Fed Daily News Sentiment Index, lexical analysis of 24 "
            f"major US newspapers, daily since 1980 (Buckman, Shapiro, Sudhof & "
            f"Wilson 2020). Currently {d.get('percentile', 0):.0f}th percentile. "
            "Displayed as market context - it is NOT an input to any position."
        )
