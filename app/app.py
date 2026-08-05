"""
Cross-Asset Regime Monitor — Streamlit dashboard.

Reads precomputed parquet artifacts only. Never runs GARCH, DCC, or the
Markov filter at request time — precompute.py does that offline and
GitHub Actions refreshes it weekly.
"""
from pathlib import Path
import json
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
APP_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(APP_DIR))
PRE = ROOT / "data" / "precomputed"

from tabs import this_week, market_conditions, strategy, method  # noqa: E402  (import after sys.path setup)

st.set_page_config(
    page_title="Cross-Asset Regime Monitor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- palette -----------------------------------------------------------------
NAVY  = "#0B3040"
GREY  = "#8A9199"
AMBER = "#B45309"
RED   = "#9B1C1C"
INK   = "#1A1A1A"

st.markdown(f"""
<style>
  .block-container {{ padding-top: 3.5rem !important; padding-bottom: 3rem; max-width: 1400px; }}

  .block-container h1 {{
      font-size: 1.55rem !important; font-weight: 600; color: {INK};
      letter-spacing: -0.01em; margin-bottom: 0.15rem; line-height: 1.25;
  }}
  .block-container h2 {{ font-size: 1.1rem !important; font-weight: 600; color: {INK};
      margin-top: 1.4rem; }}
  .block-container h3 {{ font-size: 0.9rem !important; font-weight: 600; color: {INK}; }}
  .block-container h1 a, .block-container h2 a, .block-container h3 a {{ display: none !important; }}

  .stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid #E3E6E8; }}
  .stTabs [data-baseweb="tab"] {{ font-size: 0.86rem; font-weight: 500; color: {GREY};
      padding: 0.4rem 0; background: transparent; }}
  .stTabs [aria-selected="true"] {{ color: {INK}; }}
  .stTabs [data-baseweb="tab-highlight"] {{ background-color: {NAVY} !important; }}

  div[data-testid="stMetricValue"] {{ font-size: 1.3rem !important; font-weight: 600; color: {INK}; }}
  div[data-testid="stMetricLabel"] p {{ font-size: 0.68rem !important; color: {GREY};
      text-transform: uppercase; letter-spacing: 0.05em; font-weight: 500; }}

  .stamp {{ font-size: 0.75rem; color: {GREY}; font-family: ui-monospace, Menlo, monospace; }}
  .sub {{ font-size: 0.72rem; color: {GREY}; margin-top: -0.35rem; }}
  .caption {{ font-size: 0.7rem; color: {GREY}; line-height: 1.5; }}
  hr {{ margin: 1.2rem 0; border-color: #E3E6E8; }}
  footer, #MainMenu, header[data-testid="stHeader"] {{ visibility: hidden; height: 0; }}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading precomputed data...")
def load_all() -> dict:
    """Load every precomputed artifact plus the metadata sidecar."""
    if not PRE.exists():
        return {}
    out = {}
    for path in sorted(PRE.glob("*.parquet")):
        out[path.stem] = pd.read_parquet(path)
    meta_path = PRE / "metadata.json"
    out["metadata"] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return out


def card(col, label, value, sub):
    """Metric card with a plain caption instead of a misleading delta arrow."""
    with col:
        st.metric(label, value)
        st.markdown(f"<div class='sub'>{sub}</div>", unsafe_allow_html=True)


D = load_all()

if not D or not D.get("metadata"):
    st.error(
        "Precomputed data not found. Run `python precompute.py` from the "
        "project root, then reload this page."
    )
    st.stop()

M = D["metadata"]

# --- header ------------------------------------------------------------------
head_l, head_r = st.columns([3, 5])

with head_l:
    st.markdown("# Cross-Asset Regime Monitor")
    st.markdown(
        f"<div class='stamp'>Data as of {M['data_as_of']} &middot; "
        f"source: {M['data_source']} &middot; rebalance {M['latest_rebalance']}</div>",
        unsafe_allow_html=True,
    )

with head_r:
    m1, m2, m3, m4 = st.columns(4)

p_high = M["p_high_latest"]
card(m1, "Regime P(high-corr)", f"{p_high:.2f}",
     "diversification impaired" if p_high > 0.70 else "diversifying")

cap = M["active_cap_latest"]
card(m2, "Strategy A tilt cap", f"±{cap:.0f}pp",
     "gated" if cap < 4 else "full")

_vol = D.get("vol_percentiles")
if _vol is not None and "SP500" in _vol.index:
    card(m3, "S&P 500 vol", f"{_vol.loc['SP500', 'vol_ann']:.0%}",
         f"{_vol.loc['SP500', 'pctile']:.0f}th pctile of own history")
else:
    card(m3, "Assets trending", "-", "")

dnsi = M["dnsi"]
card(m4, "News sentiment", f"{dnsi['percentile']:.0f}th pct",
     "improving" if dnsi["delta_1m_12m"] > 0 else "deteriorating")

st.markdown("---")

# --- tabs (filled in blocks 9.2 - 9.6) ---------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "This Week",
    "Market Conditions",
    "The Strategy",
    "Method & Limitations",
])

with tab1:
    this_week.render(D, M)

with tab2:
    market_conditions.render(D, M)

with tab3:
    strategy.render(D, M)

with tab4:
    method.render(D, M)

# --- footer ------------------------------------------------------------------
st.markdown("---")
st.markdown(
    f"<div class='caption'>Roberto Berardi &middot; MSc Finance, HEC Lausanne &middot; "
    f"Model positioning, not investment advice. "
    f"Built on EMiF Project 2 (Piras, D'Amico, Berardi 2025), extended for production. "
    f"Precomputed {M['generated_utc'][:16].replace('T', ' ')} UTC "
    f"in {M['elapsed_seconds']:.0f}s.</div>",
    unsafe_allow_html=True,
)
