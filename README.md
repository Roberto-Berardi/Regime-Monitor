# Cross-Asset Regime Monitor

> Live weekly market brief for multi-asset portfolios.
> Strategic ERC core + tactical momentum tilt + Markov regime gate.

**Author:** Roberto Berardi — MSc Finance, HEC Lausanne
**Status:** Phases 0-2 complete · Phase 3 in progress (GARCH)
**Live app:** *coming soon — deployed on Streamlit Cloud in Phase 10*
**Repo:** [Roberto-Berardi/Regime-Monitor](https://github.com/Roberto-Berardi/Regime-Monitor)

---

## What this project does

Every Monday, this pipeline pulls fresh cross-asset data, refits the models, and publishes a one-page weekly brief that answers three questions a PM asks every morning:

1. **What happened last week** — asset returns, macro data releases, surprises.
2. **Why** — short commentary linking releases to price action.
3. **What to do about it** — current portfolio positioning and this week's tilts.

The engine behind the positioning is a three-layer system:

| Layer | What it does | Method |
|-------|--------------|--------|
| Strategic core | Equal Risk Contribution weights across 9 assets | ERC on GARCH-filtered covariance |
| Tactical tilt | ± 4pp per asset around ERC weights | 12-1M time-series momentum + 200DMA confirmation |
| Regime gate | Halves tilt magnitude in high-correlation regimes | 2-state Markov switching on weekly stock-bond DCC |

## Asset universe

9 assets, chosen to span the main risk factors in a multi-asset book:

- **Equities:** S&P 500, Euro Stoxx 50, MSCI EM
- **Rates (duration proxies):** US 2Y, US 10Y
- **Credit:** US IG, US HY
- **Commodities:** Gold, WTI Oil

Data sources: Yahoo Finance (equities, commodities, credit ETFs) and FRED (yields, spreads, macro releases). Both free.

## Tech stack

- **Data & modeling:** pandas, numpy, scipy, statsmodels, arch
- **App layer:** Streamlit + Plotly
- **Deployment:** Streamlit Community Cloud + GitHub Actions (weekly refresh)
- **Environment:** Python 3.11, conda

## Repository layout

    regime-monitor/
    ├── app/          # Streamlit web app
    ├── src/          # Analytics modules (data, returns, garch, dcc, regime, erc, tilt)
    ├── data/         # Cached parquet snapshots (gitignored)
    ├── notebooks/    # Exploratory work
    ├── tests/        # Sanity checks
    ├── config.py     # Central configuration — every model assumption
    └── requirements.txt

Every model constant (GARCH spec, DCC parameters, tilt cap, regime threshold, transaction costs, return cap) lives in `config.py`. If a choice needs defending, the answer is one line in that file.

## Foundations

Built on empirical methods coursework at HEC Lausanne (EMiF Project 2), extended for production: recursive parameter estimation to prevent look-ahead in the regime gate, transaction-cost accounting in the backtest, cached data with stale-data fallback so the deployed app degrades gracefully.

## Build log

- **Phase 0** — Project skeleton, environment (Python 3.11 conda), config, git+GitHub setup.
- **Phase 1** — Data layer: Yahoo + FRED downloaders, parquet cache, 4-check validator, graceful cache fallback (tested against simulated Yahoo outage).
- **Phase 2** — Returns module: log returns for prices, modified-duration proxies for bonds, ±25% winsorization for extreme events. Reconciled against EMiF Project 2 — 5/5 comparable assets within 5% relative deviation.

## Data notes

- **Free-data limitation:** FRED restricted ICE BofA credit spread series to a rolling 3-year window in April 2026 (licensing change with ICE Data Indices). Long-history context uses Moody's Baa/Aaa Fed-computed proxies. Documented in `config.py`.
- **Winsorization:** daily returns are capped at ±25% per asset before entering any model. The 2020-04-20 WTI negative-price event and 2008 EM equity outliers survive as extreme days but no longer dominate volatility estimation. Sign and direction preserved.
- **Instrument choice:** US IG and US HY use LQD and HYG ETFs (Yahoo) rather than bond total-return indices; MSCI EM uses the EEM ETF. ETF vols run 2-6pp higher than the index equivalents — a documented trade-off in favour of intraday-refreshable data.

## Limitations

Pedagogical demonstration, not investment advice. Bonds are represented via modified-duration return proxies, not investable instruments. DCC parameters follow Engle (2002) convention rather than QMLE estimation in the base spec. Full limitations section is displayed inside the deployed app.

## References

- Engle, R. (2002). *Dynamic conditional correlation.* JBES 20(3).
- Maillard, S., Roncalli, T., & Teiletche, J. (2010). *The properties of equally weighted risk contribution portfolios.* J. of Portfolio Management.
- Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012). *Time series momentum.* JFE 104(2).

---

*This project is a public portfolio piece for MSc Finance internship applications. Feedback welcome via Issues.*