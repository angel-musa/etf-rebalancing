import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import datetime
import os
import hashlib

from src.config import DEFAULT_BENCHMARK, DEFAULT_START_DATE_OFFSET_YEARS, TRADING_DAYS_PER_YEAR
from src.data_loader import load_data, calculate_returns
from src.metrics import (
    annualized_return, annualized_volatility, sharpe_ratio, calculate_beta,
    max_drawdown, rolling_volatility, rolling_correlation, cumulative_returns, rolling_return
)
from src.signals import generate_signals
from src.portfolio import calculate_portfolio_returns

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ETF Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get help": None, "Report a bug": None, "About": None}
)

ASSET_COLORS = [
    "#00D4FF", "#FF6B35", "#7ED321", "#BD10E0",
    "#F5A623", "#50C878", "#FF85A1", "#A8E6CF",
]

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.gradient-title {
    font-size:1.65rem; font-weight:800; margin:0; line-height:1.2;
    background:linear-gradient(90deg,#00D4FF 0%,#60a5fa 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.subtitle    { color:#64748b; font-size:.82rem; margin:4px 0 0; }
.section-lbl { font-size:10px; font-weight:700; letter-spacing:.12em;
               text-transform:uppercase; color:#475569; margin-bottom:6px; display:block; }

.regime-badge { display:inline-block; padding:4px 14px; border-radius:20px;
                font-size:11px; font-weight:700; letter-spacing:.06em; text-transform:uppercase; }
.risk-on  { background:rgba(16,185,129,.15); border:1px solid #10b981; color:#10b981; }
.risk-off { background:rgba(239,68,68,.15);  border:1px solid #ef4444; color:#ef4444; }
.neutral  { background:rgba(100,116,139,.15);border:1px solid #64748b; color:#94a3b8; }

.score-wrap   { background:rgba(17,24,39,.85); border:1px solid #1e2a4a;
                border-radius:10px; padding:14px 16px; margin-bottom:12px; }
.score-bar-bg { background:#1e2a4a; border-radius:999px; height:8px; margin:8px 0 4px; }
.score-bar    { height:8px; border-radius:999px; }

.insight { padding:10px 14px; margin:5px 0; border-radius:0 8px 8px 0;
           font-size:13px; line-height:1.6; color:#cbd5e1; }
.insight.info    { border-left:3px solid #00D4FF; background:rgba(0,212,255,.05); }
.insight.success { border-left:3px solid #10b981; background:rgba(16,185,129,.05); }
.insight.warn    { border-left:3px solid #f59e0b; background:rgba(245,158,11,.05); }
.insight.danger  { border-left:3px solid #ef4444; background:rgba(239,68,68,.05); }

.rebal-card { background:rgba(17,24,39,.7); border:1px solid #1e2a4a;
              border-radius:10px; padding:14px 16px; }
</style>
""", unsafe_allow_html=True)


# ─── HELPERS ─────────────────────────────────────────────────────────────────
def fmt_pct(v, d=1):
    return f"{v:.{d}%}" if pd.notna(v) else "—"

def fmt_f(v, d=2):
    return f"{v:.{d}f}" if pd.notna(v) else "—"

def signal_style(val):
    return {
        "Increase": "background-color:rgba(16,185,129,.12);color:#10b981;font-weight:700",
        "Review":   "background-color:rgba(239,68,68,.12);color:#ef4444;font-weight:700",
        "Hold":     "background-color:rgba(245,158,11,.12);color:#f59e0b;font-weight:700",
    }.get(val, "")

def ret_style(val):
    if not isinstance(val, float) or np.isnan(val): return ""
    return "color:#10b981;font-weight:600" if val >= 0 else "color:#ef4444;font-weight:600"

def health_score(port_sharpe, bench_sharpe, port_ret, bench_ret, sigs, regime, max_dd, beta):
    s = 50
    s += np.clip(port_sharpe - bench_sharpe, -1, 1) * 20
    s += 10 if port_ret > bench_ret else -10
    s += {"Risk-On": 5, "Risk-Off": -5}.get(regime, 0)
    s += (sigs == "Increase").sum() * 5 - (sigs == "Review").sum() * 8
    if max_dd < -0.30: s -= 10
    elif max_dd < -0.20: s -= 5
    if not (0.6 <= beta <= 1.4): s -= 5
    return int(np.clip(s, 0, 100))

def score_color(s):
    return "#10b981" if s >= 70 else ("#f59e0b" if s >= 45 else "#ef4444")

def ai_insights(port_ret, bench_ret, port_sharpe, max_dd, beta, sigs, regime, bm):
    items = []
    if regime == "Risk-On":
        items.append(("info",    "Market is in a <b>Risk-On</b> regime — recent vol is below the 12-month average, supporting equity exposure."))
    elif regime == "Risk-Off":
        items.append(("warn",    "Market is in a <b>Risk-Off</b> regime — volatility is elevated. Consider hedging or trimming high-beta positions."))
    else:
        items.append(("info",    "Market regime is <b>Neutral</b> — volatility in line with historical norms."))

    alpha = port_ret - bench_ret
    if alpha > 0:
        items.append(("success", f"Portfolio is <b>outperforming {bm}</b> by <b>{alpha:.1%}</b> annualized."))
    else:
        items.append(("warn",    f"Portfolio is <b>underperforming {bm}</b> by <b>{abs(alpha):.1%}</b> annualized."))

    if port_sharpe >= 1.0:
        items.append(("success", f"Sharpe of <b>{port_sharpe:.2f}</b> reflects strong risk-adjusted returns."))
    elif port_sharpe >= 0.5:
        items.append(("info",    f"Sharpe of <b>{port_sharpe:.2f}</b> is adequate but has room to improve."))
    else:
        items.append(("danger",  f"Sharpe of <b>{port_sharpe:.2f}</b> is weak — risk may not be adequately compensated."))

    inc = sigs[sigs == "Increase"].index.tolist()
    rev = sigs[sigs == "Review"].index.tolist()
    if inc:
        items.append(("success", f"<b>Opportunity:</b> {', '.join(inc)} outperforming with below-benchmark vol — consider increasing allocation."))
    if rev:
        items.append(("danger",  f"<b>Risk Alert:</b> {', '.join(rev)} lagging {bm} with elevated vol — consider reducing exposure."))

    if max_dd < -0.20:
        items.append(("warn",    f"Max drawdown of <b>{max_dd:.1%}</b> exceeds 20% — evaluate downside protection."))
    if beta > 1.2:
        items.append(("warn",    f"Beta of <b>{beta:.2f}</b> amplifies market moves — especially risky in Risk-Off regimes."))
    elif beta < 0.8:
        items.append(("info",    f"Beta of <b>{beta:.2f}</b> positions the portfolio defensively vs {bm}."))
    return items


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style="padding:10px 0 18px;border-bottom:1px solid #1e2a4a;margin-bottom:18px;">
            <div style="font-size:1.05rem;font-weight:800;color:#00D4FF;">📊 ETF Intelligence</div>
            <div style="font-size:.7rem;color:#475569;margin-top:3px;">Portfolio Analytics Engine</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="section-lbl">Portfolio Source</span>', unsafe_allow_html=True)
    uploaded = st.file_uploader("CSV with Ticker & Weight columns", type=["csv"], label_visibility="collapsed")

    default_csv = "test_portfolio.csv"
    src = uploaded if uploaded else (default_csv if os.path.exists(default_csv) else None)
    if src is None:
        st.error("No portfolio CSV found. Upload one above.")
        st.stop()

    try:
        pf_df = pd.read_csv(src)
        if "Ticker" not in pf_df.columns or "Weight" not in pf_df.columns:
            st.error("CSV must have 'Ticker' and 'Weight' columns.")
            st.stop()
        tickers     = pf_df["Ticker"].astype(str).str.upper().str.strip().tolist()
        raw_weights = dict(zip(tickers, pf_df["Weight"].astype(float)))
        st.caption(f"{len(tickers)} assets · {'upload' if uploaded else default_csv}")
    except Exception as e:
        st.error(f"CSV error: {e}")
        st.stop()

    # ── Reset sliders when the portfolio changes ──────────────────────────────
    pkey = hashlib.md5(str(sorted(raw_weights.items())).encode()).hexdigest()
    if st.session_state.get("_pkey") != pkey:
        st.session_state["_pkey"] = pkey
        for t, w in raw_weights.items():
            st.session_state[f"w_{t}"] = max(1, round(w * 100))

    st.markdown("---")
    st.markdown('<span class="section-lbl">Settings</span>', unsafe_allow_html=True)
    benchmark   = st.text_input("Benchmark", value=DEFAULT_BENCHMARK).strip().upper()
    show_assets = st.checkbox("Show individual assets on chart", value=True)

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start_date = st.date_input(
            "From",
            value=datetime.date.today() - datetime.timedelta(days=DEFAULT_START_DATE_OFFSET_YEARS * 365)
        )
    with col_d2:
        end_date = st.date_input("To", value=datetime.date.today())

    st.markdown("---")
    st.markdown('<span class="section-lbl">Portfolio Weights</span>', unsafe_allow_html=True)

    use_equal = st.toggle("Equal Weights", value=False)

    if use_equal:
        slider_raw = {t: 1 for t in tickers}
        st.caption(f"Each position: {1/len(tickers):.1%}")
    else:
        c_reset, _ = st.columns([1, 1])
        with c_reset:
            if st.button("↺ Reset", help="Reset to CSV weights"):
                for t, w in raw_weights.items():
                    st.session_state[f"w_{t}"] = max(1, round(w * 100))
                st.rerun()

        slider_raw = {}
        for t in tickers:
            # value intentionally omitted — session_state[key] is set by the hash-check above
            slider_raw[t] = st.slider(t, 0, 100, key=f"w_{t}")

    # Normalize
    _total = sum(slider_raw.values()) or 1
    eff_weights = {t: v / _total for t, v in slider_raw.items()}

    # Normalized weight bars
    if not use_equal:
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        for t, w in eff_weights.items():
            st.markdown(f"""
                <div style="display:flex;justify-content:space-between;font-size:11px;
                            color:#64748b;margin:2px 0;">
                    <span>{t}</span>
                    <span style="color:#00D4FF;font-weight:600">{w:.1%}</span>
                </div>
                <div style="background:#1e2a4a;border-radius:3px;height:3px;margin-bottom:4px;">
                    <div style="width:{int(w*100)}%;background:#00D4FF;height:3px;border-radius:3px;"></div>
                </div>
            """, unsafe_allow_html=True)


# ─── DATA LOADING ─────────────────────────────────────────────────────────────
tickers_to_dl = list(set(tickers + [benchmark]))
with st.spinner("Fetching market data..."):
    prices = load_data(tuple(sorted(tickers_to_dl)), start_date, end_date)

if prices.empty:
    st.error("No data returned. Check tickers and date range.")
    st.stop()

returns = calculate_returns(prices)

asset_tickers = [t for t in tickers if t in returns.columns and t != benchmark]
asset_returns = returns[asset_tickers]

if benchmark not in returns.columns:
    st.error(f"Benchmark '{benchmark}' data unavailable.")
    st.stop()

bench_df  = returns[[benchmark]]
bench_ser = returns[benchmark]

port_returns = calculate_portfolio_returns(asset_returns, eff_weights)


# ─── METRICS ──────────────────────────────────────────────────────────────────
port_ann_ret = annualized_return(port_returns)
port_ann_vol = annualized_volatility(port_returns)
port_sharpe  = sharpe_ratio(port_returns)
port_beta    = calculate_beta(port_returns, bench_ser)
port_max_dd  = max_drawdown(port_returns)

bench_ann_ret = annualized_return(bench_ser)
bench_ann_vol = annualized_volatility(bench_ser)
bench_sharpe  = sharpe_ratio(bench_ser)
bench_max_dd  = max_drawdown(bench_ser)

n20  = min(20,  len(asset_returns))
n252 = min(252, len(asset_returns))
rv20  = asset_returns.iloc[-n20:].std().mean()  * np.sqrt(TRADING_DAYS_PER_YEAR)
rv252 = asset_returns.iloc[-n252:].std().mean() * np.sqrt(TRADING_DAYS_PER_YEAR)
if n20 < 10 or n252 < 30:
    regime = "Neutral"
elif rv20 < rv252 * 0.9:
    regime = "Risk-On"
elif rv20 > rv252 * 1.1:
    regime = "Risk-Off"
else:
    regime = "Neutral"

signals = generate_signals(asset_returns, bench_df)
h_score = health_score(port_sharpe, bench_sharpe, port_ann_ret, bench_ann_ret,
                       signals, regime, port_max_dd, port_beta)
h_color = score_color(h_score)

port_cum  = cumulative_returns(port_returns)
bench_cum = cumulative_returns(bench_ser)

# Suggested rebalancing weights (for Holdings tab + sidebar button)
_sig_adj = {"Increase": 5, "Review": -5, "Hold": 0}
_sug_raw = {}
for t in asset_tickers:
    cur_slider = st.session_state.get(f"w_{t}", max(1, round(raw_weights.get(t, 1/len(tickers)) * 100)))
    _sug_raw[t] = max(1, cur_slider + _sig_adj.get(signals.get(t, "Hold"), 0))
_sug_total = sum(_sug_raw.values()) or 1
suggested_weights = {t: v / _sug_total for t, v in _sug_raw.items()}


# ─── SIDEBAR – Apply Suggested Weights button ─────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown('<span class="section-lbl">Rebalancing</span>', unsafe_allow_html=True)
    if st.button("✦ Apply Suggested Weights", help="Adjusts sliders based on signal recommendations"):
        for t in asset_tickers:
            st.session_state[f"w_{t}"] = _sug_raw[t]
        st.rerun()


# ─── HEADER ──────────────────────────────────────────────────────────────────
regime_cls = {"Risk-On": "risk-on", "Risk-Off": "risk-off"}.get(regime, "neutral")
c_title, c_regime = st.columns([5, 1])
with c_title:
    st.markdown(f"""
        <div style="padding:0 0 .8rem;">
            <p class="gradient-title">ETF Intelligence Dashboard</p>
            <p class="subtitle">Institutional-grade analytics &nbsp;·&nbsp;
               {datetime.date.today().strftime('%B %d, %Y')}</p>
        </div>
    """, unsafe_allow_html=True)
with c_regime:
    st.markdown(f"""
        <div style="text-align:right;padding-top:8px;">
            <div style="font-size:9px;color:#475569;text-transform:uppercase;
                        letter-spacing:.1em;margin-bottom:5px;">Market Regime</div>
            <span class="regime-badge {regime_cls}">{regime}</span>
        </div>
    """, unsafe_allow_html=True)


# ─── TOP-LEVEL TABS ───────────────────────────────────────────────────────────
tab_dash, tab_holdings = st.tabs(["📊  Dashboard", "📋  Holdings"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_dash:

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Portfolio Return", fmt_pct(port_ann_ret),
                  f"{port_ann_ret - bench_ann_ret:+.1%} vs {benchmark}")
    with k2:
        st.metric("Sharpe Ratio", fmt_f(port_sharpe),
                  f"{port_sharpe - bench_sharpe:+.2f} vs {benchmark}")
    with k3:
        st.metric("Ann. Volatility", fmt_pct(port_ann_vol),
                  f"{port_ann_vol - bench_ann_vol:+.1%} vs {benchmark}", delta_color="inverse")
    with k4:
        st.metric("Max Drawdown", fmt_pct(port_max_dd),
                  f"{port_max_dd - bench_max_dd:+.1%} vs {benchmark}", delta_color="inverse")
    with k5:
        st.metric("Portfolio Beta", fmt_f(port_beta))

    st.markdown("---")

    # Performance chart
    st.markdown('<span class="section-lbl">Cumulative Performance</span>', unsafe_allow_html=True)
    fig = go.Figure()
    if show_assets:
        for i, t in enumerate(asset_tickers):
            ac = cumulative_returns(asset_returns[t])
            fig.add_trace(go.Scatter(
                x=ac.index, y=ac.values * 100, name=t, opacity=0.45,
                line=dict(color=ASSET_COLORS[i % len(ASSET_COLORS)], width=1, dash="dot"),
                hovertemplate=f"{t}: %{{y:.1f}}%<extra></extra>"
            ))
    fig.add_trace(go.Scatter(
        x=bench_cum.index, y=bench_cum.values * 100, name=benchmark,
        line=dict(color="#64748b", width=1.5, dash="dash"),
        hovertemplate=f"{benchmark}: %{{y:.1f}}%<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=port_cum.index, y=port_cum.values * 100, name="Portfolio",
        line=dict(color="#00D4FF", width=2.5),
        fill="tozeroy", fillcolor="rgba(0,212,255,.07)",
        hovertemplate="Portfolio: %{y:.1f}%<extra></extra>"
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=8, b=0), height=340, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, font=dict(size=11)),
        yaxis=dict(ticksuffix="%", gridcolor="rgba(30,42,74,.6)", zeroline=False),
        xaxis=dict(gridcolor="rgba(30,42,74,.4)", zeroline=False),
    )
    st.plotly_chart(fig, width="stretch")

    # Signals + AI insights
    col_sig, col_ai = st.columns([6, 4], gap="large")

    b60_ret = (1 + bench_ser.tail(60)).prod() - 1
    b60_vol = bench_ser.tail(60).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    with col_sig:
        st.markdown('<span class="section-lbl">Signal Dashboard · 60-Day Trailing</span>', unsafe_allow_html=True)
        sig_rows = []
        for t in asset_tickers:
            a60     = asset_returns[t].tail(60)
            a60_ret = (1 + a60).prod() - 1
            a60_vol = a60.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            sig_rows.append({
                "Ticker":       t,
                "Signal":       signals.get(t, "N/A"),
                "Weight":       fmt_pct(eff_weights.get(t, 0)),
                "60D Ret":      fmt_pct(a60_ret),
                "vs Bench":     f"{a60_ret - b60_ret:+.1%}",
                "60D Vol":      fmt_pct(a60_vol),
                "Vol vs Bench": f"{a60_vol - b60_vol:+.1%}",
            })
        sig_df = pd.DataFrame(sig_rows)
        st.dataframe(sig_df.style.map(signal_style, subset=["Signal"]),
                     width="stretch", hide_index=True)
        st.download_button("Download Signals CSV", sig_df.to_csv(index=False).encode(),
                           "etf_signals.csv", "text/csv")

    with col_ai:
        st.markdown('<span class="section-lbl">AI Portfolio Intelligence</span>', unsafe_allow_html=True)
        st.markdown(f"""
            <div class="score-wrap">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-size:13px;color:#94a3b8;font-weight:600">Portfolio Health Score</span>
                    <span style="font-size:24px;font-weight:800;color:{h_color}">
                        {h_score}<span style="font-size:13px;font-weight:400;color:#475569">/100</span>
                    </span>
                </div>
                <div class="score-bar-bg">
                    <div class="score-bar"
                         style="width:{h_score}%;background:linear-gradient(90deg,{h_color}88,{h_color})">
                    </div>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:9px;color:#334155;">
                    <span>Distressed</span><span>Neutral</span><span>Optimal</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        for kind, text in ai_insights(port_ann_ret, bench_ann_ret, port_sharpe,
                                       port_max_dd, port_beta, signals, regime, benchmark):
            st.markdown(f'<div class="insight {kind}">{text}</div>', unsafe_allow_html=True)

    # Rolling metrics
    st.markdown("---")
    tab_vol, tab_ret, tab_corr = st.tabs(["Rolling Volatility (60D)", "Rolling Returns (60D)", "Correlation Matrix"])
    _cl = dict(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=8, b=0), height=290, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, font=dict(size=11)),
        yaxis=dict(gridcolor="rgba(30,42,74,.6)", zeroline=False),
        xaxis=dict(gridcolor="rgba(30,42,74,.4)", zeroline=False),
    )
    with tab_vol:
        fv = go.Figure()
        for i, t in enumerate(asset_tickers):
            rv = rolling_volatility(asset_returns[t]) * 100
            fv.add_trace(go.Scatter(x=rv.index, y=rv.values, name=t,
                line=dict(color=ASSET_COLORS[i % len(ASSET_COLORS)], width=1.5)))
        fv.add_trace(go.Scatter(x=(prv := rolling_volatility(port_returns) * 100).index,
            y=prv.values, name="Portfolio", line=dict(color="#00D4FF", width=2.5)))
        fv.add_trace(go.Scatter(x=(brv := rolling_volatility(bench_ser) * 100).index,
            y=brv.values, name=benchmark, line=dict(color="#64748b", width=1.5, dash="dash")))
        fv.update_layout(**_cl, yaxis_ticksuffix="%")
        st.plotly_chart(fv, width="stretch")

    with tab_ret:
        fr = go.Figure()
        for i, t in enumerate(asset_tickers):
            rr = rolling_return(asset_returns[t]) * 100
            fr.add_trace(go.Scatter(x=rr.index, y=rr.values, name=t,
                line=dict(color=ASSET_COLORS[i % len(ASSET_COLORS)], width=1.5)))
        fr.add_trace(go.Scatter(x=(prr := rolling_return(port_returns) * 100).index,
            y=prr.values, name="Portfolio", line=dict(color="#00D4FF", width=2.5)))
        fr.add_trace(go.Scatter(x=(brr := rolling_return(bench_ser) * 100).index,
            y=brr.values, name=benchmark, line=dict(color="#64748b", width=1.5, dash="dash")))
        fr.add_hline(y=0, line_color="#334155", line_dash="dot")
        fr.update_layout(**_cl, yaxis_ticksuffix="%")
        st.plotly_chart(fr, width="stretch")

    with tab_corr:
        if len(asset_tickers) >= 2:
            fc = px.imshow(asset_returns.corr(), text_auto=".2f", aspect="auto",
                           color_continuous_scale=[[0,"#ef4444"],[0.5,"#1e2a4a"],[1,"#00D4FF"]],
                           zmin=-1, zmax=1)
            fc.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=8,b=0), height=340,
                             coloraxis_colorbar=dict(title="Corr", tickformat=".1f"))
            fc.update_traces(textfont=dict(size=11))
            st.plotly_chart(fc, width="stretch")
        else:
            st.info("Add more assets to view the correlation matrix.")

    # Allocation + stats
    st.markdown("---")
    col_pie, col_stats = st.columns([3, 7], gap="large")
    with col_pie:
        st.markdown('<span class="section-lbl">Allocation</span>', unsafe_allow_html=True)
        fp = go.Figure(go.Pie(
            labels=asset_tickers,
            values=[eff_weights.get(t, 0) for t in asset_tickers],
            hole=0.52, textinfo="label+percent", textposition="outside",
            marker=dict(colors=ASSET_COLORS[:len(asset_tickers)],
                        line=dict(color="rgba(0,0,0,0)", width=1))
        ))
        fp.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0), height=300, showlegend=False,
            annotations=[dict(text="Portfolio", x=.5, y=.5, font_size=12,
                              showarrow=False, font_color="#94a3b8")]
        )
        st.plotly_chart(fp, width="stretch")

    with col_stats:
        st.markdown('<span class="section-lbl">Asset Statistics</span>', unsafe_allow_html=True)
        stats_rows = [{
            "Ticker": t,
            "Signal": signals.get(t, "N/A"),
            "Ann. Return": fmt_pct(annualized_return(asset_returns[t])),
            "Volatility":  fmt_pct(annualized_volatility(asset_returns[t])),
            "Sharpe":      fmt_f(sharpe_ratio(asset_returns[t])),
            "Beta":        fmt_f(calculate_beta(asset_returns[t], bench_ser)),
            "Max DD":      fmt_pct(max_drawdown(asset_returns[t])),
        } for t in asset_tickers]
        st.dataframe(pd.DataFrame(stats_rows).style.map(signal_style, subset=["Signal"]),
                     width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#334155;font-size:11px;padding:6px 0 2px;">
        ETF Intelligence &nbsp;·&nbsp; Data via Yahoo Finance &nbsp;·&nbsp; Not financial advice
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — HOLDINGS
# ══════════════════════════════════════════════════════════════════════════════
with tab_holdings:

    # ── Summary strip ─────────────────────────────────────────────────────────
    h_s1, h_s2, h_s3, h_s4 = st.columns(4)
    with h_s1:
        st.metric("Positions", len(asset_tickers))
    with h_s2:
        w_avg_ret = sum(eff_weights.get(t, 0) * annualized_return(asset_returns[t])
                        for t in asset_tickers)
        st.metric("Wtd. Avg Return", fmt_pct(w_avg_ret))
    with h_s3:
        w_avg_vol = sum(eff_weights.get(t, 0) * annualized_volatility(asset_returns[t])
                        for t in asset_tickers)
        st.metric("Wtd. Avg Vol", fmt_pct(w_avg_vol))
    with h_s4:
        port_value = st.number_input(
            "Total Portfolio Value ($)",
            min_value=0, value=1_000_000, step=100_000, format="%d",
            label_visibility="visible"
        )

    st.markdown("---")

    # ── Holdings table ────────────────────────────────────────────────────────
    st.markdown('<span class="section-lbl">Current Holdings</span>', unsafe_allow_html=True)

    ytd_start = pd.Timestamp(datetime.date.today().year, 1, 1)

    h_rows = []
    for t in asset_tickers:
        ret_1d  = asset_returns[t].iloc[-1] if len(asset_returns[t]) > 0 else np.nan
        ret_60d = (1 + asset_returns[t].tail(60)).prod() - 1
        ret_1y  = (1 + asset_returns[t].tail(252)).prod() - 1
        ytd_ser = asset_returns[t][asset_returns[t].index >= ytd_start]
        ret_ytd = (1 + ytd_ser).prod() - 1 if len(ytd_ser) > 0 else np.nan

        price   = prices[t].iloc[-1] if t in prices.columns else np.nan
        wt      = eff_weights.get(t, 0)
        notional = wt * port_value if port_value > 0 else np.nan

        h_rows.append({
            "Ticker":    t,
            "Signal":    signals.get(t, "N/A"),
            "Weight":    wt,
            "Notional":  notional,
            "Price":     price,
            "1D %":      ret_1d,
            "YTD %":     ret_ytd,
            "60D %":     ret_60d,
            "1Y %":      ret_1y,
            "Ann. Vol":  annualized_volatility(asset_returns[t]),
            "Sharpe":    sharpe_ratio(asset_returns[t]),
            "Beta":      calculate_beta(asset_returns[t], bench_ser),
            "Max DD":    max_drawdown(asset_returns[t]),
        })

    h_df = pd.DataFrame(h_rows)

    fmt_map = {
        "Weight":   "{:.1%}",
        "Price":    "${:.2f}",
        "1D %":     "{:+.2%}",
        "YTD %":    "{:+.1%}",
        "60D %":    "{:+.1%}",
        "1Y %":     "{:+.1%}",
        "Ann. Vol": "{:.1%}",
        "Sharpe":   "{:.2f}",
        "Beta":     "{:.2f}",
        "Max DD":   "{:.1%}",
    }
    if port_value > 0:
        fmt_map["Notional"] = "${:,.0f}"
    else:
        h_df = h_df.drop(columns=["Notional"])

    styled_h = (
        h_df.style
        .map(signal_style, subset=["Signal"])
        .map(ret_style, subset=["1D %", "YTD %", "60D %", "1Y %"])
        .format(fmt_map, na_rep="—")
    )
    st.dataframe(styled_h, width="stretch", hide_index=True)

    # ── Weight distribution chart ─────────────────────────────────────────────
    st.markdown("---")
    col_wt_chart, col_rebal = st.columns([5, 5], gap="large")

    with col_wt_chart:
        st.markdown('<span class="section-lbl">Weight Distribution</span>', unsafe_allow_html=True)

        sig_colors = {
            "Increase": "#10b981",
            "Review":   "#ef4444",
            "Hold":     "#f59e0b",
        }
        bar_colors = [sig_colors.get(signals.get(t, "Hold"), "#64748b") for t in asset_tickers]

        fw = go.Figure(go.Bar(
            x=[eff_weights.get(t, 0) * 100 for t in asset_tickers],
            y=asset_tickers,
            orientation="h",
            marker=dict(
                color=bar_colors,
                opacity=0.85,
                line=dict(width=0),
            ),
            text=[f"{eff_weights.get(t, 0):.1%}" for t in asset_tickers],
            textposition="outside",
            textfont=dict(size=12, color="#cbd5e1"),
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        ))
        fw.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=60, t=8, b=0), height=max(200, len(asset_tickers) * 42),
            xaxis=dict(ticksuffix="%", gridcolor="rgba(30,42,74,.5)", range=[0, 45]),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            showlegend=False,
        )
        st.plotly_chart(fw, width="stretch")

    # ── Rebalancing suggestion ────────────────────────────────────────────────
    with col_rebal:
        st.markdown('<span class="section-lbl">AI Rebalancing Suggestion</span>', unsafe_allow_html=True)

        rebal_rows = []
        for t in asset_tickers:
            cur = eff_weights.get(t, 0)
            sug = suggested_weights.get(t, cur)
            delta = sug - cur
            rebal_rows.append({
                "Ticker":   t,
                "Signal":   signals.get(t, "N/A"),
                "Current":  cur,
                "Suggested": sug,
                "Change":   delta,
            })
        rebal_df = pd.DataFrame(rebal_rows)

        def change_style(v):
            if not isinstance(v, float) or np.isnan(v): return ""
            return "color:#10b981;font-weight:600" if v > 0.001 else (
                   "color:#ef4444;font-weight:600" if v < -0.001 else "color:#64748b")

        styled_rebal = (
            rebal_df.style
            .map(signal_style, subset=["Signal"])
            .map(change_style, subset=["Change"])
            .format({
                "Current":   "{:.1%}",
                "Suggested": "{:.1%}",
                "Change":    "{:+.1%}",
            }, na_rep="—")
        )
        st.dataframe(styled_rebal, width="stretch", hide_index=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("""
            <div style="background:rgba(0,212,255,.05);border:1px solid rgba(0,212,255,.15);
                        border-radius:8px;padding:10px 14px;font-size:12px;color:#94a3b8;
                        line-height:1.6;">
                <b style="color:#00D4FF">How this works:</b> Signal-based nudge of ±5% per
                position (Increase +5, Review −5, Hold 0), then renormalized to 100%.
                Click <b>✦ Apply Suggested Weights</b> in the sidebar to use these weights.
            </div>
        """, unsafe_allow_html=True)
