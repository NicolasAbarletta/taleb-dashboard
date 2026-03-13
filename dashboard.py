# -*- coding: utf-8 -*-
"""
dashboard.py -- Streamlit Live Dashboard (Cloud-Ready)
========================================================
Works both locally (with run.py) and on Streamlit Community Cloud.
When no data exists, triggers an inline fetch with a spinner.
Cached for 30 minutes via st.cache_data.
"""

import json
import os
import sqlite3
import time
import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s")

DB_PATH = "market_data.db"
REFRESH_SEC = 300

# ── Secrets: support both .env and st.secrets (for Streamlit Cloud) ──────────

def _get_secret(key: str) -> str:
    """Try st.secrets first (cloud), then os.getenv (.env / local)."""
    try:
        return st.secrets.get(key, "") or os.getenv(key, "")
    except Exception:
        return os.getenv(key, "")


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Taleb Trade Advisor", page_icon="@", layout="wide", initial_sidebar_state="expanded")

# ── CSS ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Force all text to be readable on dark background */
[data-testid="stAppViewContainer"] { background-color: #0a0a0a; color: #e8e8e8; }
[data-testid="stSidebar"] { background-color: #0f0f0f; color: #e0e0e0; }
[data-testid="stAppViewContainer"] p, [data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] label, [data-testid="stAppViewContainer"] div,
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] div { color: #e0e0e0; }
[data-testid="stAppViewContainer"] h1, [data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3, [data-testid="stAppViewContainer"] h4 { color: #f0f0f0; }
[data-testid="stMetricValue"] { color: #ffffff !important; }
[data-testid="stMetricLabel"] { color: #bbbbbb !important; }
[data-testid="stMarkdownContainer"] p { color: #dddddd; }
[data-testid="stExpander"] summary { color: #cccccc !important; }
[data-testid="stExpander"] summary span { color: #cccccc !important; }
.stSlider label { color: #cccccc !important; }
.stCheckbox label span { color: #cccccc !important; }
/* Streamlit dataframe header and cells */
[data-testid="stDataFrame"] { color: #e0e0e0; }
/* Cards */
.card-high { background: linear-gradient(135deg,#1a1400,#2a2000); border:1px solid #b8860b; border-radius:10px; padding:18px 22px; margin-bottom:16px; }
.card-watch { background: linear-gradient(135deg,#0a0a1a,#0d0d2a); border:1px solid #1e3a8a; border-radius:10px; padding:18px 22px; margin-bottom:16px; }
.badge-high { background:#b8860b; color:#000; font-weight:700; padding:3px 12px; border-radius:12px; font-size:12px; }
.badge-watch { background:#1e3a8a; color:#fff; font-weight:700; padding:3px 12px; border-radius:12px; font-size:12px; }
.trade-ticket { background:#151515; border:1px solid #444; border-radius:8px; padding:14px; margin:8px 0; font-family:monospace; color:#e0e0e0; }
.action-buy { color:#00ff88; font-size:20px; font-weight:800; letter-spacing:1px; }
.action-sell { color:#ff4466; font-size:20px; font-weight:800; letter-spacing:1px; }
.thesis-text { font-style:italic; color:#dddddd; font-size:14px; line-height:1.6; margin:8px 0; }
.risk-box { background:#1a0a0a; border-left:3px solid #ff4444; padding:10px 14px; margin:8px 0; font-size:13px; color:#ffaaaa; }
.stress-red { color:#ff5555; font-weight:700; } .stress-yellow { color:#ffcc44; font-weight:700; } .stress-green { color:#22dd77; font-weight:700; }
.section-hdr { font-size:12px; color:#999999; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:6px; }
.log-row { font-family:monospace; font-size:11px; color:#aaaaaa; border-bottom:1px solid #222; padding:3px 0; }
</style>
""", unsafe_allow_html=True)


# ── DB + inline data fetch ───────────────────────────────────────────────────

@st.cache_resource
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_tables(conn):
    for tbl, cols in [
        ("agent_runs", "id INTEGER PRIMARY KEY, run_id TEXT, started_at TEXT, finished_at TEXT, assets_scanned INTEGER DEFAULT 0, opportunities INTEGER DEFAULT 0, errors TEXT, status TEXT DEFAULT 'running'"),
        ("taleb_scores", "id INTEGER PRIMARY KEY, run_id TEXT, scored_at TEXT, ticker TEXT, total_score INTEGER, convexity_score INTEGER, antifragility_score INTEGER, fragility_avoidance INTEGER, tail_risk_score INTEGER, conviction_tier TEXT, catalyst TEXT, rationale TEXT, price REAL, iv_percentile REAL, score_details TEXT"),
        ("macro_snapshots", "id INTEGER PRIMARY KEY, run_id TEXT, timestamp TEXT, series_key TEXT, series_id TEXT, value REAL, prev_value REAL, change REAL, z_score_2y REAL, stress_flag INTEGER DEFAULT 0"),
        ("trade_recommendations", "id INTEGER PRIMARY KEY, run_id TEXT, created_at TEXT, ticker TEXT, total_score INTEGER, conviction_tier TEXT, trade_action TEXT, instrument TEXT, strike REAL, expiry TEXT, premium REAL, delta REAL, gamma REAL, vega REAL, underlying_price REAL, thesis TEXT, triggers TEXT, pnl_scenarios TEXT, risk_check TEXT, position_pct REAL"),
        ("equity_snapshots", "id INTEGER PRIMARY KEY, run_id TEXT, timestamp TEXT, ticker TEXT, price REAL, price_change_1d REAL, volume REAL, avg_volume_20d REAL, volume_ratio REAL, week52_high REAL, week52_low REAL, week52_position REAL, iv_30d REAL, iv_60d REAL, iv_90d REAL, iv_1y_percentile REAL, short_interest REAL, market_cap REAL, beta REAL"),
        ("news_signals", "id INTEGER PRIMARY KEY, run_id TEXT, timestamp TEXT, keyword TEXT, article_count INTEGER, novel_count INTEGER, consensus_count INTEGER, novelty_score REAL, consensus_penalty REAL"),
    ]:
        conn.execute(f"CREATE TABLE IF NOT EXISTS {tbl} ({cols})")
    conn.commit()


@st.cache_data(ttl=1800, show_spinner=False)
def _run_full_cycle() -> str:
    """
    Run data pull + scoring + trade building inline.
    Cached for 30 minutes so it doesn't re-fetch on every page load.
    Returns the run_id.
    """
    # Inject secrets into env so backend modules pick them up
    for key in ["ANTHROPIC_API_KEY", "FRED_API_KEY", "NEWS_API_KEY"]:
        val = _get_secret(key)
        if val:
            os.environ[key] = val

    from database import init_db
    from agent import run_agent
    from scorer import score_all
    from trade_builder import build_trades

    conn = init_db()
    run_id = run_agent(conn)
    score_all(conn, run_id)
    build_trades(conn, run_id)
    return run_id


def _latest_run(conn):
    r = conn.execute("SELECT run_id, finished_at, assets_scanned, opportunities FROM agent_runs WHERE status='completed' ORDER BY finished_at DESC LIMIT 1").fetchone()
    return dict(r) if r else {}


def _load_trades(conn, run_id):
    return conn.execute("SELECT * FROM trade_recommendations WHERE run_id=? ORDER BY total_score DESC", (run_id,)).fetchall()


def _load_macro(conn, run_id):
    return conn.execute("SELECT * FROM macro_snapshots WHERE run_id=?", (run_id,)).fetchall()


def _load_history(conn, key, limit=180):
    return pd.read_sql_query("""
        SELECT m.timestamp, m.value FROM macro_snapshots m
        JOIN agent_runs r ON m.run_id=r.run_id
        WHERE m.series_key=? AND r.status='completed'
        ORDER BY m.timestamp DESC LIMIT ?
    """, conn, params=(key, limit))


def _load_log(conn):
    return conn.execute("SELECT * FROM agent_runs ORDER BY started_at DESC LIMIT 10").fetchall()


# ── Render: Payoff chart ─────────────────────────────────────────────────────

def _render_payoff_chart(ticker, strike, expiry, premium, opt_type, price, position_size):
    from trade_builder import OptionLeg, build_payoff_curve
    opt = None
    if strike and premium:
        opt = OptionLeg(option_type=opt_type or "call", strike=strike, expiry=expiry or "",
                        premium=premium, delta=0, gamma=0, vega=0, dte=30)
    curve = build_payoff_curve(opt, price, position_size)

    fig = go.Figure()
    prices_arr, pnl_arr = curve["prices"], curve["pnl"]
    pos_pnl = [max(0, p) for p in pnl_arr]
    neg_pnl = [min(0, p) for p in pnl_arr]

    fig.add_trace(go.Scatter(x=prices_arr, y=pos_pnl, fill='tozeroy', fillcolor='rgba(0,200,100,0.15)',
                             line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=prices_arr, y=neg_pnl, fill='tozeroy', fillcolor='rgba(255,60,60,0.15)',
                             line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=prices_arr, y=pnl_arr, mode='lines', name='P&L',
                             line=dict(color='#5599ff', width=2.5)))
    fig.add_vline(x=curve["current_price"], line_dash="dot", line_color="#ffffff", line_width=1,
                  annotation_text=f"Current ${curve['current_price']:.2f}", annotation_font_color="#aaa")
    fig.add_vline(x=curve["breakeven"], line_dash="dash", line_color="#ffbb33", line_width=1,
                  annotation_text=f"BE ${curve['breakeven']:.2f}", annotation_font_color="#ffbb33",
                  annotation_position="top left")
    fig.add_hline(y=0, line_color="#444", line_width=1)
    fig.update_layout(title=f"{ticker} Payoff at Expiration",
                      xaxis_title="Underlying Price ($)", yaxis_title="Profit / Loss ($)",
                      plot_bgcolor="#0a0a0a", paper_bgcolor="#0a0a0a", font_color="#ccc",
                      height=320, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis=dict(showgrid=True, gridcolor="#1a1a1a"),
                      yaxis=dict(showgrid=True, gridcolor="#1a1a1a"))
    st.plotly_chart(fig, use_container_width=True, key=f"payoff_{ticker}")


# ── Render: Trade card ───────────────────────────────────────────────────────

def _render_trade_card(trade, position_size):
    tier = trade["conviction_tier"]
    card_cls = "card-high" if tier == "HIGH" else "card-watch"
    badge_cls = "badge-high" if tier == "HIGH" else "badge-watch"
    tier_label = "HIGH CONVICTION" if tier == "HIGH" else "WATCH LIST"
    action_cls = "action-sell" if "PUT" in (trade["trade_action"] or "") or "SELL" in (trade["trade_action"] or "") else "action-buy"
    price = trade["underlying_price"] or 0

    st.markdown(f"""
    <div class="{card_cls}">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <span style="font-size:24px;font-weight:800;letter-spacing:1px">{trade['ticker']}</span>
        <span class="{badge_cls}">{tier_label} {trade['total_score']}/100</span>
      </div>
    </div>""", unsafe_allow_html=True)

    # Trade ticket
    opt_info = ""
    if trade["strike"]:
        cost_1 = (trade["premium"] or 0) * 100
        opt_info = f"""
        Strike: **${trade['strike']:.2f}** | Expiry: **{trade['expiry']}** | Premium: **${trade['premium']:.2f}**/contract
        Cost: **${cost_1:.0f}** (1 contract) | **${cost_1 * 10:.0f}** (10 contracts)
        Delta: {trade['delta']:.3f} | Gamma: {trade['gamma']:.4f} | Vega: {trade['vega']:.3f}"""

    pos_str = ""
    if trade["strike"] and trade["premium"] and trade["premium"] > 0:
        n = max(1, int(position_size / (trade["premium"] * 100)))
        pos_str = f"**{n} contract{'s' if n > 1 else ''}** (${n * trade['premium'] * 100:.0f})"
    elif price > 0:
        n = max(1, int(position_size / price))
        pos_str = f"**{n} shares** (${n * price:.0f})"

    st.markdown(f"""
    <div class="trade-ticket">
      <span class="{action_cls}">{trade['trade_action']}</span>
      <span style="color:#ccc;font-size:14px"> {trade['instrument']}</span><br>
      <span style="color:#bbb;font-size:13px">Underlying: ${price:.2f} | Position: {pos_str}</span>
    </div>""", unsafe_allow_html=True)

    if opt_info:
        st.markdown(opt_info)
    if trade["thesis"]:
        st.markdown(f'<div class="thesis-text">{trade["thesis"]}</div>', unsafe_allow_html=True)

    # Expandable sections
    with st.expander("Trigger Checklist", expanded=False):
        try:
            triggers = json.loads(trade["triggers"]) if trade["triggers"] else []
        except Exception:
            triggers = []
        for trig in triggers:
            st.checkbox(trig, value=False, key=f"trg_{trade['ticker']}_{hash(trig)}", disabled=True)

    with st.expander("P&L Scenarios", expanded=False):
        try:
            scenarios = json.loads(trade["pnl_scenarios"]) if trade["pnl_scenarios"] else []
        except Exception:
            scenarios = []
        if scenarios:
            if trade["strike"] and trade["premium"] and trade["premium"] > 0:
                contracts = max(1, int(position_size / (trade["premium"] * 100)))
                cost = contracts * trade["premium"] * 100
                strike, prem = trade["strike"], trade["premium"]
                is_call = "CALL" in (trade["trade_action"] or "")
                if is_call:
                    be = strike + prem; bt = strike * 1.10; ht = strike * 1.25
                    bp = max(0, bt - strike) * contracts * 100 - cost
                    hp = max(0, ht - strike) * contracts * 100 - cost
                else:
                    be = strike - prem; bt = strike * 0.90; ht = strike * 0.75
                    bp = max(0, strike - bt) * contracts * 100 - cost
                    hp = max(0, strike - ht) * contracts * 100 - cost
                scenarios = [
                    {"Scenario": "Total Loss", "What Happens": "Expires worthless", "Your P&L": f"${-cost:+,.0f}", "Return %": "-100%"},
                    {"Scenario": "Partial Loss", "What Happens": "IV crush / time decay", "Your P&L": f"${-cost*0.5:+,.0f}", "Return %": "-50%"},
                    {"Scenario": "Break Even", "What Happens": f"At ${be:.2f}", "Your P&L": "$0", "Return %": "0%"},
                    {"Scenario": "Base Case", "What Happens": f"At ${bt:.2f}", "Your P&L": f"${bp:+,.0f}", "Return %": f"{bp/cost*100:+.0f}%" if cost > 0 else "0%"},
                    {"Scenario": "Home Run", "What Happens": f"At ${ht:.2f}", "Your P&L": f"${hp:+,.0f}", "Return %": f"{hp/cost*100:+.0f}%" if cost > 0 else "0%"},
                ]
            else:
                for s in scenarios:
                    s["Your P&L"] = f"${s.get('pnl', 0):+,.0f}" if isinstance(s.get("pnl"), (int, float)) else s.get("pnl", "")
                    s["Return %"] = f"{s.get('return_pct', 0):+.0f}%" if isinstance(s.get("return_pct"), (int, float)) else s.get("return_pct", "")
                    s["Scenario"] = s.pop("scenario", "")
                    s["What Happens"] = s.pop("description", "")
                    s.pop("pnl", None)
                    s.pop("return_pct", None)
            st.dataframe(pd.DataFrame(scenarios), hide_index=True, use_container_width=True)

    with st.expander("Payoff Chart", expanded=False):
        opt_type = "call" if "CALL" in (trade["trade_action"] or "") else "put"
        _render_payoff_chart(trade["ticker"], trade["strike"], trade["expiry"],
                            trade["premium"], opt_type, price, position_size)

    with st.expander("Risk Reality Check", expanded=False):
        if trade["risk_check"]:
            st.markdown(f'<div class="risk-box">{trade["risk_check"]}</div>', unsafe_allow_html=True)

    st.markdown("---")


# ── Macro charts ──────────────────────────────────────────────────────────────

def _chart(df, title, color="#5599ff", fill=True, hline=None, zones=None):
    if df.empty:
        st.info(f"No {title} data yet.")
        return
    df = df.sort_values("timestamp")
    fig = go.Figure()
    if zones:
        for y0, y1, c in zones:
            fig.add_hrect(y0=y0, y1=y1, fillcolor=c, line_width=0)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["value"], mode="lines",
                             line=dict(color=color, width=2), name=title,
                             fill="tozeroy" if fill else None,
                             fillcolor=f"rgba{tuple(list(bytes.fromhex(color[1:])) + [0.08])}"))
    if hline is not None:
        fig.add_hline(y=hline, line_dash="dash", line_color="#ff4444", line_width=1)
    fig.update_layout(title=title, plot_bgcolor="#0a0a0a", paper_bgcolor="#0a0a0a", font_color="#ccc",
                      height=200, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#1a1a1a"))
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{title}")


def _macro_narrative(macro_rows) -> str:
    signals = []
    for r in macro_rows:
        k, v, z, f = r["series_key"], r["value"], r["z_score_2y"], r["stress_flag"]
        if k == "yield_curve_10y2y" and v is not None:
            signals.append("Yield curve inverted" if v < 0 else ("Yield curve flattening" if v < 0.3 else None))
        if k == "baa_aaa_spread" and f:
            signals.append("Credit spreads widening")
        if k == "vix" and v is not None:
            if v < 15: signals.append("Volatility suppressed")
            elif v > 30: signals.append("Fear elevated")
        if k == "unemployment_claims" and f:
            signals.append("Jobless claims spiking")
    signals = [s for s in signals if s]
    if not signals:
        return "Markets calm. No macro stress signals. Options are priced for complacency."
    return ". ".join(signals[:3]) + ". " + ("Classic pre-stress setup." if len(signals) >= 2 else "Monitor closely.")


SERIES_LABELS = {
    "fed_funds_rate": "Fed Funds Rate", "baa_aaa_spread": "Credit Spread (BAA-10Y)",
    "ted_spread": "TED Spread", "yield_curve_10y2y": "Yield Curve (10Y-2Y)",
    "unemployment_claims": "Jobless Claims", "vix": "VIX",
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    conn = get_conn()
    _ensure_tables(conn)

    # Check if we have data -- if not, fetch inline
    meta = _latest_run(conn)
    if not meta:
        with st.spinner("Scanning markets... first load takes 2-3 minutes"):
            _run_full_cycle()
        meta = _latest_run(conn)

    run_id = meta.get("run_id")

    macro_rows = _load_macro(conn, run_id) if run_id else []
    stress_count = sum(1 for r in macro_rows if r["stress_flag"])
    stress_level = "RED" if stress_count >= 4 else ("YELLOW" if stress_count >= 2 else "GREEN")
    stress_cls = {"RED": "stress-red", "YELLOW": "stress-yellow", "GREEN": "stress-green"}[stress_level]
    stress_icon = {"RED": "!!!", "YELLOW": "!!", "GREEN": "OK"}[stress_level]
    narrative = _macro_narrative(macro_rows) if macro_rows else "No data yet."

    # ── HEADER ──
    h1, h2, h3 = st.columns([3, 2, 2])
    with h1:
        st.markdown("<h1 style='font-size:26px;letter-spacing:2px;margin-bottom:0'>TALEB TRADE ADVISOR</h1>"
                    "<p style='color:#999;font-size:11px;margin-top:0'>Convexity | Antifragility | Tail Risk</p>",
                    unsafe_allow_html=True)
    with h2:
        st.markdown(f'<div class="section-hdr">Macro Stress</div><span class="{stress_cls}" style="font-size:20px">'
                    f'{stress_icon} {stress_level}</span>', unsafe_allow_html=True)
    with h3:
        col_ts, col_btn = st.columns([3, 1])
        with col_ts:
            st.markdown(f'<div class="section-hdr">Last Updated</div>'
                        f'<span style="font-size:13px;color:#ddd">{meta.get("finished_at", "Never")[:19]}</span>',
                        unsafe_allow_html=True)
        with col_btn:
            if st.button("Refresh", type="secondary"):
                _run_full_cycle.clear()
                st.rerun()

    st.markdown(f'<div style="background:#111;padding:8px 14px;border-radius:6px;font-size:13px;color:#cccccc;margin-bottom:12px">{narrative}</div>',
                unsafe_allow_html=True)

    # ── POSITION SIZE SLIDER ──
    position_size = st.slider("Risk per trade ($)", 100, 5000, 500, 50,
                              help="All P&L tables and payoff charts update dynamically")
    st.markdown("---")

    # ── LAYOUT ──
    main_col, side_col = st.columns([2, 1])

    with main_col:
        if not run_id:
            st.warning("No data available. Click Refresh above.")
        else:
            trades = _load_trades(conn, run_id)
            if not trades:
                st.info("No opportunities scored 50+ in the latest scan. Will rescan in 30 minutes.")
            else:
                high_t = [t for t in trades if t["conviction_tier"] == "HIGH"]
                watch_t = [t for t in trades if t["conviction_tier"] == "WATCH"]
                if high_t:
                    st.markdown('<div style="color:#b8860b;font-weight:700;font-size:15px;margin:8px 0">HIGH CONVICTION TRADES</div>',
                                unsafe_allow_html=True)
                    for t in high_t:
                        _render_trade_card(dict(t), position_size)
                if watch_t:
                    st.markdown('<div style="color:#5599ff;font-weight:700;font-size:15px;margin:8px 0">WATCH LIST</div>',
                                unsafe_allow_html=True)
                    for t in watch_t:
                        _render_trade_card(dict(t), position_size)

    with side_col:
        st.markdown('<div class="section-hdr">Macro Indicators</div>', unsafe_allow_html=True)
        if macro_rows:
            for r in macro_rows:
                key, val, z, flag = r["series_key"], r["value"], r["z_score_2y"], r["stress_flag"]
                light = "!!!" if flag else ("!" if z and abs(z) > 1.5 else "OK")
                label = SERIES_LABELS.get(key, key)
                val_str = f"{val/1000:.0f}k" if key == "unemployment_claims" and val else (f"{val:.2f}%" if val is not None else "N/A")
                z_str = f" (z={z:.1f})" if z is not None else ""
                color = "#ff4444" if flag else ("#ffbb33" if z and abs(z) > 1.5 else "#00cc66")
                st.markdown(f'<div style="padding:5px 0;border-bottom:1px solid #1a1a1a">'
                            f'<span style="color:{color};font-weight:700">{light}</span> <b>{label}</b><br>'
                            f'<span style="color:#ccc">{val_str}<span style="color:#999;font-size:11px">{z_str}</span></span></div>',
                            unsafe_allow_html=True)
        else:
            st.info("No FRED data yet.")

        st.markdown("<br>", unsafe_allow_html=True)
        _chart(_load_history(conn, "yield_curve_10y2y"), "Yield Curve (10Y-2Y)", "#5599ff", hline=0)
        _chart(_load_history(conn, "vix"), "VIX", "#ffbb33", fill=False,
               zones=[(0, 15, "rgba(0,200,100,0.06)"), (15, 25, "rgba(255,200,0,0.04)"), (25, 100, "rgba(255,50,50,0.04)")])

    # ── PORTFOLIO SIMULATOR ──
    if run_id:
        trades = _load_trades(conn, run_id)
        if trades and len(trades) > 1:
            st.markdown("---")
            st.markdown('<div class="section-hdr">Portfolio Simulator</div>', unsafe_allow_html=True)
            total_cost, total_base = 0, 0
            tickers = []
            for t in trades:
                t = dict(t)
                if t["strike"] and t["premium"] and t["premium"] > 0:
                    n = max(1, int(position_size / (t["premium"] * 100)))
                    c = n * t["premium"] * 100
                else:
                    c = position_size
                total_cost += c
                try:
                    scens = json.loads(t["pnl_scenarios"]) if t["pnl_scenarios"] else []
                    base = next((s for s in scens if s["scenario"] == "Base Case"), None)
                    if base:
                        total_base += base["pnl"]
                except Exception:
                    pass
                tickers.append(t["ticker"])

            c1, c2, c3 = st.columns(3)
            c1.metric("Combined Cost", f"${total_cost:,.0f}")
            c2.metric("Max Loss", f"-${total_cost:,.0f}")
            c3.metric("Base Case P&L", f"${total_base:+,.0f}")
            if len(set(tickers)) < len(tickers):
                st.warning("Correlated trades detected.")

    # ── ACTIVITY LOG ──
    st.markdown("---")
    st.markdown('<div class="section-hdr">Agent Activity Log</div>', unsafe_allow_html=True)
    log_rows = _load_log(conn)
    if log_rows:
        for r in log_rows:
            r = dict(r)
            icon = {"completed": "OK", "running": "...", "error": "ERR"}.get(r["status"], "?")
            errs = ""
            if r["errors"] and r["errors"] != "null":
                try:
                    e = json.loads(r["errors"])
                    if e: errs = f" | {', '.join(str(x)[:30] for x in e[:2])}"
                except Exception:
                    pass
            st.markdown(f'<div class="log-row">[{icon}] <b>{r["run_id"]}</b> '
                        f'{str(r["started_at"])[:19]} assets={r["assets_scanned"]} opps={r["opportunities"]}{errs}</div>',
                        unsafe_allow_html=True)
    else:
        st.info("No runs yet.")

    # ── Auto-refresh ──
    if "t0" not in st.session_state:
        st.session_state.t0 = time.time()
    elapsed = int(time.time() - st.session_state.t0)
    remaining = max(0, REFRESH_SEC - elapsed)
    st.markdown(f'<div style="text-align:center;color:#777;font-size:10px;margin-top:20px">Refresh in {remaining}s</div>',
                unsafe_allow_html=True)
    if remaining <= 0:
        st.session_state.t0 = time.time()
        _run_full_cycle.clear()
        st.rerun()
    else:
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    main()
