# -*- coding: utf-8 -*-
"""
dashboard.py -- Streamlit Live Dashboard (Premium v3)
======================================================
Works locally (with run.py) and on Streamlit Community Cloud.
"""

import json
import os
import sqlite3
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s")

DB_PATH = "market_data.db"


def _get_secret(key: str) -> str:
    try:
        return st.secrets.get(key, "") or os.getenv(key, "")
    except Exception:
        return os.getenv(key, "")


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Taleb Trade Advisor",
    page_icon="T",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Premium CSS ───────────────────────────────────────────────────────────────

st.markdown("""<style>
/* === GLOBAL === */
html, body { background:#07090f; }
[data-testid="stAppViewContainer"] { background:#07090f; }
[data-testid="stAppViewContainer"] > .main { background:#07090f; }
[data-testid="stSidebar"] { background:#0a0e1a; border-right:1px solid #1a2540; }
section[data-testid="stSidebar"] > div { background:#0a0e1a; }

/* === TEXT === */
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] div,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label { color:#cbd5e1; }
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3 { color:#f1f5f9; }
[data-testid="stMetricValue"] { color:#f1f5f9 !important; font-size:26px !important; font-weight:800 !important; }
[data-testid="stMetricLabel"] { color:#64748b !important; font-size:11px !important; text-transform:uppercase; letter-spacing:1px; }
[data-testid="stMarkdownContainer"] p { color:#94a3b8; }
[data-testid="stExpander"] { border:1px solid #1a2540 !important; border-radius:10px !important; overflow:hidden; }
[data-testid="stExpander"] > div:first-child { background:#090d18 !important; }
[data-testid="stExpander"] summary { color:#64748b !important; font-size:11px !important; text-transform:uppercase; letter-spacing:1.5px; }
[data-testid="stExpander"] summary span { color:#64748b !important; }
.stSlider label { color:#64748b !important; font-size:11px !important; text-transform:uppercase; letter-spacing:1px; }
[data-testid="stDataFrame"] th { background:#090d18 !important; color:#64748b !important; font-size:11px !important; text-transform:uppercase; }
[data-testid="stDataFrame"] td { color:#cbd5e1 !important; background:#07090f !important; }
[data-testid="stCheckbox"] span { color:#64748b !important; }
[data-baseweb="checkbox"] span[aria-disabled="true"] { color:#64748b !important; }

/* === BUTTON === */
[data-testid="stButton"] > button {
  background:transparent; border:1px solid #1a2540; color:#64748b;
  font-size:10px; text-transform:uppercase; letter-spacing:1.5px;
  border-radius:6px; padding:5px 16px; transition:all 0.2s;
}
[data-testid="stButton"] > button:hover { border-color:#d4a017; color:#d4a017; background:rgba(212,160,23,0.08); }

/* === DIVIDER === */
hr { border:none; height:1px; background:linear-gradient(90deg,transparent,#1a2540,transparent); margin:24px 0; }

/* === TRADE CARDS === */
.tcard {
  background:linear-gradient(150deg,#0b1022,#0f1828);
  border:1px solid #1a2540;
  border-radius:16px;
  padding:24px 26px;
  margin-bottom:20px;
  box-shadow:0 8px 40px rgba(0,0,0,0.6);
  position:relative;
}
.tcard-high { border-left:3px solid #d4a017; }
.tcard-watch { border-left:3px solid #3b82f6; }
.tcard::after {
  content:''; position:absolute; top:0; left:0; right:0; height:1px;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,0.04),transparent);
}

/* === BADGES === */
.badge-h {
  background:linear-gradient(90deg,#7a4e0a,#b8860b); color:#fef3c7;
  padding:3px 14px; border-radius:20px; font-size:10px; font-weight:800;
  letter-spacing:1px; text-transform:uppercase; border:1px solid rgba(212,160,23,0.4);
  display:inline-block;
}
.badge-w {
  background:linear-gradient(90deg,#1e3a8a,#2563eb); color:#bfdbfe;
  padding:3px 14px; border-radius:20px; font-size:10px; font-weight:800;
  letter-spacing:1px; text-transform:uppercase; border:1px solid rgba(59,130,246,0.4);
  display:inline-block;
}

/* === SCORE DISPLAY === */
.score-num-h { font-size:44px; font-weight:900; color:#d4a017; letter-spacing:-2px; line-height:1; display:inline-block; }
.score-num-w { font-size:44px; font-weight:900; color:#3b82f6; letter-spacing:-2px; line-height:1; display:inline-block; }
.score-denom { font-size:16px; color:#374151; font-weight:400; }
.sbar { height:5px; background:#111827; border-radius:3px; overflow:hidden; display:flex; margin:8px 0 3px; }
.sbar div { height:100%; }
.sc-c { background:#d4a017; }
.sc-a { background:#10b981; }
.sc-f { background:#3b82f6; }
.sc-t { background:#f43f5e; }
.sbar-legend { font-size:10px; }
.sbar-legend span { margin-right:8px; }
.lc { color:#d4a017; } .la { color:#10b981; } .lf { color:#3b82f6; } .lt { color:#f43f5e; }

/* === ACTION BADGE === */
.act-call { display:inline-block; padding:5px 18px; background:rgba(16,185,129,0.1); color:#34d399; border:1px solid rgba(16,185,129,0.3); border-radius:8px; font-weight:800; font-size:13px; letter-spacing:1px; }
.act-put  { display:inline-block; padding:5px 18px; background:rgba(239,68,68,0.1); color:#f87171; border:1px solid rgba(239,68,68,0.3); border-radius:8px; font-weight:800; font-size:13px; letter-spacing:1px; }
.act-buy  { display:inline-block; padding:5px 18px; background:rgba(99,102,241,0.1); color:#a5b4fc; border:1px solid rgba(99,102,241,0.3); border-radius:8px; font-weight:800; font-size:13px; letter-spacing:1px; }

/* === OPTION TICKET === */
.opt-box {
  background:#060810; border:1px solid #1a2540; border-radius:10px;
  padding:16px 20px; margin:14px 0; font-family:monospace;
}
.opt-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:18px; }
.opt-lbl { font-size:10px; color:#374151; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px; }
.opt-val { font-size:16px; font-weight:700; color:#e2e8f0; }
.opt-gold { font-size:16px; font-weight:700; color:#d4a017; }
.opt-sep { border-top:1px solid #111827; margin-top:14px; padding-top:14px; display:grid; grid-template-columns:repeat(3,1fr); gap:18px; }

/* === THESIS === */
.thesis {
  padding:14px 18px; border-left:2px solid #1a2540; margin:14px 0;
  color:#94a3b8; font-style:italic; font-size:14px; line-height:1.8;
}
.thesis-h { border-left-color:rgba(212,160,23,0.5); }
.thesis-w { border-left-color:rgba(59,130,246,0.4); }

/* === RISK BOX === */
.risk-box {
  background:rgba(239,68,68,0.05); border:1px solid rgba(239,68,68,0.15);
  border-radius:8px; padding:12px 16px; color:#fca5a5; font-size:13px; line-height:1.6;
}

/* === SECTION HEADERS === */
.sec-h {
  font-size:11px; font-weight:700; color:#374151; text-transform:uppercase;
  letter-spacing:2px; margin:24px 0 14px; padding-bottom:8px;
  border-bottom:1px solid #111827;
}
.sec-h-gold { color:#d4a017; border-bottom-color:rgba(212,160,23,0.2); }
.sec-h-blue { color:#3b82f6; border-bottom-color:rgba(59,130,246,0.2); }

/* === MACRO PANEL === */
.mac-row {
  display:flex; justify-content:space-between; align-items:center;
  padding:8px 0; border-bottom:1px solid #0f1829;
}
.mac-lbl { font-size:12px; color:#64748b; }
.mac-val { font-size:13px; font-weight:600; }
.mac-r { color:#f87171; } .mac-y { color:#fbbf24; } .mac-g { color:#34d399; }

/* === STRESS PILL === */
.pill-r { background:rgba(239,68,68,0.12); border:1px solid rgba(239,68,68,0.3); color:#f87171; padding:6px 16px; border-radius:20px; font-weight:700; font-size:13px; display:inline-block; }
.pill-y { background:rgba(245,158,11,0.12); border:1px solid rgba(245,158,11,0.3); color:#fbbf24; padding:6px 16px; border-radius:20px; font-weight:700; font-size:13px; display:inline-block; }
.pill-g { background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.2); color:#34d399; padding:6px 16px; border-radius:20px; font-weight:700; font-size:13px; display:inline-block; }

/* === NARRATIVE === */
.narr { background:#090d18; border:1px solid #1a2540; border-radius:8px; padding:11px 16px; font-size:13px; color:#64748b; line-height:1.7; margin-bottom:20px; }

/* === HELPER TEXT === */
.help-micro { font-size:9px; color:#374151; text-transform:none; letter-spacing:0; margin-top:1px; font-weight:400; }
.help-inline { font-size:11px; color:#475569; font-style:italic; margin-top:2px; }
.help-block { background:#090d18; border:1px solid #1a2540; border-radius:8px; padding:10px 14px; font-size:12px; color:#64748b; line-height:1.7; margin:6px 0 12px; }

/* === GUIDE === */
.guide-section { margin-bottom:14px; }
.guide-title { font-size:13px; font-weight:700; color:#94a3b8; margin-bottom:6px; }
.guide-text { font-size:12px; color:#64748b; line-height:1.7; }
.guide-term { display:inline-block; background:#0d1424; border:1px solid #1a2540; border-radius:6px; padding:3px 10px; margin:3px 4px 3px 0; font-size:11px; color:#94a3b8; }
.guide-term b { color:#cbd5e1; }

/* === LOG ROW === */
.log-r { font-family:monospace; font-size:11px; color:#374151; padding:3px 0; border-bottom:1px solid #0f1829; }

/* === PORTFOLIO METRICS === */
[data-testid="metric-container"] {
  background:#0b1022 !important; border:1px solid #1a2540 !important;
  border-radius:12px !important; padding:16px 20px !important;
}
</style>""", unsafe_allow_html=True)


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
    return conn.execute("""
        SELECT tr.id, tr.run_id, tr.created_at, tr.ticker, tr.total_score, tr.conviction_tier,
               tr.trade_action, tr.instrument, tr.strike, tr.expiry, tr.premium,
               tr.delta, tr.gamma, tr.vega, tr.underlying_price,
               tr.thesis, tr.triggers, tr.pnl_scenarios, tr.risk_check, tr.position_pct,
               ts.convexity_score, ts.antifragility_score,
               ts.fragility_avoidance, ts.tail_risk_score, ts.catalyst
        FROM trade_recommendations tr
        LEFT JOIN taleb_scores ts ON tr.ticker = ts.ticker AND tr.run_id = ts.run_id
        WHERE tr.run_id=? ORDER BY tr.total_score DESC
    """, (run_id,)).fetchall()


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


def _compute_dte(expiry_str):
    try:
        exp = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        return max(0, (exp - datetime.now().date()).days)
    except Exception:
        return 0


# ── Render: Payoff chart ──────────────────────────────────────────────────────

def _render_payoff_chart(ticker, strike, expiry, premium, opt_type, price, position_size):
    from trade_builder import OptionLeg, build_payoff_curve
    opt = None
    if strike and premium:
        opt = OptionLeg(option_type=opt_type or "call", strike=strike, expiry=expiry or "",
                        premium=premium, delta=0, gamma=0, vega=0, dte=30)
    curve = build_payoff_curve(opt, price, position_size)

    prices_arr, pnl_arr = curve["prices"], curve["pnl"]
    pos_pnl = [max(0, p) for p in pnl_arr]
    neg_pnl = [min(0, p) for p in pnl_arr]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices_arr, y=pos_pnl, fill='tozeroy',
                             fillcolor='rgba(16,185,129,0.12)', line=dict(width=0),
                             showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=prices_arr, y=neg_pnl, fill='tozeroy',
                             fillcolor='rgba(239,68,68,0.12)', line=dict(width=0),
                             showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=prices_arr, y=pnl_arr, mode='lines', name='P&L',
                             line=dict(color='#60a5fa', width=2.5),
                             hovertemplate='$%{x:.2f} → %{y:+.0f}<extra></extra>'))
    fig.add_vline(x=curve["current_price"], line_dash="dot", line_color="#475569", line_width=1,
                  annotation_text=f"Now ${curve['current_price']:.2f}",
                  annotation_font=dict(color="#64748b", size=11))
    fig.add_vline(x=curve["breakeven"], line_dash="dash", line_color="#f59e0b", line_width=1,
                  annotation_text=f"BE ${curve['breakeven']:.2f}",
                  annotation_font=dict(color="#f59e0b", size=11),
                  annotation_position="top left")
    if strike:
        fig.add_vline(x=strike, line_dash="dot", line_color="#374151", line_width=1)
    fig.add_hline(y=0, line_color="#1e2a3a", line_width=1)
    dte_label = f" | {_compute_dte(expiry)}d to exp" if expiry else ""
    fig.update_layout(
        title=dict(text=f"{ticker} Payoff{dte_label}", font=dict(color="#94a3b8", size=13)),
        xaxis_title="Underlying Price ($)", yaxis_title="P&L ($)",
        plot_bgcolor="#07090f", paper_bgcolor="#07090f",
        font=dict(color="#64748b", size=11),
        height=300, margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(showgrid=True, gridcolor="#0f1829", color="#64748b"),
        yaxis=dict(showgrid=True, gridcolor="#0f1829", color="#64748b"),
        legend=dict(font=dict(color="#64748b")),
    )
    st.plotly_chart(fig, width="stretch", key=f"payoff_{ticker}")


# ── Render: Trade card ────────────────────────────────────────────────────────

def _render_trade_card(trade, position_size, idx=0):
    tier = trade["conviction_tier"]
    price = trade["underlying_price"] or 0
    action = trade["trade_action"] or "BUY"
    total = trade["total_score"]

    # Sub-scores
    c = trade.get("convexity_score") or 0
    a = trade.get("antifragility_score") or 0
    f = trade.get("fragility_avoidance") or 0
    t = trade.get("tail_risk_score") or 0

    # Action badge class
    if "CALL" in action:
        act_cls = "act-call"
    elif "PUT" in action:
        act_cls = "act-put"
    else:
        act_cls = "act-buy"

    # Tier
    card_cls = "tcard tcard-high" if tier == "HIGH" else "tcard tcard-watch"
    badge_cls = "badge-h" if tier == "HIGH" else "badge-w"
    score_cls = "score-num-h" if tier == "HIGH" else "score-num-w"
    tier_label = "HIGH CONVICTION" if tier == "HIGH" else "WATCH LIST"
    thesis_cls = "thesis thesis-h" if tier == "HIGH" else "thesis thesis-w"

    # Catalyst
    catalyst = trade.get("catalyst") or ""

    # Card header
    st.markdown(f"""
<div class="{card_cls}">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:16px">
    <div>
      <div style="font-size:32px;font-weight:900;letter-spacing:2px;color:#f1f5f9;line-height:1">{trade['ticker']}</div>
      <div style="font-size:11px;color:#374151;text-transform:uppercase;letter-spacing:1px;margin-top:4px">{trade['instrument']}</div>
      {f'<div style="font-size:12px;color:#64748b;margin-top:6px">{catalyst}</div>' if catalyst else ''}
    </div>
    <div style="text-align:right;flex-shrink:0">
      <div><span class="{score_cls}">{total}</span><span class="score-denom">/100</span></div>
      <div style="margin-top:6px"><span class="{badge_cls}">{tier_label}</span></div>
    </div>
  </div>
  <div style="margin-top:14px">
    <div class="sbar">
      <div class="sc-c" style="width:{c}%"></div>
      <div class="sc-a" style="width:{a}%"></div>
      <div class="sc-f" style="width:{f}%"></div>
      <div class="sc-t" style="width:{t}%"></div>
    </div>
    <div class="sbar-legend">
      <span class="lc">Asymmetry {c}</span>
      <span class="la">Strength {a}</span>
      <span class="lf">Avoids Crowds {f}</span>
      <span class="lt">Black Swan {t}</span>
    </div>
  </div>
  <div style="margin-top:16px">
    <span class="{act_cls}">{action}</span>
    <span style="color:#374151;font-size:13px;margin-left:12px">@ ${price:.2f}</span>
    <span style="color:#475569;font-size:11px;margin-left:8px">{"Betting price goes UP" if "CALL" in action else ("Betting price goes DOWN" if "PUT" in action else "Buy the asset directly")}</span>
  </div>
</div>""", unsafe_allow_html=True)

    # Option ticket
    if trade["strike"] and trade["premium"]:
        cost_1 = round((trade["premium"] or 0) * 100, 2)
        n = max(1, int(position_size / cost_1)) if cost_1 > 0 else 1
        dte = _compute_dte(trade["expiry"] or "")
        st.markdown(f"""
<div class="opt-box">
  <div class="opt-grid">
    <div><div class="opt-lbl">Strike</div><div class="opt-gold">${trade['strike']:.2f}</div><div class="help-micro">Price option activates</div></div>
    <div><div class="opt-lbl">Expiry</div><div class="opt-val">{trade['expiry']}</div><div class="help-micro">Option deadline</div></div>
    <div><div class="opt-lbl">Premium</div><div class="opt-val">${trade['premium']:.2f}</div><div class="help-micro">Cost per share (x100)</div></div>
    <div><div class="opt-lbl">DTE</div><div class="opt-val">{dte}d</div><div class="help-micro">Days until expiry</div></div>
  </div>
  <div class="opt-sep">
    <div><div class="opt-lbl">1 Contract</div><div class="opt-val">${cost_1:.0f}</div><div class="help-micro">Minimum trade cost</div></div>
    <div><div class="opt-lbl">{n} Contracts (${position_size})</div><div class="opt-val">${n * cost_1:.0f}</div><div class="help-micro">Your actual cost</div></div>
    <div><div class="opt-lbl">Delta / Vega</div><div class="opt-val">{trade['delta']:.3f} / {trade['vega']:.3f}</div><div class="help-micro">Price / volatility sensitivity</div></div>
  </div>
</div>""", unsafe_allow_html=True)

    if trade["thesis"]:
        st.markdown(f'<div class="{thesis_cls}">{trade["thesis"]}</div>', unsafe_allow_html=True)

    # Expandable sections
    col_a, col_b = st.columns(2)
    with col_a:
        with st.expander("Trigger Checklist -- What Needs to Happen"):
            try:
                triggers = json.loads(trade["triggers"]) if trade["triggers"] else []
            except Exception:
                triggers = []
            if triggers:
                for trig in triggers:
                    st.checkbox(trig, value=False, key=f"trg_{idx}_{hash(trig)}", disabled=True)
            else:
                st.markdown('<span style="color:#374151">No triggers defined.</span>', unsafe_allow_html=True)

        with st.expander("Risk Reality Check -- Honest Odds"):
            if trade["risk_check"]:
                st.markdown(f'<div class="risk-box">{trade["risk_check"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<span style="color:#374151">No risk data.</span>', unsafe_allow_html=True)

    with col_b:
        with st.expander("P&L Scenarios -- What Could Happen"):
            st.markdown('<div class="help-inline">Your money in 5 different market outcomes, from worst to best case.</div>', unsafe_allow_html=True)
            try:
                scenarios = json.loads(trade["pnl_scenarios"]) if trade["pnl_scenarios"] else []
            except Exception:
                scenarios = []
            if scenarios and trade["strike"] and trade["premium"] and trade["premium"] > 0:
                strike, prem = trade["strike"], trade["premium"]
                contracts = max(1, int(position_size / (prem * 100)))
                cost = contracts * prem * 100
                is_call = "CALL" in (action or "")
                if is_call:
                    be = strike + prem
                    bt = strike * 1.10; ht = strike * 1.25
                    bp = max(0, bt - strike) * contracts * 100 - cost
                    hp = max(0, ht - strike) * contracts * 100 - cost
                else:
                    be = strike - prem
                    bt = strike * 0.90; ht = strike * 0.75
                    bp = max(0, strike - bt) * contracts * 100 - cost
                    hp = max(0, strike - ht) * contracts * 100 - cost
                rows = [
                    {"Scenario": "Total Loss", "Trigger": "Expires worthless", "P&L": f"${-cost:+,.0f}", "Return": "-100%"},
                    {"Scenario": "Partial Loss", "Trigger": "IV crush / decay", "P&L": f"${-cost*0.5:+,.0f}", "Return": "-50%"},
                    {"Scenario": "Break Even", "Trigger": f"${be:.2f}", "P&L": "$0", "Return": "0%"},
                    {"Scenario": "Base Case", "Trigger": f"${bt:.2f}", "P&L": f"${bp:+,.0f}", "Return": f"{bp/cost*100:+.0f}%" if cost > 0 else "0%"},
                    {"Scenario": "Home Run", "Trigger": f"${ht:.2f}", "P&L": f"${hp:+,.0f}", "Return": f"{hp/cost*100:+.0f}%" if cost > 0 else "0%"},
                ]
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            elif scenarios:
                st.markdown('<span style="color:#374151">No options data for scenario modeling.</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span style="color:#374151">No scenario data.</span>', unsafe_allow_html=True)

        with st.expander("Payoff Chart -- Visual Map"):
            st.markdown('<div class="help-inline">Green = profit zone, Red = loss zone. Hover to see exact P&L at any price.</div>', unsafe_allow_html=True)
            opt_type = "call" if "CALL" in (action or "") else "put"
            _render_payoff_chart(trade["ticker"], trade["strike"], trade["expiry"],
                                 trade["premium"], opt_type, price, position_size)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)


# ── Macro charts ──────────────────────────────────────────────────────────────

SERIES_LABELS = {
    "fed_funds_rate": "Fed Funds", "baa_aaa_spread": "Credit Spread",
    "ted_spread": "TED Spread", "yield_curve_10y2y": "Yield Curve 10Y-2Y",
    "unemployment_claims": "Jobless Claims", "vix": "VIX",
}

SERIES_HELP = {
    "fed_funds_rate": "Central bank interest rate",
    "baa_aaa_spread": "Fear premium in corporate bonds",
    "ted_spread": "Bank lending stress",
    "yield_curve_10y2y": "Negative = recession warning",
    "unemployment_claims": "Weekly layoff filings",
    "vix": "Market fear gauge (higher = more fear)",
}


def _chart(df, title, color="#3b82f6", fill=True, hline=None, zones=None):
    if df.empty:
        st.markdown(f'<div style="color:#374151;font-size:12px;padding:8px 0">No {title} data yet.</div>', unsafe_allow_html=True)
        return
    df = df.sort_values("timestamp")
    fig = go.Figure()
    if zones:
        for y0, y1, c in zones:
            fig.add_hrect(y0=y0, y1=y1, fillcolor=c, line_width=0)
    hex_color = color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["value"], mode="lines",
        line=dict(color=color, width=1.5), name=title,
        fill="tozeroy" if fill else None,
        fillcolor=f"rgba({r},{g},{b},0.07)",
        hovertemplate='%{y:.2f}<extra></extra>',
    ))
    if hline is not None:
        fig.add_hline(y=hline, line_dash="dash", line_color="#374151", line_width=1)
    fig.update_layout(
        title=dict(text=title, font=dict(color="#64748b", size=11)),
        plot_bgcolor="#07090f", paper_bgcolor="#07090f", font=dict(color="#64748b", size=10),
        height=180, margin=dict(l=0, r=0, t=28, b=0),
        xaxis=dict(showgrid=False, color="#374151", showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor="#0f1829", color="#374151"),
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch", key=f"chart_{title}")


def _macro_narrative(macro_rows) -> str:
    signals = []
    for r in macro_rows:
        k, v, z, f = r["series_key"], r["value"], r["z_score_2y"], r["stress_flag"]
        if k == "yield_curve_10y2y" and v is not None:
            if v < 0: signals.append("Yield curve inverted")
            elif v < 0.3: signals.append("Yield curve flattening")
        if k == "baa_aaa_spread" and f: signals.append("Credit spreads widening")
        if k == "vix" and v is not None:
            if v < 15: signals.append("Volatility suppressed -- options cheap")
            elif v > 30: signals.append("Fear elevated")
        if k == "unemployment_claims" and f: signals.append("Jobless claims spiking")
    signals = [s for s in signals if s]
    if not signals:
        return "No macro stress signals. Markets priced for calm. Classic setup for buying cheap tail protection."
    return ". ".join(signals[:3]) + (". Pre-stress setup." if len(signals) >= 2 else ". Monitor closely.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    conn = get_conn()
    _ensure_tables(conn)

    # Always call _run_full_cycle (cached 30 min, clears on Refresh click)
    need_spinner = not _latest_run(conn)
    if need_spinner:
        with st.spinner("Scanning markets... first load takes 3-5 minutes"):
            _run_full_cycle()
    else:
        _run_full_cycle()

    # Get a fresh connection after the pipeline completes (avoids SQLite thread issues)
    conn = get_conn()

    meta = _latest_run(conn)
    run_id = meta.get("run_id")
    macro_rows = _load_macro(conn, run_id) if run_id else []
    stress_count = sum(1 for r in macro_rows if r["stress_flag"])
    stress_level = "RED" if stress_count >= 4 else ("YELLOW" if stress_count >= 2 else "GREEN")
    stress_pill = {"RED": "pill-r", "YELLOW": "pill-y", "GREEN": "pill-g"}[stress_level]
    stress_icon = {"RED": "RISK", "YELLOW": "CAUTION", "GREEN": "CALM"}[stress_level]
    narrative = _macro_narrative(macro_rows) if macro_rows else "No macro data loaded yet."

    # ── HEADER ──────────────────────────────────────────────────────────────
    h1, h2, h3, h4 = st.columns([3, 1.5, 2, 1])
    with h1:
        st.markdown(
            "<div style='font-size:28px;font-weight:900;letter-spacing:3px;color:#f1f5f9;line-height:1'>TALEB TRADE ADVISOR</div>"
            "<div style='font-size:10px;color:#475569;letter-spacing:0.5px;margin-top:6px'>AI-powered market scanner finding cheap bets with massive upside potential</div>",
            unsafe_allow_html=True,
        )
    stress_explain = {"RED": "Multiple stress signals -- high alert", "YELLOW": "Some signals flashing -- stay alert", "GREEN": "Low stress -- good time to buy cheap protection"}[stress_level]
    with h2:
        st.markdown(
            f'<div style="font-size:10px;color:#374151;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px">Macro Stress</div>'
            f'<span class="{stress_pill}">{stress_icon}</span>'
            f'<div style="font-size:10px;color:#374151;margin-top:5px">{stress_explain}</div>',
            unsafe_allow_html=True,
        )
    with h3:
        assets = meta.get("assets_scanned", 0)
        opps = meta.get("opportunities", 0)
        ts = str(meta.get("finished_at", ""))[:16].replace("T", " ")
        st.markdown(
            f'<div style="font-size:10px;color:#374151;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px">Last Scan</div>'
            f'<div style="font-size:13px;color:#94a3b8">{ts or "Never"}</div>'
            f'<div style="font-size:11px;color:#374151;margin-top:2px">{assets} assets &nbsp;|&nbsp; {opps} opportunities</div>',
            unsafe_allow_html=True,
        )
    with h4:
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        if st.button("Refresh", type="secondary"):
            _run_full_cycle.clear()
            st.rerun()

    st.markdown(f'<div class="narr">{narrative}</div>', unsafe_allow_html=True)

    # ── HOW THIS WORKS GUIDE ─────────────────────────────────────────────────
    with st.expander("How This Works -- Click to Learn"):
        st.markdown("""
<div class="guide-section">
  <div class="guide-title">What is this dashboard?</div>
  <div class="guide-text">
    An AI agent scans 33 assets (stocks, ETFs, commodities) every 30 minutes and scores each one based on
    Nassim Taleb's investment philosophy: find cheap bets where you risk a little but could gain a lot.
    The top 10 opportunities are shown below as actionable trade ideas.
  </div>
</div>
<div class="guide-section">
  <div class="guide-title">How to read a trade card</div>
  <div class="guide-text">
    Each card shows one trade idea. The <b style="color:#cbd5e1">score (0-100)</b> tells you how well it fits the strategy.
    Higher is better. The colored bar breaks the score into 4 dimensions:<br>
    <span style="color:#d4a017">Asymmetry</span> = small cost, big potential payoff |
    <span style="color:#10b981">Strength</span> = benefits from chaos |
    <span style="color:#3b82f6">Avoids Crowds</span> = not a popular/obvious trade |
    <span style="color:#f43f5e">Black Swan Upside</span> = profits from extreme events<br><br>
    <b style="color:#34d399">BUY CALL</b> = betting the price goes UP |
    <b style="color:#f87171">BUY PUT</b> = betting the price goes DOWN |
    <b style="color:#a5b4fc">BUY</b> = buy the asset directly
  </div>
</div>
<div class="guide-section">
  <div class="guide-title">Key terms explained</div>
  <div class="guide-text">
    <span class="guide-term"><b>Strike</b> -- Price the option activates at</span>
    <span class="guide-term"><b>Premium</b> -- Cost per share (x100 for one contract)</span>
    <span class="guide-term"><b>DTE</b> -- Days until the option expires</span>
    <span class="guide-term"><b>Delta</b> -- How much the option moves per $1 of stock movement</span>
    <span class="guide-term"><b>Vega</b> -- How much the option gains when fear/volatility rises</span>
    <span class="guide-term"><b>VIX</b> -- The market's "fear gauge" (higher = more fear)</span>
    <span class="guide-term"><b>Yield Curve</b> -- Recession signal (negative = warning)</span>
    <span class="guide-term"><b>OTM</b> -- "Out of the money" = cheap option that needs a big move to pay off</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── POSITION SIZE ────────────────────────────────────────────────────────
    position_size = st.slider(
        "How much to risk per trade ($)", 100, 5000, 500, 50,
        help="Drag to adjust. All profit/loss tables and charts below update automatically to show your real dollar amounts.",
    )
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── MAIN LAYOUT ─────────────────────────────────────────────────────────
    main_col, side_col = st.columns([2, 1], gap="large")

    with main_col:
        if not run_id:
            st.warning("No data available. Click Refresh.")
        else:
            trades = _load_trades(conn, run_id)
            if not trades:
                st.markdown(
                    '<div style="background:#090d18;border:1px solid #1a2540;border-radius:10px;padding:20px;color:#64748b;font-size:13px">'
                    'No opportunities scoring 40+ in the latest scan. Click Refresh to re-scan.</div>',
                    unsafe_allow_html=True,
                )
            else:
                high_t = [t for t in trades if t["conviction_tier"] == "HIGH"]
                watch_t = [t for t in trades if t["conviction_tier"] == "WATCH"]

                if high_t:
                    st.markdown('<div class="sec-h sec-h-gold">High Conviction Trades</div>', unsafe_allow_html=True)
                    for idx, t in enumerate(high_t):
                        _render_trade_card(dict(t), position_size, idx)

                if watch_t:
                    st.markdown('<div class="sec-h sec-h-blue">Watch List</div>', unsafe_allow_html=True)
                    for idx, t in enumerate(watch_t, start=len(high_t)):
                        _render_trade_card(dict(t), position_size, idx)

    with side_col:
        # Macro indicators panel
        st.markdown('<div class="sec-h">Macro Environment</div>', unsafe_allow_html=True)
        if macro_rows:
            for r in macro_rows:
                key, val, z, flag = r["series_key"], r["value"], r["z_score_2y"], r["stress_flag"]
                light = "!!!" if flag else ("!" if z and abs(z) > 1.5 else "OK")
                flag_cls = "mac-r" if flag else ("mac-y" if z and abs(z) > 1.5 else "mac-g")
                label = SERIES_LABELS.get(key, key)
                val_str = (f"{val/1000:.0f}k" if key == "unemployment_claims" and val
                           else (f"{val:.2f}%" if val is not None else "N/A"))
                z_str = f" z={z:.1f}" if z is not None else ""
                help_txt = SERIES_HELP.get(key, "")
                st.markdown(
                    f'<div class="mac-row">'
                    f'<div><span class="mac-lbl">{label}</span><div class="help-micro">{help_txt}</div></div>'
                    f'<span class="mac-val {flag_cls}">{val_str}<span style="font-size:10px;color:#374151"> {z_str}</span></span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<div style="color:#374151;font-size:12px">No FRED data yet.</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        _chart(_load_history(conn, "yield_curve_10y2y"), "Yield Curve (10Y-2Y)", "#3b82f6", hline=0)
        _chart(_load_history(conn, "vix"), "VIX", "#f59e0b", fill=False,
               zones=[(0, 15, "rgba(16,185,129,0.06)"), (15, 25, "rgba(245,158,11,0.04)"), (25, 100, "rgba(239,68,68,0.04)")])

    # ── PORTFOLIO SIMULATOR ──────────────────────────────────────────────────
    if run_id:
        trades = _load_trades(conn, run_id)
        if trades and len(trades) > 1:
            st.markdown("---")
            st.markdown('<div class="sec-h">Portfolio Simulator</div>', unsafe_allow_html=True)
            st.markdown('<div class="help-block">If you placed all trades below with your selected risk amount, here is the combined picture. Max Loss is the absolute worst case (all options expire worthless). Base Case assumes moderate moves in your favor.</div>', unsafe_allow_html=True)
            total_cost = 0
            total_base = 0
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
                    base = next((s for s in scens if s.get("scenario") == "Base Case"), None)
                    if base:
                        total_base += base.get("pnl", 0)
                except Exception:
                    pass
                tickers.append(t["ticker"])

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Trades", str(len(trades)))
            c2.metric("Combined Cost", f"${total_cost:,.0f}")
            c3.metric("Max Loss", f"-${total_cost:,.0f}")
            c4.metric("Base Case P&L", f"${total_base:+,.0f}")
            if len(set(tickers)) < len(tickers):
                st.warning("Duplicate tickers detected -- correlated exposure.")

    # ── ACTIVITY LOG ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sec-h">Agent Activity Log</div>', unsafe_allow_html=True)
    log_rows = _load_log(conn)
    if log_rows:
        for r in log_rows:
            r = dict(r)
            icon = {"completed": "OK", "running": "...", "error": "ERR"}.get(r["status"], "?")
            errs = ""
            if r["errors"] and r["errors"] != "null":
                try:
                    e = json.loads(r["errors"])
                    if e:
                        errs = f" | {', '.join(str(x)[:30] for x in e[:2])}"
                except Exception:
                    pass
            st.markdown(
                f'<div class="log-r">[{icon}] <b style="color:#64748b">{r["run_id"]}</b>'
                f' {str(r["started_at"])[:19]}'
                f' assets={r["assets_scanned"]} opps={r["opportunities"]}{errs}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<div style="color:#374151;font-size:12px">No runs yet.</div>', unsafe_allow_html=True)

    st.markdown(
        '<div style="text-align:center;color:#1e2a3a;font-size:10px;margin-top:24px;letter-spacing:1px">DATA REFRESHES EVERY 30 MINUTES &nbsp;|&nbsp; CLICK REFRESH TO FORCE UPDATE</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
