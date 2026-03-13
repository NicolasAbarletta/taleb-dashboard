# -*- coding: utf-8 -*-
"""
trade_builder.py -- Phase 3: Trade Recommendation Engine
==========================================================
For every opportunity scoring 50+, generates a complete trade ticket:
exact trade, thesis, triggers, P&L scenarios, payoff chart data,
and risk reality check.
"""

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yfinance as yf
import anthropic
from dotenv import load_dotenv

load_dotenv(override=True)
log = logging.getLogger("trade_builder")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


@dataclass
class OptionLeg:
    option_type: str  # "call" or "put"
    strike: float
    expiry: str
    premium: float
    delta: float
    gamma: float
    vega: float
    dte: int
    contract_symbol: str = ""


@dataclass
class TradeRec:
    ticker: str
    run_id: str
    total_score: int
    conviction_tier: str
    # Section A: The exact trade
    action: str  # BUY CALL, BUY PUT, BUY CALL SPREAD, BUY
    instrument: str  # "OTM Call Option", "ETF shares", etc.
    underlying_price: float
    option_leg: OptionLeg | None = None
    position_pct: float = 0.5  # % of portfolio
    # Section B: Thesis
    thesis: str = ""
    # Section C: Triggers
    triggers: list = field(default_factory=list)
    # Section D: P&L scenarios
    pnl_scenarios: list = field(default_factory=list)
    # Section F: Risk check
    risk_check: str = ""
    # Metadata
    catalyst: str = ""
    rationale: str = ""


# ── Options chain analysis ────────────────────────────────────────────────────

def _find_best_option(ticker: str, price: float, direction: str = "call") -> OptionLeg | None:
    """
    Find the best asymmetric options play for a Taleb-style trade.
    Targets: OTM options 30-90 DTE, 5-15% OTM for calls, 5-15% OTM for puts.
    Prefers: low IV, decent open interest, reasonable premium.
    """
    try:
        t = yf.Ticker(ticker)
        exps = t.options
        if not exps:
            return None

        today = datetime.now().date()
        best = None
        best_score = -1

        for exp_str in exps:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if dte < 20 or dte > 120:
                continue

            try:
                chain = t.option_chain(exp_str)
                df = chain.calls if direction == "call" else chain.puts
                if df.empty:
                    continue

                # OTM filter
                if direction == "call":
                    # 3-15% OTM calls
                    otm = df[(df["strike"] >= price * 1.03) & (df["strike"] <= price * 1.15)]
                else:
                    # 3-15% OTM puts
                    otm = df[(df["strike"] >= price * 0.85) & (df["strike"] <= price * 0.97)]

                if otm.empty:
                    continue

                # Filter for decent liquidity
                otm = otm[otm["openInterest"] >= 10]
                if otm.empty:
                    continue

                for _, row in otm.iterrows():
                    iv = row.get("impliedVolatility", 0)
                    premium = row.get("lastPrice", 0) or row.get("ask", 0)
                    if premium <= 0 or iv <= 0:
                        continue

                    # Score: low IV + reasonable dte + decent OI
                    oi = row.get("openInterest", 0) or 0
                    option_score = (1 / (iv + 0.01)) * (oi ** 0.3) * (1 if 30 <= dte <= 75 else 0.6)

                    if option_score > best_score:
                        best_score = option_score
                        delta = abs(row.get("delta", 0.15) if not pd.isna(row.get("delta")) else 0.15)
                        gamma = row.get("gamma", 0.01) if not pd.isna(row.get("gamma")) else 0.01
                        vega_val = row.get("vega", 0.05) if not pd.isna(row.get("vega")) else 0.05

                        best = OptionLeg(
                            option_type=direction,
                            strike=float(row["strike"]),
                            expiry=exp_str,
                            premium=round(float(premium), 2),
                            delta=round(float(delta), 3),
                            gamma=round(float(gamma), 4),
                            vega=round(float(vega_val), 3),
                            dte=dte,
                            contract_symbol=row.get("contractSymbol", ""),
                        )
            except Exception as e:
                log.debug(f"Chain error {exp_str}: {e}")

        return best
    except Exception as e:
        log.error(f"[options] {ticker}: {e}")
        return None


# ── Trade direction logic ─────────────────────────────────────────────────────

def _decide_direction(ticker: str, snap_data: dict, macro_data: dict) -> str:
    """Decide whether this is a bullish (call) or bearish (put) setup."""
    tk = ticker.upper()

    # VIX products: always calls (betting on vol spike)
    if tk in ("VIXY", "UVXY"):
        return "call"

    # Inverse VIX: always puts (betting on vol spike)
    if tk == "SVXY":
        return "put"

    # Broad market near highs + macro stress: puts
    w52 = snap_data.get("week52_position")
    stress = macro_data.get("stress_level", "GREEN")

    if tk in ("SPY", "QQQ", "IWM") and stress in ("YELLOW", "RED"):
        return "put"

    # Near 52w low: calls (mean reversion / recovery)
    if w52 is not None and w52 <= 0.25:
        return "call"

    # Near 52w high: puts (crowded long)
    if w52 is not None and w52 >= 0.80:
        return "put"

    # Commodities in stress: calls
    if tk in ("GLD", "SLV", "CPER", "USO", "UNG") and stress != "GREEN":
        return "call"

    # Default: calls (asymmetric upside)
    return "call"


# ── P&L scenario builder ─────────────────────────────────────────────────────

def _build_pnl_scenarios(opt: OptionLeg | None, price: float, position_size: float = 500.0) -> list[dict]:
    """
    Build 5-scenario P&L table. Returns list of dicts with
    scenario, description, pnl, return_pct.
    """
    if opt is None:
        # Stock/ETF trade (no options)
        shares = int(position_size / price) if price > 0 else 1
        cost = shares * price
        return [
            {"scenario": "Total Loss", "description": "Asset goes to zero", "pnl": round(-cost, 2), "return_pct": -100},
            {"scenario": "Partial Loss", "description": "-15% drawdown", "pnl": round(-cost * 0.15, 2), "return_pct": -15},
            {"scenario": "Break Even", "description": "Flat", "pnl": 0, "return_pct": 0},
            {"scenario": "Base Case", "description": "+20% gain", "pnl": round(cost * 0.20, 2), "return_pct": 20},
            {"scenario": "Home Run", "description": "+50% gain", "pnl": round(cost * 0.50, 2), "return_pct": 50},
        ]

    # Options trade
    contracts = max(1, int(position_size / (opt.premium * 100)))
    cost = contracts * opt.premium * 100

    if opt.option_type == "call":
        breakeven = opt.strike + opt.premium
        base_target = opt.strike * 1.10  # 10% above strike
        home_run = opt.strike * 1.25     # 25% above strike
        base_profit = max(0, base_target - opt.strike) * contracts * 100 - cost
        home_profit = max(0, home_run - opt.strike) * contracts * 100 - cost
    else:
        breakeven = opt.strike - opt.premium
        base_target = opt.strike * 0.90
        home_run = opt.strike * 0.75
        base_profit = max(0, opt.strike - base_target) * contracts * 100 - cost
        home_profit = max(0, opt.strike - home_run) * contracts * 100 - cost

    return [
        {"scenario": "Total Loss", "description": "Expires worthless",
         "pnl": round(-cost, 2), "return_pct": -100},
        {"scenario": "Partial Loss", "description": "IV crush / time decay",
         "pnl": round(-cost * 0.50, 2), "return_pct": -50},
        {"scenario": "Break Even", "description": f"Underlying at ${breakeven:.2f}",
         "pnl": 0, "return_pct": 0},
        {"scenario": "Base Case", "description": f"Underlying at ${base_target:.2f}",
         "pnl": round(base_profit, 2),
         "return_pct": round(base_profit / cost * 100, 0) if cost > 0 else 0},
        {"scenario": "Home Run", "description": f"Underlying at ${home_run:.2f}",
         "pnl": round(home_profit, 2),
         "return_pct": round(home_profit / cost * 100, 0) if cost > 0 else 0},
    ]


# ── Payoff curve data ─────────────────────────────────────────────────────────

def build_payoff_curve(opt: OptionLeg | None, price: float, position_size: float = 500.0) -> dict:
    """
    Returns dict with 'prices' and 'pnl' arrays for Plotly charting.
    X: underlying price from -40% to +40% of current.
    Y: profit/loss in dollars.
    """
    if opt is None:
        shares = max(1, int(position_size / price)) if price > 0 else 1
        prices = np.linspace(price * 0.6, price * 1.4, 100)
        pnl = [(p - price) * shares for p in prices]
        breakeven = price
    else:
        contracts = max(1, int(position_size / (opt.premium * 100)))
        cost = contracts * opt.premium * 100
        prices = np.linspace(price * 0.6, price * 1.4, 100)

        if opt.option_type == "call":
            pnl = [max(0, p - opt.strike) * contracts * 100 - cost for p in prices]
            breakeven = opt.strike + opt.premium
        else:
            pnl = [max(0, opt.strike - p) * contracts * 100 - cost for p in prices]
            breakeven = opt.strike - opt.premium

    return {
        "prices": [round(float(p), 2) for p in prices],
        "pnl": [round(float(v), 2) for v in pnl],
        "current_price": round(price, 2),
        "breakeven": round(breakeven, 2),
        "max_loss": round(-position_size if opt else -(price * (max(1, int(position_size / price)) if price > 0 else 1)), 2),
    }


# ── Claude-powered thesis & triggers ─────────────────────────────────────────

def _generate_thesis_and_triggers(rec: TradeRec, macro: dict) -> tuple[str, list[str], str]:
    """
    Generate trader-voice thesis, trigger checklist, and risk reality check.
    Returns (thesis, triggers, risk_check).
    """
    opt = rec.option_leg
    opt_details = ""
    if opt:
        opt_details = f"""
Option: {opt.option_type.upper()} {rec.ticker} ${opt.strike} exp {opt.expiry}
Premium: ${opt.premium}/contract, Delta: {opt.delta}, DTE: {opt.dte}"""

    prompt = f"""You are a senior options trader giving a friend specific trade advice.
Generate THREE things for this trade (output them as JSON, nothing else):

Trade: {rec.action} {rec.ticker} at ${rec.underlying_price:.2f}
{opt_details}
Score: {rec.total_score}/100 ({rec.conviction_tier})
Catalyst: {rec.catalyst}
Macro stress: {macro.get('stress_level', 'GREEN')}
VIX: {macro.get('vix_value', 'N/A')}
Yield curve: {macro.get('yield_curve', 'N/A')}

Return valid JSON with exactly these keys:
{{
  "thesis": "3 sentences max. Like explaining to a smart friend at a bar. Specific. Why NOW. What the market is missing. What could go wrong. No jargon. Examples: 'Oil is being ignored right now. If Iran escalates, this triples. If not, you lose what you paid for lunch.'",
  "triggers": ["3-5 specific, measurable conditions that must happen for this trade to work. Be precise with numbers and timeframes. Like: 'VIX closes above 25 within 30 days' or 'Gold breaks $2100 resistance'"],
  "risk_check": "1-2 sentences. Blunt reality check. How often does this type of trade work? Size it like what? Be honest."
}}"""

    if not ANTHROPIC_API_KEY:
        direction = "up" if (opt and opt.option_type == "call") or rec.action == "BUY" else "down"
        return (
            f"{rec.ticker} has an asymmetric setup right now. If it moves {direction}, the payoff is nonlinear. If it doesn't, you lose the premium.",
            [
                f"{rec.ticker} moves {'above' if direction == 'up' else 'below'} ${rec.underlying_price * (1.1 if direction == 'up' else 0.9):.2f}",
                f"VIX stays below 25 (options remain affordable)",
                f"No adverse macro surprise in next 30 days",
                f"Position sized at max {rec.position_pct}% of portfolio",
            ],
            "This is a defined-risk trade. Most options expire worthless. Size it like a lottery ticket.",
        )

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()
        # Extract JSON from response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            return (
                data.get("thesis", ""),
                data.get("triggers", []),
                data.get("risk_check", ""),
            )
    except Exception as e:
        log.error(f"[Claude] thesis for {rec.ticker}: {e}")

    return (
        f"Asymmetric setup in {rec.ticker}. Catalyst: {rec.catalyst}.",
        [f"Monitor {rec.ticker} price action", "Watch macro developments"],
        "Size appropriately. Most options trades lose. The winners pay for the losers.",
    )


# ── Master builder ────────────────────────────────────────────────────────────

def build_trades(conn, run_id: str, position_size: float = 500.0) -> list[TradeRec]:
    """
    Build complete trade recommendations for all scored opportunities (50+).
    """
    log.info(f"[trades] Building recommendations for {run_id}")

    # Load scored opportunities
    rows = conn.execute("""
        SELECT ts.ticker, ts.total_score, ts.conviction_tier, ts.catalyst, ts.rationale,
               ts.price, ts.iv_percentile, ts.score_details,
               es.week52_position, es.iv_30d, es.beta
        FROM taleb_scores ts
        LEFT JOIN equity_snapshots es ON ts.ticker = es.ticker AND ts.run_id = es.run_id
        WHERE ts.run_id = ? AND ts.total_score >= 50
        ORDER BY ts.total_score DESC
    """, (run_id,)).fetchall()

    if not rows:
        log.info("[trades] No opportunities >= 50")
        return []

    # Load macro context
    macro_row = conn.execute("""
        SELECT series_key, value, stress_flag FROM macro_snapshots WHERE run_id=?
    """, (run_id,)).fetchall()
    macro = {"stress_level": "GREEN", "vix_value": None, "yield_curve": None}
    stress_count = 0
    for r in macro_row:
        if r["stress_flag"]:
            stress_count += 1
        if r["series_key"] == "vix":
            macro["vix_value"] = r["value"]
        if r["series_key"] == "yield_curve_10y2y":
            macro["yield_curve"] = r["value"]
    macro["stress_level"] = "RED" if stress_count >= 4 else ("YELLOW" if stress_count >= 2 else "GREEN")

    results = []

    for row in rows:
        ticker = row["ticker"]
        price = row["price"] or 0
        score = row["total_score"]
        tier = row["conviction_tier"]

        snap_data = {
            "week52_position": row["week52_position"],
            "iv_30d": row["iv_30d"],
            "beta": row["beta"],
        }

        # Decide trade direction
        direction = _decide_direction(ticker, snap_data, macro)

        # Find best option
        opt = _find_best_option(ticker, price, direction)

        if opt:
            action = f"BUY {direction.upper()}"
            instrument = f"OTM {direction.title()} Option"
        else:
            action = "BUY"
            instrument = "ETF Shares"

        rec = TradeRec(
            ticker=ticker,
            run_id=run_id,
            total_score=score,
            conviction_tier=tier,
            action=action,
            instrument=instrument,
            underlying_price=price,
            option_leg=opt,
            position_pct=0.5 if tier == "WATCH" else 1.0,
            catalyst=row["catalyst"] or "",
            rationale=row["rationale"] or "",
        )

        # Generate P&L scenarios
        rec.pnl_scenarios = _build_pnl_scenarios(opt, price, position_size)

        # Generate thesis, triggers, risk check via Claude
        rec.thesis, rec.triggers, rec.risk_check = _generate_thesis_and_triggers(rec, macro)

        # Persist to DB
        conn.execute("""
            INSERT INTO trade_recommendations (
                run_id, created_at, ticker, total_score, conviction_tier,
                trade_action, instrument, strike, expiry, premium,
                delta, gamma, vega, underlying_price,
                thesis, triggers, pnl_scenarios, risk_check, position_pct
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            run_id, datetime.now(timezone.utc).isoformat(),
            ticker, score, tier,
            action, instrument,
            opt.strike if opt else None,
            opt.expiry if opt else None,
            opt.premium if opt else None,
            opt.delta if opt else None,
            opt.gamma if opt else None,
            opt.vega if opt else None,
            price,
            rec.thesis,
            json.dumps(rec.triggers),
            json.dumps(rec.pnl_scenarios),
            rec.risk_check,
            rec.position_pct,
        ))
        results.append(rec)
        log.info(f"[trades] {ticker}: {action} {instrument}" + (f" ${opt.strike} {opt.expiry}" if opt else ""))

    conn.commit()
    log.info(f"[trades] Built {len(results)} recommendations")
    return results
