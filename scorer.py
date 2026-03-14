# -*- coding: utf-8 -*-
"""
scorer.py -- Phase 2: Taleb Scoring Engine
============================================
Four filters: Convexity, Antifragility, Fragility Avoidance, Tail Risk.
Writes scored results to taleb_scores table.
"""

import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone

import anthropic
from dotenv import load_dotenv

load_dotenv(override=True)
log = logging.getLogger("scorer")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


@dataclass
class Snap:
    ticker: str
    price: float | None
    price_change_1d: float | None
    volume_ratio: float | None
    week52_position: float | None
    iv_30d: float | None
    iv_1y_percentile: float | None
    short_interest: float | None
    beta: float | None


@dataclass
class Macro:
    stress_count: int = 0
    stress_level: str = "GREEN"
    vix_value: float | None = None
    yield_curve: float | None = None
    credit_spread: float | None = None
    geo_news_score: float = 0.0
    indicators: dict = field(default_factory=dict)


@dataclass
class Score:
    ticker: str
    run_id: str
    convexity: int = 0
    antifragility: int = 0
    fragility_avoidance: int = 0
    tail_risk: int = 0
    total: int = 0
    tier: str = "IGNORE"
    catalyst: str = ""
    rationale: str = ""
    price: float | None = None
    iv_percentile: float | None = None
    details: dict = field(default_factory=dict)


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_snaps(conn, run_id: str) -> list[Snap]:
    rows = conn.execute("""
        SELECT ticker, price, price_change_1d, volume_ratio,
               week52_position, iv_30d, iv_1y_percentile,
               short_interest, beta
        FROM equity_snapshots WHERE run_id=?
    """, (run_id,)).fetchall()
    return [Snap(**dict(r)) for r in rows]


def _load_macro(conn, run_id: str) -> Macro:
    rows = conn.execute(
        "SELECT series_key, value, z_score_2y, stress_flag FROM macro_snapshots WHERE run_id=?",
        (run_id,),
    ).fetchall()
    m = Macro()
    for r in rows:
        m.indicators[r["series_key"]] = {"value": r["value"], "z_score": r["z_score_2y"], "stress_flag": r["stress_flag"]}
        if r["stress_flag"]:
            m.stress_count += 1
        if r["series_key"] == "vix":
            m.vix_value = r["value"]
        if r["series_key"] == "yield_curve_10y2y":
            m.yield_curve = r["value"]
        if r["series_key"] == "baa_aaa_spread":
            m.credit_spread = r["value"]

    m.stress_level = "RED" if m.stress_count >= 4 else ("YELLOW" if m.stress_count >= 2 else "GREEN")

    row = conn.execute(
        "SELECT AVG(novelty_score) as ns, AVG(1 - consensus_penalty) as cp FROM news_signals WHERE run_id=?",
        (run_id,),
    ).fetchone()
    if row and row["ns"] is not None:
        m.geo_news_score = round(float(row["ns"]) * 0.6 + float(row["cp"] or 0) * 0.4, 3)
    return m


# ── Filter 1: Convexity (0-30) ───────────────────────────────────────────────

def _score_convexity(s: Snap) -> tuple[int, dict]:
    sc, d = 15, {}
    if s.week52_position is not None:
        p = s.week52_position
        if p <= 0.15:
            sc += 12; d["w52"] = "+12 (at 52w low)"
        elif p <= 0.30:
            sc += 8; d["w52"] = "+8 (near 52w low)"
        elif p <= 0.50:
            sc += 3; d["w52"] = "+3 (mid-range)"
        elif p >= 0.85:
            sc -= 10; d["w52"] = "-10 (near 52w high)"
        elif p >= 0.70:
            sc -= 5; d["w52"] = "-5 (elevated)"
    if s.iv_30d is not None:
        if s.iv_30d < 0.15:
            sc += 5; d["iv"] = "+5 (very low IV)"
        elif s.iv_30d < 0.25:
            sc += 3; d["iv"] = "+3 (low IV)"
        elif s.iv_30d > 0.60:
            sc -= 5; d["iv"] = "-5 (expensive options)"
    return max(0, min(30, sc)), d


# ── Filter 2: Antifragility (0-25) ───────────────────────────────────────────

def _score_antifragility(s: Snap, m: Macro) -> tuple[int, dict]:
    sc, d = 10, {}
    tk = s.ticker.upper()

    if tk in ("VIXY", "UVXY", "VIXM"):
        if m.vix_value is not None and m.vix_value < 18:
            sc += 12; d["vix_cheap"] = f"+12 (VIX={m.vix_value:.1f} cheap fear insurance)"
        elif m.vix_value is not None and m.vix_value < 22:
            sc += 7; d["vix_low"] = f"+7 (VIX={m.vix_value:.1f})"
        else:
            sc -= 3; d["vix_high"] = "-3 (VIX elevated)"

    if tk in ("ITA", "XAR", "HACK"):
        g = round(m.geo_news_score * 15)
        sc += g; d["defense"] = f"+{g} (geo={m.geo_news_score:.2f})"

    if tk in ("GLD", "SLV", "CPER", "USO", "UNG", "PDBC", "GUNR",
              "PALL", "PPLT", "SLX"):
        if m.stress_level in ("YELLOW", "RED"):
            sc += 8; d["commodity"] = f"+8 (stress={m.stress_level})"
        elif m.geo_news_score > 0.4:
            sc += 5; d["commodity"] = "+5 (geo elevated)"

    # Agriculture: benefits from supply chain disruption and geo conflict
    if tk in ("WEAT", "CORN", "DBA", "MOO"):
        if m.geo_news_score > 0.4:
            sc += 8; d["agri"] = f"+8 (food supply geo risk={m.geo_news_score:.2f})"
        elif m.stress_level in ("YELLOW", "RED"):
            sc += 5; d["agri"] = f"+5 (stress={m.stress_level})"

    # Uranium/Nuclear: neglected sector with massive optionality
    if tk in ("URA", "URNM"):
        sc += 6; d["nuclear"] = "+6 (neglected sector, structural demand)"
        if m.geo_news_score > 0.5:
            sc += 3; d["nuclear_geo"] = "+3 (energy security catalyst)"

    # Strategic materials: supply chain fragility = optionality
    if tk in ("REMX", "LIT"):
        sc += 5; d["strategic"] = "+5 (critical supply chain optionality)"
        if m.geo_news_score > 0.4:
            sc += 3; d["strategic_geo"] = "+3 (supply chain disruption risk)"

    # Bonds: flight-to-quality during stress (antifragile in crises)
    if tk in ("TLT", "TIP"):
        if m.stress_level in ("YELLOW", "RED"):
            sc += 8; d["bonds"] = f"+8 (flight to quality, stress={m.stress_level})"
        elif m.yield_curve is not None and m.yield_curve < 0:
            sc += 5; d["bonds"] = "+5 (inverted yield curve)"

    # Real assets / Infrastructure: tangible, inflation-resistant
    if tk in ("IFRA", "WOOD"):
        sc += 3; d["real_asset"] = "+3 (tangible real asset)"

    if s.beta is not None:
        if s.beta > 1.3:
            sc -= 8; d["beta"] = f"-8 (beta={s.beta:.2f})"
        elif s.beta > 0.9:
            sc -= 4; d["beta"] = f"-4 (beta={s.beta:.2f})"
        elif s.beta < 0.3:
            sc += 5; d["beta"] = f"+5 (beta={s.beta:.2f})"
        elif s.beta < 0:
            sc += 8; d["beta"] = f"+8 (beta={s.beta:.2f})"

    if tk in ("EIDO", "EWZ", "EWY", "EWT", "TUR", "EZA", "ARGT") and m.geo_news_score > 0.5:
        sc += 4; d["frontier"] = "+4 (geo catalyst)"

    return max(0, min(25, sc)), d


# ── Filter 3: Fragility Avoidance (0-25) ─────────────────────────────────────

def _score_fragility(s: Snap, m: Macro) -> tuple[int, dict]:
    sc, d = 15, {}
    tk = s.ticker.upper()

    if s.short_interest is not None:
        if s.short_interest > 0.20:
            sc -= 10; d["si"] = f"-10 (SI={s.short_interest:.1%})"
        elif s.short_interest > 0.10:
            sc -= 5; d["si"] = f"-5 (SI={s.short_interest:.1%})"
        elif s.short_interest < 0.03:
            sc += 5; d["si"] = "+5 (very low SI)"

    if tk in ("SPY", "QQQ", "VTI", "EFA"):
        sc -= 8; d["consensus"] = "-8 (consensus ETF)"
    elif tk in ("GLD", "IWM", "TLT"):
        sc -= 3; d["popular"] = "-3 (well-covered)"

    if tk in ("CPER", "DJP", "GUNR", "MOO", "WEAT", "CORN", "DBA",
              "URA", "URNM", "REMX", "LIT", "PALL", "PPLT",
              "IFRA", "WOOD", "SLX", "FM", "ARGT"):
        sc += 7; d["uncrowded"] = "+7 (low coverage)"
    elif tk in ("TUR", "EZA", "EIDO"):
        sc += 5; d["uncrowded"] = "+5 (under-followed EM)"

    if s.volume_ratio is not None:
        if s.volume_ratio > 3.0:
            sc -= 5; d["vol"] = f"-5 (volume {s.volume_ratio:.1f}x avg)"
        elif s.volume_ratio > 2.0:
            sc -= 2; d["vol"] = f"-2 (volume {s.volume_ratio:.1f}x avg)"

    return max(0, min(25, sc)), d


# ── Filter 4: Tail Risk Asymmetry (0-20) ─────────────────────────────────────

def _score_tail_risk(s: Snap) -> tuple[int, dict]:
    sc, d = 5, {}
    pct = s.iv_1y_percentile

    if pct is None:
        iv = s.iv_30d
        if iv is not None:
            if iv < 0.15:
                sc += 8; d["iv_raw"] = f"+8 (raw IV={iv:.2f} low, no history)"
            elif iv < 0.25:
                sc += 4; d["iv_raw"] = f"+4 (raw IV={iv:.2f}, no history)"
            else:
                d["iv_raw"] = f"0 (raw IV={iv:.2f})"
        else:
            d["no_iv"] = "0 (no IV data)"
        return max(0, min(20, sc)), d

    if pct <= 10:
        sc += 15; d["pct"] = f"+15 (IV {pct:.0f}th pct)"
    elif pct <= 20:
        sc += 12; d["pct"] = f"+12 (IV {pct:.0f}th pct)"
    elif pct <= 35:
        sc += 8; d["pct"] = f"+8 (IV {pct:.0f}th pct)"
    elif pct <= 50:
        sc += 4; d["pct"] = f"+4 (IV {pct:.0f}th pct)"
    elif pct >= 80:
        sc -= 3; d["pct"] = f"-3 (IV {pct:.0f}th pct expensive)"

    return max(0, min(20, sc)), d


# ── Catalyst ──────────────────────────────────────────────────────────────────

def _catalyst(s: Snap, m: Macro) -> str:
    parts = []
    tk = s.ticker.upper()
    if s.week52_position is not None and s.week52_position <= 0.20:
        parts.append("Near 52w low")
    if s.iv_1y_percentile is not None and s.iv_1y_percentile <= 20:
        parts.append(f"IV {s.iv_1y_percentile:.0f}th pct")
    elif s.iv_30d is not None and s.iv_30d < 0.20:
        parts.append(f"Low IV ({s.iv_30d:.2f})")
    if tk in ("VIXY", "UVXY", "VIXM") and m.vix_value and m.vix_value < 18:
        parts.append(f"VIX={m.vix_value:.1f}")
    if tk in ("ITA", "XAR", "HACK") and m.geo_news_score > 0.4:
        parts.append("Geo risk elevated")
    if m.stress_level in ("YELLOW", "RED") and tk in ("GLD", "SLV", "CPER", "PALL", "PPLT"):
        parts.append(f"Macro stress {m.stress_level}")
    if tk in ("URA", "URNM"):
        parts.append("Nuclear optionality")
    if tk in ("REMX", "LIT"):
        parts.append("Supply chain fragility")
    if tk in ("WEAT", "CORN", "DBA") and m.geo_news_score > 0.3:
        parts.append("Food supply risk")
    if tk in ("TLT", "TIP") and m.stress_level in ("YELLOW", "RED"):
        parts.append("Flight to quality")
    if tk in ("TUR", "EZA", "ARGT", "FM") and m.geo_news_score > 0.4:
        parts.append("Frontier geo catalyst")
    return " | ".join(parts[:2]) if parts else "Technical setup"


# ── Claude rationale ──────────────────────────────────────────────────────────

def _rationale(sc: Score, m: Macro) -> str:
    if not ANTHROPIC_API_KEY:
        return (
            f"{sc.ticker} scores {sc.total}/100. Catalyst: {sc.catalyst}. "
            f"Macro: {m.stress_level}, VIX={m.vix_value}. "
            f"[Set ANTHROPIC_API_KEY for AI rationales]"
        )
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=250,
            messages=[{"role": "user", "content": f"""Write a 3-sentence trade thesis for {sc.ticker} like a senior trader explaining to a smart friend.
No jargon. Be specific about WHY now and what could go wrong.
Context: Score {sc.total}/100, Catalyst: {sc.catalyst}, VIX: {m.vix_value}, Macro: {m.stress_level}, Yield curve: {m.yield_curve}.
Tone: blunt, direct, skeptical of consensus. Not a memo -- a conversation."""}],
        )
        return msg.content[0].text.strip()
    except Exception as e:
        log.error(f"[Claude] {sc.ticker}: {e}")
        return f"Asymmetric setup in {sc.ticker} ({sc.total}/100). {sc.catalyst}. [Rationale failed: {e}]"


# ── Main scorer ───────────────────────────────────────────────────────────────

def score_all(conn, run_id: str | None = None) -> list[Score]:
    from database import latest_run_id
    if run_id is None:
        run_id = latest_run_id(conn)
    if not run_id:
        log.warning("[scorer] No completed run found")
        return []

    log.info(f"[scorer] Scoring {run_id}")
    snaps = _load_snaps(conn, run_id)
    macro = _load_macro(conn, run_id)
    log.info(f"[scorer] {len(snaps)} assets | stress={macro.stress_level} | geo={macro.geo_news_score:.2f}")

    results = []
    opps = 0

    for s in snaps:
        c, cd = _score_convexity(s)
        a, ad = _score_antifragility(s, macro)
        f, fd = _score_fragility(s, macro)
        t, td = _score_tail_risk(s)
        total = c + a + f + t
        tier = "HIGH" if total >= 65 else ("WATCH" if total >= 40 else "IGNORE")
        cat = _catalyst(s, macro)

        sc = Score(
            ticker=s.ticker, run_id=run_id,
            convexity=c, antifragility=a, fragility_avoidance=f, tail_risk=t,
            total=total, tier=tier, catalyst=cat,
            price=s.price, iv_percentile=s.iv_1y_percentile,
            details={**cd, **ad, **fd, **td},
        )

        if tier in ("WATCH", "HIGH"):
            sc.rationale = _rationale(sc, macro)
            opps += 1

        conn.execute("""
            INSERT INTO taleb_scores (run_id,scored_at,ticker,total_score,convexity_score,
                antifragility_score,fragility_avoidance,tail_risk_score,
                conviction_tier,catalyst,rationale,price,iv_percentile,score_details)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            run_id, datetime.now(timezone.utc).isoformat(), s.ticker, total,
            c, a, f, t, tier, cat, sc.rationale, s.price, sc.iv_percentile,
            json.dumps(sc.details),
        ))
        results.append(sc)

    conn.commit()
    conn.execute("UPDATE agent_runs SET opportunities=? WHERE run_id=?", (opps, run_id))
    conn.commit()

    hi = sum(1 for r in results if r.tier == "HIGH")
    wa = sum(1 for r in results if r.tier == "WATCH")
    log.info(f"[scorer] Done -- {hi} HIGH, {wa} WATCH, {len(results) - hi - wa} ignored")
    return results
