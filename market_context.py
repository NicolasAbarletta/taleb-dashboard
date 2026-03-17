# -*- coding: utf-8 -*-
"""
market_context.py -- Intelligence Layer
=========================================
Computes rich analytical context for each trade recommendation:
- Market regime detection (macro environment classification)
- Historical analogs (similar setups in the past year and their outcomes)
- Cross-asset signals (divergences, rotations, non-obvious relationships)
- Intelligence brief assembly (structured text for Claude prompts)

This module sits between data collection (agent.py) and AI analysis
(scorer.py / trade_builder.py), giving Claude the context it needs
to make genuinely insightful, non-obvious arguments.
"""

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger("market_context")


# ── Regime Detection ─────────────────────────────────────────────────────────

@dataclass
class MarketRegime:
    vix_level: float | None = None
    vix_regime: str = "unknown"          # complacent / calm / elevated / fear / panic
    credit_regime: str = "unknown"       # tight / normal / loose
    yield_curve_regime: str = "unknown"  # steep / normal / flat / inverted
    overall_regime: str = "unknown"      # risk-on complacency / normal / stress building / crisis
    stress_count: int = 0
    narrative: str = ""


def detect_regime(macro_indicators: dict) -> MarketRegime:
    """Classify current market environment from FRED macro data.

    Args:
        macro_indicators: dict of series_key -> {value, z_score, stress_flag}
    """
    r = MarketRegime()

    # VIX regime
    vix = macro_indicators.get("vix", {})
    r.vix_level = vix.get("value")
    if r.vix_level is not None:
        v = r.vix_level
        if v < 15:
            r.vix_regime = "complacent"
        elif v < 20:
            r.vix_regime = "calm"
        elif v < 30:
            r.vix_regime = "elevated"
        elif v < 50:
            r.vix_regime = "fear"
        else:
            r.vix_regime = "panic"

    # Credit regime (BAA spread z-score)
    credit = macro_indicators.get("baa_aaa_spread", {})
    cz = credit.get("z_score")
    if cz is not None:
        if cz > 1.5:
            r.credit_regime = "stressed"
        elif cz > 0.5:
            r.credit_regime = "tightening"
        elif cz < -1.0:
            r.credit_regime = "loose"
        else:
            r.credit_regime = "normal"

    # Yield curve regime
    yc = macro_indicators.get("yield_curve_10y2y", {})
    yc_val = yc.get("value")
    if yc_val is not None:
        if yc_val > 1.5:
            r.yield_curve_regime = "steep"
        elif yc_val > 0:
            r.yield_curve_regime = "normal"
        elif yc_val > -0.2:
            r.yield_curve_regime = "flat"
        else:
            r.yield_curve_regime = "inverted"

    # Count stress flags
    r.stress_count = sum(
        1 for v in macro_indicators.values()
        if isinstance(v, dict) and v.get("stress_flag")
    )

    # Overall regime classification
    if r.stress_count >= 4 or r.vix_regime in ("fear", "panic"):
        r.overall_regime = "crisis mode"
    elif r.stress_count >= 2 or r.vix_regime == "elevated":
        r.overall_regime = "stress building"
    elif r.vix_regime == "complacent" and r.credit_regime in ("normal", "loose"):
        r.overall_regime = "risk-on complacency"
    else:
        r.overall_regime = "normal"

    # Build narrative
    parts = []
    if r.vix_level is not None:
        parts.append(f"VIX at {r.vix_level:.1f} ({r.vix_regime})")
    if r.credit_regime != "unknown":
        cv = credit.get("value")
        parts.append(f"Credit {r.credit_regime}" + (f" (spread {cv:.2f})" if cv else ""))
    if yc_val is not None:
        parts.append(f"Yield curve {r.yield_curve_regime} ({yc_val:+.2f})")

    fed = macro_indicators.get("fed_funds_rate", {})
    if fed.get("value") is not None:
        parts.append(f"Fed rate {fed['value']:.2f}%")

    if r.overall_regime == "risk-on complacency":
        parts.append("Markets pricing zero tail risk -- cheap insurance available")
    elif r.overall_regime == "crisis mode":
        parts.append("Multiple stress signals active -- tail hedges paying off")
    elif r.overall_regime == "stress building":
        parts.append("Early warning signs -- smart money hedging")

    r.narrative = ". ".join(parts) + "."
    return r


# ── Historical Analogs ───────────────────────────────────────────────────────

@dataclass
class HistoricalAnalog:
    match_count: int = 0
    avg_return_20d: float | None = None
    avg_return_40d: float | None = None
    avg_return_60d: float | None = None
    win_rate_40d: float | None = None
    best_return: float | None = None
    worst_return: float | None = None
    narrative: str = "Insufficient historical data for analog matching."


def compute_historical_analog(
    ticker: str,
    hist_df: pd.DataFrame | None,
    current_w52_position: float | None,
    current_iv: float | None,
) -> HistoricalAnalog:
    """Find similar setups in the trailing 1-year history and compute forward returns.

    Looks for past moments where the asset was at a similar position in its
    52-week range AND had similar implied volatility, then measures what
    happened 20/40/60 trading days later.
    """
    analog = HistoricalAnalog()

    if hist_df is None or hist_df.empty or current_w52_position is None:
        return analog

    try:
        closes = hist_df["Close"].dropna()
        if len(closes) < 120:
            analog.narrative = f"Only {len(closes)} days of history -- too short for reliable analog matching."
            return analog

        # Compute rolling 52-week position for each historical day
        window = min(252, len(closes))
        rolling_hi = closes.rolling(window=window, min_periods=60).max()
        rolling_lo = closes.rolling(window=window, min_periods=60).min()
        w52_pos_series = (closes - rolling_lo) / (rolling_hi - rolling_lo)
        w52_pos_series = w52_pos_series.dropna()

        if len(w52_pos_series) < 80:
            return analog

        # Find similar setups: within 10% of current 52w position
        tolerance = 0.10
        similar_mask = (
            (w52_pos_series >= current_w52_position - tolerance) &
            (w52_pos_series <= current_w52_position + tolerance)
        )

        # Exclude the last 20 days (too recent, overlaps with current)
        similar_mask.iloc[-20:] = False

        similar_indices = w52_pos_series[similar_mask].index
        if len(similar_indices) < 3:
            analog.narrative = f"Only {len(similar_indices)} similar setups found -- need at least 3 for reliable analogs."
            return analog

        # Compute forward returns from each similar setup
        fwd_20, fwd_40, fwd_60 = [], [], []

        for idx in similar_indices:
            pos = closes.index.get_loc(idx)
            if pos + 60 < len(closes):
                fwd_60.append(float((closes.iloc[pos + 60] / closes.iloc[pos] - 1) * 100))
            if pos + 40 < len(closes):
                fwd_40.append(float((closes.iloc[pos + 40] / closes.iloc[pos] - 1) * 100))
            if pos + 20 < len(closes):
                fwd_20.append(float((closes.iloc[pos + 20] / closes.iloc[pos] - 1) * 100))

        analog.match_count = len(similar_indices)

        if fwd_20:
            analog.avg_return_20d = round(np.mean(fwd_20), 1)
        if fwd_40:
            analog.avg_return_40d = round(np.mean(fwd_40), 1)
            analog.win_rate_40d = round(sum(1 for r in fwd_40 if r > 0) / len(fwd_40) * 100, 0)
            analog.best_return = round(max(fwd_40), 1)
            analog.worst_return = round(min(fwd_40), 1)
        if fwd_60:
            analog.avg_return_60d = round(np.mean(fwd_60), 1)

        # Build narrative
        if analog.avg_return_40d is not None:
            direction = "positive" if analog.avg_return_40d > 0 else "negative"
            analog.narrative = (
                f"{analog.match_count} similar setups in the past year "
                f"(similar 52-week position within {tolerance*100:.0f}%). "
                f"Average 40-day forward return: {analog.avg_return_40d:+.1f}% "
                f"(win rate: {analog.win_rate_40d:.0f}%). "
                f"Best: {analog.best_return:+.1f}%, Worst: {analog.worst_return:+.1f}%. "
                f"Historical bias: {direction}."
            )
        else:
            analog.narrative = (
                f"{analog.match_count} similar setups found but insufficient forward data "
                f"to compute reliable return statistics."
            )

    except Exception as e:
        log.debug(f"[analog] {ticker}: {e}")

    return analog


# ── Cross-Asset Signals ──────────────────────────────────────────────────────

@dataclass
class CrossAssetSignals:
    gold_divergence: str = ""
    vol_term_structure: str = ""
    sector_rotation: str = ""
    commodity_complex: str = ""
    em_vs_dm: str = ""
    crowd_focus: str = ""
    narrative: str = ""


def compute_cross_asset_signals(all_snaps: list[dict], regime: MarketRegime) -> CrossAssetSignals:
    """Identify non-obvious cross-asset relationships from current snapshots."""
    sig = CrossAssetSignals()

    if not all_snaps:
        return sig

    # Index snapshots by ticker for easy lookup
    by_ticker = {s["ticker"]: s for s in all_snaps}

    # ── Gold divergence ──
    # Gold at highs while other assets at lows = hidden stress
    gld = by_ticker.get("GLD", {})
    spy = by_ticker.get("SPY", {})
    gld_pos = gld.get("week52_position")
    spy_pos = spy.get("week52_position")

    if gld_pos is not None and spy_pos is not None:
        if gld_pos > 0.75 and spy_pos < 0.40:
            sig.gold_divergence = "Gold near highs while equities are weak -- hidden stress signal. Smart money is hedging."
        elif gld_pos > 0.75 and spy_pos > 0.75:
            sig.gold_divergence = "Gold AND equities both near highs -- unusual. Something has to give."
        elif gld_pos < 0.30 and spy_pos > 0.75:
            sig.gold_divergence = "Gold weak while equities strong -- pure risk-on environment. No fear pricing."
        else:
            sig.gold_divergence = f"Gold at {gld_pos:.0%} of range, SPY at {spy_pos:.0%} -- no extreme divergence."

    # ── VIX term structure ──
    # VIXY (short-term) vs VIXM (mid-term) -- contango vs backwardation
    vixy = by_ticker.get("VIXY", {})
    vixm = by_ticker.get("VIXM", {})
    vixy_ret = vixy.get("return_20d")
    vixm_ret = vixm.get("return_20d")

    if vixy_ret is not None and vixm_ret is not None:
        if vixy_ret < vixm_ret - 3:
            sig.vol_term_structure = "VIX term structure in steep contango (short-term vol crushed). Tail protection is CHEAP."
        elif vixy_ret > vixm_ret + 3:
            sig.vol_term_structure = "VIX term structure in backwardation (near-term fear > long-term). Market already pricing risk."
        else:
            sig.vol_term_structure = "VIX term structure normal -- no extreme signal."

    # ── Sector rotation ──
    sector_tickers = {
        "Energy": "XLE", "Financials": "XLF", "Utilities": "XLU",
        "Healthcare": "XLV", "Biotech": "XBI", "Defense": "ITA",
    }
    sectors_at_highs, sectors_at_lows = [], []
    for name, tk in sector_tickers.items():
        s = by_ticker.get(tk, {})
        pos = s.get("week52_position")
        if pos is not None:
            if pos > 0.80:
                sectors_at_highs.append(name)
            elif pos < 0.25:
                sectors_at_lows.append(name)

    if sectors_at_highs or sectors_at_lows:
        parts = []
        if sectors_at_highs:
            parts.append(f"Money crowding into: {', '.join(sectors_at_highs)}")
        if sectors_at_lows:
            parts.append(f"Money abandoning: {', '.join(sectors_at_lows)}")
        sig.sector_rotation = ". ".join(parts) + "."
    else:
        sig.sector_rotation = "No extreme sector rotation detected."

    # ── Commodity complex ──
    commodity_tickers = ["GLD", "SLV", "USO", "UNG", "CPER", "PALL", "PPLT", "WEAT", "CORN"]
    bottoming, topping = [], []
    for tk in commodity_tickers:
        s = by_ticker.get(tk, {})
        pos = s.get("week52_position")
        if pos is not None:
            if pos < 0.25:
                bottoming.append(tk)
            elif pos > 0.80:
                topping.append(tk)

    if len(bottoming) >= 3:
        sig.commodity_complex = f"Multiple commodities bottoming together ({', '.join(bottoming)}) -- potential inflation signal the crowd is ignoring."
    elif len(topping) >= 3:
        sig.commodity_complex = f"Multiple commodities at highs ({', '.join(topping)}) -- inflation already priced in."
    elif bottoming:
        sig.commodity_complex = f"Some commodities bottoming ({', '.join(bottoming)}) -- selective opportunities."
    else:
        sig.commodity_complex = "Commodity complex showing no extreme cluster signals."

    # ── EM vs DM ──
    em_tickers = ["EEM", "EIDO", "EWZ", "EWY", "EWT", "TUR", "EZA", "ARGT"]
    dm_tickers = ["SPY", "QQQ", "EWU", "EWG", "EWJ"]
    em_positions = [by_ticker.get(t, {}).get("week52_position") for t in em_tickers]
    dm_positions = [by_ticker.get(t, {}).get("week52_position") for t in dm_tickers]
    em_avg = np.mean([p for p in em_positions if p is not None]) if any(p is not None for p in em_positions) else None
    dm_avg = np.mean([p for p in dm_positions if p is not None]) if any(p is not None for p in dm_positions) else None

    if em_avg is not None and dm_avg is not None:
        gap = dm_avg - em_avg
        if gap > 0.25:
            sig.em_vs_dm = f"Developed markets outperforming EM by wide margin (DM avg {dm_avg:.0%} vs EM avg {em_avg:.0%}). EM is unloved -- contrarian opportunity."
        elif gap < -0.15:
            sig.em_vs_dm = f"EM outperforming DM -- unusual risk appetite for frontier markets."
        else:
            sig.em_vs_dm = f"DM and EM roughly in sync (DM {dm_avg:.0%}, EM {em_avg:.0%})."

    # ── Crowd focus ──
    # Identify what's near highs with high volume (crowd is there)
    crowded = []
    neglected = []
    for s in all_snaps:
        pos = s.get("week52_position")
        vol_trend = s.get("volume_trend_20d")
        if pos is not None and pos > 0.80 and vol_trend is not None and vol_trend > 0:
            crowded.append(s["ticker"])
        elif pos is not None and pos < 0.30 and vol_trend is not None and vol_trend < -0.01:
            neglected.append(s["ticker"])

    if crowded:
        sig.crowd_focus = f"Crowd is chasing: {', '.join(crowded[:5])}. "
    if neglected:
        sig.crowd_focus += f"Nobody is watching: {', '.join(neglected[:5])}."
    if not crowded and not neglected:
        sig.crowd_focus = "No extreme crowd positioning detected."

    # Build composite narrative
    parts = [s for s in [
        sig.gold_divergence, sig.vol_term_structure, sig.sector_rotation,
        sig.commodity_complex, sig.em_vs_dm, sig.crowd_focus,
    ] if s]
    sig.narrative = " ".join(parts)
    return sig


# ── Intelligence Brief Builder ───────────────────────────────────────────────

def build_intelligence_brief(
    ticker: str,
    snap: dict,
    regime: MarketRegime,
    analog: HistoricalAnalog,
    cross_signals: CrossAssetSignals,
) -> str:
    """Assemble all intelligence into a structured brief for Claude prompts.

    This is the key output -- a rich text block that gives Claude enough
    context to make genuinely insightful, non-obvious arguments rather
    than generic commentary.
    """
    sections = []

    # ── REGIME section ──
    sections.append(f"MARKET REGIME: {regime.overall_regime.upper()}. {regime.narrative}")

    # ── ASSET PROFILE section ──
    profile_parts = []
    w52 = snap.get("week52_position")
    if w52 is not None:
        profile_parts.append(f"52-week position: {w52:.0%}")

    dd = snap.get("drawdown_from_peak")
    if dd is not None:
        profile_parts.append(f"Drawdown from peak: {dd:+.1f}%")

    rv = snap.get("realized_vol_20d")
    iv = snap.get("iv_30d")
    if rv is not None and iv is not None:
        spread = rv - iv
        if spread < -0.05:
            profile_parts.append(f"Realized vol {rv:.1%} vs implied {iv:.1%} -- options are CHEAP relative to actual moves")
        elif spread > 0.05:
            profile_parts.append(f"Realized vol {rv:.1%} vs implied {iv:.1%} -- options are expensive")
        else:
            profile_parts.append(f"Realized vol {rv:.1%} vs implied {iv:.1%} -- fairly priced")
    elif rv is not None:
        profile_parts.append(f"Realized vol: {rv:.1%}")

    corr = snap.get("spy_correlation_60d")
    if corr is not None:
        if corr < 0.3:
            profile_parts.append(f"SPY correlation: {corr:.2f} (LOW -- excellent portfolio diversifier)")
        elif corr > 0.8:
            profile_parts.append(f"SPY correlation: {corr:.2f} (HIGH -- moves with broad market)")
        else:
            profile_parts.append(f"SPY correlation: {corr:.2f}")

    skew = snap.get("skewness_60d")
    if skew is not None:
        if skew > 0.3:
            profile_parts.append(f"Return skewness: {skew:+.2f} (positive -- Taleb-friendly convex profile)")
        elif skew < -0.3:
            profile_parts.append(f"Return skewness: {skew:+.2f} (negative -- concave, anti-Taleb)")
        else:
            profile_parts.append(f"Return skewness: {skew:+.2f}")

    vol_trend = snap.get("volume_trend_20d")
    if vol_trend is not None:
        if vol_trend < -0.02:
            profile_parts.append("Volume declining -- crowd losing interest (contrarian positive)")
        elif vol_trend > 0.02:
            profile_parts.append("Volume rising -- attention increasing")

    ret_20 = snap.get("return_20d")
    ret_60 = snap.get("return_60d")
    if ret_20 is not None:
        profile_parts.append(f"20-day return: {ret_20:+.1f}%")
    if ret_60 is not None:
        profile_parts.append(f"60-day return: {ret_60:+.1f}%")

    if profile_parts:
        sections.append(f"ASSET PROFILE ({ticker}): " + ". ".join(profile_parts) + ".")

    # ── HISTORICAL ANALOG section ──
    sections.append(f"HISTORICAL ANALOG: {analog.narrative}")

    # ── CROSS-ASSET section ──
    # Pick the most relevant cross-asset signals (not all)
    cross_parts = []
    if cross_signals.gold_divergence and "extreme" not in cross_signals.gold_divergence.lower():
        cross_parts.append(cross_signals.gold_divergence)
    if cross_signals.vol_term_structure and "normal" not in cross_signals.vol_term_structure.lower():
        cross_parts.append(cross_signals.vol_term_structure)
    if cross_signals.commodity_complex and "no extreme" not in cross_signals.commodity_complex.lower():
        cross_parts.append(cross_signals.commodity_complex)
    if cross_signals.crowd_focus:
        cross_parts.append(cross_signals.crowd_focus)
    if cross_signals.em_vs_dm and "sync" not in cross_signals.em_vs_dm.lower():
        cross_parts.append(cross_signals.em_vs_dm)

    if cross_parts:
        sections.append("CROSS-ASSET CONTEXT: " + " ".join(cross_parts[:3]))

    return "\n\n".join(sections)


# ── Convenience: load all from DB ────────────────────────────────────────────

def load_snap_dicts(conn, run_id: str) -> list[dict]:
    """Load equity snapshots as list of dicts (including new derived columns)."""
    rows = conn.execute("""
        SELECT ticker, price, price_change_1d, volume_ratio,
               week52_position, iv_30d, iv_1y_percentile,
               short_interest, beta,
               realized_vol_20d, return_20d, return_60d,
               skewness_60d, spy_correlation_60d,
               volume_trend_20d, drawdown_from_peak
        FROM equity_snapshots WHERE run_id=?
    """, (run_id,)).fetchall()
    return [dict(r) for r in rows]


def load_macro_dict(conn, run_id: str) -> dict:
    """Load macro snapshots as dict of series_key -> {value, z_score, stress_flag}."""
    rows = conn.execute(
        "SELECT series_key, value, z_score_2y, stress_flag FROM macro_snapshots WHERE run_id=?",
        (run_id,),
    ).fetchall()
    return {
        r["series_key"]: {
            "value": r["value"],
            "z_score": r["z_score_2y"],
            "stress_flag": r["stress_flag"],
        }
        for r in rows
    }
