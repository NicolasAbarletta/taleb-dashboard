# -*- coding: utf-8 -*-
"""
agent.py -- Phase 1: Live Data Agent
======================================
Pulls live market data from yfinance, FRED, and NewsAPI every 30 minutes.
Stores everything in SQLite via database.py.
Resilience: each source is isolated -- if one fails, the others continue.
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv(override=True)

log = logging.getLogger("agent")

FRED_API_KEY = os.getenv("FRED_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# ── Asset Universe ────────────────────────────────────────────────────────────

EQUITY_UNIVERSE = [
    # Broad market
    "SPY", "QQQ", "IWM", "VTI", "EFA", "EEM",
    # Volatility
    "VIXY", "UVXY", "SVXY",
    # Commodities
    "GLD", "SLV", "USO", "UNG", "CPER", "PDBC",
    # Defense / Cyber
    "ITA", "XAR", "HACK",
    # International
    "EIDO", "EWZ", "EWY", "EWT", "EWU", "EWG", "EWJ",
    # Sector
    "XLE", "XLF", "XLU", "XLV", "XBI",
    # Hard assets
    "DJP", "GUNR", "MOO",
]

FRED_SERIES = {
    "fed_funds_rate":      "DFF",
    "baa_aaa_spread":      "BAA10Y",
    "ted_spread":          "TEDRATE",
    "yield_curve_10y2y":   "T10Y2Y",
    "unemployment_claims": "ICSA",
    "vix":                 "VIXCLS",
}

NEWS_KEYWORDS = [
    "geopolitical risk", "supply chain", "sanctions", "conflict",
    "central bank", "systemic risk", "black swan", "tail risk",
    "regime change",
]

MAJOR_OUTLETS = {
    "reuters.com", "bloomberg.com", "wsj.com", "ft.com", "cnbc.com",
    "nytimes.com", "bbc.com", "apnews.com", "theguardian.com", "economist.com",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(val) -> float | None:
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return None


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")


# ── yfinance ──────────────────────────────────────────────────────────────────

def _fetch_iv_from_options(ticker_obj: yf.Ticker, price: float) -> dict:
    result = {"iv_30d": None, "iv_60d": None, "iv_90d": None}
    try:
        expirations = ticker_obj.options
        if not expirations:
            return result

        today = datetime.now().date()
        buckets = {"iv_30d": None, "iv_60d": None, "iv_90d": None}
        bucket_ranges = {"iv_30d": (15, 45), "iv_60d": (46, 75), "iv_90d": (76, 120)}

        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if dte < 1:
                continue
            for key, (lo, hi) in bucket_ranges.items():
                if lo <= dte <= hi and buckets[key] is None:
                    try:
                        chain = ticker_obj.option_chain(exp_str)
                        calls, puts = chain.calls, chain.puts
                        atm_calls = calls[
                            (calls["strike"] >= price * 0.97) &
                            (calls["strike"] <= price * 1.03) &
                            (calls["impliedVolatility"] > 0)
                        ]
                        atm_puts = puts[
                            (puts["strike"] >= price * 0.97) &
                            (puts["strike"] <= price * 1.03) &
                            (puts["impliedVolatility"] > 0)
                        ]
                        ivs = pd.concat([atm_calls["impliedVolatility"], atm_puts["impliedVolatility"]])
                        if len(ivs) > 0:
                            buckets[key] = float(ivs.median())
                    except Exception:
                        pass
        return buckets
    except Exception:
        return result


def _iv_percentile(ticker: str, current_iv: float | None, conn) -> float | None:
    if current_iv is None:
        return None
    try:
        one_year_ago = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
        cur = conn.execute(
            "SELECT iv_30d FROM equity_snapshots WHERE ticker=? AND timestamp>=? AND iv_30d IS NOT NULL",
            (ticker, one_year_ago),
        )
        rows = cur.fetchall()
        if len(rows) < 10:
            return None
        historical = [r["iv_30d"] for r in rows]
        return round(sum(1 for v in historical if v < current_iv) / len(historical) * 100, 1)
    except Exception:
        return None


def fetch_equity_data(conn, run_id: str) -> int:
    log.info(f"[yfinance] Fetching {len(EQUITY_UNIVERSE)} tickers")
    count = 0
    for sym in EQUITY_UNIVERSE:
        try:
            t = yf.Ticker(sym)
            info = t.info or {}
            hist = t.history(period="1y", auto_adjust=True)
            if hist.empty:
                log.warning(f"[yfinance] No data for {sym}")
                continue

            price = _safe_float(hist["Close"].iloc[-1])
            if price is None:
                continue

            price_1d = _safe_float(hist["Close"].iloc[-2]) if len(hist) > 1 else None
            change_1d = round((price - price_1d) / price_1d * 100, 2) if price_1d else None
            vol = _safe_float(hist["Volume"].iloc[-1])
            avg_vol = _safe_float(hist["Volume"].tail(20).mean())
            vol_ratio = round(vol / avg_vol, 2) if (vol and avg_vol and avg_vol > 0) else None
            w52_hi = _safe_float(hist["Close"].max())
            w52_lo = _safe_float(hist["Close"].min())
            w52_pos = round((price - w52_lo) / (w52_hi - w52_lo), 3) if (w52_hi and w52_lo and w52_hi != w52_lo) else None

            iv = _fetch_iv_from_options(t, price)
            iv_pct = _iv_percentile(sym, iv.get("iv_30d"), conn)

            conn.execute("""
                INSERT INTO equity_snapshots (
                    run_id, timestamp, ticker, price, price_change_1d,
                    volume, avg_volume_20d, volume_ratio,
                    week52_high, week52_low, week52_position,
                    iv_30d, iv_60d, iv_90d, iv_1y_percentile,
                    short_interest, market_cap, beta
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                run_id, _now_utc(), sym, price, change_1d,
                vol, avg_vol, vol_ratio,
                w52_hi, w52_lo, w52_pos,
                iv.get("iv_30d"), iv.get("iv_60d"), iv.get("iv_90d"), iv_pct,
                _safe_float(info.get("shortPercentOfFloat")),
                _safe_float(info.get("marketCap")),
                _safe_float(info.get("beta")),
            ))
            conn.commit()
            count += 1
        except Exception as e:
            log.error(f"[yfinance] {sym}: {e}")

    log.info(f"[yfinance] Done -- {count}/{len(EQUITY_UNIVERSE)} fetched")
    return count


# ── FRED ──────────────────────────────────────────────────────────────────────

def _fetch_fred(series_id: str) -> pd.Series | None:
    try:
        resp = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": series_id, "api_key": FRED_API_KEY,
                    "file_type": "json", "sort_order": "desc", "limit": 500},
            timeout=15,
        )
        resp.raise_for_status()
        data = {}
        for o in resp.json().get("observations", []):
            try:
                data[o["date"]] = float(o["value"])
            except (ValueError, KeyError):
                pass
        if not data:
            return None
        s = pd.Series(data)
        s.index = pd.to_datetime(s.index)
        return s.sort_index()
    except Exception as e:
        log.error(f"[FRED] {series_id}: {e}")
        return None


def fetch_macro_data(conn, run_id: str) -> int:
    log.info("[FRED] Fetching macro data")
    if not FRED_API_KEY:
        log.warning("[FRED] No API key")
        return 0

    count = 0
    two_years_ago = pd.Timestamp.now() - pd.DateOffset(years=2)

    for key, sid in FRED_SERIES.items():
        try:
            series = _fetch_fred(sid)
            if series is None or series.empty:
                continue
            val = _safe_float(series.iloc[-1])
            prev = _safe_float(series.iloc[-2]) if len(series) > 1 else None
            change = round(val - prev, 4) if (val is not None and prev is not None) else None

            recent = series[series.index >= two_years_ago].dropna()
            z_score, stress = None, 0
            if len(recent) >= 10 and val is not None:
                mean, std = float(recent.mean()), float(recent.std())
                if std > 0:
                    z_score = round((val - mean) / std, 2)
                    stress = 1 if abs(z_score) > 2.0 else 0

            conn.execute("""
                INSERT INTO macro_snapshots (run_id,timestamp,series_key,series_id,value,prev_value,change,z_score_2y,stress_flag)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (run_id, _now_utc(), key, sid, val, prev, change, z_score, stress))
            conn.commit()
            count += 1
        except Exception as e:
            log.error(f"[FRED] {key}: {e}")

    log.info(f"[FRED] Done -- {count}/{len(FRED_SERIES)} fetched")
    return count


# ── NewsAPI ───────────────────────────────────────────────────────────────────

def _newsapi_query(keyword: str) -> dict:
    thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    # Try /everything first, fall back to /top-headlines
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q": f'"{keyword}"', "from": thirty_days_ago,
                    "sortBy": "publishedAt", "pageSize": 100,
                    "language": "en", "apiKey": NEWS_API_KEY},
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    resp = requests.get(
        "https://newsapi.org/v2/top-headlines",
        params={"q": keyword, "pageSize": 100, "language": "en", "apiKey": NEWS_API_KEY},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_news_data(conn, run_id: str) -> int:
    log.info("[NewsAPI] Fetching news signals")
    if not NEWS_API_KEY:
        log.warning("[NewsAPI] No API key")
        return 0

    count = 0
    for keyword in NEWS_KEYWORDS:
        try:
            data = _newsapi_query(keyword)
            articles = data.get("articles", [])
            article_count = len(articles)

            outlet_hits = {}
            for art in articles:
                name = (art.get("source", {}).get("name") or "").lower()
                for m in MAJOR_OUTLETS:
                    if m.split(".")[0] in name:
                        outlet_hits[m] = outlet_hits.get(m, 0) + 1

            consensus_count = sum(1 for v in outlet_hits.values() if v >= 1)

            # Novelty: check if keyword has appeared before in DB
            thirty_d_ago = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
            prior = conn.execute(
                "SELECT COUNT(*) as c FROM news_signals WHERE keyword=? AND timestamp>=? AND article_count>0",
                (keyword, thirty_d_ago),
            ).fetchone()["c"]
            novel_count = article_count if prior == 0 else max(0, article_count - 10)
            novelty = round(novel_count / article_count, 3) if article_count > 0 else 0.0
            consensus_pen = round(consensus_count / len(MAJOR_OUTLETS), 3)

            conn.execute("""
                INSERT INTO news_signals (run_id,timestamp,keyword,article_count,novel_count,consensus_count,novelty_score,consensus_penalty)
                VALUES (?,?,?,?,?,?,?,?)
            """, (run_id, _now_utc(), keyword, article_count, novel_count, consensus_count, novelty, consensus_pen))
            conn.commit()
            count += 1
        except Exception as e:
            log.error(f"[NewsAPI] '{keyword}': {e}")

    log.info(f"[NewsAPI] Done -- {count}/{len(NEWS_KEYWORDS)} fetched")
    return count


# ── Main run ──────────────────────────────────────────────────────────────────

def run_agent(conn) -> str:
    run_id = _run_id()
    conn.execute("INSERT INTO agent_runs (run_id, started_at, status) VALUES (?,?,'running')",
                 (run_id, _now_utc()))
    conn.commit()
    log.info(f"=== Agent run: {run_id} ===")

    errors = []
    try:
        eq_count = fetch_equity_data(conn, run_id)
    except Exception as e:
        eq_count = 0
        errors.append(f"yfinance: {e}")

    try:
        fetch_macro_data(conn, run_id)
    except Exception as e:
        errors.append(f"FRED: {e}")

    try:
        fetch_news_data(conn, run_id)
    except Exception as e:
        errors.append(f"NewsAPI: {e}")

    conn.execute("""
        UPDATE agent_runs SET finished_at=?, assets_scanned=?, errors=?, status='completed'
        WHERE run_id=?
    """, (_now_utc(), eq_count, json.dumps(errors) if errors else None, run_id))
    conn.commit()
    log.info(f"=== Agent done: {run_id} | errors={errors} ===")
    return run_id
