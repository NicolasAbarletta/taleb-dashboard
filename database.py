# -*- coding: utf-8 -*-
"""
database.py -- SQLite interface layer
======================================
Single source of truth for schema, connections, and queries.
All other modules import from here.
"""

import json
import sqlite3
import threading
from datetime import datetime, timezone

DB_PATH = "market_data.db"
_local = threading.local()


def get_conn(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Thread-local connection with WAL mode for concurrent read/write."""
    conn = getattr(_local, "conn", None)
    # Validate existing connection is still usable
    if conn is not None:
        try:
            conn.execute("SELECT 1")
        except Exception:
            conn = None
    if conn is None:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        _local.conn = conn
    return conn


def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Create all tables. Safe to call repeatedly."""
    conn = get_conn(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS equity_snapshots (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          TEXT NOT NULL,
            timestamp       TEXT NOT NULL,
            ticker          TEXT NOT NULL,
            price           REAL,
            price_change_1d REAL,
            volume          REAL,
            avg_volume_20d  REAL,
            volume_ratio    REAL,
            week52_high     REAL,
            week52_low      REAL,
            week52_position REAL,
            iv_30d          REAL,
            iv_60d          REAL,
            iv_90d          REAL,
            iv_1y_percentile REAL,
            short_interest  REAL,
            market_cap      REAL,
            beta            REAL,
            realized_vol_20d    REAL,
            return_20d          REAL,
            return_60d          REAL,
            skewness_60d        REAL,
            spy_correlation_60d REAL,
            volume_trend_20d    REAL,
            drawdown_from_peak  REAL
        )
    """)

    # Graceful migration for existing DBs -- add columns if missing
    _new_cols = [
        "realized_vol_20d", "return_20d", "return_60d", "skewness_60d",
        "spy_correlation_60d", "volume_trend_20d", "drawdown_from_peak",
    ]
    try:
        existing = {row[1] for row in cur.execute("PRAGMA table_info(equity_snapshots)").fetchall()}
        for col in _new_cols:
            if col not in existing:
                cur.execute(f"ALTER TABLE equity_snapshots ADD COLUMN {col} REAL")
    except Exception:
        pass

    cur.execute("""
        CREATE TABLE IF NOT EXISTS macro_snapshots (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      TEXT NOT NULL,
            timestamp   TEXT NOT NULL,
            series_key  TEXT NOT NULL,
            series_id   TEXT NOT NULL,
            value       REAL,
            prev_value  REAL,
            change      REAL,
            z_score_2y  REAL,
            stress_flag INTEGER DEFAULT 0
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS news_signals (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          TEXT NOT NULL,
            timestamp       TEXT NOT NULL,
            keyword         TEXT NOT NULL,
            article_count   INTEGER,
            novel_count     INTEGER,
            consensus_count INTEGER,
            novelty_score   REAL,
            consensus_penalty REAL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS taleb_scores (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id              TEXT NOT NULL,
            scored_at           TEXT NOT NULL,
            ticker              TEXT NOT NULL,
            total_score         INTEGER,
            convexity_score     INTEGER,
            antifragility_score INTEGER,
            fragility_avoidance INTEGER,
            tail_risk_score     INTEGER,
            conviction_tier     TEXT,
            catalyst            TEXT,
            rationale           TEXT,
            price               REAL,
            iv_percentile       REAL,
            score_details       TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS trade_recommendations (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            ticker          TEXT NOT NULL,
            total_score     INTEGER,
            conviction_tier TEXT,
            trade_action    TEXT,
            instrument      TEXT,
            strike          REAL,
            expiry          TEXT,
            premium         REAL,
            delta           REAL,
            gamma           REAL,
            vega            REAL,
            underlying_price REAL,
            thesis          TEXT,
            triggers        TEXT,
            pnl_scenarios   TEXT,
            risk_check      TEXT,
            position_pct    REAL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS agent_runs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          TEXT NOT NULL UNIQUE,
            started_at      TEXT NOT NULL,
            finished_at     TEXT,
            assets_scanned  INTEGER DEFAULT 0,
            opportunities   INTEGER DEFAULT 0,
            errors          TEXT,
            status          TEXT DEFAULT 'running'
        )
    """)

    conn.commit()
    return conn


# ── Convenience queries ──────────────────────────────────────────────────────

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def latest_run_id(conn: sqlite3.Connection) -> str | None:
    cur = conn.execute("""
        SELECT run_id FROM agent_runs
        WHERE status = 'completed'
        ORDER BY finished_at DESC LIMIT 1
    """)
    row = cur.fetchone()
    return row["run_id"] if row else None
