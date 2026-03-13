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
    if not hasattr(_local, "conn") or _local.conn is None:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        _local.conn = conn
    return _local.conn


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
            beta            REAL
        )
    """)

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
