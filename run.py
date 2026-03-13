# -*- coding: utf-8 -*-
"""
run.py -- Master Orchestrator
================================
Bootstraps data pull + scoring + trade building, then launches
the Streamlit dashboard. Agent loop runs every 30 min in background.

Usage: python run.py
"""

import logging
import os
import signal
import subprocess
import sys
import threading
import time

from dotenv import load_dotenv

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
    handlers=[logging.FileHandler("agent.log"), logging.StreamHandler()],
)
log = logging.getLogger("run")

INTERVAL = 30 * 60  # 30 minutes
PORT = 8501


def _full_cycle():
    """Run one complete cycle: pull data -> score -> build trades."""
    from database import init_db
    from agent import run_agent
    from scorer import score_all
    from trade_builder import build_trades

    conn = init_db()
    run_id = run_agent(conn)
    results = score_all(conn, run_id)
    trades = build_trades(conn, run_id)

    hi = sum(1 for r in results if r.tier == "HIGH")
    wa = sum(1 for r in results if r.tier == "WATCH")
    log.info(f"Cycle done -- {len(results)} scored, {hi} HIGH, {wa} WATCH, {len(trades)} trades built")
    return results, trades


def bootstrap():
    log.info("=== Bootstrap: initial data pull ===")
    try:
        _full_cycle()
    except Exception as e:
        log.error(f"Bootstrap error (non-fatal): {e}")


def agent_loop():
    log.info(f"Agent loop started (every {INTERVAL}s)")
    while True:
        time.sleep(INTERVAL)
        log.info("=== Scheduled cycle ===")
        try:
            _full_cycle()
        except Exception as e:
            log.error(f"Agent loop error: {e}")


def start_dashboard():
    cmd = [sys.executable, "-m", "streamlit", "run", "dashboard.py",
           "--server.port", str(PORT), "--server.headless", "true",
           "--server.runOnSave", "false", "--browser.gatherUsageStats", "false"]
    log.info(f"Starting Streamlit on port {PORT}")
    return subprocess.Popen(cmd)


def main():
    print(
        "\n"
        "============================================================\n"
        "   TALEB TRADE ADVISOR -- Starting Up                      \n"
        "============================================================\n"
    )

    # Validate keys
    for key in ["ANTHROPIC_API_KEY", "FRED_API_KEY", "NEWS_API_KEY"]:
        if not os.getenv(key):
            log.warning(f"{key} not set -- some features will be limited")

    bootstrap()

    t = threading.Thread(target=agent_loop, daemon=True)
    t.start()

    proc = start_dashboard()
    log.info(f"\nDashboard: http://localhost:{PORT}\nPress Ctrl+C to stop.\n")

    def shutdown(sig, frame):
        log.info("Shutting down...")
        proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    while True:
        if proc.poll() is not None:
            log.error("Streamlit exited. Restarting...")
            proc = start_dashboard()
        time.sleep(5)


if __name__ == "__main__":
    main()
