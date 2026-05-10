#!/usr/bin/env python3
"""
Panic 4-Factor Buy Strategy — automated scheduler with alerts.

Runs the strategy on a schedule, persists position state across runs,
and sends Telegram / email alerts when:
  • Panic score tier UPGRADES  → entry alert (add X% now)
  • Exit tranche triggers fire → exit alert (sell 1/3)

Usage:
  python scheduler.py                    # run now, then every 24h
  python scheduler.py --interval 12      # every 12 hours
  python scheduler.py --run-once         # one shot, print + send, then exit
  python scheduler.py --ticker SPY       # use SPY instead of QQQ
  python scheduler.py --no-alert         # dry run, print only (no messages sent)
  python scheduler.py --reset-state      # wipe position state and exit

Required env vars for alerts:
  Telegram: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
  Email:    ALERT_EMAIL_FROM, ALERT_EMAIL_TO, ALERT_EMAIL_PASSWORD

Optional:
  FRED_API_KEY  — higher-accuracy credit stress (BAMLH0A0HYM2)
"""

import argparse
import sys
import time
from datetime import datetime

from panic4factor.data import auto_fetch
from panic4factor.strategy import run_strategy
from panic4factor.state import TradingState
from panic4factor import alerts


_TIER_ORDER = {
    "NO_SIGNAL":    0,
    "TIER_1_PROBE": 1,
    "TIER_2_CORE":  2,
    "TIER_3_HEAVY": 3,
    "TIER_4_MAX":   4,
}


def run_check(ticker: str, state: TradingState, send: bool) -> TradingState:
    """One full cycle: fetch → score → compare state → alert if needed."""

    print(f"\n{'='*54}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}  —  {ticker}")
    print(f"{'='*54}\n")

    # ── Fetch + score ────────────────────────────────────────────────────────
    try:
        snap = auto_fetch(ticker=ticker)
    except Exception as e:
        print(f"  ERROR fetching data: {e}")
        return state

    output = run_strategy(snap, current_allocation_pct=state.allocation)
    print(output.report())

    new_tier  = output.entry.tier
    new_alloc = output.entry.effective_target_pct
    add_now   = output.entry.add_now_pct

    # ── Entry alert: add_now > 0  (tier upgrade OR filter relaxation) ───────────
    old_rank = _TIER_ORDER.get(state.last_tier, 0)
    new_rank = _TIER_ORDER.get(new_tier, 0)

    if add_now > 0:
        reason = (f"tier {state.last_tier} → {new_tier}"
                  if new_rank != old_rank else
                  f"filter relaxed, same tier {new_tier}")
        print(f"\n  *** ENTRY SIGNAL — {reason} ***")
        state.update_entry(snap.index_price, new_alloc)
        state.last_tier = new_tier

        if send:
            alerts.send_alert(
                f"PANIC BUY — {ticker}  [{new_tier}]",
                alerts.fmt_entry(output, add_now),
            )
            print("  Alert sent.")

    elif new_rank < old_rank and state.in_position:
        # Tier degraded but still holding — track the drop
        print(f"\n  Score dropped ({state.last_tier} → {new_tier}). Holding position.")
        state.last_tier = new_tier

    elif not state.in_position and new_rank == 0:
        state.last_tier = "NO_SIGNAL"

    # ── Exit alerts: capture avg_entry BEFORE check_exits() can call close() ──
    entry_price_snapshot = state.avg_entry_price
    if state.in_position:
        fired = state.check_exits(
            price=snap.index_price,
            vix=snap.vix,
            fg=snap.fear_greed,
            high_52w=snap.high_52w,
        )
        for tranche_n, reason in fired:
            print(f"\n  *** EXIT SIGNAL — Tranche {tranche_n}: {reason} ***")
            if send:
                alerts.send_alert(
                    f"EXIT {ticker}  Tranche {tranche_n}/3",
                    alerts.fmt_exit(tranche_n, reason, snap, entry_price_snapshot),
                )
                print("  Alert sent.")

    # ── Weekly summary (always send on Sunday or if no-alert flag) ───────────
    elif not state.in_position and datetime.now().weekday() == 6:  # Sunday
        if send:
            alerts.send_alert(
                f"Weekly check — {ticker}  (no signal)",
                alerts.fmt_no_signal(snap),
            )

    # ── Print current position ────────────────────────────────────────────────
    print(f"\n  {state.summary()}\n")

    state.last_run_ts = time.time()
    state.save()
    return state


def main():
    parser = argparse.ArgumentParser(description="Panic 4-Factor scheduler")
    parser.add_argument("--ticker",      default="QQQ")
    parser.add_argument("--interval",    type=float, default=24.0,
                        help="Hours between checks (default 24)")
    parser.add_argument("--run-once",    action="store_true",
                        help="Run one check then exit")
    parser.add_argument("--no-alert",    action="store_true",
                        help="Print results but do not send Telegram/email")
    parser.add_argument("--reset-state", action="store_true",
                        help="Wipe persisted position state and exit")
    args = parser.parse_args()

    if args.reset_state:
        state = TradingState()
        state.save()
        print("State reset to FLAT.")
        sys.exit(0)

    state = TradingState.load()
    print(f"Loaded state: {state.summary()}")

    if args.run_once:
        run_check(args.ticker.upper(), state, send=not args.no_alert)
        return

    interval_secs = args.interval * 3600
    print(f"\nScheduler running — check every {args.interval:.0f}h. Ctrl+C to stop.")

    while True:
        state = run_check(args.ticker.upper(), state, send=not args.no_alert)
        next_run = datetime.fromtimestamp(time.time() + interval_secs)
        print(f"  Next check: {next_run.strftime('%Y-%m-%d %H:%M')}")
        time.sleep(interval_secs)


if __name__ == "__main__":
    main()
