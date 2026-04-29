#!/usr/bin/env python3
"""
Panic 4-Factor Buy Strategy — weekly check CLI.

Usage (auto-fetch VIX + price data, manual sentiment inputs):
  python run.py

Usage (fully manual, no network):
  python run.py --manual

Usage (example back-test snapshot, e.g. 2022-10 trough):
  python run.py --example
"""

import argparse
import sys

from panic4factor.data import fetch_market_data, build_snapshot_from_manual, MarketSnapshot
from panic4factor.strategy import run_strategy


# ── Manual data sources ─────────────────────────────────────────────────────
# CNN Fear & Greed : https://edition.cnn.com/markets/fear-and-greed
# AAII Sentiment   : https://www.aaii.com/sentimentsurvey
# NAAIM Exposure   : https://www.naaim.org/programs/naaim-exposure-index/
# ────────────────────────────────────────────────────────────────────────────


def prompt_float(label: str, lo: float, hi: float) -> float:
    while True:
        raw = input(f"  {label} [{lo}–{hi}]: ").strip()
        try:
            val = float(raw)
            if lo <= val <= hi:
                return val
            print(f"    Enter a value between {lo} and {hi}.")
        except ValueError:
            print("    Invalid number, try again.")


def prompt_bool(label: str) -> bool:
    while True:
        raw = input(f"  {label} [y/n]: ").strip().lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("    Enter y or n.")


def interactive_mode(ticker: str) -> MarketSnapshot:
    """Try auto-fetch; prompt for sentinel / manual fields."""
    print(f"\nFetching live data for {ticker} …", end=" ", flush=True)
    snap = fetch_market_data(ticker)

    if snap is None:
        print("failed (yfinance not installed or no network).")
        print("Switching to fully manual mode.\n")
        return manual_mode(ticker)

    print("done.\n")
    print(f"  VIX          : {snap.vix:.2f}")
    print(f"  {ticker} Price  : {snap.index_price:,.2f}")
    print(f"  52-wk High   : {snap.high_52w:,.2f}")
    print(f"  200-day MA   : {snap.ma_200:,.2f}")
    print(f"  Drawdown     : -{snap.drawdown_pct:.1f}%\n")

    print("Manual sentiment inputs (check links above):")
    snap.fear_greed          = prompt_float("CNN Fear & Greed (0=extreme fear, 100=extreme greed)", 0, 100)
    snap.aaii_bull_bear_spread = prompt_float("AAII Bull-Bear Spread (Bullish% − Bearish%, e.g. -30)", -70, 70)
    snap.naaim_exposure      = prompt_float("NAAIM Exposure Index (0–200, typical 0–100)", 0, 200)
    snap.credit_crisis       = prompt_bool("Credit/macro crisis active? (bank stress, spreads spiking)")
    return snap


def manual_mode(ticker: str) -> MarketSnapshot:
    print(f"Manual data entry for {ticker}:\n")
    vix         = prompt_float("VIX", 5, 90)
    price       = prompt_float(f"{ticker} current price", 1, 100_000)
    high_52w    = prompt_float(f"{ticker} 52-week high", 1, 100_000)
    ma_200      = prompt_float(f"{ticker} 200-day MA", 1, 100_000)
    fg          = prompt_float("CNN Fear & Greed (0–100)", 0, 100)
    aaii        = prompt_float("AAII Bull-Bear Spread (e.g. -30)", -70, 70)
    naaim       = prompt_float("NAAIM Exposure (0–200)", 0, 200)
    crisis      = prompt_bool("Credit/macro crisis active?")
    return build_snapshot_from_manual(ticker, vix, price, high_52w, ma_200, fg, aaii, naaim, crisis)


def example_snapshot() -> MarketSnapshot:
    """
    Approximate conditions around Oct 2022 SPY trough.
    VIX ~33, F&G ~20, AAII spread ~ -42, NAAIM ~30, SPY ~-25% from ATH.
    """
    return build_snapshot_from_manual(
        ticker="QQQ",
        vix=33.0,
        index_price=267.0,
        high_52w=399.0,
        ma_200=330.0,
        fear_greed=20.0,
        aaii_bull_bear_spread=-42.0,
        naaim_exposure=30.0,
        credit_crisis=False,
    )


TICKERS = ["QQQ", "SPY", "^GSPC", "^NDX"]


def main():
    parser = argparse.ArgumentParser(description="Panic 4-Factor Buy Strategy")
    parser.add_argument("--manual",   action="store_true", help="Skip auto-fetch, enter all values manually")
    parser.add_argument("--example",  action="store_true", help="Run with Oct-2022-like example data")
    parser.add_argument("--ticker",   default="QQQ",       help="Index ticker (QQQ / SPY / ^GSPC)")
    parser.add_argument("--held",     type=float, default=0.0,
                        help="Fraction of strategy capital already deployed, e.g. 0.2 = 20%%")
    parser.add_argument("--entry-price", type=float, default=None,
                        help="Average entry price if already in position (shows exit levels)")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    print("=" * 54)
    print("  Panic 4-Factor Buy Strategy — Index Edition")
    print("  Instruments: QQQ / SPY / ES / SPX")
    print("=" * 54)

    if args.example:
        snap = example_snapshot()
        print("\n[Running with Oct-2022 example snapshot]\n")
    elif args.manual:
        snap = manual_mode(ticker)
    else:
        snap = interactive_mode(ticker)

    output = run_strategy(
        snapshot=snap,
        current_allocation_pct=args.held,
        avg_entry_price=args.entry_price,
    )

    print(output.report())


if __name__ == "__main__":
    main()
