#!/usr/bin/env python3
"""
Panic 4-Factor Buy Strategy — fully automated weekly check.

All data is fetched automatically:
  VIX / price / HYG / LQD / TLT  → yfinance
  CNN Fear & Greed                → CNN dataviz API
  AAII Bull-Bear Spread           → aaii.com Excel
  NAAIM Exposure                  → naaim.org Excel
  Credit stress                   → computed from HYG/LQD/TLT (+ FRED if key set)

Usage:
  python run.py                          # live fetch, QQQ default
  python run.py --ticker SPY             # use SPY instead
  python run.py --held 0.20              # already 20% deployed
  python run.py --entry-price 430.5      # show exit levels
  python run.py --no-cache               # bypass 8h disk cache
  python run.py --example                # Oct-2022 back-test snapshot (no network)

Optional — set for higher-accuracy credit stress (FRED HY OAS):
  export FRED_API_KEY=your_free_key
  (register at https://fred.stlouisfed.org/docs/api/api_key.html)
"""

import argparse
import sys

from panic4factor.data import auto_fetch, build_snapshot_from_manual
from panic4factor.strategy import run_strategy


def example_snapshot():
    """Approximate Oct-2022 conditions — QQQ trough."""
    return build_snapshot_from_manual(
        ticker="QQQ",
        vix=33.0,
        index_price=267.0,
        high_52w=399.0,
        ma_200=330.0,
        fear_greed=20.0,
        aaii_bull_bear_spread=-42.0,
        naaim_exposure=30.0,
        # Credit: HYG was ~71 vs ~80 200MA, LQD also weak but less so
        hyg_price=71.0,
        hyg_ma200=80.0,
        hyg_price_4w=74.0,
        lqd_price=100.0,
        lqd_price_4w=104.0,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Panic 4-Factor Buy Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ticker",       default="QQQ",  help="Index ticker (QQQ / SPY / ^GSPC)")
    parser.add_argument("--held",         type=float, default=0.0,
                        help="Fraction of strategy capital already deployed, e.g. 0.2 = 20%%")
    parser.add_argument("--entry-price",  type=float, default=None,
                        help="Average entry price if already in position (shows exit levels)")
    parser.add_argument("--no-cache",     action="store_true", help="Bypass disk cache, re-fetch all")
    parser.add_argument("--example",      action="store_true", help="Run Oct-2022 example (no network)")
    args = parser.parse_args()

    print("=" * 54)
    print("  Panic 4-Factor Buy Strategy")
    print("  Instruments: QQQ / SPY / ES / SPX")
    print("=" * 54)

    if args.example:
        print("\n[Oct-2022 example snapshot — no network]\n")
        snap = example_snapshot()
    else:
        print(f"\nFetching live data for {args.ticker.upper()} …\n")
        try:
            snap = auto_fetch(ticker=args.ticker.upper(), no_cache=args.no_cache)
        except Exception as e:
            print(f"\nERROR: {e}", file=sys.stderr)
            print("Try --example to run with a local snapshot.", file=sys.stderr)
            sys.exit(1)

    output = run_strategy(
        snapshot=snap,
        current_allocation_pct=args.held,
        avg_entry_price=args.entry_price,
    )
    print(output.report())


if __name__ == "__main__":
    main()
