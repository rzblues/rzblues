#!/usr/bin/env python3
"""
Panic 4-Factor Buy Strategy — historical backtest.

Simulates the strategy week-by-week from 2007 to today, tracking
staged entries (tier upgrades → buy) and exits (VIX<20 + F&G normal).

Data availability determines the start:
  NAAIM : 2006+   HYG/LQD (credit): 2007+   ← binding constraints
  AAII  : 1987+   VIX / SPY / QQQ : 1990+

CNN Fear & Greed was created in 2012. For all periods we approximate:
  fg_approx = clip(130 − VIX × 3, 0, 100)
  VIX=10 → F&G=100  |  VIX=27 → F&G=49  |  VIX=40 → F&G=10
This is a reasonable proxy (VIX is the dominant driver of the index).

Exit rule (simplified for backtest):
  VIX < 20 AND F&G_approx > 50  →  full exit at week-end close
  (Live trading uses 3-tranche exits; backtest uses combined to avoid
  over-engineering a simulation of execution timing.)

Usage:
  python backtest.py                      # QQQ, 2007-01-01 to today
  python backtest.py --ticker SPY
  python backtest.py --start 2010-01-01
  python backtest.py --csv results.csv    # save trade log to CSV
  python backtest.py --no-cache           # force re-download all data
"""

import argparse
import sys
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from panic4factor.scorer import compute_panic_score
from panic4factor.filters import apply_filters
from panic4factor.credit import (
    CreditStressData, score_credit_stress,
    _score_hyg_vs_200ma, _score_hyg_lqd_relative, _score_tlt_vol,
)
from panic4factor import fetch, cache

_TIER_TARGETS = {
    "NO_SIGNAL":    0.00,
    "TIER_1_PROBE": 0.20,
    "TIER_2_CORE":  0.50,
    "TIER_3_HEAVY": 0.80,
    "TIER_4_MAX":   1.00,
}


# ── Trade record ──────────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_date: date
    entry_price: float
    entry_tier: str
    exit_date: Optional[date] = None
    exit_price: Optional[float] = None
    max_drawdown_pct: float = 0.0
    open_trade: bool = False

    @property
    def return_pct(self) -> Optional[float]:
        if self.exit_price is None:
            return None
        return (self.exit_price / self.entry_price - 1) * 100

    @property
    def hold_weeks(self) -> Optional[int]:
        if self.exit_date is None:
            return None
        return max(1, (self.exit_date - self.entry_date).days // 7)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(ticker: str, start: str, no_cache: bool) -> pd.DataFrame:
    if no_cache:
        cache.clear()

    print(f"\nDownloading market data ({ticker} / ^VIX / HYG / LQD / TLT)…", end=" ", flush=True)
    raw = yf.download(
        [ticker, "^VIX", "HYG", "LQD", "TLT"],
        start=start,
        auto_adjust=True,
        progress=False,
    )
    closes = raw["Close"].copy()
    closes.columns = [c.lower().replace("^", "") for c in closes.columns]
    idx_col = ticker.lower().replace("^", "")
    print("done")

    # 200-day MA and 52-week high computed from daily data
    closes["ma200"]   = closes[idx_col].rolling(200, min_periods=100).mean()
    closes["high52w"] = closes[idx_col].rolling(252, min_periods=126).max()

    # HYG 200-day MA and 20-day-ago prices for credit stress
    closes["hyg_ma200"]  = closes["hyg"].rolling(200, min_periods=100).mean()
    closes["hyg_4w"]     = closes["hyg"].shift(20)   # ~4 trading weeks
    closes["lqd_4w"]     = closes["lqd"].shift(20)

    # TLT 10-day realized vol (annualized) as MOVE proxy
    tlt_rets = closes["tlt"].pct_change()
    closes["tlt_vol10"] = tlt_rets.rolling(10).std() * (252 ** 0.5) * 100

    # Approximate F&G from VIX (CNN F&G not available pre-2012)
    closes["fg_approx"] = (130 - closes["vix"] * 3).clip(0, 100)

    # Resample to weekly (Friday close)
    weekly = closes.resample("W-FRI").last()

    # Merge AAII history
    print("Downloading AAII sentiment history…", end=" ", flush=True)
    try:
        aaii = fetch.fetch_aaii_history()
        aaii_w = aaii["spread"].resample("W-FRI").last().rename("aaii_spread")
        weekly = weekly.join(aaii_w, how="left")
        weekly["aaii_spread"] = weekly["aaii_spread"].ffill()
        print("done")
    except Exception as e:
        print(f"FAILED ({e}) — using neutral 0")
        weekly["aaii_spread"] = 0.0

    # Merge NAAIM history
    print("Downloading NAAIM exposure history…", end=" ", flush=True)
    try:
        naaim = fetch.fetch_naaim_history()
        naaim_w = naaim["exposure"].resample("W-FRI").last().rename("naaim")
        weekly = weekly.join(naaim_w, how="left")
        weekly["naaim"] = weekly["naaim"].ffill()
        print("done")
    except Exception as e:
        print(f"FAILED ({e}) — using neutral 50")
        weekly["naaim"] = 50.0

    weekly = weekly.dropna(subset=[idx_col, "vix", "hyg", "lqd"])
    weekly.attrs["idx_col"] = idx_col
    return weekly


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate(df: pd.DataFrame) -> list[Trade]:
    idx_col = df.attrs["idx_col"]
    trades: list[Trade] = []

    # State
    allocation: float          = 0.0
    avg_entry: Optional[float] = None
    entry_date: Optional[date] = None
    entry_tier: str            = "NO_SIGNAL"
    max_dd_current: float      = 0.0

    for ts, row in df.iterrows():
        week_date = ts.date()

        price   = row[idx_col]
        vix     = row["vix"]
        fg      = row.get("fg_approx", 50.0)
        aaii    = row.get("aaii_spread", 0.0)
        naaim   = row.get("naaim", 50.0)
        ma200   = row.get("ma200", price)
        high52w = row.get("high52w", price)
        hyg     = row["hyg"]
        lqd     = row["lqd"]
        hyg_ma  = row.get("hyg_ma200", hyg)
        hyg_4w  = row.get("hyg_4w", hyg)
        lqd_4w  = row.get("lqd_4w", lqd)
        tlt_vol = row.get("tlt_vol10", 0.0)

        # Guard against NaN in critical fields
        if any(np.isnan(v) for v in [price, vix, ma200] if v is not None):
            continue
        if np.isnan(aaii):  aaii = 0.0
        if np.isnan(naaim): naaim = 50.0
        if np.isnan(hyg_ma): hyg_ma = hyg
        if np.isnan(hyg_4w): hyg_4w = hyg
        if np.isnan(lqd_4w): lqd_4w = lqd
        if np.isnan(tlt_vol): tlt_vol = 0.0
        if np.isnan(high52w): high52w = price

        # ── Score ────────────────────────────────────────────────────────────
        panic = compute_panic_score(
            vix=vix, fear_greed=fg,
            aaii_bull_bear_spread=aaii, naaim_exposure=naaim,
        )

        # Credit stress (yfinance path, no FRED in backtest)
        hyg_ret = (hyg - hyg_4w) / hyg_4w if hyg_4w else 0.0
        lqd_ret = (lqd - lqd_4w) / lqd_4w if lqd_4w else 0.0
        spread_s   = _score_hyg_vs_200ma(hyg, hyg_ma)
        relative_s = _score_hyg_lqd_relative(hyg_ret, lqd_ret)
        vol_s      = 1 if tlt_vol > 25 else 0
        credit_score = spread_s + relative_s + vol_s
        is_crisis = credit_score >= 5

        # Filters
        drawdown = (high52w - price) / high52w * 100 if high52w > 0 else 0.0
        p_vs_ma  = price / ma200 if ma200 > 0 else 1.0
        filt = apply_filters(drawdown, p_vs_ma, is_crisis)

        tier   = panic.signal_tier()
        target = _TIER_TARGETS[tier] * filt.effective_cap()

        # ── Entry: tier upgrade ───────────────────────────────────────────────
        if target > allocation:
            add = target - allocation
            if allocation == 0:
                avg_entry    = price
                entry_date   = week_date
                entry_tier   = tier
                max_dd_current = 0.0
            else:
                avg_entry = (avg_entry * allocation + price * add) / target
                entry_tier = tier
            allocation = target

        # ── Track max drawdown during hold ────────────────────────────────────
        if allocation > 0 and avg_entry and price < avg_entry:
            dd = (avg_entry - price) / avg_entry * 100
            max_dd_current = max(max_dd_current, dd)

        # ── Exit: market normalized (VIX < 20 AND F&G > 50) ─────────────────
        if allocation > 0 and avg_entry and vix < 20 and fg > 50:
            trades.append(Trade(
                entry_date=entry_date,
                entry_price=avg_entry,
                entry_tier=entry_tier,
                exit_date=week_date,
                exit_price=price,
                max_drawdown_pct=max_dd_current,
            ))
            allocation     = 0.0
            avg_entry      = None
            entry_date     = None
            entry_tier     = "NO_SIGNAL"
            max_dd_current = 0.0

    # Close any open position at last available price
    if allocation > 0 and avg_entry:
        last_price = df[idx_col].iloc[-1]
        trades.append(Trade(
            entry_date=entry_date,
            entry_price=avg_entry,
            entry_tier=entry_tier,
            exit_date=df.index[-1].date(),
            exit_price=last_price,
            max_drawdown_pct=max_dd_current,
            open_trade=True,
        ))

    return trades


# ── Buy-and-hold benchmark ────────────────────────────────────────────────────

def buy_and_hold_return(df: pd.DataFrame) -> float:
    idx_col = df.attrs["idx_col"]
    prices = df[idx_col].dropna()
    return (prices.iloc[-1] / prices.iloc[0] - 1) * 100


# ── Report ────────────────────────────────────────────────────────────────────

def report(trades: list[Trade], df: pd.DataFrame, ticker: str) -> None:
    idx_col = df.attrs["idx_col"]
    start   = df.index[0].strftime("%Y-%m-%d")
    end     = df.index[-1].strftime("%Y-%m-%d")
    bah     = buy_and_hold_return(df)

    print(f"\n{'='*68}")
    print(f"  Panic 4-Factor Backtest  —  {ticker}  ({start} → {end})")
    print(f"{'='*68}")
    print(f"  Note: CNN F&G approximated from VIX for all periods.")
    print(f"  Entry: tier upgrade (cumulative staged buys).")
    print(f"  Exit:  VIX < 20 AND F&G_approx > 50 (market normalized).")
    print(f"{'='*68}\n")

    if not trades:
        print("  No trades triggered in this period.")
        return

    # Trade table header
    print(f"  {'#':<3}  {'Entry':>10}  {'Exit':>10}  "
          f"{'Entry $':>8}  {'Exit $':>8}  {'Return':>7}  "
          f"{'Hold':>5}  {'MaxDD':>6}  {'Tier'}")
    print(f"  {'-'*3}  {'-'*10}  {'-'*10}  "
          f"{'-'*8}  {'-'*8}  {'-'*7}  "
          f"{'-'*5}  {'-'*6}  {'-'*20}")

    returns = []
    for i, t in enumerate(trades, 1):
        ret  = t.return_pct
        tag  = " (open)" if t.open_trade else ""
        ret_s = f"{ret:+.1f}%" if ret is not None else "open"
        returns.append(ret if ret is not None else 0.0)
        print(
            f"  {i:<3}  {str(t.entry_date):>10}  {str(t.exit_date):>10}  "
            f"{t.entry_price:>8.2f}  {t.exit_price:>8.2f}  "
            f"{ret_s:>7}  {str(t.hold_weeks or '?'):>5}wk  "
            f"{t.max_drawdown_pct:>5.1f}%  {t.entry_tier}{tag}"
        )

    # Summary stats
    closed = [t for t in trades if not t.open_trade]
    wins   = [t for t in closed if t.return_pct and t.return_pct > 0]
    rets   = [t.return_pct for t in closed if t.return_pct is not None]

    print(f"\n{'='*68}")
    print(f"  Summary")
    print(f"{'='*68}")
    print(f"  Trades (closed)      : {len(closed)}")
    if rets:
        print(f"  Win rate             : {len(wins)}/{len(closed)}  "
              f"({len(wins)/len(closed)*100:.0f}%)")
        print(f"  Avg return           : {np.mean(rets):+.1f}%")
        print(f"  Median return        : {np.median(rets):+.1f}%")
        print(f"  Best trade           : {max(rets):+.1f}%")
        print(f"  Worst trade          : {min(rets):+.1f}%")
        avg_hold = np.mean([t.hold_weeks for t in closed if t.hold_weeks])
        print(f"  Avg hold time        : {avg_hold:.0f} weeks")
        avg_dd = np.mean([t.max_drawdown_pct for t in closed])
        print(f"  Avg max drawdown     : {avg_dd:.1f}%")

    # Strategy vs buy-and-hold (strategy only deployed during trade periods)
    total_weeks = len(df)
    strategy_weeks = sum(
        t.hold_weeks or 0 for t in closed
    )
    deployed_pct = strategy_weeks / total_weeks * 100 if total_weeks else 0
    print(f"\n  Buy-and-hold return  : {bah:+.1f}%")
    print(f"  Strategy deployed    : ~{deployed_pct:.0f}% of weeks")
    print(f"  (Strategy capital not deployed earns risk-free rate when flat)")
    print(f"\n  Caveats:")
    print(f"  - Entry/exit at Friday close (no slippage modeled)")
    print(f"  - F&G approximated from VIX (lower accuracy pre-2012)")
    print(f"  - NAAIM started 2006; credit (HYG) started 2007")
    print(f"  - Single exit rule used (live trading uses 3-tranche exits)")
    print(f"{'='*68}\n")


# ── CSV export ────────────────────────────────────────────────────────────────

def save_csv(trades: list[Trade], path: str) -> None:
    rows = [{
        "entry_date":     t.entry_date,
        "exit_date":      t.exit_date,
        "entry_price":    t.entry_price,
        "exit_price":     t.exit_price,
        "return_pct":     round(t.return_pct, 2) if t.return_pct else None,
        "hold_weeks":     t.hold_weeks,
        "max_drawdown":   round(t.max_drawdown_pct, 2),
        "entry_tier":     t.entry_tier,
        "open":           t.open_trade,
    } for t in trades]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Trade log saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Panic 4-Factor historical backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--ticker",   default="QQQ")
    parser.add_argument("--start",    default="2007-01-01",
                        help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--csv",      default=None,
                        help="Save trade log to CSV file")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-download all data")
    args = parser.parse_args()

    df = load_data(args.ticker.upper(), args.start, args.no_cache)
    trades = simulate(df)
    report(trades, df, args.ticker.upper())

    if args.csv:
        save_csv(trades, args.csv)


if __name__ == "__main__":
    main()
