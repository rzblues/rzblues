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
    entry_allocation: float = 0.0    # peak allocation fraction during this trade
    exit_date: Optional[date] = None
    exit_price: Optional[float] = None
    max_drawdown_pct: float = 0.0
    open_trade: bool = False

    @property
    def etf_return_pct(self) -> Optional[float]:
        """Raw ETF gain/loss — NOT the portfolio impact."""
        if self.exit_price is None:
            return None
        return (self.exit_price / self.entry_price - 1) * 100

    @property
    def portfolio_return_pct(self) -> Optional[float]:
        """ETF return × allocation = actual impact on strategy capital."""
        if self.etf_return_pct is None:
            return None
        return self.etf_return_pct * self.entry_allocation

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
        threads=False,
    )
    if raw is None or raw.empty or "Close" not in raw:
        raise RuntimeError("yfinance returned no usable market data")

    closes = raw["Close"].copy()
    closes.columns = [c.lower().replace("^", "") for c in closes.columns]
    idx_col = ticker.lower().replace("^", "")

    required_cols = [idx_col, "vix", "hyg", "lqd", "tlt"]
    missing_cols = [c for c in required_cols if c not in closes.columns]
    if missing_cols:
        raise RuntimeError(f"yfinance response missing required columns: {missing_cols}")
    empty_cols = [c for c in required_cols if closes[c].dropna().empty]
    if empty_cols:
        raise RuntimeError(f"yfinance returned empty data for required columns: {empty_cols}")
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
    if weekly.empty:
        raise RuntimeError("No usable weekly rows after merging market/sentiment data")
    weekly.attrs["idx_col"] = idx_col
    return weekly


# ── Simulation ────────────────────────────────────────────────────────────────

_CASH_RATE_WEEKLY = 0.04 / 52   # 4% annual risk-free rate on idle cash
_DEFAULT_COST_BPS = 5.0          # one-way execution + spread cost per traded dollar


def simulate(df: pd.DataFrame, cost_bps: float = _DEFAULT_COST_BPS) -> tuple[list[Trade], pd.Series]:
    """
    Returns (trades, equity_series).
    equity_series tracks portfolio value week-by-week starting at 1.0,
    crediting the index return on the deployed fraction and risk-free rate
    on idle cash.  This is the basis for CAGR and max-drawdown reporting.
    """
    idx_col = df.attrs["idx_col"]
    trades: list[Trade] = []

    # Trade state
    allocation: float          = 0.0
    avg_entry: Optional[float] = None
    entry_date: Optional[date] = None
    entry_tier: str            = "NO_SIGNAL"
    max_dd_current: float      = 0.0

    # Portfolio equity curve
    equity: float = 1.0
    prev_price: Optional[float] = None
    equity_index: list = []
    equity_values: list = []

    cost_rate = max(0.0, cost_bps) / 10_000

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

        # ── Portfolio equity update (previous close -> this close) ───────────
        # Signals are computed at this close, so they cannot earn this week's
        # return. The allocation entering this block is the position held during
        # the week that just ended.
        if prev_price is not None and prev_price > 0:
            price_ret     = (price - prev_price) / prev_price
            portfolio_ret = allocation * price_ret + (1 - allocation) * _CASH_RATE_WEEKLY
            equity       *= (1 + portfolio_ret)

        # ── Score at current close ───────────────────────────────────────────
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

        # ── Entry: target increased (tier upgrade or filter relaxation) ─────
        if target > allocation:
            add = target - allocation
            if cost_rate:
                equity *= (1 - add * cost_rate)
            if allocation == 0:
                avg_entry    = price
                entry_date   = week_date
                entry_tier   = tier
                max_dd_current = 0.0
            else:
                avg_entry = (avg_entry * allocation + price * add) / target
                entry_tier = tier
            allocation = target

        # ── Track max drawdown while in trade ────────────────────────────────
        if allocation > 0 and avg_entry and price < avg_entry:
            dd = (avg_entry - price) / avg_entry * 100
            max_dd_current = max(max_dd_current, dd)

        # ── Exit: market normalized (VIX < 20 AND F&G > 50) ─────────────────
        if allocation > 0 and avg_entry and vix < 20 and fg > 50:
            trades.append(Trade(
                entry_date=entry_date,
                entry_price=avg_entry,
                entry_tier=entry_tier,
                entry_allocation=allocation,
                exit_date=week_date,
                exit_price=price,
                max_drawdown_pct=max_dd_current,
            ))
            if cost_rate:
                equity *= (1 - allocation * cost_rate)
            allocation     = 0.0
            avg_entry      = None
            entry_date     = None
            entry_tier     = "NO_SIGNAL"
            max_dd_current = 0.0

        equity_index.append(ts)
        equity_values.append(equity)
        prev_price = price

    # Close any open position at last available price
    if allocation > 0 and avg_entry:
        last_price = df[idx_col].iloc[-1]
        trades.append(Trade(
            entry_date=entry_date,
            entry_price=avg_entry,
            entry_tier=entry_tier,
            entry_allocation=allocation,
            exit_date=df.index[-1].date(),
            exit_price=last_price,
            max_drawdown_pct=max_dd_current,
            open_trade=True,
        ))

    equity_series = pd.Series(equity_values, index=equity_index, name="equity")

    return trades, equity_series


# ── Buy-and-hold benchmark ────────────────────────────────────────────────────

def buy_and_hold_return(df: pd.DataFrame) -> float:
    idx_col = df.attrs["idx_col"]
    prices = df[idx_col].dropna()
    return (prices.iloc[-1] / prices.iloc[0] - 1) * 100


# ── Report ────────────────────────────────────────────────────────────────────

def report(trades: list[Trade], df: pd.DataFrame, ticker: str,
           equity_series: pd.Series, cost_bps: float = _DEFAULT_COST_BPS) -> None:
    idx_col = df.attrs["idx_col"]
    start   = df.index[0].strftime("%Y-%m-%d")
    end     = df.index[-1].strftime("%Y-%m-%d")
    bah     = buy_and_hold_return(df)
    years   = (df.index[-1] - df.index[0]).days / 365.25

    print(f"\n{'='*74}")
    print(f"  Panic 4-Factor Backtest  —  {ticker}  ({start} → {end})")
    print(f"{'='*74}")
    print(f"  Note: CNN F&G approximated from VIX.  Cash earns 4% p.a. when flat.")
    print(f"  Execution cost: {cost_bps:.1f} bps per traded dollar.")
    print(f"  Entry: tier upgrade OR filter relaxation (staged buys).")
    print(f"  Exit:  VIX < 20 AND F&G_approx > 50 (market normalized).")
    print(f"{'='*74}\n")

    if not trades:
        print("  No trades triggered in this period.")
        return

    # Trade table — show ETF return AND allocation-weighted portfolio impact
    hdr = (f"  {'#':<3}  {'Entry':>10}  {'Exit':>10}  "
           f"{'ETF Ret':>8}  {'Alloc':>6}  {'Port Ret':>9}  "
           f"{'Hold':>5}  {'MaxDD':>6}  Tier")
    sep = "  " + "-" * (len(hdr) - 2)
    print(hdr)
    print(sep)

    for i, t in enumerate(trades, 1):
        tag   = " (open)" if t.open_trade else ""
        etr   = f"{t.etf_return_pct:+.1f}%" if t.etf_return_pct is not None else "?"
        ptr   = f"{t.portfolio_return_pct:+.1f}%" if t.portfolio_return_pct is not None else "?"
        alloc = f"{t.entry_allocation*100:.0f}%"
        print(
            f"  {i:<3}  {str(t.entry_date):>10}  {str(t.exit_date):>10}  "
            f"{etr:>8}  {alloc:>6}  {ptr:>9}  "
            f"{str(t.hold_weeks or '?'):>5}wk  "
            f"{t.max_drawdown_pct:>5.1f}%  {t.entry_tier}{tag}"
        )

    # Per-trade stats (portfolio-level, not ETF-level)
    closed  = [t for t in trades if not t.open_trade]
    ptrs    = [t.portfolio_return_pct for t in closed if t.portfolio_return_pct is not None]
    wins    = [p for p in ptrs if p > 0]

    # Portfolio-level stats from equity curve
    total_return = (equity_series.iloc[-1] - 1) * 100
    cagr         = (equity_series.iloc[-1] ** (1 / years) - 1) * 100
    rolling_max  = equity_series.cummax()
    port_max_dd  = abs(((equity_series - rolling_max) / rolling_max).min() * 100)

    # Buy-and-hold CAGR for comparison
    bah_eq   = df[idx_col].dropna()
    bah_cagr = (bah_eq.iloc[-1] / bah_eq.iloc[0]) ** (1 / years) * 100 - 100

    total_weeks    = len(df)
    deployed_weeks = sum(t.hold_weeks or 0 for t in closed)
    deployed_pct   = deployed_weeks / total_weeks * 100 if total_weeks else 0

    print(f"\n{'='*74}")
    print(f"  Portfolio summary  (strategy capital = 100%, cash earns 4% p.a., costs included)")
    print(f"{'='*74}")
    print(f"  Trades (closed)          : {len(closed)}")
    if ptrs:
        print(f"  Win rate (port impact)   : {len(wins)}/{len(ptrs)}  "
              f"({len(wins)/len(ptrs)*100:.0f}%)")
        print(f"  Avg portfolio impact/trade: {np.mean(ptrs):+.1f}%  "
              f"(median {np.median(ptrs):+.1f}%)")
        avg_hold = np.mean([t.hold_weeks for t in closed if t.hold_weeks])
        print(f"  Avg hold time            : {avg_hold:.0f} weeks")
    print(f"\n  Total strategy return    : {total_return:+.1f}%")
    print(f"  Strategy CAGR            : {cagr:+.1f}% p.a.")
    print(f"  Max portfolio drawdown   : -{port_max_dd:.1f}%")
    print(f"  Time deployed            : ~{deployed_pct:.0f}% of weeks")
    print(f"\n  Buy-and-hold return      : {bah:+.1f}%  (CAGR {bah_cagr:+.1f}% p.a.)")
    print(f"\n  Caveats:")
    print(f"  - Entry/exit at Friday close; intraday timing is not modeled")
    print(f"  - Execution costs modeled as flat bps; live fills can be worse in panics")
    print(f"  - F&G approximated from VIX (lower accuracy pre-2012)")
    print(f"  - NAAIM started 2006; credit (HYG) started 2007")
    print(f"  - Backtest uses single combined exit; live uses 3-tranche exits")
    print(f"{'='*74}\n")


# ── CSV export ────────────────────────────────────────────────────────────────

def save_csv(trades: list[Trade], path: str) -> None:
    rows = [{
        "entry_date":        t.entry_date,
        "exit_date":         t.exit_date,
        "entry_price":       t.entry_price,
        "exit_price":        t.exit_price,
        "entry_allocation":  t.entry_allocation,
        "etf_return_pct":    round(t.etf_return_pct, 2) if t.etf_return_pct else None,
        "portfolio_return_pct": round(t.portfolio_return_pct, 2) if t.portfolio_return_pct else None,
        "hold_weeks":        t.hold_weeks,
        "max_drawdown":      round(t.max_drawdown_pct, 2),
        "entry_tier":        t.entry_tier,
        "open":              t.open_trade,
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
    parser.add_argument("--cost-bps", type=float, default=_DEFAULT_COST_BPS,
                        help="One-way execution cost in basis points per traded dollar")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-download all data")
    args = parser.parse_args()

    df = load_data(args.ticker.upper(), args.start, args.no_cache)
    trades, equity_series = simulate(df, cost_bps=args.cost_bps)
    report(trades, df, args.ticker.upper(), equity_series, cost_bps=args.cost_bps)

    if args.csv:
        save_csv(trades, args.csv)


if __name__ == "__main__":
    main()
