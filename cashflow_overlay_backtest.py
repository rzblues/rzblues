#!/usr/bin/env python3
"""
Cashflow crisis-buy backtest.

This tests the real personal-investor model:
  - new cash arrives every trading day
  - optional base DCA buys happen every day
  - reserve cash earns a cash rate
  - panic signals deploy reserve cash into a panic asset
  - purchased shares are held; the strategy does not sell

The benchmark is daily DCA into a chosen asset over the same dates.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import yfinance as yf

from backtest import _CASH_RATE_WEEKLY, _TIER_TARGETS, load_data
from panic4factor.credit import _score_hyg_lqd_relative, _score_hyg_vs_200ma
from panic4factor.filters import apply_filters
from panic4factor.scorer import compute_panic_score


CASH_RATE_DAILY = (1 + _CASH_RATE_WEEKLY) ** (52 / 252) - 1


@dataclass
class Buy:
    date: pd.Timestamp
    asset: str
    amount: float
    price: float
    reason: str
    tier: str = ""
    score: int = 0
    vix: float = 0.0
    drawdown: float = 0.0


def _daily_prices(tickers: list[str], start: str) -> pd.DataFrame:
    raw = yf.download(
        tickers,
        start=start,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty or "Close" not in raw:
        raise RuntimeError(f"yfinance returned no usable daily prices for {tickers}")
    closes = raw["Close"].copy()
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(tickers[0])
    missing = [t for t in tickers if t not in closes.columns or closes[t].dropna().empty]
    if missing:
        raise RuntimeError(f"missing daily price data for {missing}")
    return closes.dropna(how="all")


def _weekly_targets(signal_df: pd.DataFrame) -> pd.Series:
    idx_col = signal_df.attrs["idx_col"]
    targets: dict[pd.Timestamp, tuple[float, str, int, float, float]] = {}

    for ts, row in signal_df.iterrows():
        price = row[idx_col]
        vix = row["vix"]
        fg = row.get("fg_approx", 50.0)
        aaii = row.get("aaii_spread", 0.0)
        naaim = row.get("naaim", 50.0)
        ma200 = row.get("ma200", price)
        high52w = row.get("high52w", price)
        hyg = row["hyg"]
        lqd = row["lqd"]
        hyg_ma = row.get("hyg_ma200", hyg)
        hyg_4w = row.get("hyg_4w", hyg)
        lqd_4w = row.get("lqd_4w", lqd)
        tlt_vol = row.get("tlt_vol10", 0.0)

        if any(pd.isna(v) for v in [price, vix, ma200]):
            continue
        if pd.isna(aaii):
            aaii = 0.0
        if pd.isna(naaim):
            naaim = 50.0
        if pd.isna(hyg_ma):
            hyg_ma = hyg
        if pd.isna(hyg_4w):
            hyg_4w = hyg
        if pd.isna(lqd_4w):
            lqd_4w = lqd
        if pd.isna(tlt_vol):
            tlt_vol = 0.0
        if pd.isna(high52w):
            high52w = price

        panic = compute_panic_score(vix, fg, aaii, naaim)
        hyg_ret = (hyg - hyg_4w) / hyg_4w if hyg_4w else 0.0
        lqd_ret = (lqd - lqd_4w) / lqd_4w if lqd_4w else 0.0
        credit_score = (
            _score_hyg_vs_200ma(hyg, hyg_ma)
            + _score_hyg_lqd_relative(hyg_ret, lqd_ret)
            + (1 if tlt_vol > 25 else 0)
        )
        drawdown = (high52w - price) / high52w * 100 if high52w > 0 else 0.0
        p_vs_ma = price / ma200 if ma200 > 0 else 1.0
        filters = apply_filters(drawdown, p_vs_ma, credit_score >= 5)
        target = _TIER_TARGETS[panic.signal_tier()] * filters.effective_cap()
        targets[pd.Timestamp(ts)] = (target, panic.signal_tier(), panic.total, vix, drawdown)

    return pd.Series(targets)


def _stats(name: str, equity: pd.Series, contributions: float) -> dict:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    final = float(equity.iloc[-1])
    return {
        "name": name,
        "final": final,
        "profit": final - contributions,
        "return_on_contrib": (final / contributions - 1) * 100,
        "annualized_multiple": (final / contributions) ** (1 / years) * 100 - 100,
        "max_drawdown": float((equity / equity.cummax() - 1).min() * 100),
    }


def simulate(
    prices: pd.DataFrame,
    weekly_targets: pd.Series,
    benchmark_asset: str,
    base_asset: str,
    panic_asset: str,
    daily_contribution: float,
    base_dca_pct: float,
    min_panic_target: float,
    cost_bps: float,
) -> tuple[pd.Series, pd.Series, list[Buy], float]:
    cost = max(0.0, cost_bps) / 10_000
    cash = 0.0
    shares = {base_asset: 0.0, panic_asset: 0.0}
    benchmark_shares = 0.0
    contributions = 0.0
    buys: list[Buy] = []
    equity_rows: list[tuple[pd.Timestamp, float]] = []
    benchmark_rows: list[tuple[pd.Timestamp, float]] = []

    target_by_date = weekly_targets.to_dict()

    for ts, row in prices.iterrows():
        if any(asset not in row or pd.isna(row[asset]) for asset in [benchmark_asset, base_asset, panic_asset]):
            continue

        cash *= (1 + CASH_RATE_DAILY)
        cash += daily_contribution
        contributions += daily_contribution

        # Benchmark: daily DCA all cash into benchmark asset.
        benchmark_net = daily_contribution * (1 - cost)
        benchmark_shares += benchmark_net / float(row[benchmark_asset])

        # Strategy: buy base allocation daily, keep reserve for panic.
        base_amount = daily_contribution * base_dca_pct
        if base_amount > 0:
            net = base_amount * (1 - cost)
            shares[base_asset] += net / float(row[base_asset])
            cash -= base_amount

        if ts in target_by_date:
            target, tier, score, vix, drawdown = target_by_date[ts]
            if target >= min_panic_target and cash > 0:
                amount = cash
                net = amount * (1 - cost)
                shares[panic_asset] += net / float(row[panic_asset])
                cash = 0.0
                buys.append(Buy(
                    date=ts,
                    asset=panic_asset,
                    amount=amount,
                    price=float(row[panic_asset]),
                    reason="panic",
                    tier=tier,
                    score=int(score),
                    vix=float(vix),
                    drawdown=float(drawdown),
                ))

        value = cash + sum(shares[a] * float(row[a]) for a in shares)
        benchmark_value = benchmark_shares * float(row[benchmark_asset])
        equity_rows.append((ts, value))
        benchmark_rows.append((ts, benchmark_value))

    return (
        pd.Series([v for _, v in equity_rows], index=[d for d, _ in equity_rows]),
        pd.Series([v for _, v in benchmark_rows], index=[d for d, _ in benchmark_rows]),
        buys,
        contributions,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest daily cashflow + crisis-buy overlay")
    parser.add_argument("--start", default="2007-01-01")
    parser.add_argument("--benchmark-asset", default="QQQ")
    parser.add_argument("--base-asset", default="QQQ")
    parser.add_argument("--panic-asset", default="QLD")
    parser.add_argument("--signal-ticker", default="QQQ")
    parser.add_argument("--daily-contribution", type=float, default=100.0)
    parser.add_argument("--base-dca-pct", type=float, default=0.70)
    parser.add_argument("--min-panic-target", type=float, default=0.20)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    args = parser.parse_args()

    tickers = sorted({args.benchmark_asset, args.base_asset, args.panic_asset})
    prices = _daily_prices(tickers, args.start)
    signal_df = load_data(args.signal_ticker, args.start, no_cache=False)
    targets = _weekly_targets(signal_df)

    equity, benchmark, buys, contributions = simulate(
        prices=prices,
        weekly_targets=targets,
        benchmark_asset=args.benchmark_asset,
        base_asset=args.base_asset,
        panic_asset=args.panic_asset,
        daily_contribution=args.daily_contribution,
        base_dca_pct=args.base_dca_pct,
        min_panic_target=args.min_panic_target,
        cost_bps=args.cost_bps,
    )

    idx = equity.index.intersection(benchmark.index)
    equity = equity.loc[idx]
    benchmark = benchmark.loc[idx]

    print(f"Period: {idx[0].date()} -> {idx[-1].date()}")
    print(f"Daily contribution: ${args.daily_contribution:,.2f}")
    print(f"Benchmark: daily DCA {args.benchmark_asset}")
    print(
        "Strategy: "
        f"{args.base_dca_pct*100:.0f}% daily DCA {args.base_asset}, "
        f"reserve buys {args.panic_asset} when panic target >= {args.min_panic_target*100:.0f}%"
    )
    print("")

    for row in [
        _stats(f"daily DCA {args.benchmark_asset}", benchmark, contributions),
        _stats("crisis overlay", equity, contributions),
    ]:
        print(
            f"{row['name']:22s} final=${row['final']:,.0f} "
            f"profit=${row['profit']:,.0f} "
            f"return_on_contrib={row['return_on_contrib']:7.1f}% "
            f"annualized_multiple={row['annualized_multiple']:5.2f}% "
            f"maxDD={row['max_drawdown']:6.1f}%"
        )

    print("")
    print(f"Panic buys: {len(buys)}")
    for buy in buys[-30:]:
        print(
            f"  {buy.date.date()} buy ${buy.amount:,.0f} {buy.asset} "
            f"@ {buy.price:,.2f} {buy.tier} score={buy.score} "
            f"vix={buy.vix:.1f} dd={buy.drawdown:.1f}%"
        )


if __name__ == "__main__":
    main()
