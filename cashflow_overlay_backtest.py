#!/usr/bin/env python3
"""
Cashflow crisis-buy backtest.

This tests the real personal-investor model:
  - new cash arrives every trading day
  - optional base DCA buys happen every day
  - reserve cash earns a cash rate
  - panic signals deploy reserve cash into a panic asset
  - purchased leveraged panic shares can optionally be reduced into the base asset

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


@dataclass
class Sale:
    date: pd.Timestamp
    asset: str
    amount: float
    price: float
    reason: str
    profit_pct: float
    target: float


@dataclass
class SimulationResult:
    equity: pd.Series
    benchmark: pd.Series
    buys: list[Buy]
    sales: list[Sale]
    contributions: float


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


def _parse_pct_list(raw: str) -> list[float]:
    values: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value > 1:
            value /= 100
        if value < 0 or value > 1:
            raise argparse.ArgumentTypeError(f"percentage out of range: {item}")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("expected at least one percentage")
    return values


def _should_reduce(
    rule: str,
    target: float,
    profit_pct: float,
    hold_days: int,
    recovery_target: float,
    min_profit_pct: float,
    min_hold_days: int,
) -> tuple[bool, str]:
    if rule == "hold":
        return False, ""
    if hold_days < min_hold_days:
        return False, ""

    recovered = target <= recovery_target and profit_pct >= min_profit_pct
    profitable = profit_pct >= min_profit_pct
    base_rule = rule.removesuffix("-cap")
    if base_rule == "recovery" and recovered:
        return True, "recovery"
    if base_rule == "profit" and profitable:
        return True, "profit"
    if base_rule == "recovery-or-profit" and (recovered or profitable):
        return True, "recovery" if recovered else "profit"
    return False, ""


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
    deleveraging_rule: str,
    deleveraging_sell_pct: float,
    deleveraging_keep_pct: Optional[float],
    deleveraging_min_profit_pct: float,
    deleveraging_min_hold_days: int,
    recovery_panic_target: float,
) -> SimulationResult:
    cost = max(0.0, cost_bps) / 10_000
    cash = 0.0
    shares = {base_asset: 0.0, panic_asset: 0.0}
    panic_cost_basis = 0.0
    panic_first_buy_date: Optional[pd.Timestamp] = None
    benchmark_shares = 0.0
    contributions = 0.0
    buys: list[Buy] = []
    sales: list[Sale] = []
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
                panic_cost_basis += amount
                if panic_first_buy_date is None:
                    panic_first_buy_date = ts
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

            panic_value = shares[panic_asset] * float(row[panic_asset])
            if panic_value > 0 and panic_cost_basis > 0 and panic_first_buy_date is not None:
                profit_pct = (panic_value / panic_cost_basis - 1) * 100
                hold_days = int((ts - panic_first_buy_date).days)
                reduce_now, reason = _should_reduce(
                    deleveraging_rule,
                    target,
                    profit_pct,
                    hold_days,
                    recovery_panic_target,
                    deleveraging_min_profit_pct,
                    deleveraging_min_hold_days,
                )
                if reduce_now:
                    current_value = (
                        cash
                        + shares[base_asset] * float(row[base_asset])
                        + panic_value
                    )
                    if deleveraging_keep_pct is not None:
                        target_panic_value = current_value * min(max(deleveraging_keep_pct, 0.0), 1.0)
                        gross = max(0.0, panic_value - target_panic_value)
                        sold_shares = gross / float(row[panic_asset]) if gross > 0 else 0.0
                    else:
                        sell_fraction = min(max(deleveraging_sell_pct, 0.0), 1.0)
                        sold_shares = shares[panic_asset] * sell_fraction
                        gross = sold_shares * float(row[panic_asset])
                    if sold_shares <= 1e-9:
                        continue
                    sell_fraction = sold_shares / shares[panic_asset]
                    net = gross * (1 - cost)
                    shares[panic_asset] -= sold_shares
                    panic_cost_basis *= 1 - sell_fraction
                    shares[base_asset] += net / float(row[base_asset])
                    sales.append(Sale(
                        date=ts,
                        asset=panic_asset,
                        amount=gross,
                        price=float(row[panic_asset]),
                        reason=reason,
                        profit_pct=float(profit_pct),
                        target=float(target),
                    ))
                    if shares[panic_asset] <= 1e-9:
                        shares[panic_asset] = 0.0
                        panic_cost_basis = 0.0
                        panic_first_buy_date = None

        value = cash + sum(shares[a] * float(row[a]) for a in shares)
        benchmark_value = benchmark_shares * float(row[benchmark_asset])
        equity_rows.append((ts, value))
        benchmark_rows.append((ts, benchmark_value))

    return SimulationResult(
        equity=pd.Series([v for _, v in equity_rows], index=[d for d, _ in equity_rows]),
        benchmark=pd.Series([v for _, v in benchmark_rows], index=[d for d, _ in benchmark_rows]),
        buys=buys,
        sales=sales,
        contributions=contributions,
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
    parser.add_argument("--base-dca-pcts", type=_parse_pct_list)
    parser.add_argument("--min-panic-target", type=float, default=0.20)
    parser.add_argument(
        "--deleveraging-rule",
        choices=[
            "hold",
            "recovery",
            "profit",
            "recovery-or-profit",
            "recovery-cap",
            "profit-cap",
            "recovery-or-profit-cap",
        ],
        default="hold",
        help="How to reduce the panic asset after crisis buys.",
    )
    parser.add_argument("--deleveraging-rules", default="")
    parser.add_argument("--deleveraging-sell-pct", type=float, default=0.50)
    parser.add_argument(
        "--deleveraging-keep-pct",
        type=float,
        help="For *-cap rules, reduce panic asset down to this portfolio weight.",
    )
    parser.add_argument("--deleveraging-min-profit-pct", type=float, default=50.0)
    parser.add_argument("--deleveraging-min-hold-days", type=int, default=126)
    parser.add_argument("--recovery-panic-target", type=float, default=0.0)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    args = parser.parse_args()
    if args.deleveraging_keep_pct is not None and args.deleveraging_keep_pct > 1:
        args.deleveraging_keep_pct /= 100

    tickers = sorted({args.benchmark_asset, args.base_asset, args.panic_asset})
    prices = _daily_prices(tickers, args.start)
    signal_df = load_data(args.signal_ticker, args.start, no_cache=False)
    targets = _weekly_targets(signal_df)

    base_pcts = args.base_dca_pcts if args.base_dca_pcts is not None else [args.base_dca_pct]
    rules = [r.strip() for r in args.deleveraging_rules.split(",") if r.strip()] or [args.deleveraging_rule]
    valid_rules = {
        "hold",
        "recovery",
        "profit",
        "recovery-or-profit",
        "recovery-cap",
        "profit-cap",
        "recovery-or-profit-cap",
    }
    unknown_rules = sorted(set(rules) - valid_rules)
    if unknown_rules:
        parser.error(f"unknown deleveraging rule(s): {', '.join(unknown_rules)}")
    if any(rule.endswith("-cap") for rule in rules) and args.deleveraging_keep_pct is None:
        args.deleveraging_keep_pct = 0.30

    results: list[tuple[float, str, SimulationResult, dict, dict]] = []
    for base_pct in base_pcts:
        for rule in rules:
            result = simulate(
                prices=prices,
                weekly_targets=targets,
                benchmark_asset=args.benchmark_asset,
                base_asset=args.base_asset,
                panic_asset=args.panic_asset,
                daily_contribution=args.daily_contribution,
                base_dca_pct=base_pct,
                min_panic_target=args.min_panic_target,
                cost_bps=args.cost_bps,
                deleveraging_rule=rule,
                deleveraging_sell_pct=args.deleveraging_sell_pct,
                deleveraging_keep_pct=args.deleveraging_keep_pct,
                deleveraging_min_profit_pct=args.deleveraging_min_profit_pct,
                deleveraging_min_hold_days=args.deleveraging_min_hold_days,
                recovery_panic_target=args.recovery_panic_target,
            )
            idx = result.equity.index.intersection(result.benchmark.index)
            result.equity = result.equity.loc[idx]
            result.benchmark = result.benchmark.loc[idx]
            benchmark_stats = _stats(f"daily DCA {args.benchmark_asset}", result.benchmark, result.contributions)
            overlay_stats = _stats("crisis overlay", result.equity, result.contributions)
            results.append((base_pct, rule, result, benchmark_stats, overlay_stats))

    first_result = results[0][2]
    idx = first_result.equity.index.intersection(first_result.benchmark.index)

    print(f"Period: {idx[0].date()} -> {idx[-1].date()}")
    print(f"Daily contribution: ${args.daily_contribution:,.2f}")
    print(f"Benchmark: daily DCA {args.benchmark_asset}")
    print(f"Panic rule: reserve buys {args.panic_asset} when panic target >= {args.min_panic_target*100:.0f}%")
    print(
        "Deleveraging: "
        + (
            f"cap {args.panic_asset} at {args.deleveraging_keep_pct*100:.0f}% into {args.base_asset}, "
            if args.deleveraging_keep_pct is not None
            else f"sell {args.deleveraging_sell_pct*100:.0f}% into {args.base_asset}, "
        )
        +
        f"min profit {args.deleveraging_min_profit_pct:.0f}%, "
        f"min hold {args.deleveraging_min_hold_days}d, "
        f"recovery target <= {args.recovery_panic_target*100:.0f}%"
    )
    print("")

    if len(results) > 1:
        benchmark_stats = results[0][3]
        print(
            f"{'strategy':40s} {'final':>12s} {'ret':>8s} {'ann':>7s} "
            f"{'maxDD':>8s} {'buys':>5s} {'sells':>5s} {'vs bench':>10s}"
        )
        print(
            f"{benchmark_stats['name']:40s} ${benchmark_stats['final']:>11,.0f} "
            f"{benchmark_stats['return_on_contrib']:>7.1f}% "
            f"{benchmark_stats['annualized_multiple']:>6.2f}% "
            f"{benchmark_stats['max_drawdown']:>7.1f}% "
            f"{'':>5s} {'':>5s} {'':>10s}"
        )
        for base_pct, rule, result, _, overlay_stats in results:
            label = f"{base_pct*100:.0f}% {args.base_asset} + {args.panic_asset} {rule}"
            vs_bench = overlay_stats["final"] / benchmark_stats["final"] - 1
            print(
                f"{label:40s} ${overlay_stats['final']:>11,.0f} "
                f"{overlay_stats['return_on_contrib']:>7.1f}% "
                f"{overlay_stats['annualized_multiple']:>6.2f}% "
                f"{overlay_stats['max_drawdown']:>7.1f}% "
                f"{len(result.buys):>5d} {len(result.sales):>5d} "
                f"{vs_bench*100:>9.1f}%"
            )
        return

    base_pct, rule, result, benchmark_stats, overlay_stats = results[0]
    print(
        "Strategy: "
        f"{base_pct*100:.0f}% daily DCA {args.base_asset}, "
        f"reserve buys {args.panic_asset}, deleveraging={rule}"
    )
    print("")

    for row in [benchmark_stats, overlay_stats]:
        print(
            f"{row['name']:22s} final=${row['final']:,.0f} "
            f"profit=${row['profit']:,.0f} "
            f"return_on_contrib={row['return_on_contrib']:7.1f}% "
            f"annualized_multiple={row['annualized_multiple']:5.2f}% "
            f"maxDD={row['max_drawdown']:6.1f}%"
        )

    print("")
    print(f"Panic buys: {len(result.buys)}")
    for buy in result.buys[-30:]:
        print(
            f"  {buy.date.date()} buy ${buy.amount:,.0f} {buy.asset} "
            f"@ {buy.price:,.2f} {buy.tier} score={buy.score} "
            f"vix={buy.vix:.1f} dd={buy.drawdown:.1f}%"
        )
    print("")
    print(f"Panic reductions: {len(result.sales)}")
    for sale in result.sales[-30:]:
        print(
            f"  {sale.date.date()} sell ${sale.amount:,.0f} {sale.asset} "
            f"@ {sale.price:,.2f} {sale.reason} profit={sale.profit_pct:.1f}% "
            f"target={sale.target*100:.0f}%"
        )


if __name__ == "__main__":
    main()
