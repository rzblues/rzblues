"""Panic 4-Factor Buy Strategy — main orchestrator."""

from dataclasses import dataclass
from typing import List, Optional

from .scorer import PanicScore, compute_panic_score
from .filters import FilterResult, apply_filters
from .sizer import EntryPlan, ExitPlan, compute_entry_plan, INDEX_INSTRUMENTS
from .credit import CreditStressResult
from .data import MarketSnapshot


@dataclass
class StrategyOutput:
    snapshot: MarketSnapshot
    panic: PanicScore
    filters: FilterResult
    entry: EntryPlan
    exit_plan: Optional[ExitPlan] = None

    def report(self) -> str:
        lines = [
            "",
            "╔══════════════════════════════════════════════════╗",
            "║        PANIC 4-FACTOR BUY STRATEGY               ║",
            f"║  Instrument: {self.snapshot.ticker:<36}║",
            "╚══════════════════════════════════════════════════╝",
            "",
            f"  VIX             : {self.snapshot.vix:.2f}",
            f"  {self.snapshot.ticker} Price      : {self.snapshot.index_price:,.2f}",
            f"  52-wk High      : {self.snapshot.high_52w:,.2f}",
            f"  200-day MA      : {self.snapshot.ma_200:,.2f}",
            f"  Drawdown        : -{self.snapshot.drawdown_pct:.1f}%",
            f"  Fear & Greed    : {self.snapshot.fear_greed:.0f}",
            f"  AAII B-B Spread : {self.snapshot.aaii_bull_bear_spread:+.1f}",
            f"  NAAIM Exposure  : {self.snapshot.naaim_exposure:.0f}",
            "",
            str(self.panic),
            "",
            str(self.snapshot.credit),
            "",
            str(self.filters),
            "",
            str(self.entry),
        ]
        if self.exit_plan:
            lines += ["", self.exit_plan.levels(
                current_fg=self.snapshot.fear_greed,
                current_vix=self.snapshot.vix,
            )]
        lines.append("")
        return "\n".join(lines)


def run_strategy(
    snapshot: MarketSnapshot,
    current_allocation_pct: float = 0.0,
    avg_entry_price: Optional[float] = None,
    instruments: List[str] = None,
) -> StrategyOutput:
    panic = compute_panic_score(
        vix=snapshot.vix,
        fear_greed=snapshot.fear_greed,
        aaii_bull_bear_spread=snapshot.aaii_bull_bear_spread,
        naaim_exposure=snapshot.naaim_exposure,
    )
    filters = apply_filters(
        drawdown_pct=snapshot.drawdown_pct,
        price_vs_200ma=snapshot.price_vs_200ma,
        credit_crisis=snapshot.credit.is_crisis,   # auto-computed
    )
    entry = compute_entry_plan(
        panic=panic,
        filters=filters,
        current_allocation_pct=current_allocation_pct,
        instruments=instruments or INDEX_INSTRUMENTS,
    )
    exit_plan = ExitPlan(entry_price=avg_entry_price) if avg_entry_price else None

    return StrategyOutput(
        snapshot=snapshot,
        panic=panic,
        filters=filters,
        entry=entry,
        exit_plan=exit_plan,
    )
