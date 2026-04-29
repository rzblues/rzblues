"""
Position sizing and exit signal logic.

Instruments: QQQ, SPY (ETF cash) or ES / SPX (futures).
No individual stocks, no leveraged ETFs as primary vehicle.

Entry tiers — % of designated strategy capital (before filter caps):
  Tier 1 (score 50-64): 20%  probe
  Tier 2 (score 65-79): add to 50% total
  Tier 3 (score 80-89): add to 80% total
  Tier 4 (score ≥90):   add to 100% total

These are *cumulative targets*, not per-trade additions.
Filters cap the maximum of all tiers.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List

from .scorer import PanicScore
from .filters import FilterResult


class Instrument(str, Enum):
    QQQ = "QQQ"
    SPY = "SPY"
    ES  = "ES (E-mini futures)"
    SPX = "SPX (options / cash)"


# Target cumulative allocation by tier (fraction of strategy capital)
_TIER_TARGETS = {
    "NO_SIGNAL":      0.00,
    "TIER_1_PROBE":   0.20,
    "TIER_2_CORE":    0.50,
    "TIER_3_HEAVY":   0.80,
    "TIER_4_MAX":     1.00,
}


@dataclass
class EntryPlan:
    tier: str
    raw_target_pct: float          # before filters
    effective_target_pct: float    # after filters
    current_allocation_pct: float  # what you already hold in this trade
    add_now_pct: float             # incremental capital to deploy now
    instruments: List[str]
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.add_now_pct <= 0:
            action = "  ACTION: HOLD — already at or above target, or no signal."
        else:
            action = f"  ACTION: ADD {self.add_now_pct*100:.0f}% of strategy capital now."

        inst_line = "  Instruments: " + " / ".join(self.instruments)
        lines = [
            "── Entry Plan ───────────────────────────────────",
            f"  Signal tier              : {self.tier}",
            f"  Raw cumulative target    : {self.raw_target_pct*100:.0f}%",
            f"  After-filter target      : {self.effective_target_pct*100:.0f}%",
            f"  Current allocation held  : {self.current_allocation_pct*100:.0f}%",
            action,
            inst_line,
        ]
        for n in self.notes:
            lines.append(f"  ⚠  {n}")
        lines.append("-" * 50)
        return "\n".join(lines)


@dataclass
class ExitPlan:
    entry_price: float
    take_profit_1_pct: float = 0.22   # sell 1/3
    take_profit_2_fg_threshold: float = 50.0   # Fear & Greed back to neutral → sell 1/3
    take_profit_3_vix_threshold: float = 20.0  # VIX back below 20, index near prior high → last 1/3

    def levels(self, current_fg: float = None, current_vix: float = None) -> str:
        p1 = self.entry_price * (1 + self.take_profit_1_pct)
        lines = [
            "── Exit Plan (3-tranche) ────────────────────────",
            f"  Tranche 1 (sell 1/3): +{self.take_profit_1_pct*100:.0f}% from avg entry → ${p1:,.2f}",
            f"  Tranche 2 (sell 1/3): Fear & Greed recovers to >{self.take_profit_2_fg_threshold:.0f}",
            f"  Tranche 3 (sell 1/3): VIX <{self.take_profit_3_vix_threshold:.0f} + index near 52-wk high",
            "  (Long-term ETF holders may keep last tranche as core position)",
            "-" * 50,
        ]
        if current_fg is not None and current_fg >= self.take_profit_2_fg_threshold:
            lines.insert(-1, "  *** Tranche 2 trigger MET — consider selling 1/3 ***")
        if current_vix is not None and current_vix < self.take_profit_3_vix_threshold:
            lines.insert(-1, "  *** Tranche 3 trigger MET — consider selling final 1/3 ***")
        return "\n".join(lines)


# Preferred instruments for index-only traders (priority order)
INDEX_INSTRUMENTS = [Instrument.QQQ.value, Instrument.SPY.value,
                     Instrument.ES.value, Instrument.SPX.value]


def compute_entry_plan(
    panic: PanicScore,
    filters: FilterResult,
    current_allocation_pct: float = 0.0,
    instruments: List[str] = None,
) -> EntryPlan:
    """
    current_allocation_pct: fraction of strategy capital already deployed in
    this panic trade (from prior tier buys). Drives incremental add size.
    """
    tier = panic.signal_tier()
    raw_target = _TIER_TARGETS[tier]
    effective_target = raw_target * filters.effective_cap()

    add_now = max(0.0, effective_target - current_allocation_pct)

    notes = []
    if filters.price_vs_200ma < 1.0:
        notes.append(
            "Price is BELOW 200MA. Buy in tranches; wait for 50MA reclaim before adding more."
        )
    if filters.credit_crisis:
        notes.append(
            "Credit crisis flag active. All sizes halved. Do not force entries."
        )
    if panic.vix.score == 35:  # VIX >45
        notes.append(
            "VIX >45: systemic risk possible. Verify this is a panic dip, not a liquidity crisis."
        )
    if tier == "NO_SIGNAL":
        notes.append("Score <50 — stand aside. Watch, don't trade.")

    return EntryPlan(
        tier=tier,
        raw_target_pct=raw_target,
        effective_target_pct=effective_target,
        current_allocation_pct=current_allocation_pct,
        add_now_pct=add_now,
        instruments=instruments or INDEX_INSTRUMENTS,
        notes=notes,
    )
