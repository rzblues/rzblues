"""
Persistent trading state — tracks current allocation across runs.

Stored in ~/.panic4factor_cache/state.json

Fields:
  allocation       : fraction of strategy capital currently deployed (0.0–1.0)
  peak_allocation  : highest allocation reached in current trade
  avg_entry_price  : weighted average entry price
  entry_date       : date of first entry in current trade
  last_tier        : tier from last strategy run
  tranches_alerted : which exit tranches have already been notified (list of 3 bools)
  last_run_ts      : Unix timestamp of last run
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

_STATE_FILE = Path.home() / ".panic4factor_cache" / "state.json"


@dataclass
class TradingState:
    allocation: float = 0.0          # current deployed fraction
    peak_allocation: float = 0.0     # for tranche sizing
    avg_entry_price: Optional[float] = None
    entry_date: Optional[str] = None  # ISO date string
    last_tier: str = "NO_SIGNAL"
    tranches_alerted: list = field(default_factory=lambda: [False, False, False])
    last_run_ts: float = 0.0

    # ── persistence ──────────────────────────────────────────────────────────

    @classmethod
    def load(cls) -> "TradingState":
        if not _STATE_FILE.exists():
            return cls()
        try:
            data = json.loads(_STATE_FILE.read_text())
            return cls(**data)
        except Exception:
            return cls()

    def save(self) -> None:
        _STATE_FILE.parent.mkdir(exist_ok=True)
        _STATE_FILE.write_text(json.dumps(asdict(self), indent=2))

    # ── entry helpers ─────────────────────────────────────────────────────────

    def update_entry(self, price: float, new_allocation: float) -> float:
        """
        Record a new or additional buy.
        Returns the incremental fraction to deploy (add_now).
        """
        add_now = max(0.0, new_allocation - self.allocation)
        if add_now <= 0:
            return 0.0

        if self.allocation == 0.0:
            # Fresh trade
            self.avg_entry_price = price
            from datetime import date
            self.entry_date = date.today().isoformat()
            self.tranches_alerted = [False, False, False]
            self.peak_allocation = new_allocation
        else:
            # Add to existing — update weighted avg entry price
            total = self.allocation + add_now
            self.avg_entry_price = (
                self.avg_entry_price * self.allocation + price * add_now
            ) / total
            self.peak_allocation = max(self.peak_allocation, new_allocation)

        self.allocation = new_allocation
        self.last_tier = _tier_for_allocation(new_allocation)
        self.last_run_ts = time.time()
        return add_now

    def close(self) -> None:
        """Fully reset — position closed."""
        self.allocation = 0.0
        self.peak_allocation = 0.0
        self.avg_entry_price = None
        self.entry_date = None
        self.last_tier = "NO_SIGNAL"
        self.tranches_alerted = [False, False, False]

    # ── exit helpers ──────────────────────────────────────────────────────────

    def check_exits(self, price: float, vix: float, fg: float, high_52w: float
                    ) -> list[tuple[int, str]]:
        """
        Returns list of (tranche_number, reason) for tranches that just fired
        and haven't been alerted yet.
        """
        if not self.allocation or not self.avg_entry_price:
            return []

        fires = []
        entry = self.avg_entry_price

        if not self.tranches_alerted[0]:
            if price >= entry * 1.22:
                fires.append((1, f"Price +22% from avg entry ({entry:.2f} → {price:.2f})"))
                self.tranches_alerted[0] = True

        if not self.tranches_alerted[1]:
            if fg >= 50:
                fires.append((2, f"Fear & Greed recovered to {fg:.0f} (threshold 50)"))
                self.tranches_alerted[1] = True

        if not self.tranches_alerted[2]:
            near_high = high_52w > 0 and price >= high_52w * 0.95
            if vix < 20 and near_high:
                fires.append((3, f"VIX {vix:.1f} < 20 and price near 52-wk high"))
                self.tranches_alerted[2] = True
                self.close()   # fully out after tranche 3

        return fires

    @property
    def in_position(self) -> bool:
        return self.allocation > 0.0

    def summary(self) -> str:
        if not self.in_position:
            return "Position: FLAT"
        ret = ""
        if self.avg_entry_price:
            ret = f"  Avg entry: {self.avg_entry_price:.2f}\n"
        return (
            f"Position: {self.allocation*100:.0f}% deployed "
            f"(peak {self.peak_allocation*100:.0f}%)\n"
            f"{ret}"
            f"  Entry date: {self.entry_date}\n"
            f"  Tier: {self.last_tier}\n"
            f"  Tranches sold: {sum(self.tranches_alerted)}/3"
        )


def _tier_for_allocation(alloc: float) -> str:
    if alloc >= 1.0:  return "TIER_4_MAX"
    if alloc >= 0.8:  return "TIER_3_HEAVY"
    if alloc >= 0.5:  return "TIER_2_CORE"
    if alloc >= 0.2:  return "TIER_1_PROBE"
    return "NO_SIGNAL"
