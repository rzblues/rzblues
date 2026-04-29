"""
Three guard filters that cap allowable position size before signals are applied.

Filter 1 — Drawdown from 52-week high
Filter 2 — Price vs 200-day MA
Filter 3 — Credit / macro crisis flag (manual)
"""

from dataclasses import dataclass


@dataclass
class FilterResult:
    drawdown_pct: float          # positive = index is below 52-wk high
    price_vs_200ma: float        # current price / 200ma
    credit_crisis: bool          # manually flagged

    # derived caps (0.0 – 1.0, fraction of strategy capital)
    drawdown_cap: float = 1.0
    ma_cap: float = 1.0
    crisis_multiplier: float = 1.0

    def effective_cap(self) -> float:
        """Tightest constraint wins, then apply crisis multiplier."""
        cap = min(self.drawdown_cap, self.ma_cap)
        return cap * self.crisis_multiplier

    def __str__(self) -> str:
        above_below = "above" if self.price_vs_200ma >= 1.0 else "BELOW"
        lines = [
            "── Filters ─────────────────────────────────────",
            f"  Drawdown from 52-wk high : {self.drawdown_pct:.1f}%  → cap {self.drawdown_cap*100:.0f}%",
            f"  Price vs 200-day MA       : {self.price_vs_200ma:.3f}x ({above_below} 200MA) → cap {self.ma_cap*100:.0f}%",
            f"  Credit / macro crisis flag: {'YES — all positions ÷2' if self.credit_crisis else 'no'}",
            f"  Effective position cap    : {self.effective_cap()*100:.0f}% of strategy capital",
            "-" * 50,
        ]
        return "\n".join(lines)


def apply_filters(
    drawdown_pct: float,
    price_vs_200ma: float,
    credit_crisis: bool,
) -> FilterResult:
    """
    drawdown_pct: % drop from 52-week high (positive number, e.g. 15 = -15%).
    price_vs_200ma: current_price / ma200 (e.g. 0.92 = 8% below 200MA).
    credit_crisis: True when bank stress / credit spreads spiking / liquidity seize.
    """
    result = FilterResult(
        drawdown_pct=drawdown_pct,
        price_vs_200ma=price_vs_200ma,
        credit_crisis=credit_crisis,
    )

    # Filter 1 — drawdown gate
    if drawdown_pct < 10:
        result.drawdown_cap = 0.20   # barely pulled back — probe only
    elif drawdown_pct < 20:
        result.drawdown_cap = 0.50
    elif drawdown_pct < 30:
        result.drawdown_cap = 0.80
    else:
        result.drawdown_cap = 1.00   # deep bear — full strategy capital allowed

    # Filter 2 — 200MA gate
    # Below 200MA: staged entries only, never load up in one shot.
    # The cap itself doesn't restrict total size but the sizer enforces incremental adds.
    if price_vs_200ma >= 1.0:
        result.ma_cap = 1.00   # above 200MA — normal operation
    else:
        result.ma_cap = 0.80   # below 200MA — leave 20% dry powder for deeper dip

    # Filter 3 — credit / systemic crisis halves everything
    result.crisis_multiplier = 0.50 if credit_crisis else 1.00

    return result
