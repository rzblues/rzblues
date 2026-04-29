"""
4-factor panic scoring.

Each function returns (score, max_score, label) for display.
Total max = 100.
"""

from dataclasses import dataclass


@dataclass
class FactorScore:
    name: str
    score: int
    max_score: int
    level: str  # human-readable interpretation


def score_vix(vix: float) -> FactorScore:
    """VIX fear score — max 35 pts."""
    if vix >= 45:
        s, level = 35, "Extreme panic (systemic risk possible)"
    elif vix >= 35:
        s, level = 32, "True panic — strong edge"
    elif vix >= 30:
        s, level = 22, "Elevated fear — some edge"
    elif vix >= 25:
        s, level = 10, "Nervousness — not enough"
    else:
        s, level = 0, "Complacency — no signal"
    return FactorScore("VIX", s, 35, level)


def score_fear_greed(fg: float) -> FactorScore:
    """CNN Fear & Greed score — max 25 pts.

    fg: 0-100, lower = more fear.
    """
    if fg < 15:
        s, level = 25, "Extreme fear"
    elif fg < 25:
        s, level = 17, "Fear zone"
    elif fg < 40:
        s, level = 8, "Mild fear"
    else:
        s, level = 0, "Neutral / greed — no signal"
    return FactorScore("Fear & Greed", s, 25, level)


def score_aaii(bull_bear_spread: float) -> FactorScore:
    """AAII Bull-Bear Spread score — max 20 pts.

    bull_bear_spread = Bullish% - Bearish% (e.g. -30 means very bearish).
    """
    if bull_bear_spread < -25:
        s, level = 20, "Extreme retail capitulation"
    elif bull_bear_spread < -10:
        s, level = 12, "Retail clearly bearish"
    elif bull_bear_spread < 0:
        s, level = 5, "Slight bearish lean"
    else:
        s, level = 0, "Retail still bullish — no signal"
    return FactorScore("AAII Bull-Bear Spread", s, 20, level)


def score_naaim(exposure: float) -> FactorScore:
    """NAAIM Exposure Index score — max 20 pts.

    exposure: 0-200 (can exceed 100 if leveraged).
    Typical range is 0-100 for this strategy.
    """
    if exposure < 30:
        s, level = 20, "Institutions heavily de-risked"
    elif exposure < 50:
        s, level = 12, "Institutions underweight equities"
    elif exposure < 70:
        s, level = 5, "Modest institutional caution"
    else:
        s, level = 0, "Institutions fully invested — no signal"
    return FactorScore("NAAIM Exposure", s, 20, level)


@dataclass
class PanicScore:
    vix: FactorScore
    fear_greed: FactorScore
    aaii: FactorScore
    naaim: FactorScore

    @property
    def total(self) -> int:
        return self.vix.score + self.fear_greed.score + self.aaii.score + self.naaim.score

    @property
    def max_total(self) -> int:
        return 100

    def signal_tier(self) -> str:
        t = self.total
        if t >= 90:
            return "TIER_4_MAX"
        if t >= 80:
            return "TIER_3_HEAVY"
        if t >= 65:
            return "TIER_2_CORE"
        if t >= 50:
            return "TIER_1_PROBE"
        return "NO_SIGNAL"

    def __str__(self) -> str:
        lines = [
            "=" * 52,
            f"  Panic 4-Factor Score: {self.total} / {self.max_total}",
            "=" * 52,
            f"  VIX              {self.vix.score:>3} / {self.vix.max_score}  {self.vix.level}",
            f"  Fear & Greed     {self.fear_greed.score:>3} / {self.fear_greed.max_score}  {self.fear_greed.level}",
            f"  AAII B-B Spread  {self.aaii.score:>3} / {self.aaii.max_score}  {self.aaii.level}",
            f"  NAAIM Exposure   {self.naaim.score:>3} / {self.naaim.max_score}  {self.naaim.level}",
            "-" * 52,
            f"  Tier: {self.signal_tier()}",
            "=" * 52,
        ]
        return "\n".join(lines)


def compute_panic_score(
    vix: float,
    fear_greed: float,
    aaii_bull_bear_spread: float,
    naaim_exposure: float,
) -> PanicScore:
    return PanicScore(
        vix=score_vix(vix),
        fear_greed=score_fear_greed(fear_greed),
        aaii=score_aaii(aaii_bull_bear_spread),
        naaim=score_naaim(naaim_exposure),
    )
