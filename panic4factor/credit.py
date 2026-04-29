"""
Credit Stress Quantification — replaces the manual credit_crisis flag.

Three market-based signals (all via yfinance, no API key required):

  Signal 1 — HYG vs 200-day MA
    High yield ETF below its long-term trend = credit markets under stress.

  Signal 2 — HYG vs LQD 4-week relative performance
    HY underperforming investment grade = investors fleeing junk, not just equities.
    This is the key tell: in a pure equity correction HYG/LQD spread is modest;
    in a true credit crisis HY blows out while IG holds better.

  Signal 3 — MOVE Index proxy (TLT realized vol)
    MOVE measures bond market implied vol. Proxy: 10-day realized vol of TLT.
    Spiking bond vol = forced deleveraging / liquidity seizure in credit markets.

Optional (set FRED_API_KEY env var for higher accuracy):
  Signal 1 replaced by BAMLH0A0HYM2 HY OAS level + 4-week change.
  This is the definitive, direct credit spread measure.
  Free FRED key: https://fred.stlouisfed.org/docs/api/api_key.html

Crisis threshold: score ≥ 5 out of 8 possible points.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class CreditStressData:
    """Raw inputs for credit stress scoring."""
    hyg_price: float
    hyg_ma200: float
    lqd_price: float
    lqd_price_4w: float
    hyg_price_4w: float
    tlt_hist_10d: pd.Series   # 10 daily closes for realized vol
    vix: float
    # FRED optional
    hy_oas: Optional[float] = None
    hy_oas_4w: Optional[float] = None


@dataclass
class CreditStressResult:
    score: int
    max_score: int
    is_crisis: bool
    source: str               # "FRED" or "yfinance"

    # breakdown
    spread_score: int         # HY spread level
    spread_change_score: int  # HY spread widening speed
    relative_score: int       # HY vs IG relative performance
    vol_score: int            # bond market vol

    # raw values
    hyg_vs_200ma_pct: float        # negative = below 200MA
    hyg_lqd_4w_rel_pct: float      # negative = HYG underperformed LQD
    tlt_realized_vol_10d: float    # annualized
    hy_oas: Optional[float]
    hy_oas_4w_change: Optional[float]

    details: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        crisis_tag = "*** CREDIT CRISIS ***" if self.is_crisis else "no crisis"
        lines = [
            "── Credit Stress ────────────────────────────────",
            f"  Source                  : {self.source}",
            f"  Credit stress score     : {self.score} / {self.max_score}  ({crisis_tag})",
            f"  HY spread score         : {self.spread_score}  ({_fmt_oas(self.hy_oas)})",
            f"  HY spread change score  : {self.spread_change_score}  ({_fmt_oas_chg(self.hy_oas_4w_change)})",
            f"  HYG vs IG (4wk rel)     : {self.relative_score}  ({self.hyg_lqd_4w_rel_pct:+.1f}%)",
            f"  Bond vol (TLT 10d real) : {self.vol_score}  ({self.tlt_realized_vol_10d:.0f}% ann.)",
            f"  HYG vs 200MA            : {self.hyg_vs_200ma_pct:+.1f}%",
        ]
        for d in self.details:
            lines.append(f"  ⚠  {d}")
        lines.append("-" * 50)
        return "\n".join(lines)


def _fmt_oas(oas: Optional[float]) -> str:
    return f"{oas:.0f} bps" if oas is not None else "n/a (no FRED key)"


def _fmt_oas_chg(chg: Optional[float]) -> str:
    if chg is None:
        return "n/a"
    return f"{chg:+.0f} bps in 4 wks"


# ── Scoring tables ────────────────────────────────────────────────────────────

def _score_hy_oas_level(oas: float) -> int:
    """FRED BAMLH0A0HYM2 — basis points. Max 4 pts."""
    if oas > 800:  return 4
    if oas > 650:  return 3
    if oas > 500:  return 2
    if oas > 400:  return 1
    return 0


def _score_hy_oas_change(change_bps: float) -> int:
    """4-week widening. Max 3 pts."""
    if change_bps > 200:  return 3
    if change_bps > 100:  return 2
    if change_bps > 50:   return 1
    return 0


def _score_hyg_vs_200ma(hyg: float, ma200: float) -> int:
    """
    HYG price vs its 200-day MA. Supporting signal only — max 2 pts.

    Intentionally capped low because in a pure rate bear market (2022) HYG can
    be 10%+ below its 200MA with zero credit spread widening. This signal alone
    should never be enough to declare a credit crisis.
    """
    pct = (hyg - ma200) / ma200 * 100   # negative = below MA
    if pct < -10:  return 2
    if pct < -4:   return 1
    return 0


def _score_hyg_lqd_relative(hyg_ret: float, lqd_ret: float) -> int:
    """
    HY vs IG 4-week total return differential. This is the decisive signal.

    In a pure rate sell-off (2022), HYG has SHORTER duration than LQD so it
    actually OUTPERFORMS — rel > 0, score = 0. Only when credit spreads blow
    out do HY bonds underperform IG (HYG -15% vs LQD -4% in Mar 2020).

    Max 4 pts.
    """
    rel = (hyg_ret - lqd_ret) * 100   # pct points; negative = HYG underperforms LQD
    if rel < -8:   return 4   # massive spread blow-out (COVID Mar 2020)
    if rel < -5:   return 3   # severe stress (2008 peak)
    if rel < -3:   return 2
    if rel < -1.5: return 1
    return 0


def _score_tlt_vol(tlt_series: pd.Series) -> int:
    """
    10-day realized vol of TLT as MOVE index proxy.
    Annualized. Normal ~10-15%, stress >20%, crisis >30%.
    Max 1 pt. (Tiebreaker — bond vol alone doesn't mean credit crisis.)
    """
    rets = tlt_series.pct_change().dropna()
    if len(rets) < 5:
        return 0
    ann_vol = rets.std() * (252 ** 0.5) * 100
    return 1 if ann_vol > 25 else 0


# ── Main scoring function ─────────────────────────────────────────────────────

def score_credit_stress(data: CreditStressData) -> CreditStressResult:
    """
    Returns a CreditStressResult.
    Uses FRED data when available, yfinance proxies otherwise.

    yfinance mode — max 7 pts (2 + 4 + 1), crisis threshold ≥ 5.
    FRED mode     — max 8 pts (4 + 3 + 1), crisis threshold ≥ 5.

    Key design principle: the HYG vs LQD 4-week relative is the decisive signal.
    In a rate-only sell-off (2022) HYG outperforms LQD (shorter duration),
    so the relative score = 0 and a credit crisis is NOT declared even when
    HYG is well below its 200MA.
    """
    hyg_vs_200ma_pct = (data.hyg_price - data.hyg_ma200) / data.hyg_ma200 * 100
    hyg_4w_ret = (data.hyg_price - data.hyg_price_4w) / data.hyg_price_4w
    lqd_4w_ret = (data.lqd_price - data.lqd_price_4w) / data.lqd_price_4w
    hyg_lqd_rel_pct = (hyg_4w_ret - lqd_4w_ret) * 100

    tlt_vol = _score_tlt_vol(data.tlt_hist_10d)
    tlt_ann_vol = (
        data.tlt_hist_10d.pct_change().dropna().std() * (252 ** 0.5) * 100
        if len(data.tlt_hist_10d) >= 5 else 0.0
    )

    details = []

    if data.hy_oas is not None and data.hy_oas_4w is not None:
        # FRED path — direct credit spread measurement, most accurate
        source = "FRED + yfinance"
        oas_chg = data.hy_oas - data.hy_oas_4w
        spread_score  = _score_hy_oas_level(data.hy_oas)
        spread_change = _score_hy_oas_change(oas_chg)
        rel_score     = _score_hyg_lqd_relative(hyg_4w_ret, lqd_4w_ret)
        vol_score     = tlt_vol
        total = spread_score + spread_change + rel_score + vol_score
        hy_oas_4w_change = oas_chg
    else:
        # yfinance-only path
        source = "yfinance"
        spread_score  = _score_hyg_vs_200ma(data.hyg_price, data.hyg_ma200)   # max 2
        spread_change = 0   # no direct OAS change without FRED
        rel_score     = _score_hyg_lqd_relative(hyg_4w_ret, lqd_4w_ret)       # max 4
        vol_score     = tlt_vol                                                  # max 1
        total = spread_score + spread_change + rel_score + vol_score
        details.append("Set FRED_API_KEY for direct HY OAS data (more accurate)")
        hy_oas_4w_change = None

    is_crisis = total >= 5

    if is_crisis:
        details.append(
            "All position sizes HALVED — credit markets show systemic stress."
        )
    if hyg_lqd_rel_pct < -3:
        details.append(
            f"HYG underperforming LQD by {abs(hyg_lqd_rel_pct):.1f}% in 4 wks — "
            "junk bonds selling specifically, not just equities."
        )
    if tlt_ann_vol > 25:
        details.append(
            f"Bond vol elevated ({tlt_ann_vol:.0f}% ann.) — forced deleveraging risk."
        )

    max_score = 8 if (data.hy_oas is not None) else 7
    return CreditStressResult(
        score=total,
        max_score=max_score,
        is_crisis=is_crisis,
        source=source,
        spread_score=spread_score,
        spread_change_score=spread_change,
        relative_score=rel_score,
        vol_score=vol_score,
        hyg_vs_200ma_pct=hyg_vs_200ma_pct,
        hyg_lqd_4w_rel_pct=hyg_lqd_rel_pct,
        tlt_realized_vol_10d=tlt_ann_vol,
        hy_oas=data.hy_oas,
        hy_oas_4w_change=hy_oas_4w_change,
        details=details,
    )


# ── Optional FRED fetch ───────────────────────────────────────────────────────

def _try_fetch_fred_oas() -> tuple[Optional[float], Optional[float]]:
    """
    Returns (current_oas_bps, oas_4w_ago_bps) or (None, None) if FRED unavailable.
    Requires FRED_API_KEY environment variable.
    FRED series: BAMLH0A0HYM2 (ICE BofA US High Yield Option-Adjusted Spread).
    """
    api_key = os.environ.get("FRED_API_KEY", "").strip()
    if not api_key:
        return None, None
    try:
        import requests
        url = (
            "https://api.stlouisfed.org/fred/series/observations"
            f"?series_id=BAMLH0A0HYM2&api_key={api_key}"
            "&file_type=json&sort_order=desc&limit=30"
        )
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        obs = [
            o for o in resp.json()["observations"]
            if o["value"] != "."
        ]
        if len(obs) < 2:
            return None, None
        current = float(obs[0]["value"]) * 100    # FRED stores in %, convert to bps
        four_wk = float(obs[min(19, len(obs)-1)]["value"]) * 100
        return current, four_wk
    except Exception:
        return None, None
