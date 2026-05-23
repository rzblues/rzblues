"""
MarketSnapshot — fully auto-fetched.

Call auto_fetch() to get a complete, ready-to-use snapshot.
Manual construction (build_snapshot_from_manual) retained for back-testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from . import fetch
from .credit import (
    CreditStressData,
    CreditStressResult,
    score_credit_stress,
    _try_fetch_fred_oas,
)


@dataclass
class MarketSnapshot:
    # price / vol
    ticker: str
    vix: float
    index_price: float
    high_52w: float
    ma_200: float

    # sentiment (auto-fetched)
    fear_greed: float             # 0–100 (CNN)
    aaii_bull_bear_spread: float  # Bullish% − Bearish%
    naaim_exposure: float         # 0–200

    # credit stress (quantified)
    credit: CreditStressResult

    @property
    def drawdown_pct(self) -> float:
        return (self.high_52w - self.index_price) / self.high_52w * 100

    @property
    def price_vs_200ma(self) -> float:
        return self.index_price / self.ma_200


def auto_fetch(ticker: str = "QQQ", no_cache: bool = False) -> MarketSnapshot:
    """
    Fetch all data sources automatically.
    Raises on hard failures (VIX, price).
    Warns but continues on soft failures (AAII, NAAIM, F&G).
    """
    if no_cache:
        from . import cache
        cache.clear()

    print(f"  Fetching VIX + {ticker} price … ", end="", flush=True)
    vix = fetch.fetch_vix()
    price_snap = fetch.fetch_price_snapshot(ticker)
    print("done")

    print("  Fetching CNN Fear & Greed … ", end="", flush=True)
    try:
        fg = fetch.fetch_fear_greed()
        print(f"done ({fg:.0f})")
    except Exception as e:
        print(f"FAILED ({e})")
        raise

    print("  Fetching AAII sentiment … ", end="", flush=True)
    try:
        aaii = fetch.fetch_aaii_bull_bear_spread()
        print(f"done ({aaii:+.1f})")
    except Exception as e:
        print(f"FAILED ({e})")
        raise

    print("  Fetching NAAIM exposure … ", end="", flush=True)
    try:
        naaim = fetch.fetch_naaim_exposure()
        print(f"done ({naaim:.0f})")
    except Exception as e:
        print(f"FAILED ({e})")
        raise

    print("  Fetching credit market data (HYG/LQD/TLT) … ", end="", flush=True)
    credit_raw = fetch.fetch_credit_market_data()
    print("done")

    # Optional FRED HY OAS
    fred_oas, fred_oas_4w = _try_fetch_fred_oas()
    if fred_oas:
        print(f"  FRED HY OAS: {fred_oas:.0f} bps (4wk chg {fred_oas - fred_oas_4w:+.0f} bps)")

    credit_data = CreditStressData(
        hyg_price=credit_raw["hyg_price"],
        hyg_ma200=credit_raw["hyg_ma200"],
        hyg_price_4w=credit_raw["hyg_price_4w"],
        lqd_price=credit_raw["lqd_price"],
        lqd_price_4w=credit_raw["lqd_price_4w"],
        tlt_hist_10d=pd.Series(credit_raw["tlt_last12"]),
        vix=vix,
        hy_oas=fred_oas,
        hy_oas_4w=fred_oas_4w,
    )
    credit = score_credit_stress(credit_data)

    return MarketSnapshot(
        ticker=ticker,
        vix=vix,
        index_price=price_snap.price,
        high_52w=price_snap.high_52w,
        ma_200=price_snap.ma_200,
        fear_greed=fg,
        aaii_bull_bear_spread=aaii,
        naaim_exposure=naaim,
        credit=credit,
    )


def build_snapshot_from_manual(
    ticker: str,
    vix: float,
    index_price: float,
    high_52w: float,
    ma_200: float,
    fear_greed: float,
    aaii_bull_bear_spread: float,
    naaim_exposure: float,
    # credit crisis: provide raw values or pre-computed result
    hyg_price: Optional[float] = None,
    hyg_ma200: Optional[float] = None,
    hyg_price_4w: Optional[float] = None,
    lqd_price: Optional[float] = None,
    lqd_price_4w: Optional[float] = None,
    credit_crisis_override: Optional[bool] = None,
) -> MarketSnapshot:
    """For back-testing or example scenarios."""
    if credit_crisis_override is not None:
        # Build a fake CreditStressResult to carry the override
        credit = _manual_credit_result(credit_crisis_override)
    elif hyg_price and hyg_ma200 and hyg_price_4w and lqd_price and lqd_price_4w:
        data = CreditStressData(
            hyg_price=hyg_price,
            hyg_ma200=hyg_ma200,
            hyg_price_4w=hyg_price_4w,
            lqd_price=lqd_price,
            lqd_price_4w=lqd_price_4w,
            tlt_hist_10d=pd.Series(dtype=float),
            vix=vix,
        )
        credit = score_credit_stress(data)
    else:
        credit = _manual_credit_result(False)

    return MarketSnapshot(
        ticker=ticker,
        vix=vix,
        index_price=index_price,
        high_52w=high_52w,
        ma_200=ma_200,
        fear_greed=fear_greed,
        aaii_bull_bear_spread=aaii_bull_bear_spread,
        naaim_exposure=naaim_exposure,
        credit=credit,
    )


def _manual_credit_result(is_crisis: bool) -> CreditStressResult:
    from .credit import CreditStressResult
    return CreditStressResult(
        score=6 if is_crisis else 1,
        max_score=7,
        is_crisis=is_crisis,
        source="manual override",
        spread_score=0, spread_change_score=0,
        relative_score=0, vol_score=0,
        hyg_vs_200ma_pct=0.0,
        hyg_lqd_4w_rel_pct=0.0,
        tlt_realized_vol_10d=0.0,
        hy_oas=None,
        hy_oas_4w_change=None,
    )
