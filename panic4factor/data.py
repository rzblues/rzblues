"""
Market data fetcher.

Auto-fetched (via yfinance):
  - VIX (^VIX)
  - Index price, 52-week high, 200-day MA  (QQQ / SPY / ^GSPC / ^NDX)

Manual inputs (no free real-time API):
  - CNN Fear & Greed Index (https://edition.cnn.com/markets/fear-and-greed)
  - AAII Bull-Bear Spread  (https://www.aaii.com/sentimentsurvey)
  - NAAIM Exposure Index   (https://www.naaim.org/programs/naaim-exposure-index/)
  - Credit crisis flag
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MarketSnapshot:
    # auto-fetched
    vix: float
    index_price: float
    high_52w: float
    ma_200: float
    ticker: str

    # manual inputs
    fear_greed: float
    aaii_bull_bear_spread: float
    naaim_exposure: float
    credit_crisis: bool

    @property
    def drawdown_pct(self) -> float:
        """Percentage drop from 52-week high (positive = down)."""
        return (self.high_52w - self.index_price) / self.high_52w * 100

    @property
    def price_vs_200ma(self) -> float:
        return self.index_price / self.ma_200


def fetch_market_data(ticker: str = "QQQ") -> Optional["MarketSnapshot"]:
    """
    Fetch VIX and price/MA data for the given ticker via yfinance.
    Returns a partial snapshot; caller must fill in manual fields.
    """
    try:
        import yfinance as yf
    except ImportError:
        return None

    vix_ticker = yf.Ticker("^VIX")
    idx_ticker = yf.Ticker(ticker)

    vix_hist = vix_ticker.history(period="1d")
    idx_hist  = idx_ticker.history(period="1y")

    if vix_hist.empty or idx_hist.empty:
        return None

    vix_now     = float(vix_hist["Close"].iloc[-1])
    idx_now     = float(idx_hist["Close"].iloc[-1])
    high_52w    = float(idx_hist["High"].max())
    ma_200      = float(idx_hist["Close"].tail(200).mean())

    # Return a partially filled snapshot — manual fields left as sentinels
    return MarketSnapshot(
        vix=vix_now,
        index_price=idx_now,
        high_52w=high_52w,
        ma_200=ma_200,
        ticker=ticker,
        fear_greed=-1.0,           # sentinel — must be filled by user
        aaii_bull_bear_spread=999, # sentinel
        naaim_exposure=-1.0,       # sentinel
        credit_crisis=False,
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
    credit_crisis: bool = False,
) -> MarketSnapshot:
    """Construct a snapshot entirely from user-supplied values (no network)."""
    return MarketSnapshot(
        vix=vix,
        index_price=index_price,
        high_52w=high_52w,
        ma_200=ma_200,
        ticker=ticker,
        fear_greed=fear_greed,
        aaii_bull_bear_spread=aaii_bull_bear_spread,
        naaim_exposure=naaim_exposure,
        credit_crisis=credit_crisis,
    )
