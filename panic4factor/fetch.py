"""
All external data fetching — fully automatic, no manual inputs.

Sources:
  VIX / price / HYG / LQD / TLT  : yfinance
  CNN Fear & Greed               : CNN dataviz API (unofficial, widely used)
  AAII Bull-Bear Spread          : aaii.com Excel download (weekly)
  NAAIM Exposure Index           : naaim.org Excel download (weekly)
  HY OAS (optional)              : FRED BAMLH0A0HYM2 (requires FRED_API_KEY)

All results are disk-cached for 8 hours (AAII/NAAIM publish weekly, no point
hitting them repeatedly in a day). Use --no-cache to bypass.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional
import warnings

import pandas as pd
import requests
import yfinance as yf

from . import cache

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


# ── Market data (yfinance) ────────────────────────────────────────────────────

@dataclass
class PriceSnapshot:
    ticker: str
    price: float
    high_52w: float
    ma_200: float


def _yf_history(ticker: str, period: str) -> "pd.DataFrame":
    """Wrap yfinance history() to surface a clean RuntimeError on any failure."""
    try:
        hist = yf.Ticker(ticker).history(period=period)
        if hist is None or (hasattr(hist, "empty") and hist.empty):
            raise RuntimeError(f"Empty response for {ticker}")
        return hist
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"yfinance failed for {ticker}: {exc}") from exc


def fetch_vix() -> float:
    cached = cache.load("vix", ttl=900)   # 15-min TTL for real-time VIX
    if cached is not None:
        return float(cached)
    hist = _yf_history("^VIX", "5d")
    val = float(hist["Close"].iloc[-1])
    cache.save("vix", val)
    return val


def fetch_price_snapshot(ticker: str) -> PriceSnapshot:
    cache_key = f"price_{ticker}"
    cached = cache.load(cache_key, ttl=900)
    if cached is not None:
        return PriceSnapshot(**cached)

    hist = _yf_history(ticker, "1y")
    snap = PriceSnapshot(
        ticker=ticker,
        price=float(hist["Close"].iloc[-1]),
        high_52w=float(hist["High"].max()),
        ma_200=float(hist["Close"].tail(200).mean()),
    )
    cache.save(cache_key, snap.__dict__)
    return snap


def fetch_credit_market_data() -> dict:
    """Fetch HYG, LQD, TLT for credit stress computation."""
    cached = cache.load("credit_market", ttl=900)
    if cached is not None:
        return cached

    try:
        tickers = yf.download(
            ["HYG", "LQD", "TLT"],
            period="1y",
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        raise RuntimeError(f"yfinance multi-download failed: {exc}") from exc

    if tickers is None or tickers.empty:
        raise RuntimeError("yfinance returned empty data for HYG/LQD/TLT")

    closes = tickers["Close"]
    hyg = closes["HYG"].dropna()
    lqd = closes["LQD"].dropna()
    tlt = closes["TLT"].dropna()

    data = {
        "hyg_price":   float(hyg.iloc[-1]),
        "hyg_ma200":   float(hyg.tail(200).mean()),
        "hyg_price_4w": float(hyg.iloc[-20]) if len(hyg) >= 20 else float(hyg.iloc[0]),
        "lqd_price":   float(lqd.iloc[-1]),
        "lqd_price_4w": float(lqd.iloc[-20]) if len(lqd) >= 20 else float(lqd.iloc[0]),
        # store last 12 TLT closes as list for realized vol calculation
        "tlt_last12":  tlt.tail(12).tolist(),
    }
    cache.save("credit_market", data)
    return data


# ── CNN Fear & Greed ──────────────────────────────────────────────────────────

_CNN_FG_URL = (
    "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
)


def fetch_fear_greed() -> float:
    """Return CNN Fear & Greed score (0=extreme fear, 100=extreme greed)."""
    cached = cache.load("fear_greed", ttl=3600)   # update hourly
    if cached is not None:
        return float(cached)

    resp = requests.get(_CNN_FG_URL, headers=_HEADERS, timeout=12)
    resp.raise_for_status()
    score = float(resp.json()["fear_and_greed"]["score"])
    cache.save("fear_greed", score)
    return score


# ── AAII Sentiment ────────────────────────────────────────────────────────────

_AAII_URL = "https://www.aaii.com/files/surveys/sentiment.xls"


def fetch_aaii_bull_bear_spread() -> float:
    """
    Return most recent AAII Bull-Bear Spread (Bullish% − Bearish%).
    Positive = net bullish, negative = net bearish (bearish contrarian buy signal).
    """
    cached = cache.load("aaii")
    if cached is not None:
        return float(cached)

    resp = requests.get(_AAII_URL, headers=_HEADERS, timeout=20)
    resp.raise_for_status()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # AAII Excel: row 0-2 are header/metadata, row 3+ are data
        df = pd.read_excel(
            io.BytesIO(resp.content),
            engine="xlrd",
            skiprows=3,
            header=0,
        )

    df.columns = [str(c).strip() for c in df.columns]

    # Drop rows without a parseable date
    df = df[pd.to_datetime(df.iloc[:, 0], errors="coerce").notna()].copy()
    df = df.dropna(how="all")

    # Identify Bullish and Bearish columns (robust to column name variations)
    bull_col = _find_col(df, ["Bullish", "Bull"])
    bear_col = _find_col(df, ["Bearish", "Bear"])

    latest = df.iloc[-1]
    bull = float(latest[bull_col])
    bear = float(latest[bear_col])

    # AAII stores values as decimals (0.35) or as pct integers (35); normalise
    if bull < 1.1:   # decimal form
        bull *= 100
        bear *= 100

    spread = round(bull - bear, 1)
    cache.save("aaii", spread)
    return spread


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in df.columns:
        for c in candidates:
            if c.lower() in col.lower():
                return col
    raise KeyError(f"Cannot find column matching {candidates} in {list(df.columns)}")


# ── NAAIM Exposure Index ──────────────────────────────────────────────────────

# They update the Excel file on their server; URL has been stable but may shift.
_NAAIM_URLS = [
    "https://www.naaim.org/wp-content/uploads/NAAIM-Data-for-Download.xlsx",
    "https://www.naaim.org/wp-content/uploads/2024/10/NAAIM-Data-for-Download.xlsx",
    "https://www.naaim.org/wp-content/uploads/2024/01/NAAIM-Data-for-Download.xlsx",
]


def fetch_naaim_exposure() -> float:
    """Return the most recent NAAIM Exposure Index value (0–200, typical 0–100)."""
    cached = cache.load("naaim")
    if cached is not None:
        return float(cached)

    content = _try_naaim_excel()
    if content is not None:
        val = _parse_naaim_excel(content)
        cache.save("naaim", val)
        return val

    # Fallback: scrape HTML table from NAAIM page
    val = _scrape_naaim_html()
    cache.save("naaim", val)
    return val


def _try_naaim_excel() -> Optional[bytes]:
    for url in _NAAIM_URLS:
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=20)
            if resp.status_code == 200 and len(resp.content) > 1000:
                return resp.content
        except Exception:
            continue
    return None


def _parse_naaim_excel(content: bytes) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_excel(io.BytesIO(content), engine="openpyxl")

    # Find the NAAIM Number / Exposure column
    exposure_col = _find_col(df, ["NAAIM Number", "Exposure", "Mean", "Average"])
    df = df.dropna(subset=[exposure_col])
    return float(df.iloc[-1][exposure_col])


def _scrape_naaim_html() -> float:
    """Scrape the NAAIM exposure table from their public page."""
    url = "https://www.naaim.org/programs/naaim-exposure-index/"
    resp = requests.get(url, headers=_HEADERS, timeout=20)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    if not tables:
        raise RuntimeError("No tables found on NAAIM page")
    # The exposure table is typically the first or largest table
    df = max(tables, key=len)
    exposure_col = _find_col(df, ["NAAIM Number", "Exposure", "Mean", "Average"])
    df = df.dropna(subset=[exposure_col])
    return float(df.iloc[-1][exposure_col])
