"""Simple JSON disk cache with configurable TTL."""

import json
import time
from pathlib import Path
from typing import Any, Optional

_CACHE_DIR = Path.home() / ".panic4factor_cache"
_DEFAULT_TTL = 8 * 3600  # 8 hours — AAII/NAAIM publish weekly, F&G updates hourly


def _path(key: str) -> Path:
    _CACHE_DIR.mkdir(exist_ok=True)
    return _CACHE_DIR / f"{key}.json"


def load(key: str, ttl: int = _DEFAULT_TTL) -> Optional[Any]:
    p = _path(key)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        if time.time() - data["ts"] > ttl:
            return None
        return data["value"]
    except Exception:
        return None


def save(key: str, value: Any) -> None:
    try:
        _path(key).write_text(json.dumps({"ts": time.time(), "value": value}))
    except Exception:
        pass  # cache write failure is non-fatal


def clear() -> None:
    for p in _CACHE_DIR.glob("*.json"):
        p.unlink(missing_ok=True)
