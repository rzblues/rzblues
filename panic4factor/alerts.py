"""
Alert dispatcher — Telegram (primary) + Email (fallback).

Setup:
  Telegram (recommended):
    1. Message @BotFather on Telegram → /newbot → get BOT_TOKEN
    2. Start a chat with your new bot
    3. Visit https://api.telegram.org/bot<TOKEN>/getUpdates → copy chat_id
    4. export TELEGRAM_BOT_TOKEN=xxx  TELEGRAM_CHAT_ID=xxx

  Email (Gmail example):
    1. Enable 2FA on Google account → generate App Password (16 chars)
    2. export ALERT_EMAIL_FROM=you@gmail.com
    3. export ALERT_EMAIL_TO=you@gmail.com
    4. export ALERT_EMAIL_PASSWORD=xxxx-xxxx-xxxx-xxxx
    5. export ALERT_SMTP_HOST=smtp.gmail.com  (optional, default)
    6. export ALERT_SMTP_PORT=587             (optional, default)
"""

from __future__ import annotations

import os
import smtplib
from email.mime.text import MIMEText
from typing import Optional

import requests

_TG_URL = "https://api.telegram.org/bot{token}/sendMessage"


# ── Telegram ──────────────────────────────────────────────────────────────────

def send_telegram(message: str) -> bool:
    token   = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return False
    try:
        resp = requests.post(
            _TG_URL.format(token=token),
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"  [Telegram] send failed: {e}")
        return False


# ── Email ─────────────────────────────────────────────────────────────────────

def send_email(subject: str, body: str) -> bool:
    from_addr = os.environ.get("ALERT_EMAIL_FROM", "").strip()
    to_addr   = os.environ.get("ALERT_EMAIL_TO", "").strip()
    password  = os.environ.get("ALERT_EMAIL_PASSWORD", "").strip()
    host      = os.environ.get("ALERT_SMTP_HOST", "smtp.gmail.com")
    port      = int(os.environ.get("ALERT_SMTP_PORT", "587"))
    if not from_addr or not to_addr or not password:
        return False
    try:
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"]    = from_addr
        msg["To"]      = to_addr
        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(from_addr, password)
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"  [Email] send failed: {e}")
        return False


# ── Unified send ──────────────────────────────────────────────────────────────

def send_alert(subject: str, body: str) -> None:
    """Try Telegram first; fall back to email. Log if both unconfigured."""
    full_msg = f"*{subject}*\n\n{body}"
    tg_ok = send_telegram(full_msg)
    if not tg_ok:
        email_ok = send_email(subject, body)
        if not email_ok:
            print("  [Alert] No transport configured. Set TELEGRAM_BOT_TOKEN or ALERT_EMAIL_* env vars.")


# ── Message formatters ────────────────────────────────────────────────────────

def fmt_entry(output, add_pct: float) -> str:
    s = output.panic
    snap = output.snapshot
    cr = snap.credit
    return (
        f"PANIC BUY SIGNAL — {snap.ticker}\n\n"
        f"Score: {s.total}/100  [{s.signal_tier()}]\n"
        f"VIX: {snap.vix:.1f}  F&G: {snap.fear_greed:.0f}  "
        f"AAII: {snap.aaii_bull_bear_spread:+.0f}  NAAIM: {snap.naaim_exposure:.0f}\n\n"
        f"Credit stress: {cr.score}/{cr.max_score}  "
        f"({'CRISIS — sizes halved' if cr.is_crisis else 'no crisis'})\n"
        f"Drawdown: -{snap.drawdown_pct:.1f}%  |  "
        f"{'Above' if snap.price_vs_200ma >= 1 else 'BELOW'} 200MA "
        f"({snap.price_vs_200ma:.2f}x)\n\n"
        f"ACTION: ADD {add_pct*100:.0f}% of strategy capital\n"
        f"Instruments: QQQ / SPY / ES / SPX"
    )


def fmt_exit(tranche: int, reason: str, snap, avg_entry: float) -> str:
    ret = (snap.index_price - avg_entry) / avg_entry * 100
    return (
        f"EXIT SIGNAL — {snap.ticker}  (Tranche {tranche}/3)\n\n"
        f"Avg entry: {avg_entry:.2f}  Current: {snap.index_price:.2f}  "
        f"({ret:+.1f}%)\n"
        f"Trigger: {reason}\n\n"
        f"ACTION: Sell 1/3 of position\n"
        + ("All tranches done — position fully closed." if tranche == 3
           else f"Remaining: watch for Tranche {tranche+1}")
    )


def fmt_no_signal(snap) -> str:
    from .scorer import compute_panic_score
    s = compute_panic_score(snap.vix, snap.fear_greed,
                            snap.aaii_bull_bear_spread, snap.naaim_exposure)
    return (
        f"Weekly check — {snap.ticker}  (no signal)\n\n"
        f"Score: {s.total}/100  VIX: {snap.vix:.1f}  "
        f"F&G: {snap.fear_greed:.0f}  "
        f"AAII: {snap.aaii_bull_bear_spread:+.0f}  "
        f"NAAIM: {snap.naaim_exposure:.0f}\n"
        f"Drawdown from ATH: -{snap.drawdown_pct:.1f}%"
    )
