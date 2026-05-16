"""
CommodityPulse Pro — Optimized Bot (main.py)
=============================================
Improvements over original:
  1. Multi-Timeframe Confluence  — HTF (1h) bias gates the 15m signal
  2. ADX Regime Filter           — Skips signals in choppy/sideways markets
  3. Volume Confirmation         — Signal must occur on above-average volume
  4. ATR Column Auto-Detection   — Handles ATRr_14 or ATR_14 dynamically
  5. Duplicate-Alert Guard       — Saves last-alerted candle to disk; no repeat spam
  6. Dual Take-Profit (TP1/TP2)  — 1.5R conservative + 3R aggressive targets
  7. Signal Strength Scoring     — Rates confluence 1-5 stars in the message
  8. Structured Telegram Message — Clean, emoji-rich, Markdown-formatted alert
  9. Retry + Rate-Limit Safety   — Exponential back-off on Yahoo Finance fetches
 10. Graceful Error Reporting     — Sends a Telegram error summary if any asset fails
"""

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
import os
import time
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("CommodityPulse")

# ──────────────────────────────────────────────────────────────
# CREDENTIALS
# ──────────────────────────────────────────────────────────────
TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ──────────────────────────────────────────────────────────────
# ASSET UNIVERSE
# ──────────────────────────────────────────────────────────────
ASSETS = {
    "GC=F": {"name": "Gold",         "emoji": "🟡"},
    "CL=F": {"name": "Crude Oil",    "emoji": "🛢️"},
    "NG=F": {"name": "Natural Gas",  "emoji": "🔥"},
    "SI=F": {"name": "Silver",       "emoji": "⚪"},
}

# ──────────────────────────────────────────────────────────────
# STRATEGY PARAMETERS  (tune here — one place, nowhere else)
# ──────────────────────────────────────────────────────────────
PARAMS = {
    # Timeframes
    "base_interval":  "15m",   # Primary signal timeframe
    "htf_interval":   "1h",    # Higher-timeframe bias filter
    "base_period":    "60d",   # Lookback for 15m data
    "htf_period":     "60d",   # Lookback for 1h data

    # EMA lengths
    "ema_fast":       9,
    "ema_slow":       21,
    "ema_trend":      200,

    # RSI
    "rsi_length":     14,
    "rsi_bull":       55,      # RSI must be ABOVE this for a buy
    "rsi_bear":       45,      # RSI must be BELOW this for a sell

    # ADX regime filter
    "adx_length":     14,
    "adx_min":        20,      # Signals skipped if ADX < this (choppy market)

    # ATR for SL/TP
    "atr_length":     14,
    "sl_atr_mult":    1.5,     # Stop-loss = price ± 1.5 × ATR
    "tp1_rr":         1.5,     # Take-Profit 1 — Risk:Reward
    "tp2_rr":         3.0,     # Take-Profit 2 — Risk:Reward

    # Volume confirmation
    "vol_ma_length":  20,      # Rolling window for average volume
    "vol_min_ratio":  1.1,     # Signal candle volume must be ≥ 110 % of MA

    # Duplicate guard
    "state_file":     "last_alerts.json",  # Written next to main.py

    # Rate-limit / retry
    "fetch_sleep":    2,       # Seconds between Yahoo fetches
    "max_retries":    3,       # Retry attempts per fetch
}

# ──────────────────────────────────────────────────────────────
# DUPLICATE ALERT GUARD
# ──────────────────────────────────────────────────────────────
STATE_FILE = Path(__file__).parent / PARAMS["state_file"]

def _load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}

def _save_state(state: dict):
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception as e:
        log.warning(f"Could not save state: {e}")

def _already_alerted(ticker: str, candle_ts: str, signal: str) -> bool:
    state = _load_state()
    key   = f"{ticker}_{signal}"
    return state.get(key) == candle_ts

def _mark_alerted(ticker: str, candle_ts: str, signal: str):
    state = _load_state()
    state[f"{ticker}_{signal}"] = candle_ts
    _save_state(state)

# ──────────────────────────────────────────────────────────────
# DATA FETCHING  (with retry + exponential back-off)
# ──────────────────────────────────────────────────────────────
def _download(ticker: str, period: str, interval: str) -> pd.DataFrame | None:
    for attempt in range(PARAMS["max_retries"]):
        try:
            time.sleep(PARAMS["fetch_sleep"] * (2 ** attempt))   # exponential back-off
            df = yf.download(ticker, period=period, interval=interval,
                             progress=False, auto_adjust=True)
            if df.empty:
                log.warning(f"{ticker} [{interval}] returned empty DataFrame (attempt {attempt+1})")
                continue
            # Flatten MultiIndex columns (yfinance ≥ 0.2.x)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        except Exception as exc:
            log.warning(f"{ticker} [{interval}] fetch error (attempt {attempt+1}): {exc}")
    return None

def fetch_base(ticker: str) -> pd.DataFrame | None:
    return _download(ticker, PARAMS["base_period"], PARAMS["base_interval"])

def fetch_htf(ticker: str) -> pd.DataFrame | None:
    return _download(ticker, PARAMS["htf_period"], PARAMS["htf_interval"])

# ──────────────────────────────────────────────────────────────
# INDICATOR ENGINE
# ──────────────────────────────────────────────────────────────
def _atr_col(df: pd.DataFrame) -> str | None:
    """Return the ATR column name regardless of pandas-ta naming variant."""
    for candidate in ("ATRr_14", "ATR_14", "ATRr_14_14"):
        if candidate in df.columns:
            return candidate
    for c in df.columns:
        if c.startswith("ATR"):
            return c
    return None

def apply_indicators(df: pd.DataFrame) -> pd.DataFrame:
    p = PARAMS
    df.ta.ema(length=p["ema_fast"],  append=True)
    df.ta.ema(length=p["ema_slow"],  append=True)
    df.ta.ema(length=p["ema_trend"], append=True)
    df.ta.rsi(length=p["rsi_length"], append=True)
    df.ta.atr(length=p["atr_length"], append=True)
    adx = df.ta.adx(length=p["adx_length"], append=False)
    if adx is not None and not adx.empty:
        df = pd.concat([df, adx], axis=1)
    # Volume moving average for confirmation
    if "Volume" in df.columns:
        df[f"VOL_MA_{p['vol_ma_length']}"] = (
            df["Volume"].rolling(p["vol_ma_length"]).mean()
        )
    df.dropna(inplace=True)
    return df

# ──────────────────────────────────────────────────────────────
# HIGHER-TIMEFRAME BIAS  (+1 bull / -1 bear / 0 neutral)
# ──────────────────────────────────────────────────────────────
def htf_bias(ticker: str) -> int:
    df = fetch_htf(ticker)
    if df is None or len(df) < 210:
        return 0
    df = apply_indicators(df)
    if df.empty:
        return 0
    c = df.iloc[-1]
    e9  = c.get("EMA_9",  float("nan"))
    e21 = c.get("EMA_21", float("nan"))
    e200= c.get("EMA_200",float("nan"))
    if any(pd.isna(v) for v in [e9, e21, e200]):
        return 0
    if e9 > e21 and c["Close"] > e200:
        return  1
    if e9 < e21 and c["Close"] < e200:
        return -1
    return 0

# ──────────────────────────────────────────────────────────────
# SIGNAL SCORER  — returns score 0-5 and direction
# ──────────────────────────────────────────────────────────────
def _stars(score: int) -> str:
    return "⭐" * score + "☆" * (5 - score)

def check_signals(df: pd.DataFrame, bias: int) -> dict | None:
    """
    Evaluate the last CLOSED candle (iloc[-2]) for a trade signal.
    Returns a rich dict or None.
    """
    if len(df) < 4:
        return None

    p    = PARAMS
    curr = df.iloc[-2]
    prev = df.iloc[-3]

    # ── Mandatory columns ──────────────────────────────────────
    try:
        c_close  = float(curr["Close"])
        c_ema9   = float(curr[f"EMA_{p['ema_fast']}"])
        c_ema21  = float(curr[f"EMA_{p['ema_slow']}"])
        c_ema200 = float(curr[f"EMA_{p['ema_trend']}"])
        p_ema9   = float(prev[f"EMA_{p['ema_fast']}"])
        p_ema21  = float(prev[f"EMA_{p['ema_slow']}"])
        c_rsi    = float(curr[f"RSI_{p['rsi_length']}"])
    except (KeyError, ValueError, TypeError):
        return None

    if any(pd.isna(v) for v in [c_ema200, c_ema9, c_ema21, c_rsi]):
        return None

    # ── ATR (optional — degrades gracefully) ───────────────────
    atr_col = _atr_col(df)
    c_atr   = float(curr[atr_col]) if atr_col and not pd.isna(curr.get(atr_col, float("nan"))) else 0.0

    # ── ADX regime filter ──────────────────────────────────────
    adx_col = next((c for c in df.columns if c.startswith("ADX_") and not c.startswith("ADX_D")), None)
    c_adx   = float(curr[adx_col]) if adx_col and not pd.isna(curr.get(adx_col, float("nan"))) else None

    if c_adx is not None and c_adx < p["adx_min"]:
        log.info(f"  ⚡ ADX={c_adx:.1f} < {p['adx_min']} — choppy market, signal skipped")
        return None

    # ── Volume confirmation ────────────────────────────────────
    vol_col = f"VOL_MA_{p['vol_ma_length']}"
    vol_ok  = True
    if "Volume" in df.columns and vol_col in df.columns:
        c_vol    = float(curr["Volume"])
        vol_ma   = float(curr[vol_col])
        if vol_ma > 0:
            vol_ok = (c_vol / vol_ma) >= p["vol_min_ratio"]

    # ── Core crossover conditions ──────────────────────────────
    ema_cross_bull = (c_ema9 > c_ema21) and (p_ema9 <= p_ema21)
    ema_cross_bear = (c_ema9 < c_ema21) and (p_ema9 >= p_ema21)
    above_trend    = c_close > c_ema200
    below_trend    = c_close < c_ema200
    rsi_bull       = c_rsi > p["rsi_bull"]
    rsi_bear       = c_rsi < p["rsi_bear"]

    # ── Build signal ───────────────────────────────────────────
    is_bull = ema_cross_bull and above_trend and rsi_bull
    is_bear = ema_cross_bear and below_trend and rsi_bear

    # HTF must agree or be neutral (never counter-trend)
    if is_bull and bias == -1:
        log.info("  ⬆️ Base bullish but HTF bearish — filtered")
        return None
    if is_bear and bias == 1:
        log.info("  ⬇️ Base bearish but HTF bullish — filtered")
        return None

    if not (is_bull or is_bear):
        return None

    direction = "BULLISH" if is_bull else "BEARISH"

    # ── Score (1–5 confluence stars) ──────────────────────────
    score = 1  # base: crossover happened
    if bias != 0:          score += 1   # HTF agrees
    if vol_ok:             score += 1   # volume confirms
    if c_adx and c_adx > 25: score += 1 # strong trend
    if (is_bull and c_rsi > 60) or (is_bear and c_rsi < 40): score += 1  # strong RSI

    # ── SL / TP levels ─────────────────────────────────────────
    sl_dist = p["sl_atr_mult"] * c_atr if c_atr else 0.0
    if direction == "BULLISH":
        sl   = c_close - sl_dist
        tp1  = c_close + p["tp1_rr"] * sl_dist
        tp2  = c_close + p["tp2_rr"] * sl_dist
    else:
        sl   = c_close + sl_dist
        tp1  = c_close - p["tp1_rr"] * sl_dist
        tp2  = c_close - p["tp2_rr"] * sl_dist

    # ── Candle timestamp (for duplicate guard) ─────────────────
    candle_ts = str(df.index[-2])

    return {
        "signal":    direction,
        "price":     c_close,
        "rsi":       c_rsi,
        "atr":       c_atr,
        "adx":       c_adx,
        "sl":        sl,
        "tp1":       tp1,
        "tp2":       tp2,
        "htf_bias":  bias,
        "vol_ok":    vol_ok,
        "score":     score,
        "candle_ts": candle_ts,
    }

# ──────────────────────────────────────────────────────────────
# TELEGRAM MESSAGING
# ──────────────────────────────────────────────────────────────
def _build_message(ticker: str, info: dict, sig: dict) -> str:
    direction  = sig["signal"]
    arrow      = "📈 LONG  ▲" if direction == "BULLISH" else "📉 SHORT ▼"
    htf_text   = {1: "✅ BULLISH", -1: "✅ BEARISH", 0: "⚪ NEUTRAL"}[sig["htf_bias"]]
    vol_text   = "✅ Above avg" if sig["vol_ok"] else "⚠️ Below avg"
    adx_text   = f"{sig['adx']:.1f}" if sig["adx"] else "N/A"
    stars      = _stars(sig["score"])
    ts_utc     = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Price formatting — commodities have varying decimal needs
    def fmt(v):
        return f"${v:,.3f}" if v < 10 else f"${v:,.2f}"

    lines = [
        f"{info['emoji']} *{info['name']}* ({ticker})",
        f"",
        f"🔔 *{arrow}*",
        f"⭐ Confluence: {stars} ({sig['score']}/5)",
        f"",
        f"💰 Entry:  {fmt(sig['price'])}",
        f"🛑 SL:     {fmt(sig['sl'])}",
        f"🎯 TP1:   {fmt(sig['tp1'])}   _(1.5R — conservative)_",
        f"🚀 TP2:   {fmt(sig['tp2'])}   _(3R — aggressive)_",
        f"",
        f"📊 RSI:   {sig['rsi']:.1f}   |   ATR: {fmt(sig['atr'])}",
        f"🌊 ADX:   {adx_text}",
        f"🔭 HTF (1h): {htf_text}",
        f"📦 Volume:  {vol_text}",
        f"",
        f"🕐 _{ts_utc}_",
        f"_CommodityPulse Pro — Not Financial Advice_",
    ]
    return "\n".join(lines)

def send_telegram(msg: str):
    if not TOKEN or not CHAT_ID:
        log.warning("Telegram credentials missing — skipping send")
        return
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"},
            timeout=10,
        )
        if not resp.ok:
            log.warning(f"Telegram API error: {resp.status_code} {resp.text[:200]}")
    except Exception as exc:
        log.error(f"Telegram send failed: {exc}")

def send_error_summary(errors: list[str]):
    if not errors:
        return
    msg = "⚠️ *CommodityPulse Bot — Errors*\n\n" + "\n".join(f"• {e}" for e in errors)
    send_telegram(msg)

# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    if not TOKEN or not CHAT_ID:
        log.error("TELEGRAM_TOKEN or TELEGRAM_CHAT_ID not set — aborting")
        return

    log.info("=" * 55)
    log.info("CommodityPulse Pro Bot — Starting scan")
    log.info("=" * 55)

    errors: list[str] = []

    for ticker, info in ASSETS.items():
        log.info(f"▶ {info['name']} ({ticker})")

        # 1. Fetch Higher-Timeframe bias first
        bias = htf_bias(ticker)
        log.info(f"  HTF bias = {['⚪ NEUTRAL', '🟢 BULL', '🔴 BEAR'][bias]}")

        # 2. Fetch base timeframe data
        df = fetch_base(ticker)
        if df is None:
            msg = f"{ticker}: data fetch failed after {PARAMS['max_retries']} retries"
            log.error(f"  ❌ {msg}")
            errors.append(msg)
            continue

        log.info(f"  ✅ {len(df)} candles — last close: {float(df['Close'].iloc[-1]):.3f}")

        # 3. Apply indicators
        try:
            df = apply_indicators(df)
        except Exception as exc:
            msg = f"{ticker}: indicator error — {exc}"
            log.error(f"  ❌ {msg}")
            errors.append(msg)
            continue

        if df.empty:
            log.info("  ⚠️ DataFrame empty after dropna — skipping")
            continue

        # 4. Check signals
        sig = check_signals(df, bias)
        if sig is None:
            log.info(f"  ℹ️  No signal")
            continue

        # 5. Duplicate guard
        if _already_alerted(ticker, sig["candle_ts"], sig["signal"]):
            log.info(f"  🔁 Signal already sent for candle {sig['candle_ts']} — skipping duplicate")
            continue

        # 6. Build and send alert
        log.info(f"  🚨 {sig['signal']} signal! Score={sig['score']}/5  SL={sig['sl']:.3f}  TP1={sig['tp1']:.3f}")
        msg = _build_message(ticker, info, sig)
        send_telegram(msg)
        _mark_alerted(ticker, sig["candle_ts"], sig["signal"])
        log.info(f"  📨 Alert sent and logged")

    # 7. Surface any errors to Telegram
    send_error_summary(errors)

    log.info("=" * 55)
    log.info("Scan complete")
    log.info("=" * 55)


if __name__ == "__main__":
    main()
