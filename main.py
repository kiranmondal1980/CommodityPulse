"""
CommodityPulse Pro — Enterprise Bot (main.py)
---------------------------------------------
Integrated Features:
- Multi-Timeframe Confluence (15m + 1h)
- ADX Regime & Volume Confirmation
- Dual Take-Profit (₹) & ATR Stop-Loss (₹)
- MCX Unit Calibration (Troy Oz to 10g/1kg) + 15% Duty
- IST Market Hours Filtering
- Duplicate Guard & Exponential Back-off Retry
"""

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
import os
import time
import json
import logging
import pytz
from datetime import datetime, timezone
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# LOGGING & TIMEZONE
# ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
log = logging.getLogger("CommodityPulse")
IST = pytz.timezone("Asia/Kolkata")

# ──────────────────────────────────────────────────────────────
# CREDENTIALS
# ──────────────────────────────────────────────────────────────
TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ──────────────────────────────────────────────────────────────
# ASSET UNIVERSE (MCX CALIBRATED)
# ──────────────────────────────────────────────────────────────
ASSETS = {
    "XAUUSD=X": {"name": "Gold (MCX)",   "emoji": "🟡", "type": "gold"},
    "XAGUSD=X": {"name": "Silver (MCX)", "emoji": "⚪", "type": "silver"},
    "BZ=F":     {"name": "Crude Oil",    "emoji": "🛢️", "type": "comm"},
    "NG=F":     {"name": "Natural Gas",  "emoji": "🔥", "type": "comm"},
    "BTC-USD":  {"name": "Bitcoin",      "emoji": "₿",  "type": "crypto"},
}

PARAMS = {
    "base_interval":  "15m",
    "htf_interval":   "1h",
    "base_period":    "30d",
    "htf_period":     "60d",
    "ema_fast": 9, "ema_slow": 21, "ema_trend": 200,
    "rsi_bull": 55, "rsi_bear": 45,
    "adx_min": 20,
    "vol_ma_length": 20, "vol_min_ratio": 1.1,
    "sl_atr_mult": 1.5, "tp1_rr": 1.5, "tp2_rr": 3.0,
    "state_file": "last_alerts.json",
    "fetch_sleep": 2, "max_retries": 3,
    "import_duty": 1.15  # 15% Indian Duty
}

# ──────────────────────────────────────────────────────────────
# HELPERS: MARKET HOURS & CURRENCY
# ──────────────────────────────────────────────────────────────
def is_mcx_open():
    """Returns True if MCX is currently trading (Mon-Fri, 9am-11:30pm IST)."""
    now_ist = datetime.now(IST)
    if now_ist.weekday() >= 5: return False
    hm = now_ist.strftime("%H:%M")
    return "09:00" <= hm <= "23:30"

def get_usdinr():
    try:
        df = yf.download("INR=X", period="1d", interval="1d", progress=False)
        return float(df['Close'].iloc[-1])
    except: return 83.80

def to_inr(usd_val, asset_type):
    """Converts USD global price to calibrated MCX INR price."""
    rate = get_usdinr()
    duty = PARAMS["import_duty"]
    if asset_type == 'gold':
        return (usd_val / 31.1034768) * 10 * rate * duty
    elif asset_type == 'silver':
        return (usd_val / 31.1034768) * 1000 * rate * duty
    else:
        return usd_val * rate

# ──────────────────────────────────────────────────────────────
# DATA ENGINE (Retry + Resilience)
# ──────────────────────────────────────────────────────────────
STATE_FILE = Path(__file__).parent / PARAMS["state_file"]

def _load_state():
    try: return json.loads(STATE_FILE.read_text())
    except: return {}

def _mark_alerted(ticker, candle_ts, signal):
    state = _load_state()
    state[f"{ticker}_{signal}"] = candle_ts
    try: STATE_FILE.write_text(json.dumps(state, indent=2))
    except: pass

def _already_alerted(ticker, candle_ts, signal):
    return _load_state().get(f"{ticker}_{signal}") == candle_ts

def _download(ticker, period, interval):
    for attempt in range(PARAMS["max_retries"]):
        try:
            time.sleep(PARAMS["fetch_sleep"] * (2 ** attempt))
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            if df.empty: continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            return df
        except Exception as e:
            log.warning(f"Fetch failed for {ticker}: {e}")
    return None

# ──────────────────────────────────────────────────────────────
# QUANT ENGINE
# ──────────────────────────────────────────────────────────────
def apply_indicators(df):
    p = PARAMS
    df.ta.ema(length=p["ema_fast"], append=True)
    df.ta.ema(length=p["ema_slow"], append=True)
    df.ta.ema(length=p["ema_trend"], append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.atr(length=14, append=True)
    adx = df.ta.adx(length=14)
    df = pd.concat([df, adx], axis=1)
    if "Volume" in df.columns:
        df["VOL_MA"] = df["Volume"].rolling(p["vol_ma_length"]).mean()
    df.dropna(inplace=True)
    return df

def get_htf_bias(ticker):
    df = _download(ticker, PARAMS["htf_period"], PARAMS["htf_interval"])
    if df is None or len(df) < 200: return 0
    df = apply_indicators(df)
    c = df.iloc[-1]
    if c['EMA_9'] > c['EMA_21'] and c['Close'] > c['EMA_200']: return 1
    if c['EMA_9'] < c['EMA_21'] and c['Close'] < c['EMA_200']: return -1
    return 0

def check_signals(df, bias):
    if len(df) < 5: return None
    p = PARAMS
    curr, prev = df.iloc[-2], df.iloc[-3] # Last closed candle
    
    # 1. ADX Filter
    adx_val = curr['ADX_14']
    if adx_val < p["adx_min"]: return None

    # 2. Volume Confirmation
    vol_ok = (curr["Volume"] > curr["VOL_MA"] * p["vol_min_ratio"]) if "Volume" in df.columns else True

    # 3. Logic
    bull = (curr['EMA_9'] > curr['EMA_21']) and (prev['EMA_9'] <= prev['EMA_21']) and (curr['Close'] > curr['EMA_200']) and (curr['RSI_14'] > p["rsi_bull"])
    bear = (curr['EMA_9'] < curr['EMA_21']) and (prev['EMA_9'] >= prev['EMA_21']) and (curr['Close'] < curr['EMA_200']) and (curr['RSI_14'] < p["rsi_bear"])

    if bull and bias == -1: return None
    if bear and bias == 1: return None
    if not (bull or bear): return None

    direction = "BULLISH" if bull else "BEARISH"
    score = 2
    if bias != 0: score += 1
    if vol_ok: score += 1
    if adx_val > 25: score += 1

    return {
        "signal": direction, "price": curr['Close'], "rsi": curr['RSI_14'],
        "atr": curr['ATRr_14'], "adx": adx_val, "vol_ok": vol_ok,
        "score": min(score, 5), "ts": str(df.index[-2]), "htf_bias": bias
    }

# ──────────────────────────────────────────────────────────────
# STRUCTURED TELEGRAM MESSAGING
# ──────────────────────────────────────────────────────────────
def send_telegram_alert(ticker, info, sig):
    p = PARAMS
    # Convert Global to INR
    entry_inr = to_inr(sig['price'], info['type'])
    sl_dist   = p["sl_atr_mult"] * sig['atr']
    sl_usd    = (sig['price'] - sl_dist) if sig['signal'] == "BULLISH" else (sig['price'] + sl_dist)
    sl_inr    = to_inr(sl_usd, info['type'])
    
    # Calculate TP targets in INR
    risk_inr = abs(entry_inr - sl_inr)
    tp1_inr  = (entry_inr + risk_inr * p["tp1_rr"]) if sig['signal'] == "BULLISH" else (entry_inr - risk_inr * p["tp1_rr"])
    tp2_inr  = (entry_inr + risk_inr * p["tp2_rr"]) if sig['signal'] == "BULLISH" else (entry_inr - risk_inr * p["tp2_rr"])

    stars = "⭐" * sig['score'] + "☆" * (5 - sig['score'])
    htf_text = {1: "✅ BULLISH", -1: "🔴 BEARISH", 0: "⚪ NEUTRAL"}[sig["htf_bias"]]
    
    msg = (
        f"{info['emoji']} *{info['name']}* ({ticker})\n"
        f"━━━━━━━━━━━━━━\n"
        f"🔔 *{'📈 LONG ▲' if sig['signal'] == 'BULLISH' else '📉 SHORT ▼'}*\n"
        f"⭐ Confluence: {stars} ({sig['score']}/5)\n\n"
        f"💰 Entry: ₹{entry_inr:,.2f}\n"
        f"🛑 SL:    ₹{sl_inr:,.2f}\n"
        f"🎯 TP1:   ₹{tp1_inr:,.2f} _(1.5R)_\n"
        f"🚀 TP2:   ₹{tp2_inr:,.2f} _(3R)_\n\n"
        f"📊 RSI: {sig['rsi']:.1f} | ADX: {sig['adx']:.1f}\n"
        f"🔭 HTF (1h): {htf_text}\n"
        f"📦 Volume: {'✅ Above Avg' if sig['vol_ok'] else '⚠️ Normal'}\n"
        f"━━━━━━━━━━━━━━\n"
        f"🕐 _{datetime.now(IST).strftime('%Y-%m-%d %H:%M IST')}_\n"
        f"_CommodityPulse Pro Enterprise_"
    )
    
    requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", 
                  json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"})

# ──────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────────────────────
def main():
    if not TOKEN or not CHAT_ID:
        log.error("Missing Telegram Credentials")
        return

    log.info("Starting CommodityPulse Pro Scan...")
    
    # Filter assets based on market status
    m_open = is_mcx_open()
    assets_to_check = ASSETS if m_open else {"BTC-USD": ASSETS["BTC-USD"]}
    
    if not m_open:
        log.info("MCX Closed. Scanning Crypto only.")

    for ticker, info in assets_to_check.items():
        log.info(f"Checking {info['name']}...")
        
        # 1. Higher Timeframe Bias
        bias = get_htf_bias(ticker)
        
        # 2. Base Timeframe Scan
        df = _download(ticker, PARAMS["base_period"], PARAMS["base_interval"])
        if df is not None:
            df = apply_indicators(df)
            sig = check_signals(df, bias)
            
            if sig and not _already_alerted(ticker, sig['ts'], sig['signal']):
                send_telegram_alert(ticker, info, sig)
                _mark_alerted(ticker, sig['ts'], sig['signal'])
                log.info(f"🚨 Alert Sent for {ticker}")
    
    log.info("Scan Complete.")

if __name__ == "__main__":
    main()
