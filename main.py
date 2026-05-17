"""
CommodityPulse Pro — Phase 4 Enterprise Bot (main.py)
======================================================
Background scanner for GitHub Actions / cron execution.

Strategies (set PARAMS["active_strategy"]):
  "trend"     → Trend Confluence   (EMA 9/21/200 + RSI, best for trending markets)
  "reversion" → Mean Reversion     (Bollinger Band Fade + RSI, best for sideways)
  "breakout"  → Volatility Breakout (Donchian Channels + ADX + Volume, best for explosive moves)

All prices in INR (₹) · All times in IST · MCX Calibrated (15% duty)
Dual Take-Profit (1.5R / 3R) · ATR Stop-Loss · IST Market Hours Filter
Duplicate Guard · Exponential Back-off Retry · Volume Confirmation
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
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# LOGGING & TIMEZONE
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s"
)
log = logging.getLogger("CommodityPulse")
IST = pytz.timezone("Asia/Kolkata")

# ──────────────────────────────────────────────────────────────
# CREDENTIALS
# ──────────────────────────────────────────────────────────────
TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ──────────────────────────────────────────────────────────────
# ★★★  MASTER CONFIGURATION  ★★★
# Change "active_strategy" to switch algorithms without editing code.
# ──────────────────────────────────────────────────────────────
PARAMS = {
    # ─── Strategy selector ───────────────────────────────────
    # Options: "trend" | "reversion" | "breakout"
    "active_strategy": "trend",

    # ─── Timeframes ──────────────────────────────────────────
    "base_interval": "15m",   # Scanning timeframe
    "htf_interval":  "1h",    # Higher timeframe bias
    "base_period":   "30d",
    "htf_period":    "60d",

    # ─── EMA settings (Trend Confluence) ─────────────────────
    "ema_fast":  9,
    "ema_slow":  21,
    "ema_trend": 200,

    # ─── RSI settings ────────────────────────────────────────
    "rsi_bull": 55,   # Trend: RSI above this for BUY
    "rsi_bear": 45,   # Trend: RSI below this for SELL
    "rsi_ob":   70,   # Mean Reversion: overbought
    "rsi_os":   30,   # Mean Reversion: oversold

    # ─── ADX settings ────────────────────────────────────────
    "adx_min_trend":     20,   # Trend Confluence: minimum ADX
    "adx_min_breakout":  25,   # Volatility Breakout: minimum ADX

    # ─── Bollinger Band settings (Mean Reversion) ────────────
    "bb_length": 20,
    "bb_std":    2,

    # ─── Donchian Channel settings (Volatility Breakout) ─────
    "dc_period": 20,

    # ─── Volume settings ─────────────────────────────────────
    "vol_ma_length":  20,
    "vol_min_ratio":  1.1,   # Volume must be 10% above its MA

    # ─── Risk / reward ───────────────────────────────────────
    "sl_atr_mult": 1.5,
    "tp1_rr":      1.5,
    "tp2_rr":      3.0,

    # ─── Operational ─────────────────────────────────────────
    "state_file":   "last_alerts.json",
    "fetch_sleep":  2,
    "max_retries":  3,
    "import_duty":  1.15,   # 15% Indian import duty for Gold/Silver
}

# ──────────────────────────────────────────────────────────────
# ASSET UNIVERSE (MCX CALIBRATED)
# ──────────────────────────────────────────────────────────────
ASSETS = {
    "XAUUSD=X": {"name": "Gold (MCX)",    "emoji": "🟡", "type": "gold"},
    "XAGUSD=X": {"name": "Silver (MCX)",  "emoji": "⚪", "type": "silver"},
    "BZ=F":     {"name": "Crude Oil MCX", "emoji": "🛢️", "type": "comm"},
    "NG=F":     {"name": "Natural Gas",   "emoji": "🔥", "type": "comm"},
    "BTC-USD":  {"name": "Bitcoin",       "emoji": "₿",  "type": "crypto"},
}

# ──────────────────────────────────────────────────────────────
# HELPERS: MARKET HOURS & CURRENCY CONVERSION
# ──────────────────────────────────────────────────────────────
def is_mcx_open() -> bool:
    """True if MCX is currently trading (Mon-Fri, 9:00–23:30 IST)."""
    now = datetime.now(IST)
    if now.weekday() >= 5: return False
    hm = now.strftime("%H:%M")
    return "09:00" <= hm <= "23:30"

_usdinr_cache: dict = {}

def get_usdinr() -> float:
    """Cached USD/INR rate (falls back to 83.80 on error)."""
    cache_key = datetime.now(IST).strftime("%Y-%m-%d-%H")
    if cache_key in _usdinr_cache: return _usdinr_cache[cache_key]
    try:
        df = yf.download("INR=X", period="1d", interval="1d", progress=False)
        rate = float(df['Close'].iloc[-1])
        _usdinr_cache[cache_key] = rate
        return rate
    except Exception:
        return 83.80

def to_inr(usd_val: float, asset_type: str) -> float:
    """Convert USD global price → calibrated MCX INR price."""
    rate = get_usdinr()
    duty = PARAMS["import_duty"]
    if asset_type == "gold":
        return (usd_val / 31.1034768) * 10 * rate * duty
    elif asset_type == "silver":
        return (usd_val / 31.1034768) * 1000 * rate * duty
    else:
        return usd_val * rate

# ──────────────────────────────────────────────────────────────
# DATA ENGINE (Retry + Resilience)
# ──────────────────────────────────────────────────────────────
STATE_FILE = Path(__file__).parent / PARAMS["state_file"]

def _load_state() -> dict:
    try:    return json.loads(STATE_FILE.read_text())
    except: return {}

def _mark_alerted(ticker: str, candle_ts: str, signal: str) -> None:
    state = _load_state()
    state[f"{ticker}_{signal}"] = candle_ts
    try: STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception as e: log.warning(f"State write failed: {e}")

def _already_alerted(ticker: str, candle_ts: str, signal: str) -> bool:
    return _load_state().get(f"{ticker}_{signal}") == candle_ts

def _download(ticker: str, period: str, interval: str) -> pd.DataFrame | None:
    for attempt in range(PARAMS["max_retries"]):
        try:
            time.sleep(PARAMS["fetch_sleep"] * (2 ** attempt))
            df = yf.download(ticker, period=period, interval=interval,
                             progress=False, auto_adjust=True)
            if df.empty: continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        except Exception as e:
            log.warning(f"Fetch attempt {attempt+1} failed for {ticker}: {e}")
    return None

# ──────────────────────────────────────────────────────────────
# ═══════════════  STRATEGY ENGINE (OOP)  ════════════════════
# ──────────────────────────────────────────────────────────────

class BaseStrategy(ABC):
    """Abstract base — all strategies implement this interface."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def key(self) -> str: ...

    @abstractmethod
    def apply_indicators(self, df: pd.DataFrame) -> pd.DataFrame: ...

    @abstractmethod
    def check_signals(self, df: pd.DataFrame, bias: int) -> dict | None: ...

    def _adx_col(self, df: pd.DataFrame) -> str | None:
        return next((c for c in df.columns if c.startswith("ADX_")), None)

    def _atr_col(self, df: pd.DataFrame) -> str | None:
        return next((c for c in df.columns if c.startswith("ATRr_")), None)


# ─────────────────────────────────────
# Strategy 1: Trend Confluence
# ─────────────────────────────────────
class TrendConfluence(BaseStrategy):
    name = "Trend Confluence (MTF)"
    key  = "trend"

    def apply_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        p = PARAMS
        df.ta.ema(length=p["ema_fast"],  append=True)
        df.ta.ema(length=p["ema_slow"],  append=True)
        df.ta.ema(length=p["ema_trend"], append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.atr(length=14, append=True)
        adx = df.ta.adx(length=14)
        if adx is not None: df = pd.concat([df, adx], axis=1)
        if "Volume" in df.columns:
            df["VOL_MA_20"] = df["Volume"].rolling(PARAMS["vol_ma_length"]).mean()
        df.dropna(inplace=True)
        return df

    def check_signals(self, df: pd.DataFrame, bias: int) -> dict | None:
        if len(df) < 5: return None
        p    = PARAMS
        curr = df.iloc[-2]   # last confirmed closed candle
        prev = df.iloc[-3]

        adx_col = self._adx_col(df)
        atr_col = self._atr_col(df)
        if not adx_col or not atr_col: return None

        adx_val = float(curr[adx_col])
        if adx_val < p["adx_min_trend"]: return None

        vol_ok = (
            float(curr["Volume"]) > float(curr["VOL_MA_20"]) * p["vol_min_ratio"]
            if "VOL_MA_20" in df.columns else True
        )

        bull = (curr['EMA_9'] > curr['EMA_21'] and prev['EMA_9'] <= prev['EMA_21'] and
                curr['Close'] > curr['EMA_200'] and curr['RSI_14'] > p["rsi_bull"])
        bear = (curr['EMA_9'] < curr['EMA_21'] and prev['EMA_9'] >= prev['EMA_21'] and
                curr['Close'] < curr['EMA_200'] and curr['RSI_14'] < p["rsi_bear"])

        if bull and bias == -1: return None
        if bear and bias ==  1: return None
        if not (bull or bear): return None

        direction = "BULLISH" if bull else "BEARISH"
        score = 2
        if bias != 0:          score += 1
        if vol_ok:             score += 1
        if adx_val > 25:       score += 1

        return {
            "signal": direction, "price": float(curr['Close']),
            "rsi": float(curr['RSI_14']), "atr": float(curr[atr_col]),
            "adx": adx_val, "vol_ok": vol_ok,
            "score": min(score, 5), "ts": str(df.index[-2]), "htf_bias": bias,
            "extra": f"EMA9={curr['EMA_9']:.2f} EMA21={curr['EMA_21']:.2f}",
        }


# ─────────────────────────────────────
# Strategy 2: Mean Reversion (Bollinger Fade)
# ─────────────────────────────────────
class MeanReversionBollinger(BaseStrategy):
    name = "Mean Reversion (Bollinger Fade)"
    key  = "reversion"

    def apply_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        p = PARAMS
        df.ta.bbands(length=p["bb_length"], std=p["bb_std"], append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.atr(length=14, append=True)
        adx = df.ta.adx(length=14)
        if adx is not None: df = pd.concat([df, adx], axis=1)
        df.dropna(inplace=True)
        return df

    def _bb_cols(self, df: pd.DataFrame):
        lower = next((c for c in df.columns if c.startswith("BBL_")), None)
        upper = next((c for c in df.columns if c.startswith("BBU_")), None)
        return lower, upper

    def check_signals(self, df: pd.DataFrame, bias: int) -> dict | None:
        if len(df) < 5: return None
        lower, upper = self._bb_cols(df)
        if not lower or not upper: return None

        atr_col = self._atr_col(df)
        adx_col = self._adx_col(df)
        if not atr_col: return None

        curr = df.iloc[-2]

        # Warn if market is strongly trending — MR works poorly then
        if adx_col:
            adx_val = float(curr[adx_col])
            if adx_val > 35:
                log.info(f"[MR] ADX={adx_val:.1f} > 35 — strong trend, skipping mean reversion signal.")
                return None
        else:
            adx_val = 0.0

        rsi = float(curr['RSI_14'])
        low = float(curr['Low']); high = float(curr['High'])
        bb_lo = float(curr[lower]); bb_hi = float(curr[upper])

        # BUY: price pierced lower band AND RSI oversold
        bull = (low <= bb_lo) and (rsi < PARAMS["rsi_os"])
        # SELL: price pierced upper band AND RSI overbought
        bear = (high >= bb_hi) and (rsi > PARAMS["rsi_ob"])

        # Respect HTF when available
        if bull and bias == -1: return None
        if bear and bias ==  1: return None
        if not (bull or bear): return None

        direction = "BULLISH" if bull else "BEARISH"
        score = 2
        if bias != 0: score += 1
        if adx_val < 20: score += 1   # Bonus: classic sideways market
        if adx_val < 15: score += 1   # Double bonus: very choppy

        bb_touch = f"Low ₹{low:.2f} ≤ BB_Lo ₹{bb_lo:.2f}" if bull else f"High ₹{high:.2f} ≥ BB_Hi ₹{bb_hi:.2f}"

        return {
            "signal": direction, "price": float(curr['Close']),
            "rsi": rsi, "atr": float(curr[atr_col]),
            "adx": adx_val, "vol_ok": True,
            "score": min(score, 5), "ts": str(df.index[-2]), "htf_bias": bias,
            "extra": bb_touch,
        }


# ─────────────────────────────────────
# Strategy 3: Volatility Breakout (Donchian)
# ─────────────────────────────────────
class VolatilityBreakoutDonchian(BaseStrategy):
    name = "Volatility Breakout (Donchian)"
    key  = "breakout"

    def apply_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        p  = PARAMS
        dc = p["dc_period"]
        df['DC_HIGH'] = df['High'].rolling(dc).max().shift(1)  # confirmed previous high
        df['DC_LOW']  = df['Low'].rolling(dc).min().shift(1)
        df.ta.atr(length=14, append=True)
        df.ta.rsi(length=14, append=True)
        adx = df.ta.adx(length=14)
        if adx is not None: df = pd.concat([df, adx], axis=1)
        if "Volume" in df.columns:
            df["VOL_MA_20"] = df["Volume"].rolling(p["vol_ma_length"]).mean()
        df.dropna(inplace=True)
        return df

    def check_signals(self, df: pd.DataFrame, bias: int) -> dict | None:
        if len(df) < PARAMS["dc_period"] + 3: return None

        atr_col = self._atr_col(df)
        adx_col = self._adx_col(df)
        if not atr_col: return None

        curr = df.iloc[-2]

        # ADX must be strong enough for a breakout to be genuine
        if adx_col:
            adx_val = float(curr[adx_col])
            if adx_val < PARAMS["adx_min_breakout"]:
                log.info(f"[BO] ADX={adx_val:.1f} < {PARAMS['adx_min_breakout']} — breakout filter failed.")
                return None
        else:
            adx_val = 0.0
            return None   # ADX is mandatory for Breakout

        # Volume must spike above average
        vol_ok = False
        if "VOL_MA_20" in df.columns:
            vol_ok = float(curr["Volume"]) > float(curr["VOL_MA_20"]) * PARAMS["vol_min_ratio"]
        if not vol_ok:
            log.info("[BO] Volume not above threshold — breakout not confirmed.")
            return None

        close  = float(curr['Close'])
        dc_hi  = float(curr['DC_HIGH'])
        dc_lo  = float(curr['DC_LOW'])

        bull = close > dc_hi
        bear = close < dc_lo

        if bull and bias == -1: return None
        if bear and bias ==  1: return None
        if not (bull or bear): return None

        direction = "BULLISH" if bull else "BEARISH"
        score = 3   # Starts at 3 because ADX + Volume are already confirmed
        if bias != 0:   score += 1
        if adx_val > 35: score += 1

        extra = (f"Close ₹{close:.2f} > DC_High ₹{dc_hi:.2f}" if bull
                 else f"Close ₹{close:.2f} < DC_Low ₹{dc_lo:.2f}")

        return {
            "signal": direction, "price": close,
            "rsi": float(curr.get('RSI_14', 50)), "atr": float(curr[atr_col]),
            "adx": adx_val, "vol_ok": vol_ok,
            "score": min(score, 5), "ts": str(df.index[-2]), "htf_bias": bias,
            "extra": extra,
        }


# Strategy registry
STRATEGIES: dict[str, BaseStrategy] = {
    "trend":     TrendConfluence(),
    "reversion": MeanReversionBollinger(),
    "breakout":  VolatilityBreakoutDonchian(),
}


# ──────────────────────────────────────────────────────────────
# HTF BIAS ENGINE (always uses Trend Confluence EMA logic)
# ──────────────────────────────────────────────────────────────
def get_htf_bias(ticker: str) -> int:
    """Higher timeframe directional bias using EMA 9/21/200."""
    df = _download(ticker, PARAMS["htf_period"], PARAMS["htf_interval"])
    if df is None or len(df) < 210: return 0
    df.ta.ema(length=9,   append=True)
    df.ta.ema(length=21,  append=True)
    df.ta.ema(length=200, append=True)
    df.dropna(inplace=True)
    if df.empty: return 0
    c = df.iloc[-1]
    if c['EMA_9'] > c['EMA_21'] and c['Close'] > c['EMA_200']: return  1
    if c['EMA_9'] < c['EMA_21'] and c['Close'] < c['EMA_200']: return -1
    return 0


# ──────────────────────────────────────────────────────────────
# RISK & DUAL-TP CALCULATION
# ──────────────────────────────────────────────────────────────
def calculate_trade_levels(sig: dict, info: dict) -> dict:
    """Return entry, SL, TP1, TP2 all in INR."""
    p = PARAMS
    asset_type = info["type"]

    entry_inr = to_inr(sig["price"], asset_type)
    sl_dist_usd = p["sl_atr_mult"] * sig["atr"]
    sl_usd = (sig["price"] - sl_dist_usd if sig["signal"] == "BULLISH"
              else sig["price"] + sl_dist_usd)
    sl_inr  = to_inr(sl_usd, asset_type)

    risk_inr = abs(entry_inr - sl_inr)
    if sig["signal"] == "BULLISH":
        tp1_inr = entry_inr + risk_inr * p["tp1_rr"]
        tp2_inr = entry_inr + risk_inr * p["tp2_rr"]
    else:
        tp1_inr = entry_inr - risk_inr * p["tp1_rr"]
        tp2_inr = entry_inr - risk_inr * p["tp2_rr"]

    return {
        "entry_inr": entry_inr,
        "sl_inr":    sl_inr,
        "tp1_inr":   tp1_inr,
        "tp2_inr":   tp2_inr,
        "risk_inr":  risk_inr,
    }


# ──────────────────────────────────────────────────────────────
# STRUCTURED TELEGRAM MESSAGING
# ──────────────────────────────────────────────────────────────
def send_telegram_alert(
    ticker: str,
    info: dict,
    sig: dict,
    strategy_name: str,
    levels: dict,
) -> None:
    stars    = "⭐" * sig["score"] + "☆" * (5 - sig["score"])
    htf_text = {1: "✅ BULLISH", -1: "🔴 BEARISH", 0: "⚪ NEUTRAL"}[sig["htf_bias"]]
    dir_icon = "📈 LONG ▲" if sig["signal"] == "BULLISH" else "📉 SHORT ▼"

    msg = (
        f"{info['emoji']} *{info['name']}* ({ticker})\n"
        f"━━━━━━━━━━━━━━\n"
        f"🧠 Strategy: _{strategy_name}_\n"
        f"🔔 *{dir_icon}*\n"
        f"⭐ Confluence: {stars} ({sig['score']}/5)\n\n"
        f"💰 Entry:  ₹{levels['entry_inr']:>12,.2f}\n"
        f"🛑 SL:     ₹{levels['sl_inr']:>12,.2f}\n"
        f"🎯 TP1:    ₹{levels['tp1_inr']:>12,.2f}  _(1.5R)_\n"
        f"🚀 TP2:    ₹{levels['tp2_inr']:>12,.2f}  _(3R)_\n"
        f"📐 Risk:   ₹{levels['risk_inr']:>12,.2f}\n\n"
        f"📊 RSI: {sig['rsi']:.1f}  |  ADX: {sig['adx']:.1f}\n"
        f"🔭 HTF (1h): {htf_text}\n"
        f"📦 Volume: {'✅ Confirmed' if sig['vol_ok'] else '⚠️ Below avg'}\n"
        f"🔎 {sig.get('extra','')}\n"
        f"━━━━━━━━━━━━━━\n"
        f"🕐 _{datetime.now(IST).strftime('%Y-%m-%d %H:%M IST')}_\n"
        f"_CommodityPulse Pro · Phase 4 Enterprise_"
    )
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"},
            timeout=10,
        )
        if not r.ok: log.warning(f"Telegram HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        log.error(f"Telegram send failed: {e}")


def send_startup_message(strategy: BaseStrategy) -> None:
    """Announce which strategy is running (optional — fired once per session)."""
    if not TOKEN or not CHAT_ID: return
    icons = {"trend": "📈", "reversion": "↔️", "breakout": "💥"}
    icon  = icons.get(strategy.key, "⚡")
    msg = (
        f"⚡ *CommodityPulse Pro — Phase 4 Bot Started*\n"
        f"Active Strategy: {icon} _{strategy.name}_\n"
        f"Scanning: MCX + Crypto\n"
        f"Interval: Every 15 minutes\n"
        f"🕐 {datetime.now(IST).strftime('%Y-%m-%d %H:%M IST')}"
    )
    try:
        requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as e:
        log.warning(f"Startup message failed: {e}")


# ──────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────────────────────
def main() -> None:
    if not TOKEN or not CHAT_ID:
        log.error("Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID environment variables.")
        return

    # ── Resolve active strategy ─────────────────────────────
    strategy_key = PARAMS.get("active_strategy", "trend").lower()
    if strategy_key not in STRATEGIES:
        log.error(
            f"Unknown strategy '{strategy_key}'. "
            f"Valid options: {list(STRATEGIES.keys())}"
        )
        return

    strategy = STRATEGIES[strategy_key]
    log.info(f"CommodityPulse Pro — Phase 4 | Active strategy: [{strategy.name}]")
    send_startup_message(strategy)

    # ── Market hours gate ───────────────────────────────────
    mcx_open = is_mcx_open()
    if not mcx_open:
        log.info("MCX is CLOSED. Scanning Crypto only (BTC-USD).")
        assets_to_scan = {"BTC-USD": ASSETS["BTC-USD"]}
    else:
        log.info("MCX is OPEN. Scanning all assets.")
        assets_to_scan = ASSETS

    # ── Main scan loop ──────────────────────────────────────
    for ticker, info in assets_to_scan.items():
        log.info(f"→ Scanning {info['name']} ({ticker}) with [{strategy.name}]…")

        # 1. Higher Timeframe Bias (always EMA-based for structural context)
        bias = get_htf_bias(ticker)
        log.info(f"  HTF bias = {bias:+d} ({'BULLISH' if bias==1 else 'BEARISH' if bias==-1 else 'NEUTRAL'})")

        # 2. Base Timeframe Download
        df_raw = _download(ticker, PARAMS["base_period"], PARAMS["base_interval"])
        if df_raw is None or df_raw.empty:
            log.warning(f"  No data for {ticker} — skipping.")
            continue

        # 3. Apply strategy indicators
        try:
            df = strategy.apply_indicators(df_raw.copy())
        except Exception as e:
            log.error(f"  Indicator error for {ticker}: {e}")
            continue

        if len(df) < 5:
            log.info(f"  Not enough data rows after indicators ({len(df)}) — skipping.")
            continue

        # 4. Check for signal
        sig = strategy.check_signals(df, bias)
        if sig is None:
            log.info(f"  No signal for {ticker}.")
            continue

        log.info(f"  🚨 Signal detected: {sig['signal']} | score={sig['score']}/5 | ADX={sig['adx']:.1f}")

        # 5. Duplicate guard
        if _already_alerted(ticker, sig["ts"], sig["signal"]):
            log.info(f"  Duplicate — alert for this candle already sent.")
            continue

        # 6. Calculate trade levels in INR
        levels = calculate_trade_levels(sig, info)
        log.info(
            f"  Entry ₹{levels['entry_inr']:,.2f} | "
            f"SL ₹{levels['sl_inr']:,.2f} | "
            f"TP1 ₹{levels['tp1_inr']:,.2f} | "
            f"TP2 ₹{levels['tp2_inr']:,.2f}"
        )

        # 7. Send Telegram alert
        send_telegram_alert(ticker, info, sig, strategy.name, levels)
        _mark_alerted(ticker, sig["ts"], sig["signal"])
        log.info(f"  ✅ Alert sent for {info['name']}.")

    log.info("Scan complete.")


if __name__ == "__main__":
    main()
