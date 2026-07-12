"""
CommodityPulse Pro — Phase 5 Enterprise Bot (main.py)
======================================================
Background scanner for GitHub Actions / cron execution.

Strategies (set PARAMS["active_strategy"]):
  "trend"     → Trend Confluence   (EMA 9/21/200 + RSI, best for trending markets)
  "reversion" → Mean Reversion     (Bollinger Band Fade + RSI, best for sideways)
  "breakout"  → Volatility Breakout (Donchian Channels + ADX + Volume, best for explosive moves)
  "auto"      → Regime-Adaptive Router (regime_engine.py) — classifies ADX +
                Donchian breakout + volume every scan, per ticker, and picks
                whichever of the three strategies above fits current conditions.
                Choice persists across cron runs via regime_state.json so the
                bot doesn't whipsaw across the ADX 20-25 transition zone.

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
import math
import logging
import pytz
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from regime_engine import compute_regime_probe, classify_regime
from quant_lab import rolling_correlation, correlated_risk_check

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
    # Options: "trend" | "reversion" | "breakout" | "auto"
    "active_strategy": "auto",

    # ─── Auto regime router (only used when active_strategy == "auto") ──
    "regime_adx_trend_min": 25.0,   # ADX >= this -> Trend Confluence
    "regime_adx_range_max": 20.0,   # ADX <  this -> Mean Reversion
    "regime_vol_ratio":     1.1,    # Volume > MA20 * this, for breakout confirmation
    "regime_state_file":    "regime_state.json",

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

    # ─── Portfolio Risk Cap (correlation-aware) ──────────────
    # The per-trade 2%-style rule doesn't protect you if two correlated
    # assets (e.g. Gold + Silver) both fire in the same direction at once —
    # that's not two independent 2% bets, it's one bigger correlated bet.
    # This caps the COMBINED risk of open positions in the same correlated
    # group, in addition to (not instead of) each trade's own sizing.
    "portfolio_capital_inr":    500_000,
    "position_risk_pct":        2.0,     # fixed % used to size the bot's own lots (Kelly needs a live backtest, which only the Streamlit terminal runs)
    "max_group_risk_pct":       4.0,     # cap on combined risk within a correlated group
    "correlation_lookback_days": 60,
    "correlation_threshold":    0.6,     # |corr| >= this counts as "correlated"
    "position_stale_hours":     72,      # auto-expire an open position that never hits SL/TP1 (data gaps, etc.)
    "open_positions_file":      "open_positions.json",
}

# ──────────────────────────────────────────────────────────────
# ASSET UNIVERSE (MCX CALIBRATED)
# ──────────────────────────────────────────────────────────────
ASSETS = {
    "XAUUSD=X": {"name": "Gold (MCX)",    "emoji": "🟡", "type": "gold",   "lot_size": 10},
    "XAGUSD=X": {"name": "Silver (MCX)",  "emoji": "⚪", "type": "silver", "lot_size": 1},
    "BZ=F":     {"name": "Crude Oil MCX", "emoji": "🛢️", "type": "comm",   "lot_size": 100},
    "NG=F":     {"name": "Natural Gas",   "emoji": "🔥", "type": "comm",   "lot_size": 10},
    "BTC-USD":  {"name": "Bitcoin",       "emoji": "₿",  "type": "crypto", "lot_size": 1},
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

# ──────────────────────────────────────────────────────────────
# REGIME STATE (persists the auto-router's per-ticker strategy choice
# across cron runs, so the transition-zone hysteresis in regime_engine
# actually holds between the 15-min scans instead of resetting each time)
# ──────────────────────────────────────────────────────────────
REGIME_STATE_FILE = Path(__file__).parent / PARAMS["regime_state_file"]

def _load_regime_state() -> dict:
    try:    return json.loads(REGIME_STATE_FILE.read_text())
    except: return {}

def _save_regime_choice(ticker: str, strategy_key: str) -> None:
    state = _load_regime_state()
    state[ticker] = strategy_key
    try: REGIME_STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception as e: log.warning(f"Regime state write failed: {e}")

# ──────────────────────────────────────────────────────────────
# OPEN POSITION TRACKING (for the correlation-aware portfolio risk cap)
# The bot fires alerts but never previously tracked whether that trade was
# still "live." Without knowing what's currently open, a correlation cap
# has nothing to sum risk across. This keeps a lightweight open-position
# ledger: entry/SL/TP1/direction/risk, closed out on next scan once price
# breaches SL or TP1, and auto-expired after position_stale_hours so a
# data gap can't permanently block future signals in that group.
# ──────────────────────────────────────────────────────────────
OPEN_POSITIONS_FILE = Path(__file__).parent / PARAMS["open_positions_file"]

def _load_open_positions() -> dict:
    try:    return json.loads(OPEN_POSITIONS_FILE.read_text())
    except: return {}

def _save_open_positions(positions: dict) -> None:
    try: OPEN_POSITIONS_FILE.write_text(json.dumps(positions, indent=2, default=str))
    except Exception as e: log.warning(f"Open-position state write failed: {e}")

def _expire_stale_positions() -> None:
    """Clears any open position older than position_stale_hours with no
    SL/TP1 hit recorded — guards against a data gap permanently blocking
    future signals in that correlation group."""
    positions = _load_open_positions()
    now = datetime.now(IST)
    changed = False
    for ticker in list(positions.keys()):
        opened_ts = datetime.fromisoformat(positions[ticker]["opened_ts"])
        age_hours = (now - opened_ts).total_seconds() / 3600
        if age_hours > PARAMS["position_stale_hours"]:
            log.info(f"  ⏱️ Open position on {ticker} expired after {age_hours:.0f}h with no SL/TP1 hit — clearing from risk pool.")
            del positions[ticker]; changed = True
    if changed:
        _save_open_positions(positions)

def _check_and_close_position(ticker: str, df_raw: pd.DataFrame) -> None:
    """Closes out ticker's open position (if any) if the latest bar's
    High/Low breached its SL or TP1. Compares in NATIVE units (same as
    df_raw), since sl/tp1 are stored native — not INR — for exactly this
    reason. Called inline, right after each ticker's data download."""
    positions = _load_open_positions()
    pos = positions.get(ticker)
    if not pos or df_raw is None or df_raw.empty:
        return

    last = df_raw.iloc[-1]
    hi, lo = float(last['High']), float(last['Low'])
    direction, sl, tp1 = pos["direction"], pos["sl"], pos["tp1"]

    hit_sl  = (lo <= sl) if direction == "BULLISH" else (hi >= sl)
    hit_tp1 = (hi >= tp1) if direction == "BULLISH" else (lo <= tp1)

    if hit_sl or hit_tp1:
        outcome = "TP1 hit ✅" if hit_tp1 else "SL hit 🛑"
        log.info(f"  📤 Closing open position on {ticker} ({direction}) — {outcome}")
        del positions[ticker]
        _save_open_positions(positions)

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
    """Return entry, SL, TP1, TP2 in INR — plus native-unit (pre-conversion)
    SL/TP1, needed because the position tracker compares against df_raw's
    High/Low, which are in the ticker's own native currency/units, not INR."""
    p = PARAMS
    asset_type = info["type"]

    entry_inr = to_inr(sig["price"], asset_type)
    sl_dist_usd = p["sl_atr_mult"] * sig["atr"]
    sl_native = (sig["price"] - sl_dist_usd if sig["signal"] == "BULLISH"
                 else sig["price"] + sl_dist_usd)
    sl_inr  = to_inr(sl_native, asset_type)

    risk_inr = abs(entry_inr - sl_inr)
    if sig["signal"] == "BULLISH":
        tp1_inr    = entry_inr + risk_inr * p["tp1_rr"]
        tp2_inr    = entry_inr + risk_inr * p["tp2_rr"]
        tp1_native = sig["price"] + sl_dist_usd * p["tp1_rr"]
    else:
        tp1_inr    = entry_inr - risk_inr * p["tp1_rr"]
        tp2_inr    = entry_inr - risk_inr * p["tp2_rr"]
        tp1_native = sig["price"] - sl_dist_usd * p["tp1_rr"]

    return {
        "entry_inr": entry_inr,
        "sl_inr":    sl_inr,
        "tp1_inr":   tp1_inr,
        "tp2_inr":   tp2_inr,
        "risk_inr":  risk_inr,   # per-unit-price risk distance in INR, NOT total capital at risk
        "sl_native":  sl_native,   # same units as df_raw's High/Low — for position-close checks
        "tp1_native": tp1_native,
    }


def compute_position_size(risk_per_unit_inr: float, lot_size: float, capital_inr: float, risk_pct: float) -> dict:
    """
    Fixed-%-of-capital lot sizing for the bot (mirrors the Streamlit
    terminal's Fixed Risk mode). Kelly sizing isn't offered here since it
    needs a live backtest to derive win-rate/payoff-ratio from — that only
    the Streamlit terminal computes. This just produces an actual rupee
    figure so the correlation risk cap below has something to sum.
    """
    risk_budget_inr = capital_inr * (risk_pct / 100.0)
    risk_per_lot    = risk_per_unit_inr * lot_size
    lots            = max(1, math.floor(risk_budget_inr / risk_per_lot)) if risk_per_lot > 0 else 1
    actual_risk_inr = lots * risk_per_lot
    return {"lots": lots, "actual_risk_inr": actual_risk_inr}


# ──────────────────────────────────────────────────────────────
# STRUCTURED TELEGRAM MESSAGING
# ──────────────────────────────────────────────────────────────
def send_telegram_alert(
    ticker: str,
    info: dict,
    sig: dict,
    strategy_name: str,
    levels: dict,
    regime_reason: str | None = None,
) -> None:
    stars    = "⭐" * sig["score"] + "☆" * (5 - sig["score"])
    htf_text = {1: "✅ BULLISH", -1: "🔴 BEARISH", 0: "⚪ NEUTRAL"}[sig["htf_bias"]]
    dir_icon = "📈 LONG ▲" if sig["signal"] == "BULLISH" else "📉 SHORT ▼"
    regime_line = f"🤖 Regime: {regime_reason}\n" if regime_reason else ""

    msg = (
        f"{info['emoji']} *{info['name']}* ({ticker})\n"
        f"━━━━━━━━━━━━━━\n"
        f"🧠 Strategy: _{strategy_name}_\n"
        f"{regime_line}"
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
        f"_CommodityPulse Pro · Phase 5 Enterprise_"
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


def send_startup_message(strategy: BaseStrategy | None) -> None:
    """Announce which strategy/mode is running (optional — fired once per session)."""
    if not TOKEN or not CHAT_ID: return
    if strategy is None:
        icon, label = "🤖", "Auto (Regime-Adaptive Router)"
    else:
        icons = {"trend": "📈", "reversion": "↔️", "breakout": "💥"}
        icon, label = icons.get(strategy.key, "⚡"), strategy.name
    msg = (
        f"⚡ *CommodityPulse Pro — Phase 5 Bot Started*\n"
        f"Active Strategy: {icon} _{label}_\n"
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
# CORRELATION MATRIX (for the portfolio risk cap)
# ──────────────────────────────────────────────────────────────
def build_correlation_matrix(assets: dict, lookback_days: int) -> pd.DataFrame:
    """
    Downloads recent daily closes for every asset in `assets` and returns a
    pairwise return-correlation matrix (via quant_lab.rolling_correlation).
    One extra download pass per ticker per scan — acceptable at a 15-min
    cadence. Silently skips tickers that fail to download; the risk cap
    treats missing correlations as "not correlated" (allow), same as any
    other conservative default in this codebase.
    """
    price_series = {}
    for ticker in assets.keys():
        df = _download(ticker, f"{lookback_days + 10}d", "1d")
        if df is not None and not df.empty and 'Close' in df.columns:
            price_series[ticker] = df['Close']
    return rolling_correlation(price_series, lookback=lookback_days)


# ──────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────────────────────
def main() -> None:
    if not TOKEN or not CHAT_ID:
        log.error("Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID environment variables.")
        return

    # ── Resolve active strategy mode ────────────────────────
    strategy_key = PARAMS.get("active_strategy", "trend").lower()
    is_auto = strategy_key == "auto"

    if not is_auto and strategy_key not in STRATEGIES:
        log.error(
            f"Unknown strategy '{strategy_key}'. "
            f"Valid options: {list(STRATEGIES.keys()) + ['auto']}"
        )
        return

    fixed_strategy = None if is_auto else STRATEGIES[strategy_key]
    mode_label = "Auto (Regime-Adaptive Router)" if is_auto else fixed_strategy.name
    log.info(f"CommodityPulse Pro — Phase 5 | Mode: [{mode_label}]")
    send_startup_message(fixed_strategy)

    # ── Portfolio risk cap setup ─────────────────────────────
    # Clear any open position that never resolved (data gap, etc.), then
    # build today's correlation matrix once for the whole scan — this is
    # what lets the risk cap tell "two independent 2% bets" apart from
    # "one bigger correlated bet wearing two tickers."
    _expire_stale_positions()
    corr_matrix = build_correlation_matrix(ASSETS, PARAMS["correlation_lookback_days"])
    if corr_matrix.empty:
        log.warning("  Correlation matrix unavailable this scan — portfolio risk cap will allow by default.")

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
        log.info(f"→ Scanning {info['name']} ({ticker})…")

        # 1. Higher Timeframe Bias (always EMA-based for structural context,
        #    regardless of which base strategy ends up active — this is a
        #    deliberate architectural boundary, unrelated to regime routing)
        bias = get_htf_bias(ticker)
        log.info(f"  HTF bias = {bias:+d} ({'BULLISH' if bias==1 else 'BEARISH' if bias==-1 else 'NEUTRAL'})")

        # 2. Base Timeframe Download
        df_raw = _download(ticker, PARAMS["base_period"], PARAMS["base_interval"])
        if df_raw is None or df_raw.empty:
            log.warning(f"  No data for {ticker} — skipping.")
            continue

        # 2a. Close out this ticker's open position (if any) before doing
        #     anything else this cycle — keeps the risk pool accurate for
        #     the correlation cap check further below.
        _check_and_close_position(ticker, df_raw)

        # 2b. Auto Regime Router — classify regime from raw OHLCV *before*
        #     any strategy-specific indicators are applied, then resolve
        #     which strategy runs for this ticker this scan. Choice is
        #     persisted to regime_state.json so the transition-zone
        #     hysteresis holds across separate 15-min cron invocations.
        regime_reason = None
        if is_auto:
            try:
                regime_state = _load_regime_state()
                prior_key    = regime_state.get(ticker)
                probe_df     = compute_regime_probe(df_raw.copy())
                regime_info  = classify_regime(
                    probe_df, prior_strategy_key=prior_key,
                    adx_trend_min=PARAMS["regime_adx_trend_min"],
                    adx_range_max=PARAMS["regime_adx_range_max"],
                    vol_ratio=PARAMS["regime_vol_ratio"],
                    confirmed_row=-2,   # last CLOSED candle, matches check_signals() below
                )
            except Exception as e:
                log.error(f"  Regime probe error for {ticker}: {e} — defaulting to Trend Confluence.")
                regime_info = {"regime": "UNKNOWN", "strategy_key": "trend", "confidence": 0,
                                "reason": "Regime probe failed.", "adx": None}

            active_strategy = STRATEGIES[regime_info["strategy_key"]]
            regime_reason   = regime_info["reason"]
            _save_regime_choice(ticker, regime_info["strategy_key"])
            log.info(
                f"  🤖 Regime={regime_info['regime']} (conf {regime_info['confidence']}%, "
                f"ADX={regime_info['adx']}) → {active_strategy.name}"
            )
            log.info(f"     {regime_reason}")
        else:
            active_strategy = fixed_strategy

        # 3. Apply strategy indicators
        try:
            df = active_strategy.apply_indicators(df_raw.copy())
        except Exception as e:
            log.error(f"  Indicator error for {ticker}: {e}")
            continue

        if len(df) < 5:
            log.info(f"  Not enough data rows after indicators ({len(df)}) — skipping.")
            continue

        # 4. Check for signal
        sig = active_strategy.check_signals(df, bias)
        if sig is None:
            log.info(f"  No signal for {ticker} with [{active_strategy.name}].")
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

        # 6b. Portfolio risk cap — combined risk within a correlated group
        #     (e.g. Gold + Silver both BUY at once) can quietly exceed your
        #     per-trade 2% rule even though each trade looks fine alone.
        lot_info = compute_position_size(levels["risk_inr"], info["lot_size"],
                                          PARAMS["portfolio_capital_inr"], PARAMS["position_risk_pct"])
        open_positions = _load_open_positions()
        risk_check = correlated_risk_check(
            ticker, sig["signal"], lot_info["actual_risk_inr"], open_positions, corr_matrix,
            capital_inr=PARAMS["portfolio_capital_inr"], max_group_risk_pct=PARAMS["max_group_risk_pct"],
            corr_threshold=PARAMS["correlation_threshold"])

        if not risk_check["allowed"]:
            log.info(f"  🚫 Correlation risk cap blocked this alert — {risk_check['reason']}")
            _mark_alerted(ticker, sig["ts"], sig["signal"])  # avoid re-logging the same candle every scan
            continue

        # 7. Send Telegram alert
        send_telegram_alert(ticker, info, sig, active_strategy.name, levels, regime_reason=regime_reason)
        _mark_alerted(ticker, sig["ts"], sig["signal"])

        # 7b. Register this as an open position (native-unit SL/TP1, so the
        #     close-check above compares apples-to-apples against df_raw).
        open_positions[ticker] = {
            "direction": sig["signal"], "sl": levels["sl_native"], "tp1": levels["tp1_native"],
            "risk_inr": lot_info["actual_risk_inr"], "lots": lot_info["lots"],
            "opened_ts": datetime.now(IST).isoformat(),
        }
        _save_open_positions(open_positions)
        log.info(f"  ✅ Alert sent for {info['name']} — {lot_info['lots']} lot(s), ₹{lot_info['actual_risk_inr']:,.0f} at risk.")

    log.info("Scan complete.")


if __name__ == "__main__":
    main()
