"""
CommodityPulse Pro — Regime Engine (Phase 1: Auto Strategy Router)
=====================================================================
Shared by app.py (Streamlit terminal) and main.py (background bot).

Classifies the current market regime from ADX + Donchian breakout +
Volume conditions, and routes to whichever of the three strategies
is best suited for it:

    BREAKOUT  -> Volatility Breakout (Donchian)   : confirmed channel
                 break + volume spike + rising ADX (explosive move)
    TREND     -> Trend Confluence (EMA 9/21/200)  : ADX >= adx_trend_min
    RANGE     -> Mean Reversion (Bollinger Fade)  : ADX <  adx_range_max

    In the "transition zone" (adx_range_max <= ADX < adx_trend_min,
    no breakout confirmed) the router HOLDS the prior strategy rather
    than flip-flopping every refresh/scan. With no prior state it
    defaults to Trend Confluence, matching CommodityPulse's house
    preference for Trend Confluence as the default lens.

This module intentionally has zero Streamlit / requests / yfinance
dependencies so it can be imported cleanly by both the UI and the bot.
"""

import pandas as pd

# Maps a resolved regime label to the strategy registry key used in
# both app.py's STRATEGIES (via STRATEGY_KEY_MAP) and main.py's STRATEGIES.
REGIME_TO_STRATEGY = {
    "BREAKOUT": "breakout",
    "TREND":    "trend",
    "RANGE":    "reversion",
}


def compute_regime_probe(df: pd.DataFrame, dc_period: int = 20,
                          adx_period: int = 14, vol_ma_length: int = 20) -> pd.DataFrame:
    """
    Adds the minimum indicator set needed to classify regime, independent
    of whichever strategy ultimately gets selected. Cheap enough to run
    every refresh / scan even when the user is on a fixed strategy.

    Expects df to have OHLCV columns (Open/High/Low/Close/Volume).
    Returns a NEW dataframe (does not mutate the input).
    """
    d = df.copy()
    adx = d.ta.adx(length=adx_period)
    if adx is not None and not adx.empty:
        for col in adx.columns:
            d[col] = adx[col]
    d['DC_HIGH_PROBE'] = d['High'].rolling(dc_period).max().shift(1)
    d['DC_LOW_PROBE']  = d['Low'].rolling(dc_period).min().shift(1)
    if 'Volume' in d.columns:
        d['VOL_MA_PROBE'] = d['Volume'].rolling(vol_ma_length).mean()
    return d


def classify_regime(probe_df: pd.DataFrame, prior_strategy_key: str | None = None,
                     adx_trend_min: float = 25.0, adx_range_max: float = 20.0,
                     vol_ratio: float = 1.1, confirmed_row: int = -1) -> dict:
    """
    Returns a dict:
        {
          "regime":       "TREND" | "RANGE" | "BREAKOUT" | "TRANSITION" | "UNKNOWN",
          "strategy_key": "trend" | "reversion" | "breakout",
          "confidence":   0-100 (int),
          "reason":       human-readable explanation (for UI card / bot logs),
          "adx":          float | None,
        }

    confirmed_row:
        -1 for app.py (it treats the latest fetched row as "current",
           matching its existing HTF-bias convention elsewhere in the app)
        -2 for main.py (it treats the last CLOSED candle as current,
           matching its existing check_signals() convention)
    """
    adx_col = next((c for c in probe_df.columns if c.startswith('ADX_')), None)
    if adx_col is None:
        return _fallback("No ADX data available — defaulting.", prior_strategy_key)

    d = probe_df.dropna(subset=[adx_col])
    if len(d) < 3:
        return _fallback("Insufficient bars for regime classification — defaulting.", prior_strategy_key)

    try:
        row  = d.iloc[confirmed_row]
        prev = d.iloc[confirmed_row - 1]
    except IndexError:
        return _fallback("Insufficient bars for regime classification — defaulting.", prior_strategy_key)

    adx_now    = float(row[adx_col])
    adx_prev   = float(prev[adx_col])
    adx_rising = adx_now > adx_prev

    vol_ok = False
    if 'VOL_MA_PROBE' in d.columns and not pd.isna(row.get('VOL_MA_PROBE', float('nan'))):
        vol_ok = float(row['Volume']) > float(row['VOL_MA_PROBE']) * vol_ratio

    dc_hi = row.get('DC_HIGH_PROBE', float('inf'))
    dc_lo = row.get('DC_LOW_PROBE', float('-inf'))
    breakout_up   = (not pd.isna(dc_hi)) and row['Close'] > dc_hi
    breakout_down = (not pd.isna(dc_lo)) and row['Close'] < dc_lo
    is_breakout   = (breakout_up or breakout_down) and vol_ok and adx_rising

    if is_breakout:
        direction = "upside" if breakout_up else "downside"
        conf = min(100, 60 + (10 if adx_now > 30 else 0) + (10 if vol_ok else 0))
        return {
            "regime": "BREAKOUT", "strategy_key": "breakout", "confidence": conf,
            "reason": (f"Donchian {direction} break + volume spike + rising ADX "
                       f"({adx_now:.1f}) -> Volatility Breakout."),
            "adx": adx_now,
        }

    if adx_now >= adx_trend_min:
        conf = min(100, 55 + int(adx_now - adx_trend_min))
        return {
            "regime": "TREND", "strategy_key": "trend", "confidence": conf,
            "reason": f"ADX {adx_now:.1f} >= {adx_trend_min:.0f} -> trending market -> Trend Confluence.",
            "adx": adx_now,
        }

    if adx_now < adx_range_max:
        conf = min(100, 55 + int(adx_range_max - adx_now))
        return {
            "regime": "RANGE", "strategy_key": "reversion", "confidence": conf,
            "reason": f"ADX {adx_now:.1f} < {adx_range_max:.0f} -> sideways market -> Mean Reversion.",
            "adx": adx_now,
        }

    # Transition zone: adx_range_max <= adx_now < adx_trend_min, no breakout.
    held_key    = prior_strategy_key if prior_strategy_key in REGIME_TO_STRATEGY.values() else "trend"
    held_regime = next((r for r, k in REGIME_TO_STRATEGY.items() if k == held_key), "TREND")
    return {
        "regime": "TRANSITION", "strategy_key": held_key, "confidence": 35,
        "reason": (f"ADX {adx_now:.1f} is in the transition zone "
                   f"({adx_range_max:.0f}-{adx_trend_min:.0f}) with no confirmed breakout — "
                   f"holding prior regime ({held_regime}) to avoid whipsaw switching."),
        "adx": adx_now,
    }


def _fallback(reason: str, prior_strategy_key: str | None) -> dict:
    key = prior_strategy_key if prior_strategy_key in REGIME_TO_STRATEGY.values() else "trend"
    return {"regime": "UNKNOWN", "strategy_key": key, "confidence": 0, "reason": reason, "adx": None}
