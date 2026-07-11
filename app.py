"""
CommodityPulse Pro — Phase 5 (Auto Regime-Adaptive Strategy Router)
======================================================
Strategies:
  1. Trend Confluence   — EMA 9/21/200 + RSI  (trending markets)
  2. Mean Reversion     — Bollinger Band Fade + RSI (sideways markets)
  3. Volatility Breakout — Donchian Channel + ADX + Volume (explosive moves)

All prices in INR (₹) · All times in IST · MCX Calibrated (15% duty)
Smart Risk Engine · Dual-TP (1.5R / 3R) · Live Backtester · Chandelier Trail
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import requests, os, time, math
from abc import ABC, abstractmethod
from datetime import datetime
import pytz
from regime_engine import compute_regime_probe, classify_regime

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CommodityPulse Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── THEME STATE (read BEFORE the CSS is built, so a toggle click on the
#    previous rerun is already reflected in session_state by the time this
#    script runs top-to-bottom again — standard Streamlit theming pattern) ──
if "theme_choice" not in st.session_state:
    st.session_state.theme_choice = "Light"
if "compact_mobile" not in st.session_state:
    st.session_state.compact_mobile = False

_THEME = st.session_state.theme_choice
_MOBILE = st.session_state.compact_mobile

# Brand constants — stay fixed across both themes (this is the one thing
# that should never change: the blue/teal/red signal palette IS the brand).
ACCENT   = "#2563eb"
ACCENT_2 = "#7c3aed"
SUCCESS  = "#089981"
DANGER   = "#F23645"
WARNING  = "#f59e0b"

if _THEME == "Dark":
    PALETTE = dict(
        bg_page="#0b0f17", bg_card="#11151d", bg_card_alt="#161b22", bg_hover="#1c2128",
        border="#21262d", border_strong="#30363d",
        text_primary="#e6edf3", text_secondary="#9aa7b8", text_muted="#6b7686",
        shadow="rgba(0,0,0,.45)", chip_shadow="rgba(37,99,235,.18)",
        plot_bg="#11151d", plot_grid="#1c2128", plot_font="#9aa7b8", plot_template="plotly_dark",
    )
else:
    PALETTE = dict(
        bg_page="#ffffff", bg_card="#ffffff", bg_card_alt="#f8faff", bg_hover="#eef2ff",
        border="#e2e8f0", border_strong="#cbd5e1",
        text_primary="#0d1117", text_secondary="#64748b", text_muted="#94a3b8",
        shadow="rgba(15,23,42,.06)", chip_shadow="rgba(37,99,235,.10)",
        plot_bg="#ffffff", plot_grid="#f1f5f9", plot_font="#64748b", plot_template="plotly_white",
    )

CHART_HEIGHT = 480 if _MOBILE else 720
CHART_MARGIN = dict(l=6, r=46, t=28, b=8) if _MOBILE else dict(l=16, r=90, t=32, b=16)
ANNOT_FONT   = 8 if _MOBILE else 10

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Syne:wght@700;800&display=swap');

:root {{
    --bg-page: {PALETTE['bg_page']};
    --bg-card: {PALETTE['bg_card']};
    --bg-card-alt: {PALETTE['bg_card_alt']};
    --bg-hover: {PALETTE['bg_hover']};
    --border: {PALETTE['border']};
    --border-strong: {PALETTE['border_strong']};
    --text-primary: {PALETTE['text_primary']};
    --text-secondary: {PALETTE['text_secondary']};
    --text-muted: {PALETTE['text_muted']};
    --shadow: {PALETTE['shadow']};
    --chip-shadow: {PALETTE['chip_shadow']};
    --accent: {ACCENT};
    --accent-2: {ACCENT_2};
    --success: {SUCCESS};
    --danger: {DANGER};
    --warning: {WARNING};
}}

html, body, [class*="css"] {{ font-family: 'IBM Plex Mono', monospace; }}
.stApp {{ background: var(--bg-page); }}
h1, h2, h3 {{ font-family: 'Syne', sans-serif; font-weight: 800; color: var(--text-primary); letter-spacing: -0.5px; }}
p, span, label, div {{ color: var(--text-primary); }}
hr {{ border-color: var(--border) !important; }}
::selection {{ background: var(--accent); color: #fff; }}

/* Header signature — thin gradient signal-line under the title */
.cp-header-rule {{
    height:3px; width:100%; margin:6px 0 14px;
    background:linear-gradient(90deg, var(--accent), var(--accent-2), var(--success));
    border-radius:2px; opacity:.85;
}}

div[data-testid="metric-container"] {{
    background: linear-gradient(135deg, var(--bg-card), var(--bg-card-alt));
    border: 1px solid var(--border); border-left: 4px solid var(--accent);
    padding: 14px 18px; border-radius: 12px;
    box-shadow: 0 2px 12px var(--chip-shadow);
    transition: transform .18s ease, box-shadow .18s ease;
}}
div[data-testid="metric-container"]:hover {{
    transform: translateY(-3px); box-shadow: 0 8px 22px var(--chip-shadow);
}}
div[data-testid="metric-container"] > div:first-child {{
    font-family:'IBM Plex Mono',monospace; font-size:11px;
    text-transform:uppercase; letter-spacing:1px; color: var(--text-secondary);
}}
div[data-testid="metric-container"] > div:nth-child(2) {{
    font-family:'Syne',sans-serif; font-weight:700; font-size:20px; color: var(--text-primary);
}}

.section-header {{
    font-family:'Syne',sans-serif; font-size:13px; font-weight:700;
    text-transform:uppercase; letter-spacing:2px; color: var(--text-secondary);
    border-bottom:1px solid var(--border); padding-bottom:6px; margin:20px 0 12px;
}}
.section-header.accent {{ color: var(--accent); border-color: var(--accent); border-bottom-width:2px; }}

/* Strategy Badge */
.strat-badge {{
    display:inline-block; border-radius:8px; padding:6px 14px;
    font-size:12px; font-weight:700; font-family:'IBM Plex Mono',monospace;
    letter-spacing:1px; margin-bottom:10px;
}}
.strat-badge.trend     {{ background:#dbeafe; color:#1e40af; border:1px solid #93c5fd; }}
.strat-badge.reversion {{ background:#fce7f3; color:#9d174d; border:1px solid #f9a8d4; }}
.strat-badge.breakout  {{ background:#fef3c7; color:#92400e; border:1px solid #fcd34d; }}

/* Confluence Matrix */
.matrix-wrap {{
    display:flex; gap:12px; align-items:stretch;
    background: var(--bg-card-alt); border:1px solid var(--border);
    border-radius:14px; padding:18px 22px; margin:12px 0; flex-wrap:wrap;
}}
.matrix-cell {{
    flex:1; min-width:110px; border-radius:10px; padding:14px 16px;
    text-align:center; border:1px solid transparent; transition:transform .15s;
}}
.matrix-cell:hover {{ transform:translateY(-2px); }}
.matrix-cell.bull {{ background:#dcfce7; border-color:#86efac; }}
.matrix-cell.bear {{ background:#fee2e2; border-color:#fca5a5; }}
.matrix-cell.neut {{ background:#f1f5f9; border-color:#cbd5e1; }}
.matrix-cell .tf-label {{ font-size:10px; letter-spacing:1.5px; text-transform:uppercase; color:#64748b; margin-bottom:6px; }}
.matrix-cell .tf-icon  {{ font-size:24px; margin:4px 0; }}
.matrix-cell .tf-text  {{ font-size:11px; font-weight:700; letter-spacing:.5px; }}
.matrix-cell.bull .tf-text {{ color:#166534; }}
.matrix-cell.bear .tf-text {{ color:#991b1b; }}
.matrix-cell.neut .tf-text {{ color:#475569; }}

.verdict-strong-buy  {{ background:#166534; color:#dcfce7; border-radius:8px; padding:10px 18px; font-weight:700; font-size:13px; letter-spacing:1px; display:inline-block; }}
.verdict-strong-sell {{ background:#991b1b; color:#fee2e2; border-radius:8px; padding:10px 18px; font-weight:700; font-size:13px; letter-spacing:1px; display:inline-block; }}
.verdict-caution     {{ background:#78350f; color:#fef9c3; border-radius:8px; padding:10px 18px; font-weight:700; font-size:13px; letter-spacing:1px; display:inline-block; }}
.verdict-neutral     {{ background:#334155; color:#e2e8f0; border-radius:8px; padding:10px 18px; font-weight:700; font-size:13px; letter-spacing:1px; display:inline-block; }}

.mkt-open   {{ background:#dcfce7; color:#166534; border:1px solid #86efac; border-radius:20px; padding:5px 14px; font-size:12px; font-weight:700; display:inline-flex; align-items:center; gap:7px; }}
.mkt-closed {{ background:#fee2e2; color:#991b1b; border:1px solid #fca5a5; border-radius:20px; padding:5px 14px; font-size:12px; font-weight:700; display:inline-block; }}
.mkt-pre    {{ background:#fef9c3; color:#92400e; border:1px solid #fde68a; border-radius:20px; padding:5px 14px; font-size:12px; font-weight:700; display:inline-block; }}

/* Live pulse dot — the one signature motion moment, reserved for "MCX OPEN" */
.pulse-dot {{
    width:8px; height:8px; border-radius:50%; background:#166534;
    box-shadow:0 0 0 0 rgba(22,101,52,.55);
    animation: pulse 1.8s infinite;
}}
@keyframes pulse {{
    0%   {{ box-shadow:0 0 0 0 rgba(22,101,52,.55); }}
    70%  {{ box-shadow:0 0 0 8px rgba(22,101,52,0); }}
    100% {{ box-shadow:0 0 0 0 rgba(22,101,52,0); }}
}}
@media (prefers-reduced-motion: reduce) {{ .pulse-dot {{ animation:none; }} }}

.fallback-banner {{
    background:#fef3c7; border:1px solid #f59e0b; border-radius:8px;
    padding:8px 14px; font-size:12px; color:#92400e; margin-bottom:10px;
}}

section[data-testid="stSidebar"] {{ background:#0d1117; border-right:1px solid #21262d; }}
section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {{ color:#e6edf3 !important; font-family:'IBM Plex Mono',monospace !important; }}
section[data-testid="stSidebar"] .stSelectbox>div>div {{ background:#161b22; border:1px solid #30363d; }}
section[data-testid="stSidebar"] button {{ transition: transform .15s ease, box-shadow .15s ease; }}
section[data-testid="stSidebar"] button:hover {{ transform: translateY(-1px); box-shadow:0 4px 14px rgba(37,99,235,.25); }}

.dark-card {{
    background:linear-gradient(135deg,#0d1117,#161b22);
    border:1px solid #30363d; border-radius:12px; padding:20px 24px; margin-top:8px;
    box-shadow: 0 4px 18px var(--shadow);
}}
.dark-card h4 {{ color:#58a6ff; margin:0 0 12px 0; font-family:'Syne',sans-serif; font-size:15px; }}
.drow {{ display:flex; justify-content:space-between; border-bottom:1px solid #21262d; padding:8px 0; font-size:13px; }}
.drow:last-child {{ border-bottom:none; }}
.dlabel {{ color:#8b949e; }}
.dval   {{ color:#e6edf3; font-weight:600; }}
.dval.g {{ color:#3fb950; }} .dval.r {{ color:#f85149; }} .dval.b {{ color:#58a6ff; }} .dval.a {{ color:#f59e0b; }}

.risk-pill {{
    display:inline-block; border-radius:20px; padding:4px 14px;
    font-size:12px; font-weight:700; font-family:'IBM Plex Mono',monospace;
}}
.risk-pill.safe   {{ background:#dcfce7; color:#166534; border:1px solid #86efac; }}
.risk-pill.medium {{ background:#fef9c3; color:#92400e; border:1px solid #fde68a; }}
.risk-pill.danger {{ background:#fee2e2; color:#991b1b; border:1px solid #fca5a5; }}

.bt-grid {{
    display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:12px 0;
}}
.bt-card {{
    background: var(--bg-card); border:1px solid var(--border); border-radius:10px;
    padding:16px; text-align:center; box-shadow:0 2px 8px var(--shadow);
    transition: transform .15s ease;
}}
.bt-card:hover {{ transform: translateY(-2px); }}
.bt-card .bt-val {{ font-family:'Syne',sans-serif; font-size:26px; font-weight:800; color: var(--text-primary); }}
.bt-card .bt-val.g {{ color:#089981; }} .bt-card .bt-val.r {{ color:#F23645; }}
.bt-card .bt-label {{ font-size:10px; letter-spacing:1px; text-transform:uppercase; color: var(--text-secondary); margin-top:4px; }}
.bt-card .bt-sub   {{ font-size:11px; color: var(--text-muted); margin-top:2px; }}

/* Strategy info box */
.strat-info {{
    background:linear-gradient(135deg,#f8faff,#eff6ff);
    border:1px solid #bfdbfe; border-left:4px solid #2563eb;
    border-radius:10px; padding:14px 18px; margin:10px 0; font-size:12px; color:#0d1117;
}}
.strat-info * {{ color:#0d1117; }}
.strat-info.reversion {{ border-color:#f9a8d4; border-left-color:#ec4899; background:linear-gradient(135deg,#fdf2f8,#fce7f3); }}
.strat-info.breakout  {{ border-color:#fcd34d; border-left-color:#d97706; background:linear-gradient(135deg,#fffbeb,#fef3c7); }}

/* ── MOBILE RESPONSIVENESS ──────────────────────────────────
   Streamlit's st.columns() doesn't reflow on its own — it just
   squeezes columns until they're unreadable. These rules force
   the underlying flex containers to wrap into a proper 2-col /
   1-col stacked grid below common phone/tablet breakpoints. */
@media (max-width: 900px) {{
    div[data-testid="stHorizontalBlock"] {{ flex-wrap: wrap !important; gap: 10px !important; }}
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {{
        min-width: 46% !important; flex: 1 1 46% !important;
    }}
    .bt-grid {{ grid-template-columns: repeat(2, 1fr); }}
    .matrix-cell {{ min-width: 45%; }}
    h1 {{ font-size: 26px !important; }}
}}
@media (max-width: 480px) {{
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {{
        min-width: 100% !important; flex: 1 1 100% !important;
    }}
    .bt-grid {{ grid-template-columns: 1fr; }}
    .matrix-cell {{ min-width: 100%; }}
    .dark-card {{ padding:14px 16px; }}
    h1 {{ font-size: 21px !important; }}
    div[data-testid="metric-container"] > div:nth-child(2) {{ font-size:17px; }}
}}

#MainMenu {{ visibility:hidden; }} footer {{ visibility:hidden; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
IST = pytz.timezone("Asia/Kolkata")

ASSETS = {
    "Global Proxy (in INR)": {
        'Crude Oil (WTI)': {'ticker':'CL=F','fallback':None,   'emoji':'🛢️','lot_size':100,'lot_unit':'barrels','mcx_lot':100},
        'Natural Gas':     {'ticker':'NG=F','fallback':None,   'emoji':'🔥','lot_size':10, 'lot_unit':'mmBtu',  'mcx_lot':10},
        'Gold':            {'ticker':'GC=F','fallback':None,   'emoji':'🟡','lot_size':10, 'lot_unit':'grams',  'mcx_lot':10},
        'Silver':          {'ticker':'SI=F','fallback':None,   'emoji':'⚪','lot_size':1,  'lot_unit':'kg',     'mcx_lot':30},
    },
    "Indian MCX (in INR)": {
        'Crude Oil (MCX)':   {'ticker':'BZ=F','fallback':'CL=F','emoji':'🛢️','lot_size':100,'lot_unit':'barrels','mcx_lot':100},
        'Natural Gas (MCX)': {'ticker':'NG=F','fallback':None,  'emoji':'🔥','lot_size':10, 'lot_unit':'mmBtu',  'mcx_lot':1250},
        'Gold (MCX)':        {'ticker':'GC=F','fallback':None,  'emoji':'🟡','lot_size':10, 'lot_unit':'grams',  'mcx_lot':100},
        'Silver (MCX)':      {'ticker':'SI=F','fallback':None,  'emoji':'⚪','lot_size':1,  'lot_unit':'kg',     'mcx_lot':30},
    }
}

MCX_DUTY = {'GC=F':1.15,'SI=F':1.15,'CL=F':1.0,'BZ=F':1.0,'NG=F':1.0}

TIMEFRAMES = {
    "15m": {"interval":"15m","higher":"1h", "daily":"1d"},
    "1h":  {"interval":"1h", "higher":"4h", "daily":"1d"},
    "4h":  {"interval":"90m","higher":"1d", "daily":"1d"},
    "1d":  {"interval":"1d", "higher":"1wk","daily":"1wk"},
}

# ─────────────────────────────────────────────────────────────
# MCX MARKET HOURS
# ─────────────────────────────────────────────────────────────
def get_market_status():
    n   = datetime.now(IST)
    mn  = n.hour * 60 + n.minute
    if n.weekday() >= 5: return "closed", n
    if 9*60 <= mn < 23*60+30: return "open", n
    if 8*60+45 <= mn < 9*60:  return "pre",  n
    return "closed", n

# ─────────────────────────────────────────────────────────────
# SELF-HEALING DATA ENGINE
# ─────────────────────────────────────────────────────────────
def _raw_dl(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty: raise ValueError(f"Empty: {ticker}")
    return df

def fetch_with_retry(ticker, period, interval, max_retries=3):
    for i in range(max_retries):
        try:    return _raw_dl(ticker, period, interval), ticker
        except:
            if i < max_retries-1: time.sleep(2**i)
    return None, ticker

@st.cache_data(ttl=3600)
def get_usdinr_rate():
    df, _ = fetch_with_retry("INR=X","1d","1d")
    if df is not None:
        try: return float(df['Close'].iloc[-1])
        except: pass
    return 83.50

def _period_for(interval):
    return {"15m":"60d","1h":"730d","90m":"60d","1d":"5y","1wk":"10y","4h":"2y"}.get(interval,"2y")

def _convert_inr(df, ticker):
    usdinr = get_usdinr_rate()
    duty   = MCX_DUTY.get(ticker, 1.0)
    mult   = usdinr * duty * (10/31.1034768)   if ticker=='GC=F' else \
             usdinr * duty * (1000/31.1034768)  if ticker=='SI=F' else \
             usdinr * duty
    for c in ['Open','High','Low','Close']:
        if c in df.columns: df[c] = df[c] * mult
    return df

def _clean(df, ticker):
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    df.index = df.index.tz_localize('UTC').tz_convert(IST) if df.index.tz is None \
               else df.index.tz_convert(IST)
    return _convert_inr(df, ticker)

@st.cache_data(ttl=600)
def fetch_data(ticker, interval, fallback=None):
    time.sleep(0.5)
    period = _period_for(interval)
    df, _  = fetch_with_retry(ticker, period, interval)
    if df is not None: return _clean(df, ticker), ticker, False
    if fallback:
        df, _ = fetch_with_retry(fallback, period, interval)
        if df is not None: return _clean(df, fallback), fallback, True
    return None, ticker, False

@st.cache_data(ttl=600)
def fetch_htf(ticker, htf_iv, fallback=None):
    df, actual, _ = fetch_data(ticker, htf_iv, fallback)
    if df is None or len(df) < 210: return 0, None, actual
    df.ta.ema(length=9,  append=True)
    df.ta.ema(length=21, append=True)
    df.ta.ema(length=200,append=True)
    df.dropna(inplace=True)
    if df.empty: return 0, df, actual
    c = df.iloc[-1]
    if pd.isna(c.get('EMA_200', float('nan'))): return 0, df, actual
    if c['EMA_9']>c['EMA_21'] and c['Close']>c['EMA_200']: return  1, df, actual
    if c['EMA_9']<c['EMA_21'] and c['Close']<c['EMA_200']: return -1, df, actual
    return 0, df, actual

# ─────────────────────────────────────────────────────────────
# ═══════════════  STRATEGY ENGINE (OOP)  ════════════════════
# ─────────────────────────────────────────────────────────────

class BaseStrategy(ABC):
    """Abstract base for all trading strategies."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def badge_class(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def info_class(self) -> str: ...

    @abstractmethod
    def apply_indicators(self, df: pd.DataFrame) -> pd.DataFrame: ...

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, htf_bias: int = 0) -> pd.DataFrame: ...

    @abstractmethod
    def add_chart_overlays(self, fig: go.Figure, df: pd.DataFrame, row: int = 1): ...


# ─────────────────────────────────────
# Strategy 1: Trend Confluence
# ─────────────────────────────────────
class TrendConfluence(BaseStrategy):
    name        = "Trend Confluence (MTF)"
    badge_class = "trend"
    info_class  = ""
    description = (
        "📈 <b>Best in: Trending markets (ADX > 25)</b><br>"
        "Signal: 9 EMA crosses 21 EMA · Price above/below 200 EMA · RSI > 55 (buy) or < 45 (sell)"
    )

    def apply_indicators(self, df):
        for L in [9, 21, 200]: df.ta.ema(length=L, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.atr(length=14, append=True)
        adx = df.ta.adx(length=14, append=False)
        if adx is not None and not adx.empty:
            for col in adx.columns: df[col] = adx[col]
        vol_ma = df['Volume'].rolling(20).mean() if 'Volume' in df.columns else None
        if vol_ma is not None: df['VOL_MA_20'] = vol_ma
        return df

    def generate_signals(self, df, htf_bias=0):
        df['Signal'] = 0
        if len(df) < 3: return df
        bull = ((df['EMA_9']>df['EMA_21']) & (df['EMA_9'].shift(1)<=df['EMA_21'].shift(1)) &
                (df['Close']>df['EMA_200']) & (df['RSI_14']>55))
        bear = ((df['EMA_9']<df['EMA_21']) & (df['EMA_9'].shift(1)>=df['EMA_21'].shift(1)) &
                (df['Close']<df['EMA_200']) & (df['RSI_14']<45))
        if htf_bias == 1:   df.loc[bull,'Signal'] =  1
        elif htf_bias ==-1: df.loc[bear,'Signal'] = -1
        else:
            df.loc[bull,'Signal'] =  1
            df.loc[bear,'Signal'] = -1
        return df

    def add_chart_overlays(self, fig, df, row=1):
        for L, clr, dash, w in [(9,'#3b82f6','solid',1.5),(21,'#f59e0b','solid',1.5),(200,'#1e293b','dot',2.5)]:
            cn = f'EMA_{L}'
            if cn in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[cn], name=f'EMA {L}',
                    line=dict(color=clr, width=w, dash=dash)
                ), row=row, col=1)


# ─────────────────────────────────────
# Strategy 2: Mean Reversion (Bollinger Fade)
# ─────────────────────────────────────
class MeanReversionBollinger(BaseStrategy):
    name        = "Mean Reversion (Bollinger Fade)"
    badge_class = "reversion"
    info_class  = "reversion"
    description = (
        "↔️ <b>Best in: Sideways / ranging markets (ADX < 20)</b><br>"
        "BUY: Price touches/pierces Lower BB AND RSI < 30 (oversold)<br>"
        "SELL: Price touches/pierces Upper BB AND RSI > 70 (overbought)"
    )

    def apply_indicators(self, df):
        df.ta.bbands(length=20, std=2, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.atr(length=14, append=True)
        adx = df.ta.adx(length=14, append=False)
        if adx is not None and not adx.empty:
            for col in adx.columns: df[col] = adx[col]
        return df

    def _bb_cols(self, df):
        """Locate the BB column names (pandas-ta names vary by version)."""
        lower = next((c for c in df.columns if c.startswith('BBL_')), None)
        upper = next((c for c in df.columns if c.startswith('BBU_')), None)
        mid   = next((c for c in df.columns if c.startswith('BBM_')), None)
        return lower, upper, mid

    def generate_signals(self, df, htf_bias=0):
        df['Signal'] = 0
        if len(df) < 3: return df
        lower, upper, _ = self._bb_cols(df)
        if lower is None or upper is None: return df

        # BUY: price touches or pierces lower band AND RSI oversold
        bull = (df['Low'] <= df[lower]) & (df['RSI_14'] < 30)
        # SELL: price touches or pierces upper band AND RSI overbought
        bear = (df['High'] >= df[upper]) & (df['RSI_14'] > 70)

        # HTF bias filter: still respect it when bias is strong
        if htf_bias == 1:   df.loc[bull,'Signal'] =  1
        elif htf_bias ==-1: df.loc[bear,'Signal'] = -1
        else:
            df.loc[bull,'Signal'] =  1
            df.loc[bear,'Signal'] = -1
        return df

    def add_chart_overlays(self, fig, df, row=1):
        lower, upper, mid = self._bb_cols(df)
        if lower is None: return

        # Upper band
        fig.add_trace(go.Scatter(
            x=df.index, y=df[upper], name='BB Upper',
            line=dict(color='#ec4899', width=1.5, dash='dot'), opacity=0.8
        ), row=row, col=1)
        # Shaded fill between upper and lower
        fig.add_trace(go.Scatter(
            x=df.index, y=df[lower], name='BB Lower',
            line=dict(color='#ec4899', width=1.5, dash='dot'), opacity=0.8,
            fill='tonexty', fillcolor='rgba(236,72,153,0.07)'
        ), row=row, col=1)
        # Middle band
        if mid:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[mid], name='BB Mid (SMA 20)',
                line=dict(color='#a78bfa', width=1, dash='dash'), opacity=0.7
            ), row=row, col=1)


# ─────────────────────────────────────
# Strategy 3: Volatility Breakout (Donchian)
# ─────────────────────────────────────
class VolatilityBreakoutDonchian(BaseStrategy):
    name        = "Volatility Breakout (Donchian)"
    badge_class = "breakout"
    info_class  = "breakout"
    description = (
        "💥 <b>Best in: News-driven / breakout events (ADX > 25, Volume spike)</b><br>"
        "BUY: Price breaks above 20-period High AND ADX > 25 AND Volume > Volume MA<br>"
        "SELL: Price breaks below 20-period Low AND ADX > 25 AND Volume > Volume MA"
    )

    DC_PERIOD = 20

    def apply_indicators(self, df):
        # Donchian Channels
        df['DC_HIGH'] = df['High'].rolling(self.DC_PERIOD).max().shift(1)  # shift(1) = confirmed previous high
        df['DC_LOW']  = df['Low'].rolling(self.DC_PERIOD).min().shift(1)
        df['DC_MID']  = (df['DC_HIGH'] + df['DC_LOW']) / 2

        df.ta.atr(length=14, append=True)
        df.ta.rsi(length=14, append=True)
        adx = df.ta.adx(length=14, append=False)
        if adx is not None and not adx.empty:
            for col in adx.columns: df[col] = adx[col]
        if 'Volume' in df.columns:
            df['VOL_MA_20'] = df['Volume'].rolling(20).mean()
        return df

    def generate_signals(self, df, htf_bias=0):
        df['Signal'] = 0
        if len(df) < self.DC_PERIOD + 2: return df

        adx_col = next((c for c in df.columns if c.startswith('ADX_')), None)
        adx_ok  = (df[adx_col] > 25) if adx_col else pd.Series(True, index=df.index)

        if 'VOL_MA_20' in df.columns:
            vol_ok = df['Volume'] > df['VOL_MA_20'] * 1.1
        else:
            vol_ok = pd.Series(True, index=df.index)

        # BUY breakout: close breaks above the previous 20-period high
        bull = (df['Close'] > df['DC_HIGH']) & adx_ok & vol_ok
        # SELL breakout: close breaks below the previous 20-period low
        bear = (df['Close'] < df['DC_LOW'])  & adx_ok & vol_ok

        if htf_bias == 1:   df.loc[bull,'Signal'] =  1
        elif htf_bias ==-1: df.loc[bear,'Signal'] = -1
        else:
            df.loc[bull,'Signal'] =  1
            df.loc[bear,'Signal'] = -1
        return df

    def add_chart_overlays(self, fig, df, row=1):
        if 'DC_HIGH' not in df.columns: return

        # Upper channel
        fig.add_trace(go.Scatter(
            x=df.index, y=df['DC_HIGH'], name=f'Donchian High ({self.DC_PERIOD})',
            line=dict(color='#f59e0b', width=1.8, dash='dot'), opacity=0.9
        ), row=row, col=1)
        # Lower channel with fill
        fig.add_trace(go.Scatter(
            x=df.index, y=df['DC_LOW'], name=f'Donchian Low ({self.DC_PERIOD})',
            line=dict(color='#f59e0b', width=1.8, dash='dot'), opacity=0.9,
            fill='tonexty', fillcolor='rgba(245,158,11,0.06)'
        ), row=row, col=1)
        # Mid channel
        fig.add_trace(go.Scatter(
            x=df.index, y=df['DC_MID'], name='Donchian Mid',
            line=dict(color='#d97706', width=1, dash='dash'), opacity=0.6
        ), row=row, col=1)


# Strategy registry — order matters for sidebar display
STRATEGIES: dict[str, BaseStrategy] = {
    "Trend Confluence (MTF)":          TrendConfluence(),
    "Mean Reversion (Bollinger Fade)": MeanReversionBollinger(),
    "Volatility Breakout (Donchian)":  VolatilityBreakoutDonchian(),
}

# Maps regime_engine's short strategy keys -> the display-name keys STRATEGIES uses here
STRATEGY_KEY_MAP = {
    "trend":     "Trend Confluence (MTF)",
    "reversion": "Mean Reversion (Bollinger Fade)",
    "breakout":  "Volatility Breakout (Donchian)",
}
AUTO_LABEL = "🤖 Auto (Regime-Adaptive)"

# ─────────────────────────────────────────────────────────────
# CHANDELIER EXIT (ATR-BASED TRAILING STOP)
# ─────────────────────────────────────────────────────────────
def compute_chandelier(df, atr_col, period=22, multiplier=3.0):
    if atr_col not in df.columns: return pd.Series(dtype=float, index=df.index)
    highest_high = df['High'].rolling(period).max()
    atr          = df[atr_col]
    return highest_high - multiplier * atr

# ─────────────────────────────────────────────────────────────
# RISK & POSITION SIZING
# ─────────────────────────────────────────────────────────────
def compute_risk(capital_inr, risk_pct, atr_inr, lot_size, price):
    risk_inr    = capital_inr * (risk_pct / 100.0)
    sl_dist     = 1.5 * atr_inr
    sl_per_lot  = sl_dist * lot_size
    lots        = max(1, math.floor(risk_inr / sl_per_lot)) if sl_per_lot > 0 else 1
    actual_risk = lots * sl_per_lot
    risk_pct_actual = (actual_risk / capital_inr) * 100
    margin_approx   = lots * lot_size * price * 0.07
    return {
        "risk_inr":         risk_inr,
        "sl_dist":          sl_dist,
        "sl_per_lot":       sl_per_lot,
        "lots":             lots,
        "actual_risk":      actual_risk,
        "risk_pct_actual":  risk_pct_actual,
        "margin_approx":    margin_approx,
    }

# ─────────────────────────────────────────────────────────────
# DUAL-TARGET TAKE PROFIT
# ─────────────────────────────────────────────────────────────
def dual_tp(entry, sl, signal):
    risk = abs(entry - sl)
    if signal == 1: return entry + 1.5 * risk, entry + 3.0 * risk
    else:           return entry - 1.5 * risk, entry - 3.0 * risk

# ─────────────────────────────────────────────────────────────
# LIVE BACKTESTER  (works for all strategies)
# ─────────────────────────────────────────────────────────────
def run_backtest(df, atr_col, lot_size, capital_start, risk_pct):
    signals = df[df['Signal'] != 0].copy()
    if signals.empty or not atr_col or atr_col not in df.columns: return None

    equity       = capital_start
    equity_curve = [capital_start]
    trades       = []
    df_idx       = list(df.index)

    for sig_time, sig_row in signals.iterrows():
        sig     = int(sig_row['Signal'])
        entry   = float(sig_row['Close'])
        atr_v   = float(sig_row.get(atr_col, 0.0))
        if atr_v <= 0: continue

        sl_dist  = 1.5 * atr_v
        sl       = entry - sl_dist if sig == 1 else entry + sl_dist
        tp1, _   = dual_tp(entry, sl, sig)

        risk_inr   = equity * (risk_pct / 100.0)
        sl_per_lot = sl_dist * lot_size
        lots       = max(1, math.floor(risk_inr / sl_per_lot)) if sl_per_lot > 0 else 1

        start_idx = df_idx.index(sig_time) + 1
        outcome   = "open"
        pnl_inr   = 0.0

        for i in range(start_idx, min(start_idx + 50, len(df_idx))):
            bar  = df.iloc[i]
            h, l = float(bar['High']), float(bar['Low'])
            if sig == 1:
                if l <= sl:  outcome = "loss"; pnl_inr = -lots * sl_dist * lot_size; break
                if h >= tp1: outcome = "win";  pnl_inr =  lots * sl_dist * lot_size * 1.5; break
            else:
                if h >= sl:  outcome = "loss"; pnl_inr = -lots * sl_dist * lot_size; break
                if l <= tp1: outcome = "win";  pnl_inr =  lots * sl_dist * lot_size * 1.5; break

        if outcome == "open": continue

        equity += pnl_inr
        equity_curve.append(equity)
        trades.append({
            "time": sig_time, "signal": sig, "entry": entry,
            "sl": sl, "tp1": tp1, "outcome": outcome, "pnl": pnl_inr, "equity": equity,
        })

    if not trades: return None

    trade_df     = pd.DataFrame(trades)
    wins         = trade_df[trade_df['outcome'] == 'win']
    losses       = trade_df[trade_df['outcome'] == 'loss']
    win_rate     = len(wins) / len(trade_df) * 100
    gross_profit = wins['pnl'].sum()   if not wins.empty   else 0.0
    gross_loss   = abs(losses['pnl'].sum()) if not losses.empty else 0.0
    profit_factor= gross_profit / gross_loss if gross_loss > 0 else float('inf')
    expectancy   = trade_df['pnl'].mean()
    eq           = pd.Series(equity_curve)
    drawdown     = eq - eq.cummax()
    max_dd       = float(drawdown.min())

    return {
        "trades": len(trade_df), "wins": len(wins), "losses": len(losses),
        "win_rate": win_rate, "profit_factor": profit_factor,
        "max_drawdown": max_dd, "expectancy": expectancy,
        "gross_profit": gross_profit, "gross_loss": gross_loss,
        "final_equity": equity, "trade_df": trade_df,
    }

# ─────────────────────────────────────────────────────────────
# CONFLUENCE MATRIX
# ─────────────────────────────────────────────────────────────
def _bias_cls(b):
    if b== 1: return "bull","🟢","BULLISH"
    if b==-1: return "bear","🔴","BEARISH"
    return "neut","⚪","NEUTRAL"

def build_verdict(tf_biases):
    v=list(tf_biases.values()); bulls=sum(1 for x in v if x==1); bears=sum(1 for x in v if x==-1)
    if bulls==len(v):              return "<span class='verdict-strong-buy'>⚡ STRONG BUY — ALL TFs ALIGNED</span>",   1
    if bears==len(v):              return "<span class='verdict-strong-sell'>⚡ STRONG SELL — ALL TFs ALIGNED</span>", -1
    if bulls>=2 and bears==0:      return "<span class='verdict-strong-buy'>✅ BUY — MAJORITY ALIGNED</span>",          1
    if bears>=2 and bulls==0:      return "<span class='verdict-strong-sell'>✅ SELL — MAJORITY ALIGNED</span>",       -1
    if bulls>0 and bears>0:        return "<span class='verdict-caution'>⚠️ CAUTION: NO CONFLUENCE</span>",            0
    return "<span class='verdict-neutral'>⚪ NEUTRAL — WAIT FOR SETUP</span>", 0

def render_matrix(tf_biases):
    cells = "".join(
        f"<div class='matrix-cell {c}'><div class='tf-label'>{lbl}</div>"
        f"<div class='tf-icon'>{ico}</div><div class='tf-text'>{txt}</div></div>"
        for lbl, b in tf_biases.items()
        for c, ico, txt in [_bias_cls(b)]
    )
    vhtml, _ = build_verdict(tf_biases)
    return (f"<div class='matrix-wrap'>{cells}"
            f"<div style='display:flex;align-items:center;padding:0 8px;min-width:240px'>{vhtml}</div></div>")

# ─────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────
def draw_dual_tp_zones(fig, df, last_sig_idx, signal, sl, tp1, tp2, row=1):
    if signal == 0: return
    entry = float(df.iloc[last_sig_idx]['Close'])
    x0    = df.index[last_sig_idx]
    x1    = df.index[min(last_sig_idx + 25, len(df)-1)]
    annot_bg = "rgba(17,21,29,.90)" if _THEME == "Dark" else "rgba(255,255,255,.92)"
    zones = [
        (min(entry,sl),  max(entry,sl),  "rgba(248,81,73,.10)",  "rgba(248,81,73,.7)", f"  SL ₹{sl:,.0f}",               "#f85149"),
        (min(entry,tp1), max(entry,tp1), "rgba(16,185,129,.12)", "rgba(16,185,129,.7)",f"  TP1 ₹{tp1:,.0f}  (1.5R — Conservative)", "#10b981"),
        (min(entry,tp2), max(entry,tp2), "rgba(5,150,105,.20)",  "rgba(5,150,105,.9)", f"  TP2 ₹{tp2:,.0f}  (3R — Aggressive)",     "#059669"),
    ]
    for y0, y1, fill, lc, label, fc in zones:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=fill, line=dict(color=lc,width=1.5,dash="dot"), layer="below", row=row, col=1)
        fig.add_annotation(x=x1, y=(y0+y1)/2, text=label, showarrow=False, font=dict(color=fc,size=ANNOT_FONT,family="IBM Plex Mono"), xanchor="left", bgcolor=annot_bg, row=row, col=1)
    fig.add_shape(type="line", x0=x0, x1=x1, y0=entry, y1=entry, line=dict(color="#58a6ff",width=1.5,dash="dash"), row=row, col=1)
    fig.add_annotation(x=x1, y=entry, text=f"  Entry ₹{entry:,.0f}", showarrow=False, font=dict(color="#58a6ff",size=ANNOT_FONT,family="IBM Plex Mono"), xanchor="left", bgcolor=annot_bg, row=row, col=1)

def draw_chandelier(fig, chan_series, df, row=1):
    valid = chan_series.dropna()
    if valid.empty: return
    fig.add_trace(go.Scatter(x=valid.index, y=valid.values, name="Chandelier Trail", line=dict(color="#e879f9",width=1.5,dash="dot"), opacity=0.85), row=row, col=1)

def add_session_highlight(fig, df):
    today = datetime.now(IST).date()
    sess  = df[df.index.date == today]
    if sess.empty: return
    fig.add_vrect(x0=sess.index[0], x1=sess.index[-1], fillcolor="rgba(37,99,235,.06)" if _THEME=="Dark" else "rgba(37,99,235,.04)",
                  layer="below", line_width=0, annotation_text="Today's Session", annotation_position="top left",
                  annotation_font=dict(size=ANNOT_FONT,color="#2563eb",family="IBM Plex Mono"))

def add_adx_panel(fig, df, adx_col, row=2):
    if not adx_col or adx_col not in df.columns: return
    s = df[adx_col].dropna()
    if s.empty: return
    annot_bg = "rgba(17,21,29,.90)" if _THEME == "Dark" else "rgba(255,255,255,.92)"
    fig.add_trace(go.Scatter(x=df.index, y=df[adx_col], name='ADX', line=dict(color='#a78bfa',width=2), fill='tozeroy', fillcolor='rgba(167,139,250,.07)'), row=row, col=1)
    for level, color, dash in [(20,'#f59e0b','dash'),(25,'#3b82f6','dash'),(40,'#3fb950','dot')]:
        fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=level, y1=level, line=dict(color=color,width=1,dash=dash), row=row, col=1)
    latest = float(s.iloc[-1])
    lbl = f"ADX {latest:.1f} · {'TRENDING' if latest>=25 else 'WEAKLY TRENDING' if latest>=20 else 'CHOPPY'}"
    clr = "#3fb950" if latest>=25 else "#3b82f6" if latest>=20 else "#f59e0b"
    fig.add_annotation(x=df.index[-1], y=latest, text=f"  {lbl}", showarrow=False, font=dict(color=clr,size=ANNOT_FONT+1,family="IBM Plex Mono"), xanchor="left", bgcolor=annot_bg, bordercolor=clr, borderwidth=1, borderpad=4, row=row, col=1)

def add_volume_panel(fig, df, row=3):
    if 'Volume' not in df.columns or df['Volume'].sum()==0: return
    colors = ['#089981' if c>=o else '#F23645' for c,o in zip(df['Close'],df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors, marker_line_width=0, opacity=0.75), row=row, col=1)
    if 'VOL_MA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['VOL_MA_20'], name='Vol MA(20)', line=dict(color='#f59e0b',width=1.5,dash='dash'), opacity=0.8), row=row, col=1)

def send_telegram(msg):
    tok = os.environ.get('TELEGRAM_TOKEN') or st.secrets.get("TELEGRAM_TOKEN","")
    cid = os.environ.get('TELEGRAM_CHAT_ID') or st.secrets.get("TELEGRAM_CHAT_ID","")
    if not tok or not cid: return
    try: requests.post(f"https://api.telegram.org/bot{tok}/sendMessage", json={"chat_id":cid,"text":msg,"parse_mode":"Markdown"})
    except: pass

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    st.markdown(
        "<h1 style='margin-bottom:2px'>⚡ CommodityPulse <span style='color:#2563eb'>Pro</span></h1>"
        "<p style='color:#64748b;font-size:13px;font-family:IBM Plex Mono,monospace;margin-top:0'>"
        "Phase 5 · Auto Regime Router · Triple Strategy Engine · Smart Risk · Dual-TP · Live Backtest · IST · INR</p>"
        "<div class='cp-header-rule'></div>",
        unsafe_allow_html=True)

    # ── SIDEBAR ─────────────────────────────
    with st.sidebar:
        st.markdown("<h3 style='color:#58a6ff;font-family:Syne,sans-serif;font-size:18px'>⚙️ Terminal</h3>", unsafe_allow_html=True)

        # Appearance controls — read at the top of the script (before CSS is
        # built) via session_state, so changing either takes effect on the
        # very next rerun. See _THEME / _MOBILE near the top of the file.
        ac1, ac2 = st.columns(2)
        with ac1:
            st.radio("Theme", ["Light", "Dark"], horizontal=True, key="theme_choice", label_visibility="collapsed")
        with ac2:
            st.toggle("📱 Compact", key="compact_mobile", help="Shorter chart + tighter margins for small screens")
        st.divider()

        mkt_status, now_ist = get_market_status()
        ts = now_ist.strftime('%H:%M IST')
        badge = (f"<span class='mkt-open'><span class='pulse-dot'></span> MCX OPEN · {ts}</span>" if mkt_status=="open"  else
                 f"<span class='mkt-pre'>🟡 PRE-MARKET · {ts}</span>"  if mkt_status=="pre"   else
                 f"<span class='mkt-closed'>🔴 MCX CLOSED · {ts}</span>")
        st.markdown(badge, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        region     = st.selectbox("🌐 Market Region", list(ASSETS.keys()))
        asset_name = st.selectbox("📌 Asset",         list(ASSETS[region].keys()))
        timeframe  = st.selectbox("⏱️ Base Timeframe", list(TIMEFRAMES.keys()))
        st.divider()

        strategy_name = st.selectbox("🧠 Algorithm", [AUTO_LABEL] + list(STRATEGIES.keys()))
        is_auto_mode  = (strategy_name == AUTO_LABEL)

        # Strategy info card — in Auto mode the actual strategy isn't known
        # until live data + regime classification runs below, so we show a
        # placeholder here and the resolved badge/reason near the chart header.
        if is_auto_mode:
            strategy = None
            st.markdown(
                "<div class='strat-info' style='border-left-color:#8b5cf6;"
                "background:linear-gradient(135deg,#f5f3ff,#ede9fe)'>"
                "🤖 <b>Auto Regime-Adaptive Mode</b><br>"
                "Reads live ADX, Donchian breakout status, and volume every refresh to pick "
                "the best-fit strategy: <b>Trend Confluence</b> (ADX ≥ 25), "
                "<b>Mean Reversion</b> (ADX &lt; 20), or <b>Volatility Breakout</b> "
                "(confirmed channel break + volume spike). The resolved strategy and "
                "reasoning appear near the chart header once data loads."
                "</div>", unsafe_allow_html=True)
        else:
            strategy = STRATEGIES[strategy_name]
            badge_html = f"<span class='strat-badge {strategy.badge_class}'>{strategy.name}</span>"
            info_html  = f"<div class='strat-info {strategy.info_class}'>{strategy.description}</div>"
            st.markdown(badge_html + info_html, unsafe_allow_html=True)
        st.divider()

        st.markdown(
            "<p style='color:#58a6ff;font-size:13px;font-weight:700;letter-spacing:1px;"
            "text-transform:uppercase;margin-bottom:4px'>💰 Risk Management</p>",
            unsafe_allow_html=True)
        capital_inr = st.number_input("Trading Capital (₹)", min_value=10_000, max_value=10_000_000, value=500_000, step=50_000, format="%d")
        risk_pct    = st.slider("Risk Per Trade (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)

        st.divider()
        st.markdown("<p style='color:#8b949e;font-size:11px;margin-bottom:4px'>🎨 CHART OPTIONS</p>", unsafe_allow_html=True)
        show_chandelier = st.toggle("📍 Chandelier Trail", value=True)
        chan_mult = st.slider("Trail Multiplier (ATR×)", 1.5, 5.0, 3.0, 0.5) if show_chandelier else 3.0

        st.divider()
        enable_alerts = st.toggle("🔔 Telegram Alerts", value=False)
        if st.button("🚀 Send Test Alert", use_container_width=True):
            _test_name = "Auto (Regime-Adaptive)" if is_auto_mode else strategy.name
            send_telegram(f"✅ CommodityPulse Pro Phase 5: [{_test_name}] — System online. (IST)")
            st.toast("Test alert sent!", icon="🚀")

        if "last_alert_time" not in st.session_state:
            st.session_state.last_alert_time = None

        st.markdown("<p style='color:#4b5563;font-size:10px;margin-top:14px'>MCX: Mon–Fri 09:00–23:30 IST</p>", unsafe_allow_html=True)

    # ── DATA FETCH ───────────────────────────
    ti          = ASSETS[region][asset_name]
    ticker      = ti['ticker'];  fallback = ti.get('fallback')
    emoji       = ti['emoji'];   lot_size = ti['lot_size'];  lot_unit = ti['lot_unit']
    tf_params   = TIMEFRAMES[timeframe]

    with st.spinner(f"📡 Fetching {asset_name} [{timeframe}]…"):
        df, actual_ticker, used_fallback = fetch_data(ticker, tf_params['interval'], fallback)

    if used_fallback:
        st.markdown(f"<div class='fallback-banner'>⚠️ {ticker} unavailable — auto-switched to <b>{actual_ticker}</b>.</div>", unsafe_allow_html=True)
    if df is None or len(df) < 50:
        st.error("⚠️ Data unavailable after retries. Wait 60s and refresh.")
        st.stop()

    # ── AUTO REGIME ROUTER ───────────────────
    # Resolve which strategy to run *before* indicators are applied below.
    # Hysteresis is kept per (region, asset, timeframe) across reruns via
    # session_state, so the router won't whipsaw across the ADX 20-25 zone.
    regime_info = None
    if is_auto_mode:
        regime_state_key = f"auto_regime::{region}::{asset_name}::{timeframe}"
        prior_key   = st.session_state.get(regime_state_key)
        probe_df    = compute_regime_probe(df)
        regime_info = classify_regime(probe_df, prior_strategy_key=prior_key, confirmed_row=-1)
        st.session_state[regime_state_key] = regime_info["strategy_key"]
        strategy      = STRATEGIES[STRATEGY_KEY_MAP[regime_info["strategy_key"]]]
        strategy_name = strategy.name

    htf_interval   = tf_params['higher']
    daily_interval = tf_params['daily']

    with st.spinner(f"🔭 HTF [{htf_interval}]…"):
        htf_bias, _, _ = fetch_htf(actual_ticker, htf_interval, fallback)
    with st.spinner(f"📅 Daily [{daily_interval}]…"):
        daily_bias, _, _ = fetch_htf(actual_ticker, daily_interval, fallback)

    # ── INDICATORS & SIGNALS ─────────────────
    df = strategy.apply_indicators(df)
    df.dropna(inplace=True)
    df = strategy.generate_signals(df, htf_bias=htf_bias)

    # MCX Time Filter — silence signals outside trading hours
    df['ist_hour']   = df.index.hour
    df['ist_minute'] = df.index.minute
    df['day_of_week']= df.index.dayofweek
    is_mcx_closed = (
        (df['ist_hour'] < 9) |
        ((df['ist_hour'] == 23) & (df['ist_minute'] > 30)) |
        (df['ist_hour'] > 23) |
        (df['day_of_week'] > 4)
    )
    df.loc[is_mcx_closed, 'Signal'] = 0

    # ── CURRENT VALUES ───────────────────────
    curr       = df.iloc[-1]
    prev_close = df.iloc[-2]['Close'] if len(df)>1 else curr['Close']
    chg_pct    = (curr['Close']-prev_close)/prev_close*100

    atr_col = next((c for c in df.columns if c.startswith('ATRr_')), None)
    adx_col = next((c for c in df.columns if c.startswith('ADX_')),  None)
    current_atr = float(curr[atr_col]) if atr_col and not pd.isna(curr.get(atr_col, float('nan'))) else 0.0
    current_adx = float(curr[adx_col]) if adx_col and not pd.isna(curr.get(adx_col, float('nan'))) else None

    regime = ("trending"       if current_adx and current_adx>=25 else
              "weakly trending" if current_adx and current_adx>=20 else
              "choppy"          if current_adx else "unknown")

    active_trend = "NEUTRAL ⚪"
    if 'EMA_200' in df.columns and not pd.isna(curr.get('EMA_200', float('nan'))):
        active_trend = "BULLISH 🟢" if curr['Close']>curr['EMA_200'] else "BEARISH 🔴"
    elif 'DC_HIGH' in df.columns:
        # For strategies without 200 EMA, use HTF bias as proxy
        active_trend = "BULLISH 🟢" if htf_bias==1 else "BEARISH 🔴" if htf_bias==-1 else "NEUTRAL ⚪"

    latest_signal = int(curr['Signal'])
    latest_time   = df.index[-1]
    signal_text   = "BUY 🟢" if latest_signal==1 else "SELL 🔴" if latest_signal==-1 else "HOLD ⚪"

    # Base bias for matrix
    base_bias = 0
    if all(c in df.columns for c in ('EMA_9','EMA_21','EMA_200')):
        e9,e21,e200 = curr['EMA_9'],curr['EMA_21'],curr['EMA_200']
        if not any(pd.isna(v) for v in [e9,e21,e200]):
            base_bias = 1 if e9>e21 and curr['Close']>e200 else (-1 if e9<e21 and curr['Close']<e200 else 0)
    else:
        base_bias = htf_bias  # Fall back to HTF for strategies without 200 EMA

    risk_data = compute_risk(capital_inr, risk_pct, current_atr, lot_size, float(curr['Close']))
    lots      = risk_data['lots']
    sl_dist   = risk_data['sl_dist']
    sl_price  = (float(curr['Close'])-sl_dist) if latest_signal>=0 else (float(curr['Close'])+sl_dist)
    tp1, tp2  = dual_tp(float(curr['Close']), sl_price, latest_signal if latest_signal!=0 else 1)

    rp_actual      = risk_data['risk_pct_actual']
    risk_level_html = (
        "<span class='risk-pill safe'>✅ SAFE</span>"         if rp_actual<2 else
        "<span class='risk-pill medium'>⚠️ MODERATE</span>"  if rp_actual<3 else
        "<span class='risk-pill danger'>🔴 HIGH RISK</span>"
    )

    chan_series = compute_chandelier(df, atr_col, period=22, multiplier=chan_mult) if show_chandelier and atr_col else pd.Series()
    bt = run_backtest(df, atr_col, lot_size, capital_inr, risk_pct)

    if enable_alerts and latest_signal!=0 and st.session_state.last_alert_time!=latest_time:
        adx_str = f"{current_adx:.1f}" if current_adx else "N/A"
        send_telegram(
            f"{emoji} *{asset_name} ({timeframe})* — {'BULLISH BUY ✅' if latest_signal==1 else 'BEARISH SELL 🔴'}\n"
            f"🤖 {strategy.name} | ADX {adx_str}\n"
            f"💰 ₹{curr['Close']:,.2f} | SL ₹{sl_price:,.2f}\n"
            f"🎯 TP1 ₹{tp1:,.2f} · TP2 ₹{tp2:,.2f}\n"
            f"📦 {lots} lot(s) to risk ₹{risk_data['actual_risk']:,.0f}"
        )
        st.session_state.last_alert_time = latest_time
        st.toast("Telegram Alert Sent!", icon="🚀")

    # ── STRATEGY BADGE ───────────────────────
    badge_prefix = "🤖 Auto → " if is_auto_mode else "🧠 Active: "
    st.markdown(
        f"<span class='strat-badge {strategy.badge_class}'>{badge_prefix}{strategy.name}</span>",
        unsafe_allow_html=True)

    if regime_info:
        adx_txt    = f"{regime_info['adx']:.1f}" if regime_info['adx'] is not None else "N/A"
        conf       = regime_info['confidence']
        conf_color = "#166534" if conf >= 60 else "#92400e" if conf >= 35 else "#991b1b"
        st.markdown(
            f"<div class='strat-info' style='border-left-color:{conf_color};margin-top:8px'>"
            f"<b>Regime Detector:</b> {regime_info['regime']} · Confidence {conf}% · ADX {adx_txt}<br>"
            f"{regime_info['reason']}</div>",
            unsafe_allow_html=True)

    # ── MARKET DATA METRICS ──────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header accent'>📊 Market Data</div>", unsafe_allow_html=True)
    mc1,mc2,mc3,mc4 = st.columns(4)
    mc1.metric("💰 Price (INR)",      f"₹{curr['Close']:,.2f}", f"{chg_pct:+.2f}%")
    mc2.metric("📈 Trend Bias",       active_trend)
    mc3.metric("📏 ATR (Volatility)", f"₹{current_atr:,.2f}")
    adx_disp = f"{current_adx:.1f} · {regime.title()}" if current_adx else "N/A"
    mc4.metric("🌊 ADX / Regime",    adx_disp)

    st.markdown("<div class='section-header accent'>🛡️ Account Risk</div>", unsafe_allow_html=True)
    rc1,rc2,rc3,rc4 = st.columns(4)
    rc1.metric("🤖 MTF Signal",       signal_text)
    rc2.metric("📦 Recommended Lots", f"{lots} lot(s)")
    rc3.metric("💸 Capital at Risk",  f"₹{risk_data['actual_risk']:,.0f}", f"{rp_actual:.2f}% of capital")
    rc4.metric("🏦 Est. Margin Req.", f"₹{risk_data['margin_approx']:,.0f}")

    st.markdown(
        f"<p style='font-size:12px;color:#64748b;font-family:IBM Plex Mono,monospace;margin:4px 0'>"
        f"Risk: {risk_level_html}&nbsp;&nbsp;Buy <b>{lots}</b> lot(s) · SL distance ₹{sl_dist:,.2f}/unit</p>",
        unsafe_allow_html=True)

    # ── CONFLUENCE MATRIX ────────────────────
    st.markdown("<div class='section-header'>🧩 Signal Confluence Matrix</div>", unsafe_allow_html=True)
    tf_biases = {timeframe: base_bias, htf_interval: htf_bias}
    if daily_interval != htf_interval: tf_biases[daily_interval] = daily_bias
    st.markdown(render_matrix(tf_biases), unsafe_allow_html=True)

    # Regime warnings specific to each strategy — only meaningful in manual
    # mode. In Auto mode the router already picked the strategy that fits
    # the current regime, so these would just contradict the badge above.
    if not is_auto_mode:
        if isinstance(strategy, TrendConfluence) and regime == "choppy":
            adx_v = f"{current_adx:.1f}" if current_adx else "N/A"
            st.warning(f"⚠️ ADX={adx_v} — CHOPPY market. Trend Confluence signals are unreliable. Wait for ADX > 20.", icon="⚠️")
        elif isinstance(strategy, MeanReversionBollinger) and regime == "trending":
            adx_v = f"{current_adx:.1f}" if current_adx else "N/A"
            st.info(f"ℹ️ ADX={adx_v} — TRENDING market. Mean Reversion works best when ADX < 20. Consider switching to Trend Confluence.", icon="ℹ️")
        elif isinstance(strategy, VolatilityBreakoutDonchian) and regime == "choppy":
            adx_v = f"{current_adx:.1f}" if current_adx else "N/A"
            st.warning(f"⚠️ ADX={adx_v} — CHOPPY market. Breakout signals require ADX > 25 + Volume spike. Waiting…", icon="⚠️")

    with st.expander("🔭 Multi-Timeframe Detail", expanded=False):
        d1,d2,d3 = st.columns(3)
        with d1:
            st.markdown(f"**Base TF ({timeframe})**")
            st.markdown(f"<h3 style='margin:4px 0'>{'🟢 BULLISH' if base_bias==1 else '🔴 BEARISH' if base_bias==-1 else '⚪ NEUTRAL'}</h3>", unsafe_allow_html=True)
            if 'RSI_14' in df.columns: st.caption(f"RSI: {curr['RSI_14']:.1f}")
        with d2:
            st.markdown(f"**HTF ({htf_interval})**")
            st.markdown(f"<h3 style='margin:4px 0'>{'🟢 BULLISH' if htf_bias==1 else '🔴 BEARISH' if htf_bias==-1 else '⚪ NEUTRAL'}</h3>", unsafe_allow_html=True)
            aligned = (htf_bias==1 and base_bias==1) or (htf_bias==-1 and base_bias==-1)
            st.caption("✅ Aligned — VALID" if aligned else "⚠️ Misaligned — FILTERED")
        with d3:
            st.markdown(f"**Daily ({daily_interval})**")
            st.markdown(f"<h3 style='margin:4px 0'>{'🟢 BULLISH' if daily_bias==1 else '🔴 BEARISH' if daily_bias==-1 else '⚪ NEUTRAL'}</h3>", unsafe_allow_html=True)

    with st.expander("📦 Smart Risk & Position Sizing", expanded=True):
        pc1,pc2,pc3 = st.columns(3)
        dir_str = "LONG 🟢" if latest_signal>=0 else "SHORT 🔴"
        with pc1:
            st.markdown(f"""<div class='dark-card'>
                <h4>Position Calculator</h4>
                <div class='drow'><span class='dlabel'>Capital at Risk ({risk_pct}%)</span><span class='dval'>₹{risk_data['risk_inr']:,.0f}</span></div>
                <div class='drow'><span class='dlabel'>ATR SL Distance</span><span class='dval'>₹{sl_dist:,.2f}/unit</span></div>
                <div class='drow'><span class='dlabel'>Risk per Lot</span><span class='dval'>₹{risk_data['sl_per_lot']:,.0f}</span></div>
                <div class='drow'><span class='dlabel'>MCX Lot Size</span><span class='dval'>{lot_size} {lot_unit}</span></div>
                <div class='drow'><span class='dlabel'>Recommended Lots</span><span class='dval b' style='font-size:22px'>{lots}</span></div>
                <div class='drow'><span class='dlabel'>Actual ₹ at Risk</span><span class='dval r'>₹{risk_data['actual_risk']:,.0f}</span></div>
            </div>""", unsafe_allow_html=True)
        with pc2:
            st.markdown(f"""<div class='dark-card'>
                <h4>Trade Blueprint · {dir_str}</h4>
                <div class='drow'><span class='dlabel'>Entry Price</span><span class='dval b'>₹{curr['Close']:,.2f}</span></div>
                <div class='drow'><span class='dlabel'>Stop Loss (1.5×ATR)</span><span class='dval r'>₹{sl_price:,.2f}</span></div>
                <div class='drow' style='border-left:3px solid #10b981;padding-left:8px'>
                    <span class='dlabel'>🎯 TP1 — Conservative (1.5R)</span><span class='dval g'>₹{tp1:,.2f}</span></div>
                <div class='drow' style='border-left:3px solid #059669;padding-left:8px'>
                    <span class='dlabel'>🚀 TP2 — Aggressive (3R)</span><span class='dval g'>₹{tp2:,.2f}</span></div>
                <div class='drow'><span class='dlabel'>Est. Margin (7%)</span><span class='dval a'>₹{risk_data['margin_approx']:,.0f}</span></div>
            </div>""", unsafe_allow_html=True)
        with pc3:
            pnl_tp1 = lots * sl_dist * lot_size * 1.5
            pnl_tp2 = lots * sl_dist * lot_size * 3.0
            pnl_sl  = -lots * sl_dist * lot_size
            st.markdown(f"""<div class='dark-card'>
                <h4>P&L Preview (₹)</h4>
                <div class='drow'><span class='dlabel'>If SL Hit</span><span class='dval r'>₹{pnl_sl:,.0f}</span></div>
                <div class='drow'><span class='dlabel'>If TP1 Hit (1.5R)</span><span class='dval g'>+₹{pnl_tp1:,.0f}</span></div>
                <div class='drow'><span class='dlabel'>If TP2 Hit (3R)</span><span class='dval g'>+₹{pnl_tp2:,.0f}</span></div>
                <div class='drow'><span class='dlabel'>Net R:R (TP1)</span><span class='dval'>1 : 1.5</span></div>
                <div class='drow'><span class='dlabel'>Net R:R (TP2)</span><span class='dval'>1 : 3.0</span></div>
                <div class='drow'><span class='dlabel'>Risk as % of Capital</span>
                    <span class='dval {"g" if rp_actual<2 else "a" if rp_actual<3 else "r"}'>{rp_actual:.2f}%</span></div>
            </div>""", unsafe_allow_html=True)

    # ── MAIN CHART ───────────────────────────
    st.markdown("<hr style='border:1px solid #e2e8f0;margin:24px 0 12px'>", unsafe_allow_html=True)
    st.markdown(f"### 📊 Advanced Chart · {strategy.name} · IST · INR · Dual-TP")

    has_vol = 'Volume' in df.columns and df['Volume'].sum()>0
    n_rows  = 3 if has_vol else 2
    heights = [0.60,0.20,0.20] if has_vol else [0.75,0.25]
    titles  = ["Price (INR)","ADX","Volume"] if has_vol else ["Price (INR)","ADX"]

    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, row_heights=heights,
                        vertical_spacing=0.03, subplot_titles=titles)

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Price (INR)',
        increasing_line_color='#089981', increasing_fillcolor='#089981',
        decreasing_line_color='#F23645', decreasing_fillcolor='#F23645'
    ), row=1, col=1)

    # Strategy-specific overlays (dynamic — changes per selection)
    strategy.add_chart_overlays(fig, df, row=1)

    # Chandelier trail
    if show_chandelier and not chan_series.empty:
        draw_chandelier(fig, chan_series, df, row=1)

    # Signal markers
    bulls = df[df['Signal']== 1]
    bears = df[df['Signal']==-1]
    if not bulls.empty:
        fig.add_trace(go.Scatter(
            x=bulls.index, y=bulls['Low'] - current_atr*0.5,
            mode='markers', name='Signal: BUY',
            marker=dict(symbol='triangle-up', color='#089981', size=14, line=dict(width=1.5,color='white'))
        ), row=1, col=1)
    if not bears.empty:
        fig.add_trace(go.Scatter(
            x=bears.index, y=bears['High'] + current_atr*0.5,
            mode='markers', name='Signal: SELL',
            marker=dict(symbol='triangle-down', color='#F23645', size=14, line=dict(width=1.5,color='white'))
        ), row=1, col=1)

    # Dual-TP zones for last signal
    sig_series = df['Signal']
    nz = sig_series[sig_series!=0]
    if not nz.empty and current_atr>0:
        li    = df.index.get_loc(nz.index[-1])
        ls    = int(sig_series.iloc[li])
        ep    = float(df.iloc[li]['Close'])
        sl_p  = ep - sl_dist if ls==1 else ep + sl_dist
        t1,t2 = dual_tp(ep, sl_p, ls)
        draw_dual_tp_zones(fig, df, li, ls, sl_p, t1, t2, row=1)

    if timeframe in ("15m","1h"): add_session_highlight(fig, df)
    add_adx_panel(fig, df, adx_col, row=2)
    if has_vol: add_volume_panel(fig, df, row=3)

    legend_bg = "rgba(17,21,29,.85)" if _THEME == "Dark" else "rgba(255,255,255,.92)"
    fig.update_layout(
        template=PALETTE['plot_template'], plot_bgcolor=PALETTE['plot_bg'], paper_bgcolor=PALETTE['plot_bg'],
        height=CHART_HEIGHT, margin=CHART_MARGIN,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1,
                    bgcolor=legend_bg,font=dict(family="IBM Plex Mono",size=9 if _MOBILE else 11)),
        font=dict(family="IBM Plex Mono", color=PALETTE['plot_font'])
    )
    fig.update_yaxes(title_text="Price (₹)",tickprefix="₹",showgrid=True,gridcolor=PALETTE['plot_grid'],row=1,col=1)
    fig.update_yaxes(title_text="ADX",showgrid=True,gridcolor=PALETTE['plot_grid'],title_font=dict(color="#a78bfa"),row=2,col=1)
    if has_vol: fig.update_yaxes(title_text="Volume",showgrid=True,gridcolor=PALETTE['plot_grid'],row=3,col=1)
    fig.update_xaxes(title_text="Indian Standard Time (IST)",showgrid=True,gridcolor=PALETTE['plot_grid'],
                     title_font=dict(color=PALETTE['plot_font'],size=11),row=n_rows,col=1)
    st.plotly_chart(fig, use_container_width=True, config={"responsive": True, "displaylogo": False})

    # ── LIVE BACKTEST ─────────────────────────
    st.markdown("<hr style='border:1px solid #e2e8f0;margin:24px 0 12px'>", unsafe_allow_html=True)
    st.markdown(f"### 🔬 Strategy Analytics — Live Backtest · {strategy.name}")

    if bt:
        wr   = bt['win_rate']
        pf   = bt['profit_factor']
        mdd  = bt['max_drawdown']
        exp  = bt['expectancy']
        n_tr = bt['trades']
        gp   = bt['gross_profit']
        gl   = bt['gross_loss']
        fe   = bt['final_equity']

        wr_cls  = "g" if wr>=55 else "a" if wr>=40 else "r"
        pf_cls  = "g" if pf>=1.5 else "a" if pf>=1.0 else "r"
        mdd_cls = "r" if abs(mdd)>capital_inr*0.10 else "a" if abs(mdd)>capital_inr*0.05 else "g"
        exp_cls = "g" if exp>0 else "r"
        pf_str  = f"{pf:.2f}" if pf != float('inf') else "∞"

        st.markdown(f"""
        <div class='bt-grid'>
            <div class='bt-card'>
                <div class='bt-val {wr_cls}'>{wr:.1f}%</div>
                <div class='bt-label'>Win Rate</div>
                <div class='bt-sub'>{bt['wins']}W / {bt['losses']}L of {n_tr} trades</div>
            </div>
            <div class='bt-card'>
                <div class='bt-val {pf_cls}'>{pf_str}</div>
                <div class='bt-label'>Profit Factor</div>
                <div class='bt-sub'>Gross profit ÷ gross loss</div>
            </div>
            <div class='bt-card'>
                <div class='bt-val {mdd_cls}'>₹{mdd:,.0f}</div>
                <div class='bt-label'>Max Drawdown</div>
                <div class='bt-sub'>{abs(mdd)/capital_inr*100:.1f}% of capital</div>
            </div>
            <div class='bt-card'>
                <div class='bt-val {exp_cls}'>₹{exp:,.0f}</div>
                <div class='bt-label'>Expectancy / Trade</div>
                <div class='bt-sub'>Avg ₹ won per completed trade</div>
            </div>
        </div>""", unsafe_allow_html=True)

        se1,se2,se3,se4 = st.columns(4)
        se1.metric("💹 Gross Profit",  f"₹{gp:,.0f}")
        se2.metric("📉 Gross Loss",    f"₹{gl:,.0f}")
        fe_delta = fe - capital_inr
        se3.metric("🏦 Final Equity",  f"₹{fe:,.0f}", f"{fe_delta:+,.0f}")
        se4.metric("📊 Total Signals", str(n_tr))

        with st.expander("📋 Trade Log (Last 20)", expanded=False):
            tdf = bt['trade_df'].copy()
            tdf['Action'] = tdf['signal'].map({1:'🟢 BUY',-1:'🔴 SELL'})
            tdf['Result'] = tdf['outcome'].map({'win':'✅ WIN','loss':'❌ LOSS'})
            tdf['time']   = pd.to_datetime(tdf['time']).dt.strftime('%Y-%m-%d %H:%M IST')
            show_cols = ['time','Action','entry','sl','tp1','Result','pnl','equity']
            tdf_show  = tdf[show_cols].rename(columns={
                'time':'Time (IST)','entry':'Entry ₹','sl':'SL ₹','tp1':'TP1 ₹','pnl':'P&L ₹','equity':'Running Equity ₹'
            }).iloc[::-1]
            st.dataframe(tdf_show, use_container_width=True, column_config={
                'Entry ₹':          st.column_config.NumberColumn(format="₹%.2f"),
                'SL ₹':             st.column_config.NumberColumn(format="₹%.2f"),
                'TP1 ₹':            st.column_config.NumberColumn(format="₹%.2f"),
                'P&L ₹':            st.column_config.NumberColumn(format="₹%.0f"),
                'Running Equity ₹': st.column_config.NumberColumn(format="₹%.0f"),
            })
    else:
        st.info("Insufficient completed signals in the current window for backtest statistics.")

    # ── SIGNAL LEDGER ─────────────────────────
    st.markdown("<hr style='border:1px solid #e2e8f0;margin:24px 0 12px'>", unsafe_allow_html=True)
    st.markdown("### 📝 Signal Ledger · MTF-Filtered")

    sig_hist = df[df['Signal']!=0].copy()
    if not sig_hist.empty:
        sig_hist['Action'] = sig_hist['Signal'].map({1:'🟢 BUY',-1:'🔴 SELL'})
        dcols = ['Action','Close']
        for x in ['RSI_14', atr_col, adx_col]:
            if x and x in sig_hist.columns: dcols.append(x)
        log_df = sig_hist[dcols].iloc[::-1].head(15)
        log_df.index = log_df.index.strftime('%Y-%m-%d %H:%M IST')
        log_df.index.name = 'Timestamp (IST)'
        cfg = {"Close": st.column_config.NumberColumn("Entry Price (₹)", format="₹%.2f"),
               "RSI_14": st.column_config.NumberColumn("RSI", format="%.1f")}
        if atr_col: cfg[atr_col] = st.column_config.NumberColumn("ATR (₹)", format="₹%.2f")
        if adx_col: cfg[adx_col] = st.column_config.NumberColumn("ADX", format="%.1f")
        st.dataframe(log_df, use_container_width=True, column_config=cfg)
    else:
        st.info("No MTF-confirmed signals yet in current window.")

    st.markdown(
        "<br><p style='text-align:center;color:#94a3b8;font-size:11px;font-family:IBM Plex Mono,monospace'>"
        "CommodityPulse Pro · Phase 5 · Auto Regime Router · IST · INR · Not Financial Advice</p>",
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
