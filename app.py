import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import requests
import os
import time
from abc import ABC, abstractmethod

# ==========================================
# PAGE CONFIGURATION & CUSTOM UI (CSS)
# ==========================================
st.set_page_config(
    page_title="CommodityPulse Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Syne:wght@700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Mono', monospace;
    }
    h1, h2, h3 {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        color: #0d1117;
        letter-spacing: -0.5px;
    }

    /* ---- METRIC CARDS ---- */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8faff 100%);
        border: 1px solid #e2e8f0;
        border-left: 4px solid #2563eb;
        padding: 16px 20px;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(37, 99, 235, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.12);
    }
    div[data-testid="metric-container"] > div:first-child {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #64748b;
    }
    div[data-testid="metric-container"] > div:nth-child(2) {
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 22px;
        color: #0d1117;
    }

    /* ---- REGIME BADGE ---- */
    .regime-trending {
        display: inline-block;
        background: #dcfce7; color: #15803d;
        border: 1px solid #86efac;
        border-radius: 20px; padding: 4px 14px;
        font-size: 12px; font-weight: 600;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.5px;
    }
    .regime-choppy {
        display: inline-block;
        background: #fef9c3; color: #92400e;
        border: 1px solid #fde68a;
        border-radius: 20px; padding: 4px 14px;
        font-size: 12px; font-weight: 600;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.5px;
    }
    .regime-neutral {
        display: inline-block;
        background: #f1f5f9; color: #475569;
        border: 1px solid #cbd5e1;
        border-radius: 20px; padding: 4px 14px;
        font-size: 12px; font-weight: 600;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.5px;
    }
    .mtf-badge-bullish {
        background: #dcfce7; color: #166534;
        border: 1px solid #86efac;
        border-radius: 6px; padding: 3px 10px;
        font-size: 11px; font-weight: 700;
        font-family: 'IBM Plex Mono', monospace;
    }
    .mtf-badge-bearish {
        background: #fee2e2; color: #991b1b;
        border: 1px solid #fca5a5;
        border-radius: 6px; padding: 3px 10px;
        font-size: 11px; font-weight: 700;
        font-family: 'IBM Plex Mono', monospace;
    }
    .mtf-badge-neutral {
        background: #f1f5f9; color: #475569;
        border: 1px solid #cbd5e1;
        border-radius: 6px; padding: 3px 10px;
        font-size: 11px; font-weight: 700;
        font-family: 'IBM Plex Mono', monospace;
    }

    /* ---- SIDEBAR STYLING ---- */
    section[data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #21262d;
    }
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: #e6edf3 !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background: #161b22;
        border: 1px solid #30363d;
        color: #e6edf3;
    }

    /* ---- POSITION SIZING CARD ---- */
    .pos-card {
        background: linear-gradient(135deg, #0d1117, #161b22);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 10px;
    }
    .pos-card h4 { color: #58a6ff; margin: 0 0 12px 0; font-family: 'Syne', sans-serif; }
    .pos-row {
        display: flex; justify-content: space-between;
        border-bottom: 1px solid #21262d;
        padding: 8px 0; font-size: 13px;
    }
    .pos-row:last-child { border-bottom: none; }
    .pos-label { color: #8b949e; }
    .pos-value { color: #e6edf3; font-weight: 600; }
    .pos-value.green { color: #3fb950; }
    .pos-value.red { color: #f85149; }
    .pos-value.blue { color: #58a6ff; }

    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# CONSTANTS & ASSET MAPPINGS
# ==========================================
ASSETS = {
    "Global Proxy (in INR)": {
        'Crude Oil (WTI)': {'ticker': 'CL=F', 'emoji': '🛢️', 'lot_size': 100, 'lot_unit': 'barrels'},
        'Natural Gas': {'ticker': 'NG=F', 'emoji': '🔥', 'lot_size': 10, 'lot_unit': 'mmBtu'},
        'Gold': {'ticker': 'GC=F', 'emoji': '🟡', 'lot_size': 10, 'lot_unit': 'grams'},
        'Silver': {'ticker': 'SI=F', 'emoji': '⚪', 'lot_size': 1, 'lot_unit': 'kg'}
    },
    "Indian MCX (in INR)": {
        'Crude Oil (MCX)': {'ticker': 'BZ=F', 'emoji': '🛢️', 'lot_size': 100, 'lot_unit': 'barrels'},
        'Natural Gas (MCX)': {'ticker': 'NG=F', 'emoji': '🔥', 'lot_size': 10, 'lot_unit': 'mmBtu'},
        'Gold (MCX)': {'ticker': 'GC=F', 'emoji': '🟡', 'lot_size': 10, 'lot_unit': 'grams'},
        'Silver (MCX)': {'ticker': 'SI=F', 'emoji': '⚪', 'lot_size': 1, 'lot_unit': 'kg'}
    }
}

# MCX import duty & unit conversion multipliers (applied on top of USDINR)
MCX_DUTY = {
    'GC=F': 1.15,   # 15% import duty on gold
    'SI=F': 1.15,   # 15% import duty on silver
    'CL=F': 1.0,
    'BZ=F': 1.0,
    'NG=F': 1.0,
}

TIMEFRAMES = {
    "15m": {"interval": "15m", "higher": "1h"},
    "1h":  {"interval": "1h",  "higher": "4h"},
    "4h":  {"interval": "90m", "higher": "1d"},
    "1d":  {"interval": "1d",  "higher": "1wk"},
}

# ==========================================
# QUANT ENGINE
# ==========================================
class BaseStrategy(ABC):
    def __init__(self): self.name = "Base Strategy"
    @abstractmethod
    def apply_indicators(self, df: pd.DataFrame) -> pd.DataFrame: pass
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame: pass
    @abstractmethod
    def add_chart_overlays(self, fig: go.Figure, df: pd.DataFrame): pass


class TrendConfluence(BaseStrategy):
    def __init__(self): self.name = "Trend Confluence (MTF)"

    def apply_indicators(self, df):
        df.ta.ema(length=9, append=True)
        df.ta.ema(length=21, append=True)
        df.ta.ema(length=200, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.atr(length=14, append=True)
        # ADX for Market Regime Detection
        adx_df = df.ta.adx(length=14, append=False)
        if adx_df is not None and not adx_df.empty:
            for col in adx_df.columns:
                df[col] = adx_df[col]
        return df

    def generate_signals(self, df, htf_bias=0):
        """
        htf_bias:  1 = higher timeframe is BULLISH
                  -1 = higher timeframe is BEARISH
                   0 = unknown / neutral (MTF filter disabled)
        """
        df['Signal'] = 0
        if len(df) < 3: return df

        bullish_base = (
            (df['EMA_9'] > df['EMA_21']) &
            (df['EMA_9'].shift(1) <= df['EMA_21'].shift(1)) &
            (df['Close'] > df['EMA_200']) &
            (df['RSI_14'] > 55)
        )
        bearish_base = (
            (df['EMA_9'] < df['EMA_21']) &
            (df['EMA_9'].shift(1) >= df['EMA_21'].shift(1)) &
            (df['Close'] < df['EMA_200']) &
            (df['RSI_14'] < 45)
        )

        # --- MTF FILTER: only fire signal if HTF agrees ---
        if htf_bias == 1:
            df.loc[bullish_base, 'Signal'] = 1          # BUY only if HTF bullish
        elif htf_bias == -1:
            df.loc[bearish_base, 'Signal'] = -1         # SELL only if HTF bearish
        else:
            df.loc[bullish_base, 'Signal'] = 1
            df.loc[bearish_base, 'Signal'] = -1

        return df

    def add_chart_overlays(self, fig, df):
        if 'EMA_9' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['EMA_9'], name='EMA 9',
                line=dict(color='#3b82f6', width=1.5, dash='solid')))
        if 'EMA_21' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['EMA_21'], name='EMA 21',
                line=dict(color='#f59e0b', width=1.5, dash='solid')))
        if 'EMA_200' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['EMA_200'], name='EMA 200',
                line=dict(color='#1e293b', width=2.5, dash='dot')))


STRATEGIES = {"Trend Confluence (MTF)": TrendConfluence()}


# ==========================================
# DATA, CURRENCY, TIMEZONE FUNCTIONS
# ==========================================
@st.cache_data(ttl=3600)
def get_usdinr_rate():
    try:
        usdinr = yf.download("INR=X", period="1d", interval="1d", progress=False)
        return float(usdinr['Close'].iloc[-1])
    except:
        return 83.50


@st.cache_data(ttl=600)
def fetch_data(ticker, timeframe):
    time.sleep(1)
    if timeframe == "15m":   fetch_period = "60d"
    elif timeframe == "1h":  fetch_period = "730d"
    elif timeframe == "4h":  fetch_period = "2y"
    elif timeframe == "90m": fetch_period = "60d"
    else:                    fetch_period = "5y"

    try:
        df = yf.download(ticker, period=fetch_period, interval=timeframe, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.dropna(inplace=True)

        # IST conversion
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
        else:
            df.index = df.index.tz_convert('Asia/Kolkata')

        # Currency & unit conversion (USD → INR with MCX calibration)
        usdinr = get_usdinr_rate()
        duty   = MCX_DUTY.get(ticker, 1.0)

        if ticker == 'GC=F':   multiplier = usdinr * duty * (10 / 31.1034768)   # per 10g
        elif ticker == 'SI=F': multiplier = usdinr * duty * (1000 / 31.1034768) # per 1 kg
        else:                  multiplier = usdinr * duty

        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = df[col] * multiplier
        return df

    except Exception as e:
        st.error(f"Data Fetching Error: {e}")
        return None


def get_htf_bias(ticker, htf_interval):
    """Fetch higher-timeframe data and return bias: 1 (bull), -1 (bear), 0 (neutral)."""
    df_htf = fetch_data(ticker, htf_interval)
    if df_htf is None or len(df_htf) < 210:
        return 0, None
    df_htf.ta.ema(length=9, append=True)
    df_htf.ta.ema(length=21, append=True)
    df_htf.ta.ema(length=200, append=True)
    df_htf.dropna(inplace=True)
    if df_htf.empty:
        return 0, df_htf
    curr = df_htf.iloc[-1]
    if pd.isna(curr.get('EMA_200', float('nan'))):
        return 0, df_htf
    if curr['EMA_9'] > curr['EMA_21'] and curr['Close'] > curr['EMA_200']:
        return 1, df_htf
    if curr['EMA_9'] < curr['EMA_21'] and curr['Close'] < curr['EMA_200']:
        return -1, df_htf
    return 0, df_htf


def compute_position_size(capital_inr, atr_inr, price_inr, lot_size, risk_pct=0.02):
    """
    Returns (lots, sl_distance, risk_amount) based on 2% capital-at-risk rule.
    SL = 1.5 × ATR (consistent with signal logic).
    """
    risk_amount = capital_inr * risk_pct
    sl_distance = 1.5 * atr_inr                           # per unit
    sl_per_lot  = sl_distance * lot_size                   # risk per 1 lot
    if sl_per_lot <= 0:
        return 0, sl_distance, risk_amount
    lots = risk_amount / sl_per_lot
    return max(1, round(lots)), sl_distance, risk_amount


def send_telegram_alert(message):
    token   = os.environ.get('TELEGRAM_TOKEN') or st.secrets.get("TELEGRAM_TOKEN", "")
    chat_id = os.environ.get('TELEGRAM_CHAT_ID') or st.secrets.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id: return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
        )
    except: pass


# ==========================================
# CHART HELPERS
# ==========================================
def add_sl_tp_boxes(fig, df, signal_row_idx, signal, atr_val, rr_ratio=2.0):
    """
    Draw SL and TP filled rectangles on the chart for the latest signal candle.
    signal:  1 (buy) or -1 (sell)
    """
    if signal == 0 or atr_val <= 0:
        return

    row   = df.iloc[signal_row_idx]
    price = float(row['Close'])
    sl    = price - 1.5 * atr_val if signal == 1 else price + 1.5 * atr_val
    tp    = price + rr_ratio * 1.5 * atr_val if signal == 1 else price - rr_ratio * 1.5 * atr_val

    # X-axis span: from signal candle to ~15 candles ahead
    x0 = df.index[signal_row_idx]
    x1_idx = min(signal_row_idx + 15, len(df) - 1)
    x1 = df.index[x1_idx]

    sl_color = "rgba(248, 81, 73, 0.15)"
    tp_color = "rgba(63, 185, 80, 0.15)"
    sl_line  = "rgba(248, 81, 73, 0.8)"
    tp_line  = "rgba(63, 185, 80, 0.8)"

    # SL box
    fig.add_shape(type="rect",
        x0=x0, x1=x1, y0=min(price, sl), y1=max(price, sl),
        fillcolor=sl_color, line=dict(color=sl_line, width=1.5, dash="dot"),
        layer="below"
    )
    fig.add_annotation(
        x=x1, y=(price + sl) / 2,
        text=f"  SL ₹{sl:,.0f}",
        showarrow=False, font=dict(color="#f85149", size=11, family="IBM Plex Mono"),
        xanchor="left", bgcolor="rgba(255,255,255,0.85)"
    )

    # TP box
    fig.add_shape(type="rect",
        x0=x0, x1=x1, y0=min(price, tp), y1=max(price, tp),
        fillcolor=tp_color, line=dict(color=tp_line, width=1.5, dash="dot"),
        layer="below"
    )
    fig.add_annotation(
        x=x1, y=(price + tp) / 2,
        text=f"  TP ₹{tp:,.0f}",
        showarrow=False, font=dict(color="#3fb950", size=11, family="IBM Plex Mono"),
        xanchor="left", bgcolor="rgba(255,255,255,0.85)"
    )

    # Entry line
    fig.add_shape(type="line",
        x0=x0, x1=x1, y0=price, y1=price,
        line=dict(color="#58a6ff", width=1.5, dash="dash")
    )
    fig.add_annotation(
        x=x1, y=price,
        text=f"  Entry ₹{price:,.0f}",
        showarrow=False, font=dict(color="#58a6ff", size=11, family="IBM Plex Mono"),
        xanchor="left", bgcolor="rgba(255,255,255,0.85)"
    )


def add_adx_panel(fig, df, row=2):
    """Add ADX subplot line + threshold band."""
    adx_col = next((c for c in df.columns if c.startswith('ADX_')), None)
    if adx_col is None:
        return
    fig.add_trace(go.Scatter(
        x=df.index, y=df[adx_col],
        name='ADX', line=dict(color='#a78bfa', width=2),
        fill='tozeroy', fillcolor='rgba(167,139,250,0.08)'
    ), row=row, col=1)
    fig.add_hline(y=20, line=dict(color='#f59e0b', width=1.5, dash='dash'),
                  annotation_text="  ADX=20 (Choppy Threshold)",
                  annotation_font=dict(color='#f59e0b', size=10),
                  row=row, col=1)
    fig.add_hline(y=40, line=dict(color='#3fb950', width=1, dash='dot'),
                  annotation_text="  ADX=40 (Strong Trend)",
                  annotation_font=dict(color='#3fb950', size=10),
                  row=row, col=1)


# ==========================================
# MAIN UI
# ==========================================
def main():
    # ---- HEADER ----
    st.markdown(
        "<h1 style='margin-bottom:2px'>⚡ CommodityPulse <span style='color:#2563eb'>Pro</span></h1>"
        "<p style='color:#64748b; font-size:13px; font-family:IBM Plex Mono,monospace; margin-top:0'>"
        "Multi-Timeframe Quant Terminal · IST · INR · MCX-Calibrated</p>",
        unsafe_allow_html=True
    )

    # ---- SIDEBAR ----
    with st.sidebar:
        st.markdown(
            "<h3 style='color:#58a6ff; font-family:Syne,sans-serif; font-size:18px'>⚙️ Terminal Settings</h3>",
            unsafe_allow_html=True
        )

        region      = st.selectbox("🌐 Market Region", list(ASSETS.keys()))
        asset_name  = st.selectbox("📌 Asset", list(ASSETS[region].keys()))
        timeframe   = st.selectbox("⏱️ Base Timeframe", list(TIMEFRAMES.keys()))

        st.divider()
        strategy_name = st.selectbox("🧠 Algorithm", list(STRATEGIES.keys()))
        rr_ratio      = st.slider("🎯 Risk:Reward Ratio", min_value=1.0, max_value=5.0, value=2.0, step=0.5)

        st.divider()
        st.markdown("<p style='color:#8b949e; font-size:12px; margin-bottom:4px'>💰 POSITION SIZING</p>", unsafe_allow_html=True)
        capital_inr = st.number_input(
            "Trading Capital (₹)",
            min_value=10_000, max_value=10_000_000,
            value=500_000, step=50_000,
            format="%d"
        )

        st.divider()
        enable_alerts = st.toggle("🔔 Telegram Alerts", value=False)
        if st.button("🚀 Send Test Alert", use_container_width=True):
            send_telegram_alert("✅ CommodityPulse Pro: MTF System online. Alerts active (IST).")
            st.toast("Test alert sent!", icon="🚀")

        if "last_alert_time" not in st.session_state:
            st.session_state.last_alert_time = None

    # ---- RESOLVE TICKER & STRATEGY ----
    ticker_info = ASSETS[region][asset_name]
    ticker      = ticker_info['ticker']
    emoji       = ticker_info['emoji']
    lot_size    = ticker_info['lot_size']
    lot_unit    = ticker_info['lot_unit']
    tf_params   = TIMEFRAMES[timeframe]
    strategy    = STRATEGIES[strategy_name]

    htf_interval = tf_params['higher']

    # ---- DATA FETCH ----
    col_spin1, col_spin2 = st.columns(2)
    with st.spinner(f"📡 Fetching {asset_name} [{timeframe}] data (IST)…"):
        df = fetch_data(ticker, tf_params['interval'])

    with st.spinner(f"🔭 Fetching higher-timeframe [{htf_interval}] for MTF filter…"):
        htf_bias, df_htf = get_htf_bias(ticker, htf_interval)

    if df is None or len(df) < 50:
        st.error("⚠️ Insufficient data. Market may be closed or rate-limited. Try again shortly.")
        st.stop()

    # ---- INDICATORS & SIGNALS ----
    df = strategy.apply_indicators(df)
    df.dropna(inplace=True)
    df = strategy.generate_signals(df, htf_bias=htf_bias)

    curr         = df.iloc[-1]
    prev_close   = df.iloc[-2]['Close'] if len(df) > 1 else curr['Close']
    chg_pct      = ((curr['Close'] - prev_close) / prev_close) * 100

    atr_col      = next((c for c in df.columns if c.startswith('ATRr_')), None)
    current_atr  = float(curr[atr_col]) if atr_col and not pd.isna(curr[atr_col]) else 0.0

    adx_col      = next((c for c in df.columns if c.startswith('ADX_')), None)
    current_adx  = float(curr[adx_col]) if adx_col and not pd.isna(curr[adx_col]) else None

    # Market Regime
    if current_adx is None:
        regime = "unknown"
    elif current_adx >= 25:
        regime = "trending"
    elif current_adx >= 20:
        regime = "weakly trending"
    else:
        regime = "choppy"

    active_trend = "NEUTRAL ⚪"
    if 'EMA_200' in df.columns and not pd.isna(curr['EMA_200']):
        active_trend = "BULLISH 🟢" if curr['Close'] > curr['EMA_200'] else "BEARISH 🔴"

    latest_signal = int(curr['Signal'])
    latest_time   = df.index[-1]
    signal_text   = "BUY 🟢" if latest_signal == 1 else "SELL 🔴" if latest_signal == -1 else "HOLD ⚪"

    # HTF label
    htf_label_html = {
        1:  "<span class='mtf-badge-bullish'>BULLISH</span>",
        -1: "<span class='mtf-badge-bearish'>BEARISH</span>",
        0:  "<span class='mtf-badge-neutral'>NEUTRAL</span>",
    }[htf_bias]

    # ---- POSITION SIZING ----
    lots, sl_dist, risk_amt = compute_position_size(
        capital_inr, current_atr, float(curr['Close']), lot_size, risk_pct=0.02
    )
    sl_price = float(curr['Close']) - 1.5 * current_atr if latest_signal >= 0 \
               else float(curr['Close']) + 1.5 * current_atr
    tp_price = float(curr['Close']) + rr_ratio * 1.5 * current_atr if latest_signal >= 0 \
               else float(curr['Close']) - rr_ratio * 1.5 * current_atr

    # ---- TELEGRAM ALERT ----
    if enable_alerts and latest_signal != 0 and st.session_state.last_alert_time != latest_time:
        sig_str  = "BULLISH BUY ✅" if latest_signal == 1 else "BEARISH SELL 🔴"
        mtf_str  = "BULLISH" if htf_bias == 1 else "BEARISH" if htf_bias == -1 else "NEUTRAL"
        sl_calc  = float(curr['Close']) - 1.5 * current_atr if latest_signal == 1 \
                   else float(curr['Close']) + 1.5 * current_atr
        tp_calc  = float(curr['Close']) + rr_ratio * 1.5 * current_atr if latest_signal == 1 \
                   else float(curr['Close']) - rr_ratio * 1.5 * current_atr
        msg = (
            f"{emoji} *{asset_name} ({timeframe})* — {sig_str}\n"
            f"🤖 Algo: {strategy.name}\n"
            f"🔭 HTF ({htf_interval}): {mtf_str}\n"
            f"📊 ADX: {current_adx:.1f if current_adx else 'N/A'} | Regime: {regime.title()}\n"
            f"💰 Price: ₹{curr['Close']:,.2f}\n"
            f"📐 SL: ₹{sl_calc:,.2f} | TP: ₹{tp_calc:,.2f}\n"
            f"📦 Suggested Lots: {lots} ({lot_unit})"
        )
        send_telegram_alert(msg)
        st.session_state.last_alert_time = latest_time
        st.toast("Telegram Alert Sent!", icon="🚀")

    # ==========================================
    # METRIC ROW
    # ==========================================
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💰 Price (INR)", f"₹{curr['Close']:,.2f}", f"{chg_pct:+.2f}%")
    c2.metric("📈 Trend Bias", active_trend)
    c3.metric("📏 ATR Volatility", f"₹{current_atr:,.2f}")
    c4.metric("🤖 MTF Signal", signal_text)
    adx_display = f"{current_adx:.1f}" if current_adx is not None else "N/A"
    c5.metric("🌊 ADX", adx_display)

    # ==========================================
    # MTF CONFLUENCE PANEL
    # ==========================================
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🔭 Multi-Timeframe Confluence Analysis", expanded=True):
        mc1, mc2, mc3 = st.columns(3)

        with mc1:
            st.markdown(f"**Base TF ({timeframe}) Signal**")
            raw_bull = (
                'EMA_9' in df.columns and 'EMA_21' in df.columns and 'EMA_200' in df.columns and
                curr['EMA_9'] > curr['EMA_21'] and curr['Close'] > curr['EMA_200']
            )
            raw_bear = (
                'EMA_9' in df.columns and 'EMA_21' in df.columns and 'EMA_200' in df.columns and
                curr['EMA_9'] < curr['EMA_21'] and curr['Close'] < curr['EMA_200']
            )
            base_bias = "🟢 BULLISH" if raw_bull else "🔴 BEARISH" if raw_bear else "⚪ NEUTRAL"
            st.markdown(f"<h3 style='margin:4px 0'>{base_bias}</h3>", unsafe_allow_html=True)
            if 'RSI_14' in df.columns:
                st.caption(f"RSI: {curr['RSI_14']:.1f}")

        with mc2:
            st.markdown(f"**Higher TF ({htf_interval}) Bias**")
            htf_disp = "🟢 BULLISH" if htf_bias == 1 else "🔴 BEARISH" if htf_bias == -1 else "⚪ NEUTRAL"
            st.markdown(f"<h3 style='margin:4px 0'>{htf_disp}</h3>", unsafe_allow_html=True)
            aligned = (htf_bias == 1 and raw_bull) or (htf_bias == -1 and raw_bear)
            st.caption("✅ Aligned — signal is VALID" if aligned else "⚠️ Misaligned — signal FILTERED")

        with mc3:
            st.markdown("**Market Regime (ADX)**")
            if regime == "choppy":
                regime_html = "<span class='regime-choppy'>⚠️ CHOPPY / SIDEWAYS</span>"
                st.markdown(regime_html, unsafe_allow_html=True)
                st.warning("ADX < 20: Trend strategies perform poorly. Avoid new entries.", icon="⚠️")
            elif regime in ("trending", "weakly trending"):
                regime_html = "<span class='regime-trending'>✅ TRENDING</span>"
                st.markdown(regime_html, unsafe_allow_html=True)
                st.caption(f"ADX={adx_display} — Trend strategies are effective.")
            else:
                st.markdown("<span class='regime-neutral'>⚪ UNKNOWN</span>", unsafe_allow_html=True)

    # ==========================================
    # POSITION SIZING CARD
    # ==========================================
    with st.expander("📦 Dynamic Position Sizing (2% Risk Rule)", expanded=True):
        ps1, ps2 = st.columns([1, 1])
        with ps1:
            direction_str = "LONG  🟢" if latest_signal >= 0 else "SHORT 🔴"
            st.markdown(f"""
            <div class='pos-card'>
                <h4>Position Calculator</h4>
                <div class='pos-row'>
                    <span class='pos-label'>Capital at Risk (2%)</span>
                    <span class='pos-value'>₹{risk_amt:,.0f}</span>
                </div>
                <div class='pos-row'>
                    <span class='pos-label'>ATR-based SL Distance</span>
                    <span class='pos-value'>₹{sl_dist:,.2f} / unit</span>
                </div>
                <div class='pos-row'>
                    <span class='pos-label'>Lot Size</span>
                    <span class='pos-value'>{lot_size} {lot_unit} / lot</span>
                </div>
                <div class='pos-row'>
                    <span class='pos-label'>Suggested Lots</span>
                    <span class='pos-value blue' style='font-size:20px'>{lots} lots</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with ps2:
            sl_display = f"₹{sl_price:,.2f}" if current_atr > 0 else "N/A"
            tp_display = f"₹{tp_price:,.2f}" if current_atr > 0 else "N/A"
            sl_color   = "red" if latest_signal >= 0 else "green"
            tp_color   = "green" if latest_signal >= 0 else "red"
            entry_disp = f"₹{curr['Close']:,.2f}"

            st.markdown(f"""
            <div class='pos-card'>
                <h4>Trade Blueprint · {direction_str}</h4>
                <div class='pos-row'>
                    <span class='pos-label'>Entry</span>
                    <span class='pos-value blue'>{entry_disp}</span>
                </div>
                <div class='pos-row'>
                    <span class='pos-label'>Stop Loss (1.5× ATR)</span>
                    <span class='pos-value {sl_color}'>{sl_display}</span>
                </div>
                <div class='pos-row'>
                    <span class='pos-label'>Take Profit ({rr_ratio}R)</span>
                    <span class='pos-value {tp_color}'>{tp_display}</span>
                </div>
                <div class='pos-row'>
                    <span class='pos-label'>R:R Ratio</span>
                    <span class='pos-value'>1 : {rr_ratio:.1f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ==========================================
    # MAIN CHART (Candlestick + EMA Overlays + ADX subplot)
    # ==========================================
    st.markdown("<hr style='border:1px solid #e2e8f0; margin:24px 0 16px'>", unsafe_allow_html=True)
    st.markdown("### 📊 Advanced Chart  ·  IST Timeline  ·  INR Prices")

    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.78, 0.22],
        vertical_spacing=0.04
    )

    # --- Candlestick ---
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Price (INR)',
        increasing_line_color='#089981', increasing_fillcolor='#089981',
        decreasing_line_color='#F23645', decreasing_fillcolor='#F23645'
    ), row=1, col=1)

    # --- EMA Overlays ---
    if 'EMA_9' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['EMA_9'], name='EMA 9',
            line=dict(color='#3b82f6', width=1.5)), row=1, col=1)
    if 'EMA_21' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['EMA_21'], name='EMA 21',
            line=dict(color='#f59e0b', width=1.5)), row=1, col=1)
    if 'EMA_200' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['EMA_200'], name='EMA 200',
            line=dict(color='#1e293b', width=2.5, dash='dot')), row=1, col=1)

    # --- Buy/Sell Markers ---
    bulls = df[df['Signal'] == 1]
    bears = df[df['Signal'] == -1]

    if not bulls.empty:
        fig.add_trace(go.Scatter(
            x=bulls.index, y=bulls['Low'] - current_atr * 0.5,
            mode='markers',
            marker=dict(symbol='triangle-up', color='#089981', size=14,
                        line=dict(width=1.5, color='white')),
            name='MTF Buy Signal'
        ), row=1, col=1)

    if not bears.empty:
        fig.add_trace(go.Scatter(
            x=bears.index, y=bears['High'] + current_atr * 0.5,
            mode='markers',
            marker=dict(symbol='triangle-down', color='#F23645', size=14,
                        line=dict(width=1.5, color='white')),
            name='MTF Sell Signal'
        ), row=1, col=1)

    # --- SL / TP Boxes for LATEST signal ---
    last_signal_idx = None
    signal_series = df['Signal']
    non_zero = signal_series[signal_series != 0]
    if not non_zero.empty:
        last_sig_time = non_zero.index[-1]
        last_signal_idx = df.index.get_loc(last_sig_time)

    if last_signal_idx is not None and current_atr > 0:
        add_sl_tp_boxes(
            fig, df,
            signal_row_idx=last_signal_idx,
            signal=int(signal_series.iloc[last_signal_idx]),
            atr_val=current_atr,
            rr_ratio=rr_ratio
        )

    # --- ADX Subplot ---
    add_adx_panel(fig, df, row=2)

    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        height=700,
        margin=dict(l=20, r=80, t=30, b=20),
        xaxis_rangeslider_visible=False,
        xaxis2=dict(
            title="Indian Standard Time (IST)",
            showgrid=True, gridcolor='#f1f5f9',
            title_font=dict(family="IBM Plex Mono", size=11, color="#64748b")
        ),
        yaxis=dict(
            showgrid=True, gridcolor='#f1f5f9',
            tickprefix="₹",
            title="Price (INR)",
            title_font=dict(family="IBM Plex Mono", size=11, color="#64748b")
        ),
        yaxis2=dict(
            title="ADX",
            showgrid=True, gridcolor='#f1f5f9',
            title_font=dict(family="IBM Plex Mono", size=11, color="#a78bfa")
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(255,255,255,0.9)", font=dict(family="IBM Plex Mono", size=11)
        ),
        font=dict(family="IBM Plex Mono")
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # HISTORICAL LEDGER
    # ==========================================
    st.markdown("<hr style='border:1px solid #e2e8f0; margin:24px 0 16px'>", unsafe_allow_html=True)
    st.markdown("### 📝 Signal Ledger  ·  MTF-Filtered Only")

    signal_history = df[df['Signal'] != 0].copy()
    if not signal_history.empty:
        signal_history['Action'] = signal_history['Signal'].map({1: '🟢 BUY', -1: '🔴 SELL'})
        display_cols = ['Action', 'Close']
        if 'RSI_14' in signal_history.columns:   display_cols.append('RSI_14')
        if atr_col and atr_col in signal_history: display_cols.append(atr_col)
        if adx_col and adx_col in signal_history: display_cols.append(adx_col)

        log_df = signal_history[display_cols].iloc[::-1].head(12)
        log_df.index = log_df.index.strftime('%Y-%m-%d %H:%M IST')
        log_df.index.name = 'Timestamp (IST)'

        col_cfg = {
            "Close": st.column_config.NumberColumn("Entry Price (₹)", format="₹%.2f"),
            "RSI_14": st.column_config.NumberColumn("RSI", format="%.1f"),
        }
        if atr_col: col_cfg[atr_col] = st.column_config.NumberColumn("ATR (₹)", format="₹%.2f")
        if adx_col: col_cfg[adx_col] = st.column_config.NumberColumn("ADX", format="%.1f")

        st.dataframe(log_df, use_container_width=True, column_config=col_cfg)
    else:
        st.info("No MTF-confirmed signals in the current window. Market is consolidating or HTF is misaligned.")

    # ---- FOOTER ----
    st.markdown(
        "<br><p style='text-align:center; color:#94a3b8; font-size:11px; font-family:IBM Plex Mono,monospace'>"
        "CommodityPulse Pro · MTF Engine · IST · INR · MCX-Calibrated · 2% Risk Rule · Not Financial Advice</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
