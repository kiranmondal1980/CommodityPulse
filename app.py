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

# Custom CSS for a "Premium Terminal" Look
st.markdown("""
    <style>
    /* Metric Cards Styling */
    div[data-testid="metric-container"] {
        background-color: #11141a;
        border: 1px solid #2b2b36;
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease-in-out;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        border: 1px solid #4a4a5a;
    }
    /* Headers */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 600;
        color: #e0e0e0;
    }
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# CONSTANTS & ASSET MAPPINGS
# ==========================================
ASSETS = {
    "Global": {
        'Crude Oil (WTI)': {'ticker': 'CL=F', 'emoji': '🛢️'},
        'Natural Gas': {'ticker': 'NG=F', 'emoji': '🔥'},
        'Gold': {'ticker': 'GC=F', 'emoji': '🟡'},
        'Silver': {'ticker': 'SI=F', 'emoji': '⚪'}
    },
    "Indian MCX": {
        'Crude Oil (MCX)': {'ticker': 'BZ=F', 'emoji': '🛢️'}, 
        'Natural Gas (MCX)': {'ticker': 'NG=F', 'emoji': '🔥'},
        'Gold (MCX)': {'ticker': 'GC=F', 'emoji': '🟡'},
        'Silver (MCX)': {'ticker': 'SI=F', 'emoji': '⚪'}
    }
}

TIMEFRAMES = {
    "15m": {"interval": "15m"},
    "1h": {"interval": "1h"},
    "4h": {"interval": "90m"}, 
    "1d": {"interval": "1d"}
}

# ==========================================
# QUANT ENGINE (OOP MULTI-STRATEGY)
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
    def __init__(self): self.name = "Trend Confluence"
    def apply_indicators(self, df):
        df.ta.ema(length=9, append=True)
        df.ta.ema(length=21, append=True)
        df.ta.ema(length=200, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.atr(length=14, append=True)
        return df
    def generate_signals(self, df):
        df['Signal'] = 0
        if len(df) < 2: return df
        bullish = (df['EMA_9'] > df['EMA_21']) & (df['EMA_9'].shift(1) <= df['EMA_21'].shift(1)) & (df['Close'] > df['EMA_200']) & (df['RSI_14'] > 55)
        bearish = (df['EMA_9'] < df['EMA_21']) & (df['EMA_9'].shift(1) >= df['EMA_21'].shift(1)) & (df['Close'] < df['EMA_200']) & (df['RSI_14'] < 45)
        df.loc[bullish, 'Signal'] = 1
        df.loc[bearish, 'Signal'] = -1
        return df
    def add_chart_overlays(self, fig, df):
        # Neon colored EMAs for premium look
        if 'EMA_9' in df: fig.add_trace(go.Scatter(x=df.index, y=df['EMA_9'], name='EMA 9', line=dict(color='#00d2ff', width=1.5)))
        if 'EMA_21' in df: fig.add_trace(go.Scatter(x=df.index, y=df['EMA_21'], name='EMA 21', line=dict(color='#ff9900', width=1.5)))
        if 'EMA_200' in df: fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], name='EMA 200', line=dict(color='#ffffff', width=2.5)))

class MeanReversion(BaseStrategy):
    def __init__(self): self.name = "Mean Reversion (BB Fade)"
    def apply_indicators(self, df):
        df.ta.bbands(length=20, std=2, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.atr(length=14, append=True)
        return df
    def generate_signals(self, df):
        df['Signal'] = 0
        if 'BBL_20_2.0' not in df.columns: return df
        bullish = (df['Close'] <= df['BBL_20_2.0']) & (df['RSI_14'] < 30)
        bearish = (df['Close'] >= df['BBU_20_2.0']) & (df['RSI_14'] > 70)
        df.loc[bullish, 'Signal'] = 1
        df.loc[bearish, 'Signal'] = -1
        return df
    def add_chart_overlays(self, fig, df):
        if 'BBU_20_2.0' in df: 
            fig.add_trace(go.Scatter(x=df.index, y=df['BBU_20_2.0'], name='Upper BB', line=dict(color='#888888', dash='dot', width=1.5)))
            fig.add_trace(go.Scatter(x=df.index, y=df['BBL_20_2.0'], name='Lower BB', line=dict(color='#888888', dash='dot', width=1.5)))

class VolatilityBreakout(BaseStrategy):
    def __init__(self): self.name = "Volatility Breakout (Donchian)"
    def apply_indicators(self, df):
        df.ta.donchian(lower_length=20, upper_length=20, append=True)
        df.ta.adx(length=14, append=True)
        df.ta.atr(length=14, append=True)
        return df
    def generate_signals(self, df):
        df['Signal'] = 0
        if 'DCU_20_20' not in df.columns or 'ADX_14' not in df.columns: return df
        bullish = (df['Close'] > df['DCU_20_20'].shift(1)) & (df['ADX_14'] > 25)
        bearish = (df['Close'] < df['DCL_20_20'].shift(1)) & (df['ADX_14'] > 25)
        df.loc[bullish, 'Signal'] = 1
        df.loc[bearish, 'Signal'] = -1
        return df
    def add_chart_overlays(self, fig, df):
        if 'DCU_20_20' in df:
            fig.add_trace(go.Scatter(x=df.index, y=df['DCU_20_20'], name='Donchian Upper', line=dict(color='#00ff88', dash='dash')))
            fig.add_trace(go.Scatter(x=df.index, y=df['DCL_20_20'], name='Donchian Lower', line=dict(color='#ff0055', dash='dash')))

STRATEGIES = {
    "Trend Confluence": TrendConfluence(),
    "Mean Reversion": MeanReversion(),
    "Volatility Breakout": VolatilityBreakout()
}

# ==========================================
# DATA & ALERTING FUNCTIONS
# ==========================================
@st.cache_data(ttl=600)
def fetch_data(ticker, region, timeframe):
    time.sleep(1) # Prevent rate limiting
    
    # CRITICAL FIX: yfinance only allows 60 days of data for 15m intervals
    if timeframe == "15m": fetch_period = "60d"
    elif timeframe == "1h": fetch_period = "730d"
    else: fetch_period = "2y"
        
    try:
        df = yf.download(ticker, period=fetch_period, interval=timeframe, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Data Fetching Error: {e}")
        return None

def send_telegram_alert(message):
    token = os.environ.get('TELEGRAM_TOKEN') or st.secrets.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get('TELEGRAM_CHAT_ID') or st.secrets.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        st.error("Telegram credentials missing in Secrets!")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        st.error(f"Connection error: {e}")

# ==========================================
# MAIN UI / FRONTEND
# ==========================================
def main():
    # Header Title
    st.markdown("<h1>⚡ CommodityPulse <span style='color:#00d2ff'>Pro</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#888; font-size:14px;'>Advanced Multi-Strategy Quant Terminal</p>", unsafe_allow_html=True)
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.image("https://img.icons8.com/nolan/96/line-chart.png", width=60)
        st.markdown("### ⚙️ Terminal Settings")
        
        region = st.selectbox("🌐 Market Region", ["Global", "Indian MCX"])
        asset_name = st.selectbox("📌 Select Asset", list(ASSETS[region].keys()))
        timeframe = st.selectbox("⏱️ Timeframe", list(TIMEFRAMES.keys()))
        
        st.divider()
        
        st.markdown("### 🧠 Quant Engine")
        strategy_name = st.selectbox("Select Algorithm", list(STRATEGIES.keys()))
        
        st.divider()
        
        st.markdown("### 📲 Notifications")
        enable_alerts = st.toggle("🔔 Enable Live Telegram Alerts", value=False)
        
        if st.button("🚀 Send Test Alert", use_container_width=True):
            test_msg = "✅ CommodityPulse Pro: System test successful! Alerts are active."
            send_telegram_alert(test_msg)
            st.toast("Test alert sent to your Telegram!", icon="🚀")
            
        if "last_alert_time" not in st.session_state:
            st.session_state.last_alert_time = None

    # Retrieve selected configurations
    ticker_info = ASSETS[region][asset_name]
    ticker = ticker_info['ticker']
    emoji = ticker_info['emoji']
    tf_params = TIMEFRAMES[timeframe]
    strategy = STRATEGIES[strategy_name]

    # Fetch Data
    with st.spinner(f"📡 Syncing market data for {asset_name}..."):
        df = fetch_data(ticker, region, tf_params['interval'])

    if df is None or len(df) < 50:
        st.error(f"⚠️ No data returned for {asset_name} ({timeframe}). Market might be closed.")
        st.stop()

    df = strategy.apply_indicators(df)
    df = strategy.generate_signals(df)
    
    curr = df.iloc[-1]
    prev_close = df.iloc[-2]['Close'] if len(df) > 1 else curr['Close']
    chg_pct = ((curr['Close'] - prev_close) / prev_close) * 100
    current_atr = curr.get(f'ATRr_14', 0.0)
    
    active_trend = "BULLISH 🟢" if ('EMA_200' in df and curr['Close'] > curr['EMA_200']) else "BEARISH 🔴"
    if 'EMA_200' not in df: active_trend = "NEUTRAL ⚪"

    # --- TOP ROW: PREMIUM METRICS ---
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Current Price", f"${curr['Close']:.2f}", f"{chg_pct:.2f}%")
    col2.metric("📈 24h Trend Bias", active_trend)
    col3.metric("📏 Volatility (ATR)", f"{current_atr:.2f}")
    
    latest_signal = curr['Signal']
    latest_time = df.index[-1]
    signal_text = "BUY 🟢" if latest_signal == 1 else "SELL 🔴" if latest_signal == -1 else "NONE ⚪"
    col4.metric("🤖 Live Algo Signal", signal_text)

    # Fire Telegram Alert
    if enable_alerts and latest_signal != 0 and st.session_state.last_alert_time != latest_time:
        sig_str = "BULLISH BUY" if latest_signal == 1 else "BEARISH SELL"
        sl_calc = curr['Close'] - (1.5 * current_atr) if latest_signal == 1 else curr['Close'] + (1.5 * current_atr)
        msg = f"{emoji} **{asset_name} ({timeframe})**: {sig_str}\n🤖 Algo: {strategy.name}\n💰 Price: ${curr['Close']:.2f}\n📐 Suggested SL: ${sl_calc:.2f}"
        send_telegram_alert(msg)
        st.session_state.last_alert_time = latest_time
        st.toast(f"Telegram Alert Sent for {asset_name}!", icon="🚀")

    # --- MIDDLE ROW: INTERACTIVE PLOTLY CHART ---
    st.markdown("<hr style='border:1px solid #2b2b36'>", unsafe_allow_html=True)
    st.markdown(f"### 📊 Advanced Charting: {strategy.name}")
    
    fig = go.Figure()
    
    # 1. Candlestick Trace (Cleaner colors)
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Price',
        increasing_line_color='#00ff88', decreasing_line_color='#ff0055'
    ))
    
    # 2. Add Strategy Overlays
    strategy.add_chart_overlays(fig, df)

    # 3. Add Signal Markers (Larger & clearer)
    bulls, bears = df[df['Signal'] == 1], df[df['Signal'] == -1]
    fig.add_trace(go.Scatter(x=bulls.index, y=bulls['Low'] - current_atr, mode='markers', marker=dict(symbol='triangle-up', color='#00ff88', size=14, line=dict(width=1, color='black')), name='Buy Signal'))
    fig.add_trace(go.Scatter(x=bears.index, y=bears['High'] + current_atr, mode='markers', marker=dict(symbol='triangle-down', color='#ff0055', size=14, line=dict(width=1, color='black')), name='Sell Signal'))

    # Premium Chart Formatting
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        xaxis_rangeslider_visible=False,
        height=650,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(showgrid=True, gridcolor='#2b2b36', gridwidth=1),
        yaxis=dict(showgrid=True, gridcolor='#2b2b36', gridwidth=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)")
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- BOTTOM ROW: LIVE SIGNAL LOG ---
    st.markdown("<hr style='border:1px solid #2b2b36'>", unsafe_allow_html=True)
    st.markdown("### 📝 Historical Ledger")
    
    signal_history = df[df['Signal'] != 0].copy()
    if not signal_history.empty:
        # Style the dataframe for a modern look
        signal_history['Action'] = signal_history['Signal'].map({1: '🟢 BUY', -1: '🔴 SELL'})
        display_cols = ['Action', 'Close']
        if 'RSI_14' in signal_history.columns: display_cols.append('RSI_14')
        if 'ATRr_14' in signal_history.columns: display_cols.append('ATRr_14')
        
        log_df = signal_history[display_cols].iloc[::-1].head(10)
        log_df.index.name = 'Timestamp'
        
        # Streamlit Native Data Editor for better aesthetics
        st.dataframe(
            log_df, 
            use_container_width=True,
            column_config={
                "Close": st.column_config.NumberColumn("Entry Price", format="$%.2f"),
                "RSI_14": st.column_config.NumberColumn("RSI Momentum", format="%.1f"),
                "ATRr_14": st.column_config.NumberColumn("Volatility (ATR)", format="%.2f")
            }
        )
    else:
        st.info("No signals generated in the current timeframe yet. Market is consolidating.")

if __name__ == "__main__":
    main()
