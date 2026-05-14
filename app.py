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
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
    }
    h1, h2, h3 {font-family: 'Helvetica Neue', sans-serif; font-weight: 600; color: #31333F;}
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# CONSTANTS & ASSET MAPPINGS
# ==========================================
ASSETS = {
    "Indian MCX (Estimated)": {
        'Gold (MCX)': {'ticker': 'XAUUSD=X', 'emoji': '🟡'},
        'Silver (MCX)': {'ticker': 'XAGUSD=X', 'emoji': '⚪'},
        'Crude Oil (MCX)': {'ticker': 'BZ=F', 'emoji': '🛢️'}, 
        'Natural Gas (MCX)': {'ticker': 'NG=F', 'emoji': '🔥'}
    },
    "Global Proxies (in INR)": {
        'Crude Oil (WTI)': {'ticker': 'CL=F', 'emoji': '🛢️'},
        'Gold (Global)': {'ticker': 'GC=F', 'emoji': '🟡'},
        'Silver (Global)': {'ticker': 'SI=F', 'emoji': '⚪'}
    }
}

TIMEFRAMES = {
    "15m": {"interval": "15m"}, "1h": {"interval": "1h"},
    "4h": {"interval": "90m"}, "1d": {"interval": "1d"}
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
        if len(df) < 5: return df
        bullish = (df['EMA_9'] > df['EMA_21']) & (df['EMA_9'].shift(1) <= df['EMA_21'].shift(1)) & (df['Close'] > df['EMA_200']) & (df['RSI_14'] > 55)
        bearish = (df['EMA_9'] < df['EMA_21']) & (df['EMA_9'].shift(1) >= df['EMA_21'].shift(1)) & (df['Close'] < df['EMA_200']) & (df['RSI_14'] < 45)
        df.loc[bullish, 'Signal'] = 1; df.loc[bearish, 'Signal'] = -1
        return df
    def add_chart_overlays(self, fig, df):
        if 'EMA_9' in df: fig.add_trace(go.Scatter(x=df.index, y=df['EMA_9'], name='EMA 9', line=dict(color='#2196F3', width=1.5)))
        if 'EMA_21' in df: fig.add_trace(go.Scatter(x=df.index, y=df['EMA_21'], name='EMA 21', line=dict(color='#FF9800', width=1.5)))
        if 'EMA_200' in df: fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], name='EMA 200', line=dict(color='#2C3E50', width=2.5)))

STRATEGIES = {"Trend Confluence": TrendConfluence()}

# ==========================================
# DATA & CALIBRATION FUNCTIONS
# ==========================================
@st.cache_data(ttl=3600)
def get_usdinr_rate():
    try:
        data = yf.download("INR=X", period="5d", interval="1d", progress=False)
        return float(data['Close'].iloc[-1])
    except: return 83.80

@st.cache_data(ttl=300) # Reduced cache time for fresher data
def fetch_data(ticker, timeframe):
    # Logic: 15m data is only reliably available for 30 days on Yahoo free tier
    period = "30d" if timeframe == "15m" else "1y"
    
    try:
        df = yf.download(ticker, period=period, interval=timeframe, auto_adjust=True, progress=False)
        
        if df is None or df.empty: return None
        
        # Repairing Yahoo Finance's New Multi-Index Format
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.copy()
        
        # Timezone conversion to IST
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
        else:
            df.index = df.index.tz_convert('Asia/Kolkata')

        # MCX Price Calibration
        usdinr = get_usdinr_rate()
        duty = 1.15 # 15% Import Duty
        
        if ticker in ['XAUUSD=X', 'GC=F']: 
            multiplier = (usdinr / 31.1034) * 10 * duty
        elif ticker in ['XAGUSD=X', 'SI=F']:
            multiplier = (usdinr / 31.1034) * 1000 * duty
        else:
            multiplier = usdinr

        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = df[col] * multiplier

        return df
    except Exception as e:
        return None

def send_telegram_alert(message):
    token = os.environ.get('TELEGRAM_TOKEN') or st.secrets.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get('TELEGRAM_CHAT_ID') or st.secrets.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id: return
    try: requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"})
    except: pass

# ==========================================
# MAIN TERMINAL
# ==========================================
def main():
    st.markdown("<h1>⚡ CommodityPulse <span style='color:#2196F3'>Pro</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#666; font-size:14px;'>Indian Quant Terminal (IST/INR)</p>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("⚙️ Settings")
        region = st.selectbox("Market", list(ASSETS.keys()))
        asset = st.selectbox("Asset", list(ASSETS[region].keys()))
        tf = st.selectbox("Timeframe", list(TIMEFRAMES.keys()))
        strategy_name = st.selectbox("Strategy", list(STRATEGIES.keys()))
        
        st.divider()
        enable_alerts = st.toggle("Enable Telegram Alerts", value=False)
        if st.button("🚀 Send Test Alert"):
            send_telegram_alert("✅ System Check: Connection established.")
            st.toast("Test Sent!")
        if "last_alert_time" not in st.session_state: st.session_state.last_alert_time = None

    ticker = ASSETS[region][asset]['ticker']
    emoji = ASSETS[region][asset]['emoji']
    strategy = STRATEGIES[strategy_name]

    with st.spinner("Syncing..."):
        df = fetch_data(ticker, tf)

    if df is None or len(df) < 20:
        st.warning(f"⚠️ Market is currently offline or Rate-Limited. Try Global Proxies or a higher timeframe (1h).")
        st.stop()

    df = strategy.generate_signals(strategy.apply_indicators(df))
    curr = df.iloc[-1]
    prev_close = df.iloc[-2]['Close']
    chg = ((curr['Close'] - prev_close) / prev_close) * 100
    atr = curr.get('ATRr_14', 0)

    # Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price (INR)", f"₹{curr['Close']:,.0f}", f"{chg:.2f}%")
    c2.metric("Trend", "BULLISH 🟢" if curr['Close'] > curr['EMA_200'] else "BEARISH 🔴")
    c3.metric("ATR (Volatility)", f"₹{atr:.1f}")
    
    sig = curr['Signal']
    c4.metric("Live Signal", "BUY 🟢" if sig == 1 else "SELL 🔴" if sig == -1 else "NONE")

    # Alert Trigger
    if enable_alerts and sig != 0 and st.session_state.last_alert_time != df.index[-1]:
        txt = "BULLISH BUY" if sig == 1 else "BEARISH SELL"
        msg = f"{emoji} **{asset}**: {txt}\n💰 Price: ₹{curr['Close']:,.0f}"
        send_telegram_alert(msg)
        st.session_state.last_alert_time = df.index[-1]

    # Chart
    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price')])
    strategy.add_chart_overlays(fig, df)
    
    bulls, bears = df[df['Signal'] == 1], df[df['Signal'] == -1]
    fig.add_trace(go.Scatter(x=bulls.index, y=bulls['Low'] - atr, mode='markers', marker=dict(symbol='triangle-up', color='#089981', size=12), name='Buy'))
    fig.add_trace(go.Scatter(x=bears.index, y=bears['High'] + atr, mode='markers', marker=dict(symbol='triangle-down', color='#F23645', size=12), name='Sell'))

    fig.update_layout(template="plotly_white", xaxis_rangeslider_visible=False, height=600, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Ledger
    st.subheader("📝 Signal History")
    history = df[df['Signal'] != 0].copy()
    if not history.empty:
        history['Action'] = history['Signal'].map({1: '🟢 BUY', -1: '🔴 SELL'})
        log = history[['Action', 'Close']].iloc[::-1].head(10)
        log.index = log.index.strftime('%H:%M %d-%b')
        st.dataframe(log, use_container_width=True)

if __name__ == "__main__": main()
