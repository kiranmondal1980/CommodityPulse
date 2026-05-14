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

# Custom CSS for a "Premium Light Mode" Look
st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease-in-out;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        border: 1px solid #c0c0c0;
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
        # Using XAU/INR and XAG/INR proxies is much more accurate for Indian Spot prices
        'Gold (MCX)': {'ticker': 'XAUUSD=X', 'emoji': '🟡'},
        'Silver (MCX)': {'ticker': 'XAGUSD=X', 'emoji': '⚪'},
        'Crude Oil (MCX)': {'ticker': 'BZ=F', 'emoji': '🛢️'}, 
        'Natural Gas (MCX)': {'ticker': 'NG=F', 'emoji': '🔥'}
    },
    "Global Markets": {
        'Crude Oil (WTI)': {'ticker': 'CL=F', 'emoji': '🛢️'},
        'Gold (COMEX)': {'ticker': 'GC=F', 'emoji': '🟡'},
        'Silver (COMEX)': {'ticker': 'SI=F', 'emoji': '⚪'}
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
        if len(df) < 2: return df
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
# DATA, CURRENCY, AND TIMEZONE FUNCTIONS
# ==========================================
@st.cache_data(ttl=3600) # Cache exchange rate for 1 hour
def get_usdinr_rate():
    try:
        usdinr = yf.download("INR=X", period="1d", interval="1d", progress=False)
        return float(usdinr['Close'].iloc[-1])
    except:
        return 83.50 # Fallback live rate if fetch fails


@st.cache_data(ttl=600)
def fetch_data(ticker, region, timeframe):
    time.sleep(1) 
    
    # Logic for 15m data limits
    fetch_period = "60d" if timeframe == "15m" else "2y"
        
    try:
        df = yf.download(ticker, period=fetch_period, interval=timeframe, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)
        df.dropna(inplace=True)

        # Convert Timezone to IST
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
        else:
            df.index = df.index.tz_convert('Asia/Kolkata')

        # --- PROFESSIONAL MCX CALIBRATION ---
        usdinr_rate = get_usdinr_rate()
        import_duty = 1.15 # 15% Indian Import Duty & Taxes
        
        if ticker == 'XAUUSD=X' or ticker == 'GC=F': 
            # 1 Troy Oz = 31.1035 grams. Indian MCX is 10 grams.
            multiplier = (usdinr_rate / 31.1034768) * 10 * import_duty
        elif ticker == 'XAGUSD=X' or ticker == 'SI=F':
            # 1 Troy Oz = 31.1035 grams. Indian MCX is 1 Kg (1000g).
            multiplier = (usdinr_rate / 31.1034768) * 1000 * import_duty
        else:
            # Crude and Gas are 1:1 with USD rate conversion
            multiplier = usdinr_rate

        # Apply the calibrated multiplier
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = df[col] * multiplier

        return df
    except Exception as e:
        st.error(f"Data Fetching Error: {e}")
        return None

def send_telegram_alert(message):
    token = os.environ.get('TELEGRAM_TOKEN') or st.secrets.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get('TELEGRAM_CHAT_ID') or st.secrets.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id: return
    try: requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"})
    except: pass

# ==========================================
# MAIN UI / FRONTEND
# ==========================================
def main():
    st.markdown("<h1>⚡ CommodityPulse <span style='color:#2196F3'>Pro</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#666666; font-size:14px;'>Advanced Multi-Strategy Quant Terminal (IST / INR)</p>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.image("https://img.icons8.com/nolan/96/line-chart.png", width=60)
        st.markdown("### ⚙️ Terminal Settings")
        
        region = st.selectbox("🌐 Market Region", list(ASSETS.keys()))
        asset_name = st.selectbox("📌 Select Asset", list(ASSETS[region].keys()))
        timeframe = st.selectbox("⏱️ Timeframe", list(TIMEFRAMES.keys()))
        
        st.divider()
        strategy_name = st.selectbox("🧠 Select Algorithm", list(STRATEGIES.keys()))
        st.divider()
        
        enable_alerts = st.toggle("🔔 Enable Live Telegram Alerts", value=False)
        if st.button("🚀 Send Test Alert", use_container_width=True):
            send_telegram_alert("✅ CommodityPulse Pro: System test successful! Alerts are active in IST.")
            st.toast("Test alert sent to your Telegram!", icon="🚀")
            
        if "last_alert_time" not in st.session_state: st.session_state.last_alert_time = None

    ticker_info = ASSETS[region][asset_name]
    ticker = ticker_info['ticker']; emoji = ticker_info['emoji']; tf_params = TIMEFRAMES[timeframe]
    strategy = STRATEGIES[strategy_name]

    with st.spinner(f"📡 Syncing live IST data for {asset_name}..."):
        df = fetch_data(ticker, region, tf_params['interval'])

    if df is None or len(df) < 50:
        st.error(f"⚠️ No data returned. Market might be closed.")
        st.stop()

    df = strategy.apply_indicators(df)
    df = strategy.generate_signals(df)
    
    curr = df.iloc[-1]
    prev_close = df.iloc[-2]['Close'] if len(df) > 1 else curr['Close']
    chg_pct = ((curr['Close'] - prev_close) / prev_close) * 100
    current_atr = curr.get(f'ATRr_14', 0.0)
    
    active_trend = "BULLISH 🟢" if ('EMA_200' in df and curr['Close'] > curr['EMA_200']) else "BEARISH 🔴"
    if 'EMA_200' not in df: active_trend = "NEUTRAL ⚪"

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    # Changed $ to ₹
    col1.metric("💰 Current Price (INR)", f"₹{curr['Close']:,.2f}", f"{chg_pct:.2f}%")
    col2.metric("📈 24h Trend Bias", active_trend)
    col3.metric("📏 Volatility (ATR in ₹)", f"₹{current_atr:.2f}")
    
    latest_signal = curr['Signal']
    latest_time = df.index[-1]
    signal_text = "BUY 🟢" if latest_signal == 1 else "SELL 🔴" if latest_signal == -1 else "NONE ⚪"
    col4.metric("🤖 Live Algo Signal", signal_text)

    if enable_alerts and latest_signal != 0 and st.session_state.last_alert_time != latest_time:
        sig_str = "BULLISH BUY" if latest_signal == 1 else "BEARISH SELL"
        sl_calc = curr['Close'] - (1.5 * current_atr) if latest_signal == 1 else curr['Close'] + (1.5 * current_atr)
        msg = f"{emoji} **{asset_name} ({timeframe})**: {sig_str}\n🤖 Algo: {strategy.name}\n💰 Price: ₹{curr['Close']:,.2f}\n📐 Suggested SL: ₹{sl_calc:,.2f}"
        send_telegram_alert(msg)
        st.session_state.last_alert_time = latest_time
        st.toast(f"Telegram Alert Sent!", icon="🚀")

    st.markdown("<hr style='border:1px solid #e6e6e6'>", unsafe_allow_html=True)
    st.markdown(f"### 📊 Advanced Charting (IST Timeline)")
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Price (INR)', increasing_line_color='#089981', decreasing_line_color='#F23645'
    ))
    strategy.add_chart_overlays(fig, df)

    bulls, bears = df[df['Signal'] == 1], df[df['Signal'] == -1]
    fig.add_trace(go.Scatter(x=bulls.index, y=bulls['Low'] - current_atr, mode='markers', marker=dict(symbol='triangle-up', color='#089981', size=14, line=dict(width=1, color='black')), name='Buy Signal'))
    fig.add_trace(go.Scatter(x=bears.index, y=bears['High'] + current_atr, mode='markers', marker=dict(symbol='triangle-down', color='#F23645', size=14, line=dict(width=1, color='black')), name='Sell Signal'))

    fig.update_layout(
        template="plotly_white", plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        xaxis_rangeslider_visible=False, height=650, margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(showgrid=True, gridcolor='#f0f2f6', gridwidth=1, title="Indian Standard Time (IST)"), 
        yaxis=dict(showgrid=True, gridcolor='#f0f2f6', gridwidth=1, tickprefix="₹"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(255,255,255,0.8)")
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr style='border:1px solid #e6e6e6'>", unsafe_allow_html=True)
    st.markdown("### 📝 Historical Ledger")
    
    signal_history = df[df['Signal'] != 0].copy()
    if not signal_history.empty:
        signal_history['Action'] = signal_history['Signal'].map({1: '🟢 BUY', -1: '🔴 SELL'})
        display_cols = ['Action', 'Close']
        if 'RSI_14' in signal_history.columns: display_cols.append('RSI_14')
        if 'ATRr_14' in signal_history.columns: display_cols.append('ATRr_14')
        
        log_df = signal_history[display_cols].iloc[::-1].head(10)
        
        # Ensure the index displays as clean IST timezone
        log_df.index = log_df.index.strftime('%Y-%m-%d %H:%M:%S IST')
        log_df.index.name = 'Timestamp (IST)'
        
        st.dataframe(
            log_df, use_container_width=True,
            column_config={
                "Close": st.column_config.NumberColumn("Entry Price", format="₹%d"),
                "RSI_14": st.column_config.NumberColumn("RSI Momentum", format="%.1f"),
                "ATRr_14": st.column_config.NumberColumn("Volatility (ATR)", format="₹%.1f")
            }
        )
    else:
        st.info("No signals generated in the current timeframe yet. Market is consolidating.")

if __name__ == "__main__": main()
