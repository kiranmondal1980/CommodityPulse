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
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="CommodityPulse Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CONSTANTS & ASSET MAPPINGS
# ==========================================
# Note: yfinance has limited support for Indian MCX. 
# We use standard proxies/standard symbols here. 
ASSETS = {
    "Global": {
        'Crude Oil (WTI)': {'ticker': 'CL=F', 'emoji': '🛢️'},
        'Natural Gas': {'ticker': 'NG=F', 'emoji': '🔥'},
        'Gold': {'ticker': 'GC=F', 'emoji': '🟡'},
        'Silver': {'ticker': 'SI=F', 'emoji': '⚪'}
    },
    "Indian MCX": {
        # Using the Yahoo Finance symbols that map to Indian Commodity Indices
        'Crude Oil (MCX)': {'ticker': 'BZ=F', 'emoji': '🛢️'}, 
        'Natural Gas (MCX)': {'ticker': 'NG=F', 'emoji': '🔥'},
        'Gold (MCX)': {'ticker': 'GC=F', 'emoji': '🟡'},
        'Silver (MCX)': {'ticker': 'SI=F', 'emoji': '⚪'}
    }
}

TIMEFRAMES = {
    "15m": {"period": "5d", "interval": "15m"},
    "1h": {"period": "1mo", "interval": "1h"},
    "4h": {"period": "60d", "interval": "90m"}, # yfinance doesn't natively support 4h well, using 90m as proxy or daily
    "1d": {"period": "1y", "interval": "1d"}
}

# ==========================================
# QUANT ENGINE (OOP MULTI-STRATEGY)
# ==========================================
class BaseStrategy(ABC):
    def __init__(self):
        self.name = "Base Strategy"
    
    @abstractmethod
    def apply_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def add_chart_overlays(self, fig: go.Figure, df: pd.DataFrame):
        pass


class TrendConfluence(BaseStrategy):
    def __init__(self):
        self.name = "Trend Confluence"
        
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
        
        # Bullish: 9 crosses above 21, Price > 200 EMA, RSI > 55
        bullish = (df['EMA_9'] > df['EMA_21']) & (df['EMA_9'].shift(1) <= df['EMA_21'].shift(1)) & \
                  (df['Close'] > df['EMA_200']) & (df['RSI_14'] > 55)
        
        # Bearish: 9 crosses below 21, Price < 200 EMA, RSI < 45
        bearish = (df['EMA_9'] < df['EMA_21']) & (df['EMA_9'].shift(1) >= df['EMA_21'].shift(1)) & \
                  (df['Close'] < df['EMA_200']) & (df['RSI_14'] < 45)
                  
        df.loc[bullish, 'Signal'] = 1
        df.loc[bearish, 'Signal'] = -1
        return df

    def add_chart_overlays(self, fig, df):
        if 'EMA_9' in df: fig.add_trace(go.Scatter(x=df.index, y=df['EMA_9'], name='EMA 9', line=dict(color='blue', width=1)))
        if 'EMA_21' in df: fig.add_trace(go.Scatter(x=df.index, y=df['EMA_21'], name='EMA 21', line=dict(color='orange', width=1)))
        if 'EMA_200' in df: fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], name='EMA 200', line=dict(color='white', width=2)))


class MeanReversion(BaseStrategy):
    def __init__(self):
        self.name = "Mean Reversion (BB Fade)"
        
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
            fig.add_trace(go.Scatter(x=df.index, y=df['BBU_20_2.0'], name='Upper BB', line=dict(color='gray', dash='dot')))
            fig.add_trace(go.Scatter(x=df.index, y=df['BBL_20_2.0'], name='Lower BB', line=dict(color='gray', dash='dot')))


class VolatilityBreakout(BaseStrategy):
    def __init__(self):
        self.name = "Volatility Breakout (Donchian)"
        
    def apply_indicators(self, df):
        df.ta.donchian(lower_length=20, upper_length=20, append=True)
        df.ta.adx(length=14, append=True)
        df.ta.atr(length=14, append=True)
        return df
        
    def generate_signals(self, df):
        df['Signal'] = 0
        if 'DCU_20_20' not in df.columns or 'ADX_14' not in df.columns: return df
        
        # Price breaks 20-period High and ADX > 25
        bullish = (df['Close'] > df['DCU_20_20'].shift(1)) & (df['ADX_14'] > 25)
        # Price breaks 20-period Low and ADX > 25
        bearish = (df['Close'] < df['DCL_20_20'].shift(1)) & (df['ADX_14'] > 25)
        
        df.loc[bullish, 'Signal'] = 1
        df.loc[bearish, 'Signal'] = -1
        return df

    def add_chart_overlays(self, fig, df):
        if 'DCU_20_20' in df:
            fig.add_trace(go.Scatter(x=df.index, y=df['DCU_20_20'], name='Donchian Upper', line=dict(color='green', dash='dash')))
            fig.add_trace(go.Scatter(x=df.index, y=df['DCL_20_20'], name='Donchian Lower', line=dict(color='red', dash='dash')))

# Available strategies dictionary mapping
STRATEGIES = {
    "Trend Confluence": TrendConfluence(),
    "Mean Reversion": MeanReversion(),
    "Volatility Breakout": VolatilityBreakout()
}

# ==========================================
# DATA & ALERTING FUNCTIONS
# ==========================================
from tvDatafeed import TvDatafeed, Interval

# Initialize the feed globally
tv = TvDatafeed()

@st.cache_data(ttl=600)
def fetch_data(ticker, region, timeframe):
    # Small pause to be nice to the servers
    time.sleep(2) 
    
    try:
        # We now use yfinance for everything, but with specific error handling
        # This avoids the "Connection Timed Out" issue of scraping TradingView
        df = yf.download(ticker, period="3mo", interval=timeframe, progress=False)
        
        if df.empty:
            return None
            
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)
            
        return df
    except Exception as e:
        st.error(f"Data Fetching Error: {e}")
        return None
            
    else:
        # Global Logic (yfinance)
        try:
            df = yf.download(ticker, period="6mo", interval=timeframe, progress=False)
            if isinstance(df.columns, pd.MultiIndex): 
                df.columns = df.columns.get_level_values(0)
            return df
        except Exception as e:
            st.error(f"yfinance Error: {e}")
            return None
    return None

def send_telegram_alert(message):
    token = os.environ.get('TELEGRAM_TOKEN') or st.secrets.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get('TELEGRAM_CHAT_ID') or st.secrets.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        st.warning("⚠️ Telegram credentials not found in environment or secrets.")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        st.error(f"Failed to send Telegram alert: {e}")

# ==========================================
# MAIN UI / FRONTEND
# ==========================================
def main():
    st.title("⚡ CommodityPulse Pro Quant Terminal")
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Terminal Settings")
        region = st.selectbox("Market Region", ["Global", "Indian MCX"])
        asset_name = st.selectbox("Select Asset", list(ASSETS[region].keys()))
        timeframe = st.selectbox("Timeframe", list(TIMEFRAMES.keys()))
        strategy_name = st.selectbox("Algorithm", list(STRATEGIES.keys()))
        
        st.markdown("---")
        enable_alerts = st.toggle("🔔 Enable Live Telegram Alerts", value=False)
        
        # State Management: Prevent duplicate alerts on Streamlit reruns
        if "last_alert_time" not in st.session_state:
            st.session_state.last_alert_time = None

    # Retrieve selected configurations
    ticker_info = ASSETS[region][asset_name]
    ticker = ticker_info['ticker']
    emoji = ticker_info['emoji']
    tf_params = TIMEFRAMES[timeframe]
    strategy = STRATEGIES[strategy_name]

    # Fetch Data - Now passing the region correctly
    with st.spinner(f"Fetching {asset_name} data..."):
        df = fetch_data(ticker, region, timeframe)

    # Validate data before proceeding
    if df is None or len(df) < 50:
        st.error(f"⚠️ No data returned for {asset_name}. If using MCX, check ticker in ASSETS dictionary.")
        st.stop()

    # Apply Quant Strategy
    df = strategy.apply_indicators(df)
    df = strategy.generate_signals(df)
    
    # Extract recent metrics
    curr = df.iloc[-1]
    prev_close = df.iloc[-2]['Close'] if len(df) > 1 else curr['Close']
    chg_pct = ((curr['Close'] - prev_close) / prev_close) * 100
    current_atr = curr.get(f'ATRr_14', 0.0)
    
    # Determine Active Trend based on simple EMA200 if available
    active_trend = "BULLISH 🟢" if ('EMA_200' in df and curr['Close'] > curr['EMA_200']) else "BEARISH 🔴"
    if 'EMA_200' not in df: active_trend = "NEUTRAL ⚪"

    # --- TOP ROW: METRICS ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"${curr['Close']:.2f}", f"{chg_pct:.2f}%")
    col2.metric("24h Trend Bias", active_trend)
    col3.metric("Current Volatility (ATR)", f"{current_atr:.2f}")
    
    # Alerting Logic Check
    latest_signal = curr['Signal']
    latest_time = df.index[-1]
    signal_text = "NONE"
    if latest_signal == 1: signal_text = "BUY 🟢"
    elif latest_signal == -1: signal_text = "SELL 🔴"
    col4.metric("Live Algo Signal", signal_text)

    # Fire Telegram Alert
    if enable_alerts and latest_signal != 0 and st.session_state.last_alert_time != latest_time:
        sig_str = "BULLISH BUY" if latest_signal == 1 else "BEARISH SELL"
        sl_calc = curr['Close'] - (1.5 * current_atr) if latest_signal == 1 else curr['Close'] + (1.5 * current_atr)
        msg = f"{emoji} **{asset_name} ({timeframe})**: {sig_str}\n" \
              f"🤖 Algo: {strategy.name}\n" \
              f"💰 Price: ${curr['Close']:.2f}\n" \
              f"📐 Suggested SL: ${sl_calc:.2f}"
        send_telegram_alert(msg)
        st.session_state.last_alert_time = latest_time
        st.toast(f"Telegram Alert Sent for {asset_name}!", icon="🚀")

    # --- MIDDLE ROW: INTERACTIVE PLOTLY CHART ---
    st.subheader(f"📊 {asset_name} - {strategy.name} Analysis")
    
    fig = go.Figure()
    # 1. Candlestick Trace
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Price'
    ))
    
    # 2. Add Strategy Overlays
    strategy.add_chart_overlays(fig, df)

    # 3. Add Signal Markers
    bulls = df[df['Signal'] == 1]
    bears = df[df['Signal'] == -1]
    
    fig.add_trace(go.Scatter(
        x=bulls.index, y=bulls['Low'] - current_atr, 
        mode='markers', marker=dict(symbol='triangle-up', color='lime', size=12),
        name='Buy Signal'
    ))
    fig.add_trace(go.Scatter(
        x=bears.index, y=bears['High'] + current_atr, 
        mode='markers', marker=dict(symbol='triangle-down', color='red', size=12),
        name='Sell Signal'
    ))

    # Formatting Chart
    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- BOTTOM ROW: LIVE SIGNAL LOG ---
    st.subheader("📝 Historical Signal Log")
    signal_history = df[df['Signal'] != 0].copy()
    if not signal_history.empty:
        # Format the dataframe for display
        signal_history['Action'] = signal_history['Signal'].map({1: 'BUY', -1: 'SELL'})
        display_cols = ['Action', 'Close']
        
        # Add RSI if it exists in the strategy
        if 'RSI_14' in signal_history.columns: display_cols.append('RSI_14')
        if 'ATRr_14' in signal_history.columns: display_cols.append('ATRr_14')
        
        # Reverse to show newest first
        log_df = signal_history[display_cols].iloc[::-1].head(10)
        log_df.index.name = 'Timestamp'
        st.dataframe(log_df, use_container_width=True)
    else:
        st.info("No signals generated in the current timeframe.")

if __name__ == "__main__":
    main()
