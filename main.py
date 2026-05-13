from keep_alive import keep_alive  # <--- Pulls in the web server from your other file
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
import time
import logging
from datetime import datetime

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
TELEGRAM_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"
POLL_INTERVAL_SECONDS = 60  # Check every 1 minute

ASSETS = {
    'CL=F': {'name': 'Crude Oil', 'emoji': '🛢️'},
    'NG=F': {'name': 'Natural Gas', 'emoji': '🔥'},
    'GC=F': {'name': 'Gold', 'emoji': '🟡'},
    'SI=F': {'name': 'Silver', 'emoji': '⚪'}
}

ALERT_STATE = {ticker: {'time': None, 'signal': None} for ticker in ASSETS.keys()}

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

# ==========================================
# DATA & QUANT ENGINE
# ==========================================
def fetch_data(ticker):
    try:
        df = yf.download(ticker, period="7d", interval="15m", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        logging.error(f"Network error fetching {ticker}: {e}")
        return None

def apply_indicators(df):
    df.ta.ema(length=9, append=True)
    df.ta.ema(length=21, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.atr(length=14, append=True)
    return df

def check_signals(df):
    if len(df) < 2: return None
    curr, prev = df.iloc[-1], df.iloc[-2]

    try:
        c_close = float(curr['Close'])
        c_ema9, c_ema21, c_ema200 = float(curr['EMA_9']), float(curr['EMA_21']), float(curr['EMA_200'])
        p_ema9, p_ema21 = float(prev['EMA_9']), float(prev['EMA_21'])
        c_rsi = float(curr['RSI_14'])
        c_atr = float(curr['ATRr_14']) if 'ATRr_14' in df.columns else 0.0
    except:
        return None
        
    if pd.isna(c_ema200): return None

    bull_cross = (c_ema9 > c_ema21) and (p_ema9 <= p_ema21)
    bear_cross = (c_ema9 < c_ema21) and (p_ema9 >= p_ema21)

    if bull_cross and (c_close > c_ema200) and (c_rsi > 55):
        return {"signal": "BULLISH", "price": c_close, "rsi": c_rsi, "atr": c_atr, "time": df.index[-1]}
    elif bear_cross and (c_close < c_ema200) and (c_rsi < 45):
        return {"signal": "BEARISH", "price": c_close, "rsi": c_rsi, "atr": c_atr, "time": df.index[-1]}
    return None

# ==========================================
# TELEGRAM ALERTING
# ==========================================
def send_telegram_alert(ticker, asset_info, sig_data):
    signal, price, rsi, atr = sig_data['signal'], sig_data['price'], sig_data['rsi'], sig_data['atr']
    emoji, name = asset_info['emoji'], asset_info['name']
    
    if signal == "BULLISH":
        direction, sl = "▲ BUY SIGNAL", price - (1.5 * atr)
    else:
        direction, sl = "▼ SELL SIGNAL", price + (1.5 * atr)

    message = f"{emoji} **{name} ({ticker})**: {direction}\n💰 Price: ${price:,.2f}\n🔥 RSI: {rsi:.1f} | Trend: {signal}\n📐 Suggested SL: ${sl:,.2f}"
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}

    try:
        requests.post(url, json=payload, timeout=10)
        logging.info(f"✅ Alert sent successfully for {ticker} ({signal})")
    except Exception as e:
        logging.error(f"❌ Failed to send Telegram alert: {e}")

# ==========================================
# MAIN ENGINE LOOP
# ==========================================
def main():
    logging.info("🚀 CommodityPulse Started...")
    while True:
        try:
            for ticker, info in ASSETS.items():
                df = fetch_data(ticker)
                if df is not None:
                    df = apply_indicators(df)
                    sig_data = check_signals(df)
                    
                    if sig_data:
                        last_alert = ALERT_STATE[ticker]
                        if (last_alert['time'] != sig_data['time']) or (last_alert['signal'] != sig_data['signal']):
                            send_telegram_alert(ticker, info, sig_data)
                            ALERT_STATE[ticker]['time'] = sig_data['time']
                            ALERT_STATE[ticker]['signal'] = sig_data['signal']

            time.sleep(POLL_INTERVAL_SECONDS)
        except Exception as e:
            logging.error(f"⚠️ Error: {e}. Restarting in 60s...")
            time.sleep(60)

if __name__ == "__main__":
    keep_alive()  # <--- This starts the hidden web server
    main()        # <--- This starts your trading bot
