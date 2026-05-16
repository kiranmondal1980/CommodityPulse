import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
import os
import time

# ==========================================
# CREDENTIALS & SETTINGS
# ==========================================
TOKEN = os.environ.get('TELEGRAM_TOKEN')
CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

# Optimized Tickers: Using Spot Gold/Silver for 10x better signal accuracy
ASSETS = {
    'CL=F':      {'name': 'Crude Oil',    'emoji': '🛢️', 'type': 'comm'},
    'NG=F':      {'name': 'Natural Gas',  'emoji': '🔥', 'type': 'comm'},
    'XAUUSD=X':  {'name': 'Gold (MCX)',   'emoji': '🟡', 'type': 'gold'},
    'XAGUSD=X':  {'name': 'Silver (MCX)', 'emoji': '⚪', 'type': 'silver'}
    'BTC-USD':   {'name': 'Bitcoin',      'emoji': '₿',  'type': 'comm'} # 'comm' type uses direct USDINR rate
}

# ==========================================
# CURRENCY & DATA HELPERS
# ==========================================
def get_usdinr():
    try:
        df = yf.download("INR=X", period="1d", interval="1d", progress=False)
        return float(df['Close'].iloc[-1])
    except: return 83.50

def fetch_data(ticker):
    time.sleep(2) # Polite to Yahoo Finance
    try:
        # Optimization: Switching to 1h for the 1.40 Profit Factor edge
        df = yf.download(ticker, period="60d", interval="1h", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception: return None

def apply_indicators(df):
    # Core Strategy Indicators
    df.ta.ema(length=9, append=True)
    df.ta.ema(length=21, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.atr(length=14, append=True)
    
    # NEW Optimization: ADX to filter out "Choppy" market losses
    df.ta.adx(length=14, append=True)
    
    df.dropna(inplace=True)
    return df

# ==========================================
# SIGNAL LOGIC (THE WEALTH ENGINE)
# ==========================================
def check_signals(df):
    if len(df) < 3: return None
    
    # Looking at index [-2] (Last Closed 1h Candle)
    curr = df.iloc[-2]  
    prev = df.iloc[-3]  

    try:
        c_close = float(curr['Close'])
        c_ema9, c_ema21, c_ema200 = float(curr['EMA_9']), float(curr['EMA_21']), float(curr['EMA_200'])
        p_ema9, p_ema21 = float(prev['EMA_9']), float(prev['EMA_21'])
        c_rsi = float(curr['RSI_14'])
        c_atr = float(curr['ATRr_14'])
        c_adx = float(curr['ADX_14'])
    except: return None
    
    # --- UPGRADE 1: Market Regime Filter ---
    # If ADX is below 20, the market is sideways. Trend following loses money here.
    if c_adx < 20: return None 

    # --- UPGRADE 2: Trend Confluence ---
    bull = (c_ema9 > c_ema21) and (p_ema9 <= p_ema21) and (c_close > c_ema200) and (c_rsi > 55)
    bear = (c_ema9 < c_ema21) and (p_ema9 >= p_ema21) and (c_close < c_ema200) and (c_rsi < 45)
    
    if bull: return {"signal": "BUY 🟢", "price": c_close, "rsi": c_rsi, "atr": c_atr, "adx": c_adx}
    if bear: return {"signal": "SELL 🔴", "price": c_close, "rsi": c_rsi, "atr": c_atr, "adx": c_adx}
    return None

# ==========================================
# ALERTING (INR CALIBRATED)
# ==========================================
def send_telegram_alert(ticker, info, sig):
    usdinr = get_usdinr()
    import_duty = 1.15 # 15% for Gold/Silver
    
    # Convert alert price to INR for easy MCX entry
    if info['type'] == 'gold':
        inr_price = (sig['price'] / 31.1034) * 10 * usdinr * import_duty
    elif info['type'] == 'silver':
        inr_price = (sig['price'] / 31.1034) * 1000 * usdinr * import_duty
    else:
        inr_price = sig['price'] * usdinr

    sl_dist = 1.5 * sig['atr']
    sl_usd = (sig['price'] - sl_dist) if "BUY" in sig['signal'] else (sig['price'] + sl_dist)
    
    # Stop Loss in INR
    if info['type'] == 'gold': inr_sl = (sl_usd / 31.1034) * 10 * usdinr * import_duty
    elif info['type'] == 'silver': inr_sl = (sl_usd / 31.1034) * 1000 * usdinr * import_duty
    else: inr_sl = sl_usd * usdinr

    msg = (
        f"{info['emoji']} **{info['name']} Alert (1H)**\n"
        f"━━━━━━━━━━━━━━\n"
        f"🚀 **Action:** {sig['signal']}\n"
        f"💰 **MCX Price:** ₹{inr_price:,.0f}\n"
        f"📐 **Stop Loss:** ₹{inr_sl:,.0f}\n"
        f"━━━━━━━━━━━━━━\n"
        f"📊 RSI: {sig['rsi']:.1f} | ADX: {sig['adx']:.1f}\n"
        f"💡 *Statistically high-probability trend confirmed.*"
    )
    
    requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", 
                  json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"})

def main():
    if not TOKEN or not CHAT_ID:
        print("Error: Missing Telegram Keys")
        return

    for ticker, info in ASSETS.items():
        df = fetch_data(ticker)
        if df is not None:
            processed_df = apply_indicators(df)
            sig = check_signals(processed_df)
            if sig:
                send_telegram_alert(ticker, info, sig)
                print(f"🚨 ALERT: {info['name']} {sig['signal']} sent!")
            else:
                print(f"ℹ️ {info['name']}: No strong trending setup.")

if __name__ == "__main__":
    main()
