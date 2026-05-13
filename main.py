import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
import os

# Get credentials from GitHub Secrets
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

ASSETS = {
    'CL=F': {'name': 'Crude Oil', 'emoji': '🛢️'},
    'NG=F': {'name': 'Natural Gas', 'emoji': '🔥'},
    'GC=F': {'name': 'Gold', 'emoji': '🟡'},
    'SI=F': {'name': 'Silver', 'emoji': '⚪'}
}

def fetch_data(ticker):
    try:
        df = yf.download(ticker, period="5d", interval="15m", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df
    except Exception: return None

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
    except: return None
    
    if pd.isna(c_ema200): return None
    bull = (c_ema9 > c_ema21) and (p_ema9 <= p_ema21) and (c_close > c_ema200) and (c_rsi > 55)
    bear = (c_ema9 < c_ema21) and (p_ema9 >= p_ema21) and (c_close < c_ema200) and (c_rsi < 45)
    
    if bull: return {"signal": "BULLISH", "price": c_close, "rsi": c_rsi, "atr": c_atr}
    if bear: return {"signal": "BEARISH", "price": c_close, "rsi": c_rsi, "atr": c_atr}
    return None

def send_telegram_alert(ticker, info, sig):
    sl = (sig['price'] - 1.5 * sig['atr']) if sig['signal'] == "BULLISH" else (sig['price'] + 1.5 * sig['atr'])
    msg = f"{info['emoji']} **{info['name']} ({ticker})**: {sig['signal']}\n💰 Price: ${sig['price']:.2f}\n🔥 RSI: {sig['rsi']:.1f}\n📐 SL: ${sl:.2f}"
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                  json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"})

def main():
    for ticker, info in ASSETS.items():
        df = fetch_data(ticker)
        if df is not None:
            sig = check_signals(apply_indicators(df))
            if sig:
                send_telegram_alert(ticker, info, sig)

if __name__ == "__main__":
    main()
