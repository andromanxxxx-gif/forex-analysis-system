# src/trading_signals.py
import pandas as pd
import numpy as np

def calculate_indicators(df):
    # EMA 200
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # OBV
    df['OBV'] = (np.sign(df['Close'] - df['Open']) * df['Volume']).cumsum()
    
    return df

def generate_signal(df):
    latest = df.iloc[-1]
    signal = "HOLD"
    if latest['Close'] > latest['EMA200'] and latest['MACD'] > latest['MACD_signal']:
        signal = "BUY"
    elif latest['Close'] < latest['EMA200'] and latest['MACD'] < latest['MACD_signal']:
        signal = "SELL"
    
    # Tentukan TP/SL sederhana
    if signal == "BUY":
        tp = latest['Close'] * 1.01
        sl = latest['Close'] * 0.995
    elif signal == "SELL":
        tp = latest['Close'] * 0.99
        sl = latest['Close'] * 1.005
    else:
        tp = sl = latest['Close']
    
    return {
        "signal": signal,
        "take_profit": round(tp, 5),
        "stop_loss": round(sl, 5)
    }
