import pandas as pd
import numpy as np

def calculate_indicators(df):
    """Tambahkan EMA200, MACD, OBV"""
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # OBV
    df['OBV'] = ((df['Volume'] * np.sign(df['Close'] - df['Open']))).cumsum()
    
    return df

def generate_signal(df):
    """Generate sinyal sederhana berdasarkan indikator"""
    last = df.iloc[-1]
    signal = "Hold"
    if last['MACD'] > last['Signal'] and last['Close'] > last['EMA200']:
        signal = "Buy"
    elif last['MACD'] < last['Signal'] and last['Close'] < last['EMA200']:
        signal = "Sell"

    # Contoh TP/SL sederhana
    last_price = last['Close']
    tp = last_price * 1.01 if signal == "Buy" else last_price * 0.99
    sl = last_price * 0.99 if signal == "Buy" else last_price * 1.01
    
    return {"signal": signal, "TP": round(tp,5), "SL": round(sl,5)}
