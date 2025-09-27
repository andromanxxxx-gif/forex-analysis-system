import pandas as pd
import numpy as np

def calculate_indicators(df):
    """Hitung indikator teknikal: EMA200, MACD, OBV"""
    # EMA 200
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # OBV
    df['Direction'] = np.where(df['Close'] > df['Open'], 1, np.where(df['Close'] < df['Open'], -1, 0))
    df['OBV'] = (df['Volume'] * df['Direction']).cumsum()
    
    return df

def generate_signal(df):
    """Generate signal Buy/Sell, TP, SL"""
    signals = {}
    last = df.iloc[-1]
    
    # Sinyal Buy/Sell dari MACD & EMA200
    if last['Close'] > last['EMA200'] and last['MACD'] > last['MACD_signal']:
        signal = "BUY"
    elif last['Close'] < last['EMA200'] and last['MACD'] < last['MACD_signal']:
        signal = "SELL"
    else:
        signal = "HOLD"
    
    # TP & SL sederhana: 1% target, 0.5% stop
    if signal == "BUY":
        tp = last['Close'] * 1.01
        sl = last['Close'] * 0.995
    elif signal == "SELL":
        tp = last['Close'] * 0.99
        sl = last['Close'] * 1.005
    else:
        tp = sl = None
    
    signals = {
        "signal": signal,
        "close": last['Close'],
        "TP": tp,
        "SL": sl,
        "EMA200": last['EMA200'],
        "MACD": last['MACD'],
        "MACD_signal": last['MACD_signal'],
        "OBV": last['OBV']
    }
    
    return signals
