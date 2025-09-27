import pandas as pd
import numpy as np

def calculate_indicators(df):
    """Hitung EMA200, MACD, OBV"""
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # OBV
    df['Direction'] = np.where(df['Close'] > df['Open'], 1, np.where(df['Close'] < df['Open'], -1, 0))
    df['OBV'] = (df['Volume'] * df['Direction']).cumsum()
    
    return df

def generate_signal(df):
    """Sinyal sederhana berdasarkan MACD crossover"""
    last = df.iloc[-1]
    signal = "Hold"
    if last['MACD'] > last['Signal'] and last['Close'] > last['EMA200']:
        signal = "Buy"
    elif last['MACD'] < last['Signal'] and last['Close'] < last['EMA200']:
        signal = "Sell"

    # Take Profit & Stop Loss sederhana
    tp = last['Close'] * (1.02 if signal=="Buy" else 0.98)
    sl = last['Close'] * (0.98 if signal=="Buy" else 1.02)
    
    return signal, tp, sl
