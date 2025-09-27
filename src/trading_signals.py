# src/trading_signals.py
import pandas as pd
import numpy as np

def calculate_indicators(df):
    # EMA 200
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # OBV (On Balance Volume)
    df['Direction'] = np.sign(df['Close'] - df['Open'])
    df['OBV'] = (df['Volume'] * df['Direction']).cumsum()
    
    return df

def generate_signal(df):
    last = df.iloc[-1]
    signal = "HOLD"
    if last['Close'] > last['EMA200'] and last['MACD'] > last['MACD_Signal']:
        signal = "BUY"
    elif last['Close'] < last['EMA200'] and last['MACD'] < last['MACD_Signal']:
        signal = "SELL"
    return signal
