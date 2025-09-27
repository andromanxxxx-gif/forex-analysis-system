# src/trading_signals.py
import pandas as pd
import numpy as np

def calculate_indicators(df):
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    # OBV
    df['OBV'] = (np.sign(df['Close'] - df['Open']) * df['Volume']).cumsum()
    return df

def generate_signal(df):
    last = df.iloc[-1]
    signal = "Neutral"
    if last['Close'] > last['EMA200'] and last['MACD'] > last['MACD_signal']:
        signal = "Buy"
    elif last['Close'] < last['EMA200'] and last['MACD'] < last['MACD_signal']:
        signal = "Sell"
    return signal
