import pandas as pd
import numpy as np

def calculate_indicators(df):
    # EMA 200
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # OBV
    df['OBV'] = (np.sign(df['Close'] - df['Open']) * df['Volume']).cumsum()
    
    return df

def generate_signal(df):
    last = df.iloc[-1]
    if last['MACD'] > last['Signal_Line'] and last['Close'] > last['EMA200']:
        return "BUY"
    elif last['MACD'] < last['Signal_Line'] and last['Close'] < last['EMA200']:
        return "SELL"
    else:
        return "HOLD"
