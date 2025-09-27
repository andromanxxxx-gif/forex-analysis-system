# src/trading_signals.py
import pandas as pd
import numpy as np

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung indikator teknikal: EMA200, MACD, OBV
    df harus punya kolom: 'Open', 'High', 'Low', 'Close', 'Volume'
    """
    # Pastikan kolom numerik
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype(float)
    
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


def generate_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghasilkan sinyal trading: Buy / Sell / Hold berdasarkan indikator
    """
    df['Signal'] = 'Hold'
    
    # Sinyal EMA200
    df.loc[df['Close'] > df['EMA200'], 'Signal'] = 'Buy'
    df.loc[df['Close'] < df['EMA200'], 'Signal'] = 'Sell'
    
    # Sinyal tambahan MACD crossover
    buy_condition = (df['MACD'] > df['MACD_signal']) & (df['Close'] > df['EMA200'])
    sell_condition = (df['MACD'] < df['MACD_signal']) & (df['Close'] < df['EMA200'])
    
    df.loc[buy_condition, 'Signal'] = 'Buy'
    df.loc[sell_condition, 'Signal'] = 'Sell'
    
    # Tambahkan prediksi TP/SL sederhana
    df['TP'] = df['Close'] * 1.01  # Target profit +1%
    df['SL'] = df['Close'] * 0.99  # Stop loss -1%
    
    return df
