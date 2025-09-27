import pandas as pd

def calculate_indicators(df):
    # EMA200
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    
    # MACD
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    
    # OBV
    df["OBV"] = (df["Volume"] * ((df["Close"] - df["Close"].shift(1)) / abs(df["Close"] - df["Close"].shift(1)).fillna(0))).fillna(0).cumsum()
    
    return df

def generate_signal(df):
    """Simple logic: Buy if MACD crosses above signal & Close > EMA200"""
    last = df.iloc[-1]
    prev = df.iloc[-2]
    signal = "Hold"
    tp = None
    sl = None
    
    if prev["MACD"] < prev["MACD_signal"] and last["MACD"] > last["MACD_signal"] and last["Close"] > last["EMA200"]:
        signal = "Buy"
        tp = last["Close"] * 1.01  # contoh TP +1%
        sl = last["Close"] * 0.995  # contoh SL -0.5%
    elif prev["MACD"] > prev["MACD_signal"] and last["MACD"] < last["MACD_signal"] and last["Close"] < last["EMA200"]:
        signal = "Sell"
        tp = last["Close"] * 0.99  # TP -1%
        sl = last["Close"] * 1.005  # SL +0.5%
    
    return signal, round(tp, 5) if tp else None, round(sl, 5) if sl else None
