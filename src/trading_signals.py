import pandas as pd

def calculate_indicators(df):
    # EMA 200
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_MACD'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # OBV
    df['OBV'] = (df['Volume'] * ((df['Close'] - df['Open']).apply(lambda x: 1 if x>0 else (-1 if x<0 else 0)))).cumsum()
    
    return df

def generate_signal(df):
    df['Signal'] = 'Hold'
    df['Take_Profit'] = None
    df['Stop_Loss'] = None
    
    for i in range(1, len(df)):
        if df['MACD'][i] > df['Signal_MACD'][i] and df['Close'][i] > df['EMA200'][i]:
            df.at[i, 'Signal'] = 'Buy'
            df.at[i, 'Take_Profit'] = df['Close'][i] * 1.01
            df.at[i, 'Stop_Loss'] = df['Close'][i] * 0.995
        elif df['MACD'][i] < df['Signal_MACD'][i] and df['Close'][i] < df['EMA200'][i]:
            df.at[i, 'Signal'] = 'Sell'
            df.at[i, 'Take_Profit'] = df['Close'][i] * 0.99
            df.at[i, 'Stop_Loss'] = df['Close'][i] * 1.005
    return df
