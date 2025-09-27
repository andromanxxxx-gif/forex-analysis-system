# dashboard/app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from streamlit_autorefresh import st_autorefresh

# --- Trading signals functions ---
def calculate_indicators(df):
    """Hitung EMA200, MACD, OBV"""
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # OBV
    df['Direction'] = 0
    df.loc[df['Close'] > df['Open'], 'Direction'] = 1
    df.loc[df['Close'] < df['Open'], 'Direction'] = -1
    df['OBV'] = (df['Volume'] * df['Direction']).cumsum()
    
    return df

def generate_signal(df):
    """Generate trading signal berdasarkan indikator"""
    df['Signal'] = 'Hold'
    df.loc[(df['MACD'] > df['Signal']) & (df['Close'] > df['EMA200']), 'Signal'] = 'Buy'
    df.loc[(df['MACD'] < df['Signal']) & (df['Close'] < df['EMA200']), 'Signal'] = 'Sell'
    
    # TP / SL sederhana (1% dari Close)
    df['TP'] = df['Close'] * (1.01)
    df['SL'] = df['Close'] * (0.99)
    
    return df

def add_ai_prediction(df):
    """Tambahkan AI prediction dummy (simulasi)"""
    df['AI_Pred'] = df['Signal']  # Bisa diganti model AI nyata
    return df

# --- Streamlit Dashboard ---
st.set_page_config(page_title="Forex Analysis Dashboard", layout="wide")
st.title("💹 Forex Analysis Dashboard")

# Pilihan pair
pair = st.selectbox("Pilih Pair", ["EURUSD=X", "GBPUSD=X", "USDJPY=X"])
refresh_interval = st.slider("Auto-refresh interval (detik)", 10, 300, 60)

# Auto-refresh
count = st_autorefresh(interval=refresh_interval*1000, limit=None, key="auto_refresh")

# Ambil data
data = yf.download(pair, period="6mo", interval="1h")
if data.empty:
    st.warning("Data tidak tersedia. Coba pair lain.")
    st.stop()
data = data.reset_index()

# Hitung indikator dan sinyal
data = calculate_indicators(data)
data = generate_signal(data)
data = add_ai_prediction(data)

# --- Candlestick Chart + EMA200 + AI Prediction ---
fig_candle = go.Figure()
fig_candle.add_trace(go.Candlestick(
    x=data['Datetime'], open=data['Open'], high=data['High'],
    low=data['Low'], close=data['Close'], name='Candlestick'
))
fig_candle.add_trace(go.Scatter(
    x=data['Datetime'], y=data['EMA200'], line=dict(color='orange', width=2), name='EMA200'
))
colors = {'Buy':'green','Sell':'red','Hold':'gray'}
fig_candle.add_trace(go.Scatter(
    x=data['Datetime'], y=data['Close'],
    mode='markers',
    marker=dict(size=8, color=data['AI_Pred'].map(colors)),
    name='AI Prediction'
))
fig_candle.update_layout(title="Candlestick + EMA200 + AI Prediction", xaxis_rangeslider_visible=False)
st.plotly_chart(fig_candle, use_container_width=True)

# --- MACD ---
fig_macd = go.Figure()
fig_macd.add_trace(go.Scatter(x=data['Datetime'], y=data['MACD'], line=dict(color='blue'), name='MACD'))
fig_macd.add_trace(go.Scatter(x=data['Datetime'], y=data['Signal'], line=dict(color='red'), name='Signal'))
fig_macd.update_layout(title="MACD Indicator")
st.plotly_chart(fig_macd, use_container_width=True)

# --- OBV ---
fig_obv = go.Figure()
fig_obv.add_trace(go.Scatter(x=data['Datetime'], y=data['OBV'], line=dict(color='green'), name='OBV'))
fig_obv.update_layout(title="On Balance Volume (OBV)")
st.plotly_chart(fig_obv, use_container_width=True)

# --- Trading Signals Table ---
st.subheader("📊 Trading Signals + AI Prediction")
signals_table = data[['Datetime','Close','Signal','AI_Pred','TP','SL']]
st.dataframe(signals_table)

st.info(f"Data akan auto-refresh setiap {refresh_interval} detik ⏱️")
